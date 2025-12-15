"""
Chatbot Verbatims Collector - Nueva implementación con API Key
Recopilación y análisis de verbatims usando el chatbot de Iberia con autenticación por API Key
"""

import pandas as pd
import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatbotVerbatimsCollector:
    """
    Recopilador de verbatims usando el chatbot de Iberia (API Key authentication)
    """
    
    def __init__(self, pbi_collector=None, environment: str = "prod", proxy: str = None):
        """
        Initialize the Chatbot Verbatims Collector with API Key authentication
        
        Args:
            pbi_collector: Power BI data collector instance (fallback mode)
            environment: "prod" or "pre" environment
            proxy: Optional proxy URL (e.g., "http://proxy.iberia.es:8080")
                   If None, will check environment variables HTTP_PROXY/HTTPS_PROXY
        """
        self.pbi_collector = pbi_collector
        self.environment = environment.lower()
        
        # Configure API endpoints and credentials based on environment
        # Note: Must use X-API-Key header (not ApiKey) - confirmed working in tests
        # Note: 'local' is treated as 'prod' for API endpoints (local only affects .env loading)
        import os
        
        effective_env = self.environment if self.environment in ["prod", "pre"] else "prod"
        
        if effective_env == "prod":
            self.endpoint = os.getenv("CHATBOT_API_ENDPOINT", "https://nps.chatbot.iberia.es/ibdp/api/question")
            self.api_key = os.getenv("CHATBOT_API_KEY")
        elif effective_env == "pre":
            self.endpoint = os.getenv("CHATBOT_API_ENDPOINT", "https://nps.chatbot.pre.iberia.es/ibdp/api/question")
            self.api_key = os.getenv("CHATBOT_API_KEY")
            # Note: PRE environment requires Iberia proxy configuration
            
        if not self.api_key:
             logger.warning(f"⚠️ Missing CHATBOT_API_KEY for environment {self.environment}")
        
        # Configure proxy
        self.proxy = proxy
        self.proxies = None
        if proxy:
            self.proxies = {
                'http': proxy,
                'https': proxy
            }
            logger.info(f"🔌 Proxy configured: {proxy}")
        else:
            # Check environment variables
            import os
            http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
            https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
            if http_proxy or https_proxy:
                self.proxies = {}
                if http_proxy:
                    self.proxies['http'] = http_proxy
                if https_proxy:
                    self.proxies['https'] = https_proxy
                logger.info(f"🔌 Proxy from environment: {self.proxies}")
        
        # Required User-Agent header (to avoid WAF blocking)
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
        
        logger.info(f"✅ ChatbotVerbatimsCollector initialized ({self.environment.upper()} environment)")
        logger.info(f"🌐 Endpoint: {self.endpoint}")
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test connection to the chatbot API
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Check API Key first
        if not self.api_key:
            return False, "❌ CHATBOT_API_KEY is missing. Cannot connect to Chatbot API."

        try:
            headers = self._get_headers()
            
            # Simple test payload
            test_payload = {
                "value": "¿Qué es NPS?",
                "filters": {
                    "date_flight_local": ["2024-01-01", "2024-01-31"]
                }
            }
            
            logger.info("🔍 Testing chatbot API connection...")
            
            response = requests.post(
                self.endpoint,
                json=test_payload,
                headers=headers,
                proxies=self.proxies,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    if 'jobId' in data or 'answer' in data:
                        return True, f"✅ Chatbot API connected successfully (status {response.status_code})"
                    else:
                        return False, f"⚠️ Unexpected response format: {data}"
                except Exception as e:
                    # Log the actual content that failed to parse
                    error_content = response.text[:500] if response.text else "Empty response"
                    logger.error(f"❌ JSON Decode Error. Content: {error_content}")
                    return False, f"❌ Invalid JSON response: {e}. Content starts with: {error_content[:100]}"
            else:
                return False, f"❌ API returned status {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.ConnectionError:
            # If chatbot fails, check if PBI fallback is available
            if self.pbi_collector:
                return True, "✅ Chatbot API unavailable, using PBI fallback"
            else:
                return False, "❌ Cannot connect to chatbot API and no PBI fallback available"
        except requests.exceptions.Timeout:
            return False, "❌ Connection timeout"
        except Exception as e:
            return False, f"❌ Connection test failed: {str(e)}"
    
    def ask_chatbot_question(
        self,
        question: str,
        start_date: str,
        end_date: str,
        node_path: str,
        filters: Optional[Dict] = None,
        max_wait_time: int = 120
    ) -> Optional[Dict]:
        """
        Ask a question to the chatbot and wait for the answer
        
        Args:
            question: The question to ask
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            node_path: Node path for context (not used in current API)
            filters: Additional filters (cabin, haul, route, nps_category, etc.)
            max_wait_time: Maximum time to wait for answer in seconds
            
        Returns:
            Dictionary with answer data if successful, None otherwise
            Expected keys: 'answer', 'toolOutput', 'jobId', 'sessionId', etc.
        """
        try:
            # Ensure dates are strings (convert datetime objects if needed)
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"🤖 Asking chatbot: {question}")
            logger.info(f"📅 Date range: {start_date} to {end_date}")
            
            headers = self._get_headers()
            
            # Prepare question payload
            payload = {
                "value": question,
                "verbatimCol": "nps_all_t",  # Required field for chatbot API
                "filters": {
                    "date_flight_local": [start_date, end_date]
                }
            }
            
            # Add additional filters if provided
            if filters:
                # Map filter names to chatbot API format (verified with API tests)
                if 'cabin' in filters:
                    payload["filters"]["cabin"] = filters['cabin']
                if 'haul' in filters:
                    payload["filters"]["haul"] = filters['haul']  # LH or SH
                if 'route' in filters:
                    payload["filters"]["route"] = filters['route']
                if 'company' in filters:
                    payload["filters"]["company"] = filters['company']  # IB or YW
                if 'nps_category' in filters:
                    payload["filters"]["nps_category"] = filters['nps_category']
            
            logger.info(f"📦 Filters: {payload['filters']}")
            
            # DEBUG: Print full request details (both logger and stdout)
            import json
            debug_msg = "\n" + "=" * 80 + "\n"
            debug_msg += "🔍 DEBUG: FULL API REQUEST\n"
            debug_msg += "=" * 80 + "\n"
            debug_msg += f"Endpoint: {self.endpoint}\n"
            debug_msg += f"Headers: {headers}\n"
            debug_msg += f"Proxies: {self.proxies}\n"
            debug_msg += "Payload JSON:\n"
            debug_msg += json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            debug_msg += "=" * 80
            
            # Print to both stdout and logger
            print(debug_msg)
            logger.info(debug_msg)
            
            # Submit question
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                proxies=self.proxies,
                timeout=30
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"❌ Question submission failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            # Parse response
            data = response.json()
            job_id = data.get('jobId')
            
            # Check if we got an immediate answer (rare case)
            answer = data.get('answer')
            if answer and answer != "null" and answer is not None and str(answer).strip():
                logger.info(f"✅ Got immediate answer!")
                return data
            
            # If we got a jobId, poll for the answer
            if job_id:
                logger.info(f"✅ Question submitted with jobId: {job_id}")
                return self._wait_for_answer(job_id, headers, max_wait_time)
            else:
                logger.warning("⚠️ No jobId or immediate answer in response")
                return data
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Connection error: Cannot connect to chatbot API")
            return None
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout error: Request timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Error asking chatbot question: {e}")
            return None
    
    def _wait_for_answer(
        self,
        job_id: str,
        headers: Dict,
        max_wait_time: int = 120
    ) -> Optional[Dict]:
        """
        Poll the chatbot API for the answer using the jobId
        
        Args:
            job_id: The jobId from the question submission
            headers: HTTP headers for the request
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Answer data if available, None otherwise
        """
        answer_endpoint = f"{self.endpoint}/{job_id}"
        
        start_time = time.time()
        attempt = 0
        
        logger.info(f"⏳ Waiting for answer (jobId: {job_id}, max {max_wait_time}s)...")
        
        while time.time() - start_time < max_wait_time:
            attempt += 1
            elapsed = int(time.time() - start_time)
            
            logger.info(f"🔍 Polling attempt {attempt} (elapsed: {elapsed}s)...")
                
            try:
                response = requests.get(
                    answer_endpoint,
                    headers=headers,
                    proxies=self.proxies,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for answer
                    answer = data.get('answer')
                    error = data.get('error')
                    
                    # Check for error
                    if error and error != "null" and error is not None:
                        logger.error(f"❌ Chatbot error: {error}")
                        return None
                        
                    # Check if we have a real answer
                    if answer and answer != "null" and answer is not None and str(answer).strip():
                        logger.info(f"✅ Got answer after {elapsed}s ({attempt} attempts)!")
                        return data
                    else:
                        logger.info(f"   ⏳ Still processing...")
                else:
                    logger.warning(f"⚠️ Poll returned status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error polling for answer: {e}")
            
            # Wait before next attempt - increasing delay
            wait_time = min(5 + attempt * 2, 15)
            logger.info(f"   ⏸️  Waiting {wait_time}s...")
            time.sleep(wait_time)
        
        logger.warning(f"⏰ Timeout after {max_wait_time}s waiting for answer")
        return None
    
    def get_verbatims_data(
        self,
        start_date: str,
        end_date: str,
        node_path: str,
        verbatim_type: str = None,
        intelligent_query: str = None
    ) -> pd.DataFrame:
        """
        Get verbatims data - fallback to PBI collector if chatbot is not available
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            node_path: Node path for filtering
            verbatim_type: Type of verbatim (optional)
            intelligent_query: Intelligent query for filtering (optional)
        
        Returns:
            DataFrame with verbatims data
        """
        try:
            logger.info(f"📊 Getting verbatims data for {start_date} to {end_date}")
            
            # For now, always fallback to PBI collector for raw verbatims data
            # The chatbot is better used for Q&A, not for bulk data extraction
            if self.pbi_collector:
                logger.info("🔄 Using PBI collector for verbatims data...")
                
                from datetime import datetime as dt
                start_dt = dt.strptime(start_date, '%Y-%m-%d')
                end_dt = dt.strptime(end_date, '%Y-%m-%d')
                
                df = self.pbi_collector.collect_verbatims_for_date_range(
                    node_path=node_path,
                    start_date=start_dt,
                    end_date=end_dt
                )
                
                if not df.empty:
                    logger.info(f"✅ Got {len(df)} verbatims from PBI")
                    return df
                else:
                    logger.warning("⚠️ No verbatims found in PBI")
                    return pd.DataFrame()
            else:
                logger.warning("⚠️ No PBI collector available for verbatims data")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error getting verbatims data: {e}")
            return pd.DataFrame()
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests
        
        Returns:
            Dictionary with required headers
        """
        return {
            "X-API-Key": self.api_key,  # Must be X-API-Key, not ApiKey
            "User-Agent": self.user_agent,
            "Content-Type": "application/json"
        }
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get information about the API configuration
        
        Returns:
            Dictionary with API configuration info
        """
        return {
            "environment": self.environment,
            "endpoint": self.endpoint,
            "api_key_set": bool(self.api_key),
            "has_pbi_fallback": bool(self.pbi_collector)
        }
