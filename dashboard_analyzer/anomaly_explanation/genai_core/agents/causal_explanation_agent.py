"""
Clean Causal Explanation Agent
==============================

A rewritten agent following the clean separated workflow pattern from the notebook.
Key improvements:
- Clean conversation flow: USER -> AI reflection -> USER instruction -> AI tool call
- For reflections, sends only system prompt + previous explanations (not full history)
- Proper handling of OpenAI tool call format
- Sequential tool execution with proper reflection phases
"""

import logging
import asyncio
import yaml
import pandas as pd
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import time
import re
import traceback
from dotenv import load_dotenv
import importlib.resources

from pydantic import BaseModel, Field

# Core imports
from dashboard_analyzer.anomaly_explanation.genai_core.llms.openai_llm import OpenAiLLM  
from dashboard_analyzer.anomaly_explanation.genai_core.llms.aws_llm import AWSLLM

# Import S3 uploader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from dashboard_analyzer.data_collection.s3_report_uploader import S3ReportUploader
from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import LLMType, MessageType, AgentName, get_default_llm_type, get_agent_conversations_folder
from dashboard_analyzer.anomaly_explanation.genai_core.utils.output_paths import (
    resolve_report_group, format_period_range,
    get_logging_path, get_s3_logging_key, build_execution_metadata, save_pretty_json,
)
from dashboard_analyzer.anomaly_explanation.genai_core.message_history import MessageHistory
from dashboard_analyzer.anomaly_explanation.genai_core.agents.agent import Agent

# Data collection imports
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector, get_touchpoint_display_name
from dashboard_analyzer.data_collection.chatbot_verbatims_collector import ChatbotVerbatimsCollector
from dashboard_analyzer.data_collection.ncs_collector import NCSDataCollector

# Data analysis imports
from dashboard_analyzer.anomaly_explanation.data_analyzer import OperationalDataAnalyzer

# Additional imports needed for tools
import time
import re
from datetime import datetime
import traceback

class CausalAnalysisResult(BaseModel):
    """Structured output for causal analysis results"""
    primary_cause: str = Field(description="Primary identified cause of the anomaly")
    confidence_level: str = Field(description="Confidence level: High/Medium/Low")
    supporting_evidence: List[str] = Field(description="List of evidence supporting the conclusion")
    next_investigation_step: Optional[str] = Field(description="Next tool/data source to investigate, if needed")
    final_explanation: str = Field(description="Complete causal explanation for the anomaly")


class CleanConversationTracker:
    """Track the clean conversation workflow"""
    
    def __init__(self):
        self.conversation_log = []
        self.iteration_count = 0
        self.previous_explanations = []
        self.identified_routes = []  # Nuevos campos para tracking
        self.last_tool_context = None  # Contexto del último tool ejecutado
        self.dax_queries = []  # Track DAX queries executed during investigation
        self.tool_executions = []  # Track structured tool executions with results

    def reset_tracker(self):
        """Reset for new investigation"""
        self.conversation_log = []
        self.iteration_count = 0
        self.previous_explanations = []
        self.identified_routes = []
        self.last_tool_context = None
        self.dax_queries = []
        self.tool_executions = []
        
    def reset(self):
        """Resets the agent's state for a new analysis run."""
        self.anomaly_type = "unknown"
        self.collected_data = {
            "explanatory_drivers": None,
            "ncs": None,
            "routes": None,
            "customer_profile": None,
            "operative_data": None,
            "verbatims": None
        }
        self.conversation_log = []
        self.iteration_count = 0
        self.previous_explanations = []
        self.identified_routes = []
        self.last_tool_context = None
        self.tool_executions = []
        
    def reset(self):
        """Resets the agent's state for a new analysis run."""
        self.anomaly_type = "unknown"
        self.collected_data = {
            "explanatory_drivers": None,
            "ncs": None,
            "routes": None,
            "customer_profile": None,
            "operative_data": None,
            "verbatims": None
        }
        self.conversation_log = []
        self.iteration_count = 0
        self.previous_explanations = []
        self.identified_routes = []
        self.last_tool_context = None
        
    def log_message(self, message_type: str, content: str, metadata: Optional[Dict] = None):
        """Log a message in the conversation"""
        self.conversation_log.append({
            'iteration': self.iteration_count,
            'type': message_type,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    def add_explanation(self, explanation: str):
        """Add an explanation to the clean context"""
        self.previous_explanations.append(explanation)
    
    def get_clean_context(self, system_prompt: str) -> List[Dict]:
        """Get clean context: system prompt + previous explanations only"""
        messages = [{"role": "system", "content": system_prompt}]
        
        for i, explanation in enumerate(self.previous_explanations):
            messages.append({
                "role": "assistant", 
                "content": f"Previous analysis {i+1}: {explanation}"
            })
            
        return messages
    
    def next_iteration(self):
        """Move to next iteration"""
        self.iteration_count += 1

    def set_tool_context(self, tool_name: str, result: str):
        """Set context of last executed tool"""
        self.last_tool_context = {
            'tool_name': tool_name,
            'result': result,
            'iteration': self.iteration_count
        }
        
        # Track structured execution
        self.tool_executions.append({
            'tool_name': tool_name,
            'result': result,
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat()
        })
        
        # Extract routes if mentioned in NCS or verbatims results
        if tool_name in ['ncs_tool', 'verbatims_tool']:
            routes = self._extract_routes_from_result(result)
            if routes:
                self.identified_routes.extend(routes)

    def get_tool_executions(self) -> List[Dict]:
        """Get the structured history of tool executions"""
        return self.tool_executions
    
    def add_dax_query(self, tool_name: str, query: str, parameters: dict = None):
        """Add a DAX query to the tracking list"""
        query_entry = {
            "tool_name": tool_name,
            "query": query,
            "parameters": parameters or {},
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "iteration": self.iteration_count
        }
        self.dax_queries.append(query_entry)
                
    def _extract_routes_from_result(self, result: str) -> List[str]:
        """Extract route identifiers from tool results including Spanish city names"""
        import re
        
        # Common route patterns: XXX-YYY (e.g., MAD-UIO, LAX-MAD)
        route_patterns = [
            r'\b[A-Z]{3}-[A-Z]{3}\b',  # Standard IATA format
            r'from [A-Z]{3} to [A-Z]{3}',  # Narrative format
            r'route [A-Z]{3}-[A-Z]{3}',  # Explicit route mention
        ]
        
        # Spanish city patterns for NCS verbatims
        spanish_city_patterns = [
            r'Madrid-Nueva York',
            r'Madrid-Bogotá',
            r'Madrid-Bogota',  # without accent
            r'Madrid-Montevideo',
            r'Madrid-Santiago',
            r'Madrid-Caracas',
            r'Madrid-Lima',
            r'Madrid-Quito',
            r'Madrid-Buenos Aires',
            r'Madrid-México',
            r'Madrid-Mexico',  # without accent
            r'Barcelona-[\w\s]+',
            r'Bilbao-[\w\s]+',
            r'[\w\s]+-Madrid',  # Routes TO Madrid
            # Generic patterns
            r'ruta Madrid-[\w\s]+',
            r'vuelos? (?:de|desde) Madrid (?:a|hacia) [\w\s]+',
            r'conexión Madrid-[\w\s]+',
        ]
        
        routes = []
        
        # Extract standard IATA routes
        for pattern in route_patterns:
            matches = re.findall(pattern, result, re.IGNORECASE)
            routes.extend(matches)
        
        # Extract Spanish city routes and try to map to IATA
        for pattern in spanish_city_patterns:
            matches = re.findall(pattern, result, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean_match = match.replace('ruta ', '').replace('vuelos de ', '').replace('vuelos desde ', '').replace(' a ', '-').replace(' hacia ', '-').replace('conexión ', '')
                routes.append(clean_match)
        
        # Simple city to IATA mapping for common Spanish cities
        city_mapping = {
            'madrid': 'MAD',
            'nueva york': 'JFK',  # or LGA/EWR
            'bogotá': 'BOG',
            'bogota': 'BOG',
            'montevideo': 'MVD',
            'santiago': 'SCL',
            'caracas': 'CCS',
            'lima': 'LIM',
            'quito': 'UIO',
            'buenos aires': 'EZE',  # or AEP
            'méxico': 'MEX',
            'mexico': 'MEX',
            'barcelona': 'BCN',
            'bilbao': 'BIO'
        }
        
        # Try to convert Spanish routes to IATA
        iata_routes = []
        for route in routes:
            if '-' in route:
                origin, destination = route.split('-', 1)
                origin_clean = origin.lower().strip()
                destination_clean = destination.lower().strip()
                
                origin_iata = city_mapping.get(origin_clean, origin.upper())
                destination_iata = city_mapping.get(destination_clean, destination.upper())
                
                # Only add if it looks like a valid route
                if len(origin_iata) <= 4 and len(destination_iata) <= 4:
                    iata_routes.append(f"{origin_iata}-{destination_iata}")
            else:
                iata_routes.append(route.upper())
        
        # Clean and deduplicate
        cleaned_routes = list(set([route.upper().replace('FROM ', '').replace(' TO ', '-').replace('ROUTE ', '') for route in iata_routes]))
        return [route for route in cleaned_routes if route and '-' in route]
        
    def has_identified_routes(self) -> bool:
        """Check if routes have been identified in previous tools"""
        return len(self.identified_routes) > 0
        
    def should_validate_routes_after_tool(self, tool_name: str) -> bool:
        """Determine if routes_tool should be called after this tool"""
        return (tool_name in ['ncs_tool', 'verbatims_tool'] and self.has_identified_routes())


class CausalExplanationAgent:
    """Clean causal explanation agent with separated workflow"""
    
    def __init__(
        self,
        llm_type: Optional[LLMType] = None,
        config_path: str = "dashboard_analyzer/anomaly_explanation/config/prompts/causal_explanation.yaml",
        logger: Optional[logging.Logger] = None,
        silent_mode: bool = False,
        custom_helper_prompts: Optional[Dict[str, Any]] = None,
        detection_mode: str = "vslast",
        causal_filter: str = "vs L7d",
        comparison_start_date: datetime = None,
        comparison_end_date: datetime = None,
        study_mode: str = "comparative",
        environment: str = "prod",
        reference_date: Optional[datetime] = None,
        focus_touchpoint: Optional[str] = None
    ):
        # Use default LLM type if none provided
        if llm_type is None:
            llm_type = get_default_llm_type()
        self.llm_type = llm_type
        self.config_path = config_path
        self.logger = logger or self._setup_logger()
        self.silent_mode = silent_mode
        self.environment = environment
        self.reference_date = reference_date  # Anchor date for fixed baseline in single mode
        self.focus_touchpoint = focus_touchpoint  # Optional touchpoint to force-investigate
        self.report_group = resolve_report_group(focus_touchpoint)

        # Transform detection_mode if needed (vslast -> vslast_dynamic when causal_filter is "vs Sel. Period")
        if detection_mode == "vslast" and causal_filter == "vs Sel. Period":
            self.detection_mode = "vslast_dynamic"
        else:
            self.detection_mode = detection_mode
        
        # Handle causal_filter conversion from string 'None' to actual None
        if causal_filter == 'None' or causal_filter == 'none':
            self.causal_filter = None
        else:
            self.causal_filter = causal_filter
        
        # Convert string dates to datetime objects if needed
        if isinstance(comparison_start_date, str):
            try:
                self.comparison_start_date = datetime.strptime(comparison_start_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                self.comparison_start_date = comparison_start_date
        else:
            self.comparison_start_date = comparison_start_date
            
        if isinstance(comparison_end_date, str):
            try:
                self.comparison_end_date = datetime.strptime(comparison_end_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                self.comparison_end_date = comparison_end_date
        else:
            self.comparison_end_date = comparison_end_date
        
        # Debug log the filter value
        self.logger.info(f"🔍 DEBUG CAUSAL_FILTER: Original: {causal_filter}, Processed: {self.causal_filter}")
        self.study_mode = study_mode
        
        # Calculate dynamic comparison dates if needed
        if self.causal_filter and self.causal_filter != "vs Sel. Period" and not self.comparison_start_date:
            self.logger.info(f"🔄 Calculating dynamic comparison dates for causal_filter: {self.causal_filter}")
            # We'll calculate them when we have the analysis dates
        
        # Load configuration
        self.config = self._load_prompt_config(config_path)
        if custom_helper_prompts:
            self._merge_helper_prompts(custom_helper_prompts)
        
        # Initialize data collectors
        self.pbi_collector = PBIDataCollector(environment=environment)
        self.chatbot_collector = self._init_chatbot_collector()
        self.ncs_collector = self._init_ncs_collector()
        
        # Initialize S3 uploader with environment
        self.s3_uploader = S3ReportUploader(environment=environment)
        
        # Create LLM and agent
        self.llm = self._create_llm(llm_type)
        self.agent = Agent(llm=self.llm, logger=self.logger)
        
        # Initialize tracker
        self.tracker = CleanConversationTracker()
        
        # Current anomaly context
        self.current_anomaly_type = None
        
        # Store data collected during investigation
        self.collected_data = {}
        
        if not self.silent_mode:
            self.logger.info(f"CausalExplanationAgent initialized with {llm_type.value} (Env: {environment})")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger for the agent."""
        logger = logging.getLogger("causal_explanation")
        logger.setLevel(logging.INFO)  # Restored back to INFO
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
    
    def calculate_dynamic_comparison_dates(self, start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
        """
        Calculate comparison dates dynamically based on causal_filter
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Tuple of (comparison_start_date, comparison_end_date)
        """
        self.logger.info(f"🗓️ calculate_dynamic_comparison_dates called:")
        self.logger.info(f"  Input dates: {start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date} → {end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date}")
        self.logger.info(f"  causal_filter: {self.causal_filter}")

        if not self.causal_filter or self.causal_filter == "vs Sel. Period":
            if self.comparison_start_date is None or self.comparison_end_date is None:
                self.logger.warning(
                    f"⚠️ causal_filter='vs Sel. Period' but comparison_start_date={self.comparison_start_date} "
                    f"and/or comparison_end_date={self.comparison_end_date} are None. "
                    f"No se pueden calcular fechas de comparación."
                )
            else:
                comp_start_str = self.comparison_start_date.strftime('%Y-%m-%d') if hasattr(self.comparison_start_date, 'strftime') else self.comparison_start_date
                comp_end_str = self.comparison_end_date.strftime('%Y-%m-%d') if hasattr(self.comparison_end_date, 'strftime') else self.comparison_end_date
                self.logger.info(f"  → Using pre-configured dates: {comp_start_str} → {comp_end_str}")
            return self.comparison_start_date, self.comparison_end_date
        
        # Calculate the period length
        period_days = (end_date - start_date).days + 1
        
        # Extract time period from causal_filter using regex
        filter_lower = self.causal_filter.lower()
        
        if "last week" in filter_lower or "lw" in filter_lower or "l7d" in filter_lower:
            # vs Last Week / vs L7d
            comp_start = start_date - timedelta(days=7)
            comp_end = end_date - timedelta(days=7)
        elif "last month" in filter_lower or "lm" in filter_lower or "l30d" in filter_lower:
            # vs Last Month / vs LM / vs L30d
            comp_start = start_date - timedelta(days=30)
            comp_end = end_date - timedelta(days=30)
        elif "same week last year" in filter_lower or "ly" in filter_lower:
            # vs Same Week Last Year / vs LY
            comp_start = start_date - timedelta(days=365)
            comp_end = end_date - timedelta(days=365)
        elif "last quarter" in filter_lower or "lq" in filter_lower:
            # vs Last Quarter / vs LQ
            comp_start = start_date - timedelta(days=90)
            comp_end = end_date - timedelta(days=90)
        elif re.search(r'l(\d+)d', filter_lower):
            # vs L[N]d (e.g., vs L14d, vs L21d)
            match = re.search(r'l(\d+)d', filter_lower)
            days = int(match.group(1))
            comp_start = start_date - timedelta(days=days)
            comp_end = end_date - timedelta(days=days)
        else:
            # Default: assume last week
            self.logger.warning(f"⚠️ Unknown causal_filter pattern: {self.causal_filter}, defaulting to last week")
            comp_start = start_date - timedelta(days=7)
            comp_end = end_date - timedelta(days=7)
        
        self.logger.info(f"🔄 Dynamic calculation for '{self.causal_filter}':")
        self.logger.info(f"  Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"  Comparison period: {comp_start.strftime('%Y-%m-%d')} to {comp_end.strftime('%Y-%m-%d')}")
        
        return comp_start, comp_end

    def _validate_analysis_dates(self, start_date, end_date, context: str = "") -> None:
        """
        Valida que las fechas de análisis sean coherentes y no futuras.
        Lanza DateValidationError si la validación falla.
        """
        from dashboard_analyzer.data_collection.chatbot_verbatims_collector import DateValidationError
        today = datetime.now().date()

        if isinstance(start_date, str):
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start = start_date.date() if hasattr(start_date, 'date') else start_date

        if isinstance(end_date, str):
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end = end_date.date() if hasattr(end_date, 'date') else end_date

        if start > end:
            msg = f"[{context}] start_date ({start_date}) es posterior a end_date ({end_date})"
            self.logger.error(f"❌ DATE VALIDATION ERROR: {msg}")
            raise DateValidationError(msg)
        if start > today:
            msg = (f"[{context}] start_date ({start_date}) es posterior a hoy ({today}). "
                   f"Posible error de propagación de fechas.")
            self.logger.error(f"❌ DATE VALIDATION ERROR: {msg}")
            raise DateValidationError(msg)
        if end > today:
            msg = (f"[{context}] end_date ({end_date}) es posterior a hoy ({today}). "
                   f"Posible error de propagación de fechas.")
            self.logger.error(f"❌ DATE VALIDATION ERROR: {msg}")
            raise DateValidationError(msg)

    def _load_prompt_config(self, config_path: str) -> Dict[str, Any]:
        """Load prompt configuration using importlib.resources for package support"""
        try:
            # Try loading as package resource first (preferred for installed package)
            filename = Path(config_path).name
            package_path = "dashboard_analyzer.anomaly_explanation.config.prompts"
            
            try:
                # Python 3.9+ style
                ref = importlib.resources.files(package_path) / filename
                with ref.open('r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from package resource: {package_path}/{filename}")
                return config
            except (ImportError, FileNotFoundError, TypeError) as e:
                # Fallback to direct file path (development mode)
                self.logger.debug(f"Could not load from package ({e}), trying file path: {config_path}")
                
                full_path = Path(config_path)
                if not full_path.exists():
                    # Try relative to workspace root if not absolute
                    full_path = Path("/workspace") / config_path
                    if not full_path.exists():
                        # Try relative to current working directory
                        full_path = Path.cwd() / config_path
                
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    self.logger.info(f"Loaded configuration from file: {full_path}")
                    return config
                else:
                    raise FileNotFoundError(f"Config file not found at {config_path} or in package resources")

        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            # Fallback configuration (minimal - should use YAML)
            default_config = {
            'investigation_flow': {
                    'max_iterations': 5,
                    'start_tool': 'explanatory_drivers_tool'
                }
            }
            return default_config
    
    # Removed _create_default_config() method - all configuration now comes from YAML file
    
    def _merge_helper_prompts(self, custom_prompts: Dict[str, Any]):
        """Merge custom helper prompts"""
        if 'helper_prompts' in custom_prompts:
            self.config['helper_prompts'].update(custom_prompts['helper_prompts'])
    
    def _init_chatbot_collector(self):
        """Initialize chatbot collector with simple token management"""
        try:
            self.logger.info("🔄 Initializing ChatbotVerbatimsCollector...")
            return ChatbotVerbatimsCollector(
                pbi_collector=self.pbi_collector,
                environment=self.environment
            )
                
        except Exception as e:
            self.logger.error(f"Failed to initialize chatbot collector: {e}")
            return None
    
    def _load_chatbot_token(self) -> Optional[str]:
        """Load chatbot token from environment or file"""
        # Try environment variable first
        token = os.getenv("CHATBOT_API_TOKEN")
        if token:
            return token
        
        # Try loading from temp_aws_credentials.env file in current working directory
        try:
            temp_creds_file = Path.cwd() / "temp_aws_credentials.env"
            if temp_creds_file.exists():
                with open(temp_creds_file, 'r') as f:
                    for line in f:
                        if line.startswith('chatbot_jwt_token ='):
                            token = line.split('=', 1)[1].strip().strip('"\'')
                            self.logger.info("✅ Chatbot token loaded from temp_aws_credentials.env")
                            return token
        except Exception as e:
            self.logger.debug(f"Could not load token from temp_aws_credentials.env: {e}")
        
        # Try loading from .env file in current working directory (fallback)
        try:
            token_file_path = Path.cwd() / ".env"
            if token_file_path.exists():
                with open(token_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('CHATBOT_API_TOKEN='):
                            token = line.split('=', 1)[1].strip().strip('"\'')
                            self.logger.info("✅ Chatbot token loaded from .env file")
                            return token
        except Exception as e:
            self.logger.debug(f"Could not load token from file: {e}")
        
        self.logger.warning("❌ No chatbot token found in environment variables or files")
        return None
    
    def _init_ncs_collector(self):
        """Initialize NCS collector using agent's environment setting"""
        try:
            # Use self.environment to ensure consistency - credentials handled by unified resolver
            collector = NCSDataCollector(environment=self.environment)
            self.logger.info(f"✅ NCS collector initialized with {self.environment} environment")
            return collector
        except Exception as e:
            self.logger.error(f"Error initializing NCS collector: {e}")
            self.logger.warning(f"Using fallback NCS collector (mode: {self.environment})")
            return NCSDataCollector(environment=self.environment)
    
    def _create_llm(self, llm_type: LLMType):
        """Create LLM instance"""
        if llm_type in [LLMType.GPT4o, LLMType.O3, LLMType.O3_MINI, LLMType.O4_MINI, LLMType.GPT_5_2]:
            return self._create_openai_llm(llm_type)
        else:
            return self._create_aws_llm(llm_type)
    
    def _create_openai_llm(self, llm_type: LLMType) -> OpenAiLLM:
        """Create OpenAI/Azure OpenAI LLM instance using unified credential strategy."""
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.openai_session import (
            get_openai_credentials, is_azure_openai_configured, is_openai_platform_configured
        )
        
        # Get credentials based on current environment (local vs prod)
        creds = get_openai_credentials(environment=self.environment)
        
        is_azure = is_azure_openai_configured(creds)
        is_platform = is_openai_platform_configured(creds)
        
        if not is_azure and not is_platform:
            raise ValueError("No OpenAI or Azure OpenAI credentials found in environment variables")
        
        # Priority: Azure (to maintain legacy) if configured, otherwise OpenAI Platform
        if is_azure:
            return OpenAiLLM(
                llm_type=llm_type,
                api_key=creds['azure_api_key'] or "",
                api_base=creds['azure_endpoint'] or "",
                api_version=creds['azure_api_version'] or "2024-12-01-preview",
                api_dep_gpt=creds['azure_deployment'] or "",
                temperature=1.0  # Default temperature for O4-MINI compatibility
            )
        else:
            return OpenAiLLM(
                llm_type=llm_type,
                api_key=creds['openai_api_key'] or "",
                project_id=creds['openai_project_id'],
                temperature=1.0
            )
    
    def _create_aws_llm(self, llm_type: LLMType) -> AWSLLM:
        """Create AWS Bedrock LLM instance using unified credential strategy."""
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.aws_session import get_aws_credentials
        
        # Get credentials based on current environment (local vs prod)
        creds = get_aws_credentials(environment=self.environment)
        
        return AWSLLM(
            llm_type=llm_type,
            region_name=creds['region_name'],
            aws_access_key_id=creds['aws_access_key_id'],
            aws_secret_access_key=creds['aws_secret_access_key'],
            aws_session_token=creds['aws_session_token'],
            profile_name=os.getenv("AWS_PROFILE"),
            environment=self.environment
        )
    
    # Removed _create_tools method - tools are now implemented directly

    async def _collect_focus_touchpoint_data(
        self,
        node_path: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Collect CSAT and target data for the focus touchpoint without the explanatory_drivers filter.
        Returns dict with keys: csat, target, gap, satisfaction_diff, shapdiff — or None on failure.
        """
        if not self.focus_touchpoint:
            return None
        try:
            return await self.pbi_collector.collect_focus_touchpoint_csat_vs_target(
                node_path=node_path,
                start_date=start_dt,
                end_date=end_dt,
                touchpoint_name=self.focus_touchpoint,
                comparison_filter=self.causal_filter,
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date,
            )
        except Exception as e:
            self.logger.warning(f"⚠️ _collect_focus_touchpoint_data failed: {e}")
            return None

    async def _explanatory_drivers_tool(self, node_path: str, start_date: str, end_date: str, min_surveys: int = 10) -> str:
        """Tool for analyzing explanatory drivers and SHAP values."""
        try:
            if not self.silent_mode:
                self.logger.info(f"Collecting explanatory drivers for {node_path} from {start_date} to {end_date}")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            # Use the existing explanatory drivers collection method with comparison filter
            print(f"         🔍 DEBUG: Causal agent using causal_filter: '{self.causal_filter}'")
            df = await self._collect_explanatory_drivers_with_query_tracking(
                node_path, start_dt, end_dt, 
                comparison_filter=self.causal_filter,
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date
            )
            
            if df.empty:
                return f"No explanatory drivers data found for {node_path} in date range {start_date} to {end_date}"
            
            # Calculate survey count
            survey_count = len(df)
            
            # Analyze SHAP values and satisfaction differences
            analysis_result = []
            analysis_result.append(f"Survey Count: {survey_count} (threshold: {min_surveys})")
            
            # Add statistical reliability warning for low sample sizes
            statistical_warning = ""
            if survey_count < 20:
                statistical_warning = "⚠️ STATISTICAL DISCLAIMER: Sample size < 20 - results subject to high statistical variability"
                analysis_result.append(statistical_warning)
            elif survey_count < min_surveys:
                statistical_warning = "⚠️ CAUTION: Sample size below recommended threshold - results may have increased variability"
                analysis_result.append(statistical_warning)
            
            # Find ALL touchpoints (SHAP values) - no threshold filtering
            # Filter out NPS and NPS Comparative as they are not drivers but period comparisons
            if 'Shapdiff' in df.columns:
                df['Shapdiff'] = pd.to_numeric(df['Shapdiff'], errors='coerce')
                
                # Filter out non-driver touchpoints (NPS values are not explanatory drivers)
                candidate_touchpoint_cols = [
                    'TouchPoint_Master[filtered_name]',
                    'TouchPoint_Master[filtered_name',
                    'filtered_name',
                    'Filtered_name',
                    'TouchPoint_Master filtered_name'
                ]
                touchpoint_col = next((c for c in candidate_touchpoint_cols if c in df.columns), None)
                if touchpoint_col:
                    # Filter out NPS comparison values that are not real touchpoints
                    df_filtered = df[~df[touchpoint_col].isin(['NPS', 'NPS Comparative'])]
                else:
                    df_filtered = df
                
                # Get ALL touchpoints, not just significant ones
                all_touchpoints = df_filtered.copy()
                all_touchpoints['Shapdiff'] = pd.to_numeric(all_touchpoints['Shapdiff'], errors='coerce')
                all_touchpoints = all_touchpoints.dropna(subset=['Shapdiff'])
                
                if len(all_touchpoints) > 0:
                    analysis_result.append("ALL SHAP drivers found (including non-significant):")
                    
                    # Store data for final summary
                    drivers_data = []
                    
                    # Categorize touchpoints for workflow routing
                    operational_drivers = []
                    product_drivers = []
                    significant_drivers = []
                    
                    # Handle both DataFrame and Series/array types
                    try:
                        # Try DataFrame methods first
                        if hasattr(all_touchpoints, 'iterrows'):
                            # Sort by absolute SHAP value (descending) to show most important first
                            all_touchpoints_sorted = all_touchpoints.sort_values(by='Shapdiff', key=lambda x: abs(x), ascending=False)
                            
                            for i, (_, row) in enumerate(all_touchpoints_sorted.iterrows()):
                                # No limit - show all SHAP values
                                if hasattr(row, 'get') and touchpoint_col and touchpoint_col in row:
                                    touchpoint = row.get(touchpoint_col)
                                else:
                                    candidate_cols_in_row = [c for c in all_touchpoints_sorted.columns if isinstance(row.get(c, None), str)]
                                    touchpoint = row.get(candidate_cols_in_row[0]) if candidate_cols_in_row else 'Unknown'
                                shap_value = row['Shapdiff'] if 'Shapdiff' in row else 0.0
                                sat_diff = row.get('Satisfaction diff', 'N/A') if hasattr(row, 'get') else 'N/A'
                                
                                # Mark significance
                                significance_marker = "⭐" if abs(shap_value) > 0.1 else "⚪"
                                analysis_result.append(f"  {significance_marker} {touchpoint}: SHAP={shap_value:.3f}, Sat_diff={sat_diff}")
                                
                                # Store for data summary
                                drivers_data.append({
                                    'touchpoint': touchpoint,
                                    'shap_value': shap_value,
                                    'satisfaction_diff': sat_diff,
                                    'significant': abs(shap_value) > 0.1
                                })
                            
                                # Categorize touchpoint type
                                if touchpoint and isinstance(touchpoint, str):
                                    touchpoint_lower = touchpoint.lower()
                                    if any(op_keyword in touchpoint_lower for op_keyword in 
                                           ['otp', 'delay', 'punctual', 'baggage', 'misconex', 'mishandling', 'load factor', 'operational']):
                                        operational_drivers.append(touchpoint)
                                        if abs(shap_value) > 0.1:
                                            significant_drivers.append(touchpoint)
                                    else:
                                        product_drivers.append(touchpoint)
                                        if abs(shap_value) > 0.1:
                                            significant_drivers.append(touchpoint)
                        else:
                            # Fallback for array-like objects
                            analysis_result.append("Found drivers (limited analysis due to data format)")
                    except Exception as e:
                        analysis_result.append(f"Error processing touchpoints: {str(e)}")
                        self.logger.warning(f"Error processing touchpoints: {e}")
                    
                    # Store collected data
                    self.collected_data['explanatory_drivers'] = {
                        'survey_count': survey_count,
                        'all_drivers': drivers_data,
                        'significant_drivers': significant_drivers,
                        'operational_drivers': operational_drivers,
                        'product_drivers': product_drivers
                    }
                    
                    # Intelligent workflow routing based on significant drivers
                    if significant_drivers:
                        if operational_drivers and product_drivers:
                            analysis_result.append(f"MIXED SIGNIFICANT DRIVERS DETECTED: Operational={operational_drivers}, Product={product_drivers}")
                            analysis_result.append("RECOMMENDATION: Start with operative_data_tool for operational validation, then use verbatims_tool for product insights")
                        elif operational_drivers:
                            analysis_result.append(f"OPERATIONAL SIGNIFICANT DRIVERS DETECTED: {operational_drivers}")
                            analysis_result.append("RECOMMENDATION: Use OPERATIVE WORKFLOW → operative_data_tool to validate operational causes")
                        elif product_drivers:
                            analysis_result.append(f"PRODUCT/SERVICE SIGNIFICANT DRIVERS DETECTED: {product_drivers}")
                            analysis_result.append("RECOMMENDATION: Use PRODUCT WORKFLOW → verbatims_tool to extract concrete service issues")
                    else:
                        analysis_result.append("No significant SHAP drivers found (threshold: 0.1) - showing all drivers for context")
                        analysis_result.append("RECOMMENDATION: Use verbatims_tool for qualitative insights and patterns")
                else:
                    analysis_result.append("No SHAP drivers found at all")
                    analysis_result.append("RECOMMENDATION: Use verbatims_tool for qualitative insights and patterns")
            
            # --- FOCUS TOUCHPOINT LOGIC ---
            if self.focus_touchpoint:
                candidate_touchpoint_cols = [
                    'TouchPoint_Master[filtered_name]',
                    'TouchPoint_Master[filtered_name',
                    'filtered_name',
                    'Filtered_name',
                    'TouchPoint_Master filtered_name'
                ]
                touchpoint_col = next((c for c in candidate_touchpoint_cols if c in df.columns), None)

                focus_in_results = (
                    touchpoint_col is not None
                    and self.focus_touchpoint in df[touchpoint_col].values
                )

                if focus_in_results:
                    # Mark existing row with 🎯 FOCUS prefix
                    df.loc[df[touchpoint_col] == self.focus_touchpoint, touchpoint_col] = (
                        f"🎯 FOCUS: {self.focus_touchpoint}"
                    )
                    analysis_result.append(f"🎯 FOCUS TOUCHPOINT '{self.focus_touchpoint}' found in normal drivers — marked.")
                    # Also fetch CSAT vs target (different from Sat_diff which is vs causal_filter)
                    focus_row = await self._collect_focus_touchpoint_data(
                        node_path,
                        start_dt if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d'),
                        end_dt if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d'),
                    )
                    if focus_row is not None:
                        analysis_result.append(
                            f"🎯 FOCUS TOUCHPOINT CSAT (vs target): "
                            f"CSAT={focus_row.get('csat')}, "
                            f"Target_diff={focus_row.get('satisfaction_diff')} pts "
                            f"(⚠️ este diff es vs TARGET, NO vs {self.causal_filter}. "
                            f"El Sat_diff de arriba es vs {self.causal_filter}.)"
                        )
                else:
                    # Query additional data without explanatory_drivers filter
                    focus_row = await self._collect_focus_touchpoint_data(
                        node_path,
                        start_dt if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d'),
                        end_dt if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d'),
                    )
                    if focus_row is not None and touchpoint_col:
                        new_row = {col: None for col in df.columns}
                        new_row[touchpoint_col] = f"🎯 FOCUS: {self.focus_touchpoint}"
                        if 'Satisfaction diff' in df.columns:
                            new_row['Satisfaction diff'] = focus_row.get('satisfaction_diff')
                        if 'Satisfaction' in df.columns:
                            new_row['Satisfaction'] = focus_row.get('csat')
                        if 'Shapdiff' in df.columns:
                            new_row['Shapdiff'] = focus_row.get('shapdiff')
                        if 'NPS diff' in df.columns:
                            new_row['NPS diff'] = None
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        analysis_result.append(
                            f"🎯 FOCUS TOUCHPOINT '{self.focus_touchpoint}' added: "
                            f"CSAT={focus_row.get('csat')}, "
                            f"Sat_diff={focus_row.get('satisfaction_diff')}, "
                            f"SHAP={focus_row.get('shapdiff')}"
                        )
                    else:
                        no_data_label = f"🎯 FOCUS: {self.focus_touchpoint} (sin datos)"
                        if touchpoint_col:
                            new_row = {col: None for col in df.columns}
                            new_row[touchpoint_col] = no_data_label
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        analysis_result.append(f"🎯 FOCUS TOUCHPOINT '{self.focus_touchpoint}': sin datos disponibles.")
            # --- END FOCUS TOUCHPOINT LOGIC ---

            return " | ".join(analysis_result)
            
        except Exception as e:
            return f"Error in explanatory drivers analysis: {str(e)}"


    
    def _get_single_period_system_prompt(self) -> str:
        """DEPRECATED: Use _get_system_prompt(mode='single') instead"""
        return self._get_system_prompt(mode="single")
    
    def _get_single_period_input_template(self) -> str:
        """DEPRECATED: Use _get_input_template(mode='single') instead"""
        return self._get_input_template(mode="single")
    
    def _get_single_period_tool_result_message(self) -> str:
        """DEPRECATED: Use _get_tool_result_message(mode='single') instead"""
        return self._get_tool_result_message(mode="single")
    

    async def _execute_single_period_tool(
        self, 
        tool_name: str, 
        node_path: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        iteration: int,
        comparison_context: str = "",
        baseline_periods: int = 7,
        anomaly_detection_mode: str = "target",
        aggregation_days: int = 7
    ) -> str:
        """Execute a tool for single period analysis"""
        try:
            # Ensure dates are handled correctly (convert to str/datetime as needed)
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                end_date_str = end_date
            else:
                end_dt = end_date
                end_date_str = end_date.strftime('%Y-%m-%d')
                
            if isinstance(start_date, str):
                start_date_str = start_date
            else:
                start_date_str = start_date.strftime('%Y-%m-%d')

            if tool_name == "operative_data_tool":
                # Use reference_date as the anchor for the fixed baseline if provided,
                # otherwise fall back to the end date of the specific period being analyzed
                anchor_dt = self.reference_date if self.reference_date else end_dt
                baseline_start_dt = anchor_dt - timedelta(days=aggregation_days * baseline_periods)
                
                return await self._operative_data_tool_single_period(
                    node_path=node_path,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    baseline_start_date=baseline_start_dt.strftime('%Y-%m-%d'),
                    comparison_context=comparison_context,
                    baseline_periods=baseline_periods,
                    anomaly_detection_mode=anomaly_detection_mode,
                    aggregation_days=aggregation_days
                )
            elif tool_name == "ncs_tool":
                return await self._ncs_tool_single_period(
                    node_path=node_path,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            elif tool_name == "routes_tool":
                return await self._routes_tool_single_period(
                    node_path=node_path,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    min_surveys=1,  # Process all routes, just show survey count
                    anomaly_type=getattr(self, 'current_anomaly_type', 'unknown')
                )
            elif tool_name == "verbatims_tool":
                return await self._verbatims_tool_single_period(
                    node_path=node_path,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            elif tool_name == "customer_profile_tool":
                return await self._customer_profile_tool(
                    node_path=node_path,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    min_surveys=1,  # Process all profiles, just show survey count
                    mode="single"
                )
            else:
                return f"Unknown tool for single period: {tool_name}"
            
        except Exception as e:
            self.logger.error(f"❌ Error executing {tool_name} for single period: {type(e).__name__}: {str(e)}")
            return f"ERROR executing {tool_name} for single period: {type(e).__name__}: {str(e)}"
    
    async def _operative_data_tool_single_period(self, node_path: str, start_date: str, end_date: str, baseline_start_date: str, comparison_context: str = "", baseline_periods: int = 7, anomaly_detection_mode: str = "target", aggregation_days: int = 7) -> str:
        """Operative data tool for single period analysis (absolute values + correlation analysis)"""
        try:
            self.logger.info(f"Collecting operative data for {node_path} from {baseline_start_date} to {end_date} for single period analysis")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            if isinstance(baseline_start_date, str):
                baseline_start_dt = datetime.strptime(baseline_start_date, '%Y-%m-%d')
            else:
                baseline_start_dt = baseline_start_date

            # Get operative data for the entire range (baseline + target day)
            # Use aggregation_days from parameters to respect the analysis flow
            operative_data = await self._collect_operative_data_with_query_tracking(
                node_path=node_path,
                target_date=end_dt,
                comparison_days=aggregation_days, # Use the aggregation from flow
                comparison_start_date=baseline_start_dt, # Start from the beginning of the baseline
                use_flexible=True
            )
            
            if operative_data.empty:
                return f"No operative data found for {node_path} in range {baseline_start_date} to {end_date}"
            
            # Pass the aggregation_days to the analyzer
            return await self._operative_data_tool_correlation_analysis(node_path, operative_data, end_date, comparison_context, baseline_periods, anomaly_detection_mode, aggregation_days)
        
        except Exception as e:
            self.logger.error(f"❌ Error in single period operative data tool: {type(e).__name__}: {str(e)}")
            return f"ERROR in operative data tool: {type(e).__name__}: {str(e)}"
    
    async def _collect_operative_data_with_query_tracking(self, node_path: str, target_date: datetime, comparison_days: int = 7, use_flexible: bool = True, comparison_start_date: datetime = None, comparison_end_date: datetime = None, current_start_date: datetime = None) -> pd.DataFrame:
        """Wrapper to collect operative data while tracking DAX queries
        
        Args:
            node_path: Node path for analysis
            target_date: End date for the current period (analysis end date)
            comparison_days: Number of days for data collection (used only if current_start_date not provided)
            use_flexible: Whether to use flexible aggregation query
            comparison_start_date: Start date of the comparison period
            comparison_end_date: End date of the comparison period
            current_start_date: Start date of the current period (if not provided, calculated from comparison_days)
        """
        try:
            # Get filters for this node
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Generate the appropriate operative query
            if use_flexible:
                # Check if we have specific comparison dates (either passed as parameters or from instance variables)
                comp_start = comparison_start_date or (self.comparison_start_date if hasattr(self, 'comparison_start_date') else None)
                comp_end = comparison_end_date or (self.comparison_end_date if hasattr(self, 'comparison_end_date') else None)
                
                if comp_start and comp_end:
                    # Use the provided current_start_date if available, otherwise calculate from comparison_days
                    if current_start_date:
                        start_dt = current_start_date
                        self.logger.info(f"🎯 Using provided current_start_date: {start_dt.strftime('%Y-%m-%d')}")
                    else:
                        start_dt = target_date - timedelta(days=comparison_days - 1)
                        self.logger.warning(f"⚠️ current_start_date not provided, calculating from comparison_days: {start_dt.strftime('%Y-%m-%d')}")
                    
                    self.logger.info(f"🎯 USING vs Sel. Period operative query!")
                    query = self.pbi_collector._get_operative_vs_sel_period_query(
                        cabins, companies, hauls,
                        start_dt, target_date,  # Current period
                        comp_start, comp_end  # Comparison period
                    )
                    query_type = "vs_sel_period_operative"
                    self.logger.info(f"🎯 Using simplified vs Sel. Period operative query")
                    self.logger.info(f"  📅 Current period: {start_dt.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}")
                    self.logger.info(f"  📅 Comparison period: {comp_start.strftime('%Y-%m-%d')} to {comp_end.strftime('%Y-%m-%d')}")
                else:
                    # Use regular flexible query
                    aggregation_days = comparison_days if comparison_days > 1 else 1
                    query = self.pbi_collector._get_flexible_operative_query(aggregation_days, cabins, companies, hauls, target_date)
                    query_type = "flexible_operative"
            else:
                query = self.pbi_collector._get_operative_query(cabins, companies, hauls, target_date, comparison_days)
                query_type = "legacy_operative"
            
            # Track the DAX query
            parameters = {
                "node_path": node_path,
                "target_date": target_date.strftime('%Y-%m-%d'),
                "comparison_days": comparison_days,
                "use_flexible": use_flexible,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("operative_data_tool", query, parameters)
            
            # Execute the query
            df = self.pbi_collector._execute_query(query)
            
            if not df.empty:
                # Clean column names
                df.columns = [col.strip('[]') for col in df.columns]
                
                # Convert date column to datetime
                if 'Date_Master[Date' in df.columns:
                    df.rename(columns={'Date_Master[Date': 'Date_Master'}, inplace=True)
                
                if 'Date_Master' in df.columns:
                    df['Date_Master'] = pd.to_datetime(df['Date_Master']).dt.date
                
                self.logger.info(f"✅ Collected {len(df)} days of operational data for analysis")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting operative data with query tracking: {type(e).__name__}: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_routes_with_query_tracking(self, node_path: str, start_date: datetime, end_date: datetime, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None) -> pd.DataFrame:
        """Wrapper to collect routes data while tracking DAX queries"""
        try:
            # Get filters for this node
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Generate the routes query for the date range
            query = self.pbi_collector._get_routes_range_query(cabins, companies, hauls, start_date, end_date, comparison_filter, comparison_start_date, comparison_end_date)
            
            # Track the DAX query
            parameters = {
                "node_path": node_path,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "comparison_filter": comparison_filter,
                "comparison_start_date": comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date else None,
                "comparison_end_date": comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date else None,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("routes_tool", query, parameters)
            
            # Execute the query
            df = self.pbi_collector._execute_query(query)
            
            if not df.empty:
                # Clean column names safely
                df = self.pbi_collector._safe_clean_columns(df)
                self.logger.info(f"✅ Collected {len(df)} routes for analysis (filter: {comparison_filter})")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting routes data with query tracking: {type(e).__name__}: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_customer_profile_with_query_tracking(self, node_path: str, start_date: datetime, end_date: datetime, dimension: str, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None, route_filter: List[str] = None) -> pd.DataFrame:
        """Wrapper to collect customer profile data while tracking DAX queries"""
        try:
            # Get filters for this node
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Generate the customer profile query (fix parameter order)
            query = self.pbi_collector._get_customer_profile_range_query(
                cabins, companies, hauls, start_date, end_date, dimension, 
                route_filter, comparison_filter, comparison_start_date, comparison_end_date
            )
            
            # Track the DAX query
            parameters = {
                "node_path": node_path,
                "start_date": start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date),
                "end_date": end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else str(end_date),
                "dimension": dimension,
                "comparison_filter": comparison_filter,
                "comparison_start_date": (comparison_start_date.strftime('%Y-%m-%d') if isinstance(comparison_start_date, datetime) else str(comparison_start_date)) if comparison_start_date else None,
                "comparison_end_date": (comparison_end_date.strftime('%Y-%m-%d') if isinstance(comparison_end_date, datetime) else str(comparison_end_date)) if comparison_end_date else None,
                "route_filter": route_filter,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("customer_profile_tool", query, parameters)
            
            # Execute the query and return results using datetime objects (as expected by pbi_collector)
            return await self.pbi_collector.collect_customer_profile_for_date_range(
                node_path, start_date, end_date, dimension, 
                comparison_filter, comparison_start_date, comparison_end_date, route_filter
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting customer profile data with query tracking: {type(e).__name__}: {str(e)}")
            return pd.DataFrame()
    
    async def _collect_explanatory_drivers_with_query_tracking(self, node_path: str, start_date: datetime, end_date: datetime, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None) -> pd.DataFrame:
        """Wrapper to collect explanatory drivers data while tracking DAX queries"""
        try:
            # Get filters for this node
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Generate the explanatory drivers query
            query = self.pbi_collector._get_explanatory_drivers_range_query(
                start_date, end_date, comparison_filter, comparison_start_date, comparison_end_date, cabins, companies, hauls
            )
            
            # Track the DAX query
            parameters = {
                "node_path": node_path,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "comparison_filter": comparison_filter,
                "comparison_start_date": comparison_start_date.strftime('%Y-%m-%d') if comparison_start_date else None,
                "comparison_end_date": comparison_end_date.strftime('%Y-%m-%d') if comparison_end_date else None,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("explanatory_drivers_tool", query, parameters)
            
            # Execute the query
            df = self.pbi_collector._execute_query(query)
            
            if not df.empty:
                # Clean column names safely
                df = self.pbi_collector._safe_clean_columns(df)
                self.logger.info(f"✅ Collected {len(df)} explanatory drivers records")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting explanatory drivers data with query tracking: {type(e).__name__}: {str(e)}")
            return pd.DataFrame()
    
    def _collect_verbatims_with_query_tracking(self, node_path: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Wrapper to collect verbatims data while tracking DAX queries"""
        try:
            # Get filters for this node
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Generate the verbatims query
            query = self.pbi_collector._get_verbatims_range_query(cabins, companies, hauls, start_date, end_date)
            
            # Track the DAX query
            parameters = {
                "node_path": node_path,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("verbatims_tool_fallback", query, parameters)
            
            # Execute the query
            df = self.pbi_collector._execute_query(query)
            
            if not df.empty:
                # Clean column names safely
                df = self.pbi_collector._safe_clean_columns(df)
                self.logger.info(f"✅ Collected {len(df)} verbatims records")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting verbatims data with query tracking: {type(e).__name__}: {str(e)}")
            return pd.DataFrame()
    
    async def _operative_data_tool_correlation_analysis(self, node_path: str, operative_data: pd.DataFrame, target_date_str: str, comparison_context: str = "", baseline_periods: int = 7, anomaly_detection_mode: str = "target", aggregation_days: int = 7) -> str:
        """Operative data tool with correlation analysis using OperationalDataAnalyzer."""
        try:
            self.logger.info(f"🔍 Starting correlation analysis for {node_path} on {target_date_str}")
            
            # Use OperationalDataAnalyzer for consistent analysis
            analyzer = OperationalDataAnalyzer(
                comparison_mode=anomaly_detection_mode,  # "target", "mean", etc.
                comparison_start_date=None,  # No specific comparison dates in single mode
                comparison_end_date=None,
                aggregation_days=aggregation_days,
                baseline_periods=baseline_periods
            )
            
            # Load the data into the analyzer
            analyzer.operative_data[node_path] = operative_data
            
            # Analyze the metrics
            analysis_result = analyzer.analyze_operative_metrics(node_path, target_date_str)
            
            if 'error' in analysis_result:
                self.logger.error(f"❌ Analyzer error: {analysis_result['error']}")
                return f"📊 **DATOS OPERATIVOS - ERROR**\n📅 Fecha: {target_date_str}\n🎯 Segmento: {node_path}\n❌ {analysis_result['error']}"
            
            # Format the results
            result_parts = []
            result_parts.append(f"📊 **DATOS OPERATIVOS - ANÁLISIS CORRELACIÓN**")
            result_parts.append(f"📅 Fecha: {target_date_str}")
            result_parts.append(f"🎯 Segmento: {node_path}")
            result_parts.append("")
            
            if 'metrics' in analysis_result:
                reference_desc = comparison_context if comparison_context else "período de referencia"
                result_parts.append(f"**VALORES ABSOLUTOS Y CORRELACIÓN (Referencia: {reference_desc}):**")
                
                for metric_name, metric_data in analysis_result['metrics'].items():
                    current_value = metric_data.get('current_value', 'N/A')
                    baseline_value = metric_data.get('baseline_value', 'N/A')
                    delta = metric_data.get('delta', 0)
                    direction = metric_data.get('direction', 'unchanged')
                    
                    # Map direction to symbols
                    direction_symbol = "📈" if direction == "increase" else "📉" if direction == "decrease" else "➡️"
                    
                    # Format the metric line
                    if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                        result_parts.append(f"   • **{metric_name}**: {current_value:.1f} (actual) vs {baseline_value:.1f} ({reference_desc}) ({direction_symbol}{delta:+.1f})")
                    else:
                        result_parts.append(f"   • **{metric_name}**: {current_value} (actual) vs {baseline_value} ({reference_desc}) ({direction_symbol})")
                    
                    # Log correlation debug info
                    self.logger.info(f"🔍 DEBUG CORRELATION: metric='{metric_name}', direction='{direction}', delta={delta}")
            
            if 'summary' in analysis_result:
                result_parts.append("")
                result_parts.append("**RESUMEN:**")
                result_parts.append(f"   {analysis_result['summary']}")
            
            result = "\n".join(result_parts)
            self.collected_data['operative_data'] = result
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in correlation analysis operative data tool: {type(e).__name__}: {str(e)}")
            return f"ERROR in correlation analysis operative data tool: {type(e).__name__}: {str(e)}"
    
    async def _ncs_tool_single_period(self, node_path: str, start_date: str, end_date: str) -> str:
        """NCS tool for single period analysis (absolute incidents only)"""
        try:
            self.logger.info(f"Collecting NCS data for {node_path} from {start_date} to {end_date} (single period)")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            # Import and create fresh NCS collector (same as comparative mode)
            from ....data_collection.ncs_collector import NCSDataCollector
            ncs_collector = NCSDataCollector(environment=self.environment)
            
            # Get NCS data for the specific period only
            ncs_data = ncs_collector.collect_ncs_data_for_date_range(
                start_date=start_dt,
                end_date=end_dt
            )
            
            if ncs_data.empty:
                return f"No NCS data found for {node_path} from {start_date} to {end_date}"

            # 2. Apply segment filtering BEFORE analysis
            filtered_ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path)

            if filtered_ncs_data.empty:
                return f"No NCS incidents found for segment {node_path} from {start_date} to {end_date} after filtering"

            # 3. Analyze the FILTERED data
            incident_analysis = self.ncs_collector.analyze_ncs_incidents_for_period(filtered_ncs_data, analysis_focus="all")

            # --- Build the result string from the analysis of filtered data ---
            result_parts = []
            result_parts.append(f"📊 **INCIDENTES NCS - PERIODO ÚNICO**")
            result_parts.append(f"📅 Período: {start_date} a {end_date}")
            result_parts.append(f"🎯 Segmento: {node_path}")
            result_parts.append(f"Total Incidentes (Segmento): {len(filtered_ncs_data)}")
            result_parts.append("")
            result_parts.append("**INCIDENTES DEL PERIODO:**")

            # Count incidents by type from the analysis result
            incident_counts = incident_analysis.get("incident_counts", {})
            for incident_type, count in incident_counts.items():
                result_parts.append(f"• {incident_type}: {count} incidentes")
            
            # Add sample incidents if available
            if "detailed_incidents" in incident_analysis:
                detailed = incident_analysis["detailed_incidents"]
                result_parts.append(f"• Incidentes con descripción detallada: {detailed['count']}")
                
                if detailed.get("sample_incidents"):
                    result_parts.append("• Ejemplos de incidentes:")
                    for i, incident in enumerate(detailed["sample_incidents"][:2], 1):
                        # Truncate long incidents
                        truncated = incident[:150] + "..." if len(incident) > 150 else incident
                        result_parts.append(f"  {i}. {truncated}")
                
                if detailed.get("incident_themes"):
                    result_parts.append("• Temas principales:")
                    for theme in detailed["incident_themes"][:3]:
                        result_parts.append(f"  - {theme}")
            
            # Add route analysis if available
            if "route_analysis" in incident_analysis:
                route_info = incident_analysis["route_analysis"]
                result_parts.append(f"• Rutas afectadas: {route_info['total_routes_affected']}")
                if route_info.get("most_affected_routes"):
                    result_parts.append("• Rutas más impactadas:")
                    for route, count in list(route_info["most_affected_routes"].items())[:3]:
                        result_parts.append(f"  - {route}: {count} incidentes")
            
            # Add summary insights
            if "summary_insights" in incident_analysis:
                result_parts.append("• Resumen de impacto:")
                for insight in incident_analysis["summary_insights"][:3]:
                    result_parts.append(f"  - {insight}")
            
            result_parts.append("")
            result_parts.append("**NOTA:** Estos son incidentes absolutos del período específico, sin comparación temporal.")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error in single period NCS tool: {type(e).__name__}: {str(e)}")
            return f"ERROR in NCS tool: {type(e).__name__}: {str(e)}"
    
    async def _routes_tool_single_period(self, node_path: str, start_date: str, end_date: str, min_surveys: int = 2, anomaly_type: str = "unknown") -> str:
        """Routes tool for single period analysis (absolute NPS values only)"""
        try:
            self.logger.info(f"Collecting routes data for {node_path} from {start_date} to {end_date} (single period)")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            result_parts = []
            result_parts.append(f"📊 **RUTAS - PERIODO ÚNICO**")
            result_parts.append(f"📅 Período: {start_date} a {end_date}")
            result_parts.append(f"🎯 Segmento: {node_path}")
            result_parts.append(f"🔍 Tipo de anomalía: {anomaly_type}")
            result_parts.append("")
            
            # 1. Get routes with NPS data (absolute values only, min_surveys >= 2)
            routes_data = await self._collect_routes_with_query_tracking(
                node_path=node_path,
                start_date=start_dt,
                end_date=end_dt,
                comparison_filter=None  # No comparison for single period
            )
            
            if routes_data.empty:
                result_parts.append("❌ No se encontraron datos de rutas con NPS para este período.")
            else:
                # Clean column names (remove brackets)
                routes_data = self._safe_clean_columns(routes_data, method="replace")
                
                # Find key columns
                route_col = self._find_column(routes_data, ['route'])
                nps_col = self._find_column(routes_data, ['nps'])
                pax_col = self._find_column(routes_data, ['pax', 'n (route)'])
                
                if route_col and nps_col:
                    # Filter by minimum surveys (>= 2)
                    if pax_col:
                        routes_data = routes_data[routes_data[pax_col].fillna(0) >= min_surveys]
                    
                    if not routes_data.empty:
                        # Sort by NPS based on anomaly type
                        if anomaly_type.lower() in ['positive', 'high', 'good']:
                            # For positive anomalies: sort high to low (best routes first)
                            routes_data = routes_data.sort_values(nps_col, ascending=False)
                            sort_direction = "mayor a menor NPS"
                        else:
                            # For negative anomalies: sort low to high (worst routes first)
                            routes_data = routes_data.sort_values(nps_col, ascending=True)
                            sort_direction = "menor a mayor NPS"
                        
                        result_parts.append(f"**RUTAS CON NPS (ordenadas {sort_direction}):**")
            
                        for _, route in routes_data.head(10).iterrows():
                            route_name = route.get(route_col, 'Unknown')
                            nps_raw = route.get(nps_col)
                            nps_value = float(nps_raw) if pd.notna(nps_raw) else 0.0
                            sample_size = int(route.get(pax_col, 0)) if pax_col and pd.notna(route.get(pax_col)) else 0
                        
                            result_parts.append(f"• {route_name}: NPS {nps_value:.1f} (n={sample_size})")
                    else:
                        result_parts.append("❌ No hay rutas con suficientes encuestas (mínimo 2).")
                else:
                    result_parts.append(f"❌ Columnas requeridas no encontradas. Disponibles: {list(routes_data.columns)}")
            
            # 2. Get NCS routes (operational incidents)
            result_parts.append("")
            result_parts.append("**RUTAS CON INCIDENTES OPERACIONALES (NCS):**")
            
            try:
                # Extract segment info from node_path
                segments = node_path.split('/')
                cabins = []
                companies = []
                hauls = []
                
                # Parse node_path to extract segment info
                if 'SH' in segments:
                    hauls.append('SH')
                if 'LH' in segments:
                    hauls.append('LH')
                if 'Economy' in segments:
                    cabins.append('Economy')
                if 'Business' in segments:
                    cabins.append('Business')
                if 'Premium' in segments:
                    cabins.append('Premium')
                if 'IB' in segments:
                    companies.append('IB')
                if 'YW' in segments:
                    companies.append('YW')
                
                # If no specific segments found, use defaults
                if not hauls:
                    hauls = ['SH', 'LH']
                if not cabins:
                    cabins = ['Economy', 'Business', 'Premium']
                if not companies:
                    companies = ['IB', 'YW']
                
                # Get NCS routes
                ncs_routes_data = await self._get_ncs_routes(node_path, cabins, companies, hauls, start_dt, end_dt)
                
                if ncs_routes_data and 'routes' in ncs_routes_data:
                    ncs_routes = ncs_routes_data['routes']
                    result_parts.append(f"📊 Rutas con incidentes NCS: {len(ncs_routes)}")
                    for route in ncs_routes[:10]:  # Show first 10
                        result_parts.append(f"• {route}")
                else:
                    result_parts.append("✅ No se encontraron incidentes operacionales en este período.")
                    
            except Exception as e:
                self.logger.warning(f"Error getting NCS routes: {str(e)}")
                result_parts.append("⚠️ No se pudieron obtener datos de incidentes operacionales.")
            
            result_parts.append("")
            result_parts.append("**NOTA:** Valores NPS absolutos del período específico, sin comparación temporal.")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error in single period routes tool: {type(e).__name__}: {str(e)}")
            return f"ERROR in routes tool: {type(e).__name__}: {str(e)}"
    
    async def _generate_single_period_synthesis(
        self, 
        message_history: MessageHistory,
        node_path: str,
        start_date: str, 
        end_date: str,
        nps_context: str = "",
        current_nps: str = "N/A",
        baseline_nps: str = "N/A",
        nps_difference: str = "N/A",
        baseline_periods: int = 7,
        aggregation_days: int = 1
    ) -> str:
        """Generate final synthesis for single period analysis"""
        try:
            # Build data summary for single period
            data_summary = self._build_single_period_data_summary()
            
            # Get synthesis prompt from YAML config (single_prompts section)
            synthesis_prompt_template = self.config.get('single_prompts', {}).get('synthesis_prompt', 
                "Basándote en el análisis de {node_path} para el período {start_date} a {end_date}, proporciona una síntesis integral.")
            
            # Create synthesis prompt for single period with all required parameters
            synthesis_prompt = synthesis_prompt_template.format(
                node_path=node_path,
                start_date=start_date,
                end_date=end_date,
                data_summary=data_summary,
                current_nps=current_nps,
                baseline_nps=baseline_nps,
                nps_difference=nps_difference,
                baseline_periods=baseline_periods,
                aggregation_days=aggregation_days
            )
            
            # Get LLM response using message history with synthesis prompt
            from langchain.schema import HumanMessage
            # Add synthesis prompt to existing conversation context
            synthesis_messages = message_history.get_messages()
            synthesis_messages.append(HumanMessage(content=synthesis_prompt))
            response = await self.llm(synthesis_messages)
            
            if response and hasattr(response, 'content') and response.content:
                # Log the synthesis to tracker so it's included in JSON export
                self.tracker.log_message("AI", f"FINAL_SYNTHESIS: {response.content}")
                self.logger.info("🎯 Single period synthesis completed successfully")
                return f"🤖 **ANÁLISIS PERIODO ÚNICO**\n\n{response.content}"
            else:
                self.logger.warning("⚠️ Empty response from LLM in single period synthesis")
                return "🤖 **ANÁLISIS PERIODO ÚNICO**\n\nAnálisis completado pero la síntesis falló."
                
        except Exception as e:
            self.logger.error(f"❌ Error en síntesis de período único: {type(e).__name__}: {str(e)}")
            return "🤖 **ANÁLISIS PERIODO ÚNICO**\n\nSíntesis falló debido a error."
    
    def _build_single_period_data_summary(self) -> str:
        """Build data summary for single period analysis"""
        summary_parts = []
        
        # Add collected data summary
        try:
            data_summary = self._build_collected_data_summary()
            summary_parts.append("📊 **DATOS RECOLECTADOS:**")
            summary_parts.append(data_summary)
        except Exception as e:
            self.logger.error(f"Error building data summary: {e}")
            summary_parts.append("📊 **DATOS RECOLECTADOS:** Error construyendo resumen")
        
        # Add tool reflections
        if self.tracker.previous_explanations:
            summary_parts.append("\n💭 **ANÁLISIS DE HERRAMIENTAS:**")
            for i, reflection in enumerate(self.tracker.previous_explanations, 1):
                summary_parts.append(f"   {i}. {reflection}")
        
        return "\n".join(summary_parts)
    
    def _build_single_period_fallback(self) -> str:
        """Build fallback response for single period analysis"""
        fallback_parts = []
        fallback_parts.append("🤖 **ANÁLISIS PERIODO ÚNICO**")
        fallback_parts.append("Análisis completado con síntesis parcial debido a error de procesamiento")
        fallback_parts.append("")
        
        # Add collected data summary
        try:
            data_summary = self._build_single_period_data_summary()
            fallback_parts.append(data_summary)
        except Exception as summary_error:
            self.logger.error(f"Error building data summary for fallback: {summary_error}")
            fallback_parts.append("❌ Error construyendo resumen de datos")
        
        return "\n".join(fallback_parts)
    
    async def investigate_anomaly(
        self,
        node_path: str,
        start_date: str,
        end_date: str,
        anomaly_type: str,
        anomaly_magnitude: float,
        nps_context: str = "",
        causal_filter: str = "vs L7d",
        comparison_start_date: datetime = None,
        comparison_end_date: datetime = None,
        anomaly_detection_mode: str = "target",
        aggregation_days: int = 7,
        comparison_context: str = "",
        baseline_periods: int = 7,
        focus_touchpoint: Optional[str] = None
    ) -> str:
        """
        Main investigation method that routes to single or comparative mode based on study_mode
        """
        # Reset all per-investigation state to prevent leakage between parallel calls
        self.tracker.reset_tracker()
        self.collected_data = {}

        # Override focus_touchpoint if provided at call time
        if focus_touchpoint is not None:
            self.focus_touchpoint = focus_touchpoint
        print(f"🔍 DEBUG CAUSAL AGENT: investigate_anomaly called with start_date='{start_date}', end_date='{end_date}'")
        
        # Convert dates to datetime objects (handle both string and datetime inputs)
        from datetime import datetime
        try:
            if isinstance(start_date, str):
                start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_date_dt = start_date
                
            if isinstance(end_date, str):
                end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date_dt = end_date
                
            print(f"🔍 DEBUG CAUSAL AGENT: Converted dates - start_date_dt={start_date_dt}, end_date_dt={end_date_dt}")
        except (ValueError, TypeError) as e:
            self.logger.error(f"❌ Error converting dates to datetime: {e}")
            return f"Error: Invalid date format. Expected YYYY-MM-DD or datetime object, got start_date='{start_date}', end_date='{end_date}'"
        
        # Update instance variables with the passed parameters
        if causal_filter:
            self.causal_filter = causal_filter
        if comparison_start_date:
            if isinstance(comparison_start_date, str):
                self.comparison_start_date = datetime.strptime(comparison_start_date, '%Y-%m-%d')
            else:
                self.comparison_start_date = comparison_start_date
        if comparison_end_date:
            if isinstance(comparison_end_date, str):
                self.comparison_end_date = datetime.strptime(comparison_end_date, '%Y-%m-%d')
            else:
                self.comparison_end_date = comparison_end_date
        
        # Route to appropriate investigation method based on study_mode
        if self.study_mode == "single":
            result = await self._investigate_anomaly_single_period(
                node_path, start_date_dt, end_date_dt, anomaly_type, anomaly_magnitude, nps_context,
                anomaly_detection_mode, aggregation_days, comparison_context, baseline_periods
            )
        else:
            result = await self._investigate_anomaly_with_comparison(
                node_path, start_date_dt, end_date_dt, anomaly_type, anomaly_magnitude, nps_context,
                causal_filter, comparison_start_date, comparison_end_date,
                anomaly_detection_mode, aggregation_days, comparison_context, baseline_periods
            )

        # --- FOCUS TOUCHPOINT ENRICHMENT: verbatims + % issues per segment ---
        # NOTE: enrichment is already collected inside the investigation methods (before synthesis).
        # This outer block only runs as a fallback if it wasn't collected there (e.g. exception path).
        effective_focus = self.focus_touchpoint
        if effective_focus and 'focus_touchpoint_enrichment' not in self.collected_data:
            # Resolve comparison dates: use stored ones or calculate dynamically
            enrichment_comp_start = self.comparison_start_date
            enrichment_comp_end = self.comparison_end_date
            if (enrichment_comp_start is None or enrichment_comp_end is None) and self.causal_filter and self.causal_filter != "vs Sel. Period":
                enrichment_comp_start, enrichment_comp_end = self.calculate_dynamic_comparison_dates(start_date_dt, end_date_dt)
                self.logger.info(f"🔄 Focus enrichment (fallback): dynamically resolved comparison dates: {enrichment_comp_start.strftime('%Y-%m-%d')} → {enrichment_comp_end.strftime('%Y-%m-%d')}")

            enrichment = await self._collect_focus_touchpoint_enrichment(
                node_path=node_path,
                start_date=start_date_dt,
                end_date=end_date_dt,
                comparison_start_date=enrichment_comp_start,
                comparison_end_date=enrichment_comp_end,
            )
            if enrichment:
                # Store in collected_data so it reaches _generate_final_synthesis via _build_collected_data_summary
                self.collected_data['focus_touchpoint_enrichment'] = enrichment
                result = f"{result}\n\n{enrichment}"
        elif effective_focus and 'focus_touchpoint_enrichment' in self.collected_data:
            # Already collected inside investigation method — just append to result string
            result = f"{result}\n\n{self.collected_data['focus_touchpoint_enrichment']}"

        return result
    
    def _get_relevant_nodes_for_segment(self, node_path: str) -> List[str]:
        """
        Return the list of nodes to cover for focus touchpoint verbatims / % issues.

        For child nodes (depth > 2, e.g. Global/SH/Business/IB), use the actual
        node_path directly — broadening to a parent segment would contaminate results
        with verbatims from sibling nodes (e.g. Economy SH verbatims bleeding into
        Business SH IB analysis).

        Only top-level nodes (depth ≤ 2) use the broader parent coverage:
        - Global (depth 1) → [Global, Global_LH, Global_SH]
        - Global/SH (depth 2) → [Global_SH]
        - Global/LH (depth 2) → [Global_LH]
        - Global/SH/Business/IB (depth 4) → [Global/SH/Business/IB]  ← exact node
        """
        parts = [p.strip() for p in node_path.split("/") if p.strip()]
        depth = len(parts)
        root = parts[0].lower() if parts else ""
        second = parts[1].lower() if len(parts) > 1 else ""

        # Child nodes (depth > 2): use the exact node to avoid cross-segment contamination
        if depth > 2:
            return [node_path]

        if root == "global":
            if second == "sh":
                return ["Global_SH"]
            elif second == "lh":
                return ["Global_LH"]
            else:
                # Full global analysis — cover all three top-level segments
                return ["Global", "Global_LH", "Global_SH"]
        # Fallback: use the node itself
        return [node_path.replace("/", "_")]

    async def _collect_focus_touchpoint_enrichment(
        self,
        node_path: str,
        start_date: datetime,
        end_date: datetime,
        comparison_start_date: Optional[datetime] = None,
        comparison_end_date: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Collect verbatims and % issues for the focus touchpoint across all relevant
        nodes for the current segment.  Returns a formatted string to append to the
        investigation result, or None if nothing was collected.
        """
        effective_focus = self.focus_touchpoint
        if not effective_focus:
            return None

        display_name = get_touchpoint_display_name(effective_focus)
        relevant_nodes = self._get_relevant_nodes_for_segment(node_path)

        parts: List[str] = []
        parts.append(f"━━━ 🎯 FOCUS TOUCHPOINT ENRICHMENT: '{effective_focus}' ━━━")

        # ── Verbatims per node (filtered by topic) ──────────────────────────
        self._focus_verbatims_by_node: Dict[str, str] = {}

        if display_name:
            for node in relevant_nodes:
                try:
                    # Collect verbatims filtered by topic
                    df_verbatims = await self.pbi_collector.collect_verbatims_by_topic(
                        node_path=node,
                        start_date=start_date,
                        end_date=end_date,
                        touchpoint_display_name=display_name,
                        top_n=10,
                    )
                    
                    if not df_verbatims.empty:
                        # Format verbatims for display
                        verbatim_col = next((c for c in df_verbatims.columns if "verbatim" in c.lower() and "sentiment" not in c.lower()), None)
                        route_col = next((c for c in df_verbatims.columns if "route" in c.lower()), None)
                        nps_col = next((c for c in df_verbatims.columns if "nps" in c.lower() and "score" in c.lower()), None)
                        category_col = next((c for c in df_verbatims.columns if "category" in c.lower()), None)
                        
                        verbatims_lines = []
                        verbatims_lines.append(f"Found {len(df_verbatims)} verbatims:")
                        
                        for idx, row in df_verbatims.iterrows():
                            route = row[route_col] if route_col and route_col in row else "N/A"
                            nps = row[nps_col] if nps_col and nps_col in row else "N/A"
                            category = row[category_col] if category_col and category_col in row else "N/A"
                            verbatim = row[verbatim_col] if verbatim_col and verbatim_col in row else "N/A"
                            
                            verbatims_lines.append(f"  • Route: {route}, NPS: {nps}, Category: {category}")
                            verbatims_lines.append(f"    Text: {str(verbatim)[:200]}...")
                        
                        verbatims_result = "\n".join(verbatims_lines)
                        self._focus_verbatims_by_node[node] = verbatims_result
                        parts.append(f"\n📝 VERBATIMS [{node}] — filtered by topic '{display_name}'")
                        parts.append(verbatims_result)
                    else:
                        self._focus_verbatims_by_node[node] = "(sin datos)"
                        parts.append(f"\n📝 VERBATIMS [{node}]: sin datos disponibles")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not collect focus verbatims for node '{node}': {e}")
                    self._focus_verbatims_by_node[node] = f"(sin datos — error: {e})"
                    parts.append(f"\n📝 VERBATIMS [{node}]: sin datos disponibles (error: {e})")

        # ── CSAT vs Target per node ─────────────────────────────────────────
        self._focus_csat_by_node: Dict[str, Dict] = {}

        if display_name:
            for node in relevant_nodes:
                try:
                    csat_result = await self.pbi_collector.collect_focus_touchpoint_csat_vs_target(
                        node_path=node,
                        start_date=start_date,
                        end_date=end_date,
                        touchpoint_name=display_name,
                        comparison_filter="vs L7d" if comparison_start_date and comparison_end_date else None,
                        comparison_start_date=comparison_start_date,
                        comparison_end_date=comparison_end_date,
                    )
                    if csat_result:
                        self._focus_csat_by_node[node] = csat_result
                        csat = csat_result.get("csat")
                        target = csat_result.get("target")
                        gap = csat_result.get("gap")
                        satisfaction_diff = csat_result.get("satisfaction_diff")
                        csat_str = f"{csat:.1f}" if csat is not None else "N/A"
                        target_str = f"{target:.1f}" if target is not None else "N/A"
                        gap_str = f"{gap:+.1f}" if gap is not None else "N/A"
                        sat_diff_str = f"{satisfaction_diff:+.1f}" if satisfaction_diff is not None else "N/A"
                        
                        # Include satisfaction_diff only in comparative mode
                        if satisfaction_diff is not None:
                            parts.append(
                                f"\n📊 CSAT vs TARGET [{node}] — '{display_name}': "
                                f"CSAT={csat_str}, Target={target_str}, Gap={gap_str}, Sat_diff={sat_diff_str}"
                            )
                        else:
                            parts.append(
                                f"\n📊 CSAT vs TARGET [{node}] — '{display_name}': "
                                f"CSAT={csat_str}, Target={target_str}, Gap={gap_str}"
                            )
                    else:
                        parts.append(f"\n📊 CSAT vs TARGET [{node}]: sin datos disponibles")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not collect CSAT vs target for node '{node}': {e}")
                    parts.append(f"\n📊 CSAT vs TARGET [{node}]: sin datos disponibles (error: {e})")

        # ── % Issues per node (only if display_name mapping exists) ─────────
        self._focus_issues_pct_by_node: Dict[str, Dict] = {}

        if display_name and comparison_start_date and comparison_end_date:
            for node in relevant_nodes:
                try:
                    # Track the DAX query for debugging
                    cabins_dbg, companies_dbg, hauls_dbg = self.pbi_collector._get_node_filters(node)
                    issues_query_dbg = self.pbi_collector._get_focus_touchpoint_issues_pct_query(
                        cabins=cabins_dbg, companies=companies_dbg, hauls=hauls_dbg,
                        start_date=start_date, end_date=end_date,
                        comparison_start_date=comparison_start_date,
                        comparison_end_date=comparison_end_date,
                        touchpoint_display_name=display_name,
                    )
                    self.tracker.add_dax_query(
                        "focus_issues_pct",
                        issues_query_dbg,
                        {"node": node, "touchpoint": display_name},
                    )
                    self.logger.info(f"🔍 focus_issues_pct query for '{node}': {issues_query_dbg[:300]}")

                    issues_result = await self.pbi_collector.collect_focus_touchpoint_issues_pct(
                        node_path=node,
                        start_date=start_date,
                        end_date=end_date,
                        comparison_start_date=comparison_start_date,
                        comparison_end_date=comparison_end_date,
                        touchpoint_display_name=display_name,
                    )
                    if issues_result:
                        self._focus_issues_pct_by_node[node] = issues_result
                        pct_cur = issues_result.get("pct_issues_current")
                        pct_prev = issues_result.get("pct_issues_prev")
                        diff = issues_result.get("diff")
                        pct_cur_str = f"{pct_cur*100:.1f}%" if pct_cur is not None else "N/A"
                        pct_prev_str = f"{pct_prev*100:.1f}%" if pct_prev is not None else "N/A"
                        diff_str = f"{diff*100:+.1f}%" if diff is not None else "N/A"
                        parts.append(
                            f"\n📊 % ISSUES [{node}] — '{display_name}': "
                            f"L7D={pct_cur_str}, prev={pct_prev_str}, diff={diff_str}"
                        )
                    else:
                        parts.append(f"\n📊 % ISSUES [{node}]: sin datos disponibles")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not collect focus issues pct for node '{node}': {e}")
                    parts.append(f"\n📊 % ISSUES [{node}]: sin datos disponibles (error: {e})")
        elif not display_name:
            self.logger.warning(
                f"⚠️ No display_name mapping for '{effective_focus}' — skipping % issues query"
            )
            parts.append(
                f"\n📊 % ISSUES: omitido (no existe mapeo display_name para '{effective_focus}')"
            )
        else:
            # Single mode — no comparison dates available
            parts.append("\n📊 % ISSUES: omitido (modo single, sin fechas de comparación)")

        return "\n".join(parts)

    async def _verbatims_tool_for_node(
        self,
        node_path: str,
        start_date: str,
        end_date: str,
        query: str,
    ) -> str:
        """
        Call the verbatims tool for a specific node and search query.
        Delegates to the existing _verbatims_tool or _verbatims_tool_single_period
        depending on study_mode.
        """
        if self.study_mode == "single":
            return await self._verbatims_tool_single_period(
                node_path=node_path,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return await self._verbatims_tool(
                node_path=node_path,
                start_date=start_date,
                end_date=end_date,
            )

    async def _investigate_anomaly_single_period(
        self,
        node_path: str,
        start_date: datetime,
        end_date: datetime,
        anomaly_type: str,
        anomaly_magnitude: float,
        nps_context: str = "",
        anomaly_detection_mode: str = "target",
        aggregation_days: int = 7,
        comparison_context: str = "",
        baseline_periods: int = 7
    ) -> str:
        """
        Single period investigation (no comparison)
        
        Flow:
        1. Analyze hierarchical tree structure (helper prompt)
        2. Skip explanatory drivers (not needed for single period)
        3. Analyze operative data with mean comparison
        4. Analyze NCS incidents for the specific period only
        5. Analyze routes filtered by NPS for the specific period only
        """
        
        try:
            self.logger.info("🎯 Starting single period investigation (no comparison)")
            
            # Reset tracker
            self.tracker.reset_tracker()
            self.current_anomaly_type = anomaly_type
            
            # Create message history for main flow
            message_history = MessageHistory(logger=self.logger)
            
            # 1. SYSTEM MESSAGE - Use single period specific prompt
            system_prompt = self._get_single_period_system_prompt()
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            self.tracker.log_message("SYSTEM", system_prompt)
            
            # 2. USER MESSAGE: Investigation parameters for single period
            # Extract NPS values from nps_context if available
            current_nps = "N/A"
            baseline_nps = "N/A" 
            nps_difference = "N/A"
            # Persist NPS context on the agent so tools (e.g., ncs_tool) can relate incidents ↔ NPS
            self.current_nps_value = None
            self.baseline_nps_value = None
            self.nps_difference_value = None
            
            if nps_context:
                # Try to extract NPS values from the context
                import re
                current_match = re.search(r'Current NPS:\s*([\d.-]+)', nps_context)
                baseline_match = re.search(r'Baseline NPS:\s*([\d.-]+)', nps_context)
                
                if current_match:
                    current_nps = current_match.group(1)
                if baseline_match:
                    baseline_nps = baseline_match.group(1)
                    
                # Calculate difference if both values are available
                if current_nps != "N/A" and baseline_nps != "N/A":
                    try:
                        current_val = float(current_nps)
                        baseline_val = float(baseline_nps)
                        nps_difference = f"{current_val - baseline_val:.1f}"
                        self.current_nps_value = current_val
                        self.baseline_nps_value = baseline_val
                        self.nps_difference_value = current_val - baseline_val
                    except ValueError:
                        nps_difference = "N/A"
            
            user_input = self._get_single_period_input_template().format(
                node_path=node_path,
                start_date=start_date,
                end_date=end_date,
                anomaly_type=anomaly_type,
                anomaly_magnitude=anomaly_magnitude,
                current_nps=current_nps,
                baseline_nps=baseline_nps,
                nps_difference=nps_difference,
                # New placeholders
                anomaly_detection_mode=anomaly_detection_mode,
                aggregation_days=aggregation_days,
                comparison_context=comparison_context,
                baseline_periods=baseline_periods
            )
            
            message_history.create_and_add_message(
                content=user_input,
                message_type=MessageType.USER
            )
            self.tracker.log_message("USER", user_input)
            
            # FIRST: Let the agent decide which tool to start with based on the initial context
            self.logger.info("🤔 First step: Agent deciding initial tool to execute (single period)...")
            
            # Create prompt for initial tool decision
            initial_decision_prompt = f"""
            Based on the investigation parameters and context, decide which tool to execute FIRST for single period analysis.
            
            INVESTIGATION CONTEXT:
            - Node Path: {node_path}
            - Period: {start_date} to {end_date}
            - Anomaly Type: {anomaly_type}
            - Anomaly Magnitude: {anomaly_magnitude}
            - Current NPS: {current_nps}
            - Baseline NPS: {baseline_nps}
            - NPS Difference: {nps_difference}
            
            AVAILABLE TOOLS FOR SINGLE PERIOD:
            - operative_data_tool: Analyze operational KPIs (OTP, Load Factor, etc.) for the specific period
            - ncs_tool: Analyze operational incidents and NCS data for the specific period
            - routes_tool: Analyze route-specific performance for the specific period
            - verbatims_tool: Analyze customer feedback and verbatims for the specific period
            - customer_profile_tool: Analyze customer profile impact for the specific period
            
            DECISION CRITERIA:
            - For single period analysis: Start with operative_data_tool to understand operational metrics
            - Consider the anomaly type and magnitude in your decision
            - Focus on tools that provide insights for the specific period without comparison
            
            Respond with ONLY the name of the tool to execute first:
            """
            
            # Get initial tool decision from LLM - NO FALLBACKS, AGENT MUST DECIDE
            try:
                from langchain.schema import HumanMessage
                initial_response = await self.llm([HumanMessage(content=initial_decision_prompt)])
                initial_tool = initial_response.content.strip()
                
                # Clean up the response and validate
                valid_tools = [
                    'operative_data_tool', 'ncs_tool', 'routes_tool',
                    'verbatims_tool', 'customer_profile_tool'
                ]
                
                if initial_tool in valid_tools:
                    current_tool = initial_tool
                    self.logger.info(f"🎯 Agent decided to start with: {current_tool}")
                else:
                    # ❌ NO FALLBACK - AGENT MUST PROVIDE VALID TOOL
                    error_msg = f"Agent provided invalid tool: '{initial_tool}'. Valid tools are: {valid_tools}"
                    self.logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                    
            except Exception as e:
                # ❌ NO FALLBACK - INVESTIGATION MUST FAIL IF AGENT CANNOT DECIDE
                error_msg = f"Agent failed to decide initial tool: {e}"
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # Now execute tools based on agent decisions
            max_iterations = 10  # Allow up to 10 iterations for dynamic tool selection
            iteration = 0
            
            while current_tool and iteration < max_iterations:
                iteration += 1
                self.tracker.next_iteration()
                
                self.logger.info(f"🔧 Iteration {iteration}: Processing {current_tool} (single period)")
                
                # Execute tool with single period logic
                tool_result = await self._execute_single_period_tool(
                    current_tool, node_path, start_date, end_date, iteration, comparison_context, baseline_periods, anomaly_detection_mode, aggregation_days
                )
                
                # Store tool context
                self.tracker.set_tool_context(current_tool, tool_result)

                # Prepend segment anchor so the LLM never loses track of which segment it's analyzing
                segment_anchor = f"⚠️ SEGMENTO BAJO ANÁLISIS: {node_path} ({self._get_segment_description(node_path)})\n\n"
                tool_result_with_anchor = segment_anchor + tool_result
                
                # AI MESSAGE: Reflection using clean context + tool results
                reflection_result = await self._get_clean_reflection(
                    system_prompt=system_prompt,
                    tool_name=current_tool,
                    tool_result=tool_result_with_anchor,
                    message_history=message_history
                )
                
                if reflection_result and isinstance(reflection_result, dict):
                    reflection = reflection_result.get("reflection", "")
                    next_tool_code = reflection_result.get("next_tool_code", "")
                
                    if reflection:
                        self.tracker.add_explanation(reflection)
                        message_history.create_and_add_message(
                            content=reflection,
                            message_type=MessageType.AI,
                            agent=AgentName.CONVERSATIONAL
                        )
                        # Include tool_name in metadata so it appears in tools_used
                        self.tracker.log_message("AI", f"REFLECTION: {reflection}", metadata={"tool_name": current_tool})
                        self.logger.info(f"💭 Reflection captured for {current_tool}")
                        
                        # Execute next tool code if provided
                        if next_tool_code:
                            self.logger.info(f"🔄 Executing next tool code: {next_tool_code}")

                            # Execute the tool based on the code from reflection
                            try:
                                if next_tool_code == "operative_data_tool":
                                    current_tool = "operative_data_tool"
                                elif next_tool_code == "ncs_tool":
                                    current_tool = "ncs_tool"
                                elif next_tool_code == "verbatims_tool":
                                    current_tool = "verbatims_tool"
                                elif next_tool_code == "routes_tool":
                                    current_tool = "routes_tool"
                                elif next_tool_code == "customer_profile_tool":
                                    current_tool = "customer_profile_tool"
                                elif next_tool_code in ["TERMINAR", "FIN", "END", "STOP"]:
                                    self.logger.info(f"✅ Agent decided to end investigation (code: {next_tool_code})")
                                    current_tool = None
                                    break
                                else:
                                    self.logger.warning(f"⚠️ Unknown tool code: {next_tool_code}")
                                    # Let agent decide the next tool based on reflection
                                    current_tool = await self._determine_next_tool_from_reflection(
                                        reflection, current_tool, iteration, max_iterations
                                    )
                            except Exception as e:
                                self.logger.error(f"❌ Error executing tool code {next_tool_code}: {e}")
                                # Let agent decide the next tool based on reflection
                                current_tool = await self._determine_next_tool_from_reflection(
                                    reflection, current_tool, iteration, max_iterations
                                )
                            
                            # Continue to next iteration with the selected tool
                            # The tool will be executed in the next iteration of the while loop
                        else:
                            # Let the agent decide the next tool based on reflection and helper prompts
                            current_tool = await self._determine_next_tool_from_reflection(
                                reflection, current_tool, iteration, max_iterations
                            )
                            
                            if current_tool:
                                self.logger.info(f"🔄 Agent decided next tool: {current_tool}")
                            else:
                                self.logger.info(f"✅ Agent decided to end investigation")
                                break
                    else:
                        self.logger.warning(f"⚠️ No reflection captured for {current_tool}")
                        # ❌ NO FALLBACK - INVESTIGATION MUST END IF AGENT CANNOT REFLECT
                        self.logger.error(f"❌ Investigation cannot continue without agent reflection")
                        break
                else:
                    self.logger.warning(f"⚠️ No reflection captured for {current_tool}")
                    # ❌ NO FALLBACK - INVESTIGATION MUST END IF AGENT CANNOT REFLECT
                    self.logger.error(f"❌ Investigation cannot continue without agent reflection")
                    break
                
            # --- FOCUS TOUCHPOINT ENRICHMENT: collect BEFORE final synthesis ---
            if self.focus_touchpoint:
                try:
                    enrichment = await self._collect_focus_touchpoint_enrichment(
                        node_path=node_path,
                        start_date=start_date,
                        end_date=end_date,
                        comparison_start_date=None,
                        comparison_end_date=None,
                    )
                    if enrichment:
                        self.collected_data['focus_touchpoint_enrichment'] = enrichment
                        self.logger.info("✅ Focus touchpoint enrichment stored in collected_data before single-period synthesis")
                except Exception as enrich_err:
                    self.logger.warning(f"⚠️ Could not collect focus touchpoint enrichment (single): {enrich_err}")

            # Final synthesis for single period
            try:
                self.logger.info("🎯 Iniciando síntesis final para análisis de período único...")
                final_response = await self._generate_single_period_synthesis(
                    message_history=message_history, 
                    node_path=node_path, 
                    start_date=start_date, 
                    end_date=end_date, 
                    nps_context=nps_context,
                    current_nps=current_nps,
                    baseline_nps=baseline_nps,
                    nps_difference=nps_difference,
                    baseline_periods=baseline_periods,
                    aggregation_days=aggregation_days
                )
                self.logger.info("✅ Investigación de período único completada")
                return final_response
                
            except Exception as e:
                self.logger.error(f"❌ Error en síntesis de período único: {type(e).__name__}: {str(e)}")
                return self._build_single_period_fallback()
            
        except Exception as e:
            self.logger.error(f"❌ Error crítico en investigación de período único: {type(e).__name__}: {e}")
            return self._build_single_period_fallback()
        finally:
            try:
                conversation_file = await self.export_conversation(node_path=node_path, start_date=start_date, end_date=end_date)
                if conversation_file:
                    print(f"🗂️ Conversación guardada: {conversation_file}")
                    self.logger.info(f"🗂️ Conversación completa guardada: {conversation_file}")
            except Exception as ex:
                self.logger.warning(f"⚠️ No se pudo guardar la conversación (single): {ex}")
    
    async def _investigate_anomaly_with_comparison(
        self,
        node_path: str,
        start_date: datetime,
        end_date: datetime,
        anomaly_type: str,
        anomaly_magnitude: float,
        nps_context: str = "",
        causal_filter: str = "vs L7d",
        comparison_start_date: datetime = None,
        comparison_end_date: datetime = None,
        anomaly_detection_mode: str = "target",
        aggregation_days: int = 7,
        comparison_context: str = "",
        baseline_periods: int = 7
    ) -> str:
        """
        Comparative investigation (with comparison to other periods)
        
        Flow:
        1. Analyze hierarchical tree structure (helper prompt)
        2. Analyze explanatory drivers (for comparison)
        3. Analyze operative data with comparison
        4. Analyze NCS incidents with temporal comparison
        5. Analyze routes with explanatory drivers
        6. Optional: Customer profile analysis
        """
        
        try:
            self.logger.info("🎯 Starting comparative investigation (with comparison)")
            
            # Reset tracker
            self.tracker.reset_tracker()
            self.current_anomaly_type = anomaly_type
            
            # Create message history for main flow
            message_history = MessageHistory(logger=self.logger)
            
            # 1. SYSTEM MESSAGE
            system_prompt = self._get_system_prompt(mode="comparative")
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            self.tracker.log_message("SYSTEM", system_prompt)
            
            # 2. USER MESSAGE: Investigation parameters
            # Extract NPS values from nps_context if available
            current_nps = "N/A"
            baseline_nps = "N/A" 
            nps_difference = "N/A"
            # Persist NPS context on the agent so tools (e.g., ncs_tool) can relate incidents ↔ NPS
            self.current_nps_value = None
            self.baseline_nps_value = None
            self.nps_difference_value = None
            
            self.logger.info(f"🔍 DEBUG COMPARATIVE: nps_context received (length={len(nps_context)}): '{nps_context}'")
            if nps_context:
                # Try to extract NPS values from the context
                import re
                self.logger.info(f"🔍 DEBUG COMPARATIVE: nps_context is not empty, processing...")
                current_match = re.search(r'Current NPS:\s*([\d.-]+)', nps_context)
                baseline_match = re.search(r'Baseline NPS:\s*([\d.-]+)', nps_context)
                
                self.logger.info(f"🔍 DEBUG COMPARATIVE: current_match: {current_match}")
                self.logger.info(f"🔍 DEBUG COMPARATIVE: baseline_match: {baseline_match}")
                
                if current_match:
                    current_nps = current_match.group(1)
                    self.logger.info(f"🔍 DEBUG COMPARATIVE: extracted current_nps: {current_nps}")
                if baseline_match:
                    baseline_nps = baseline_match.group(1)
                    self.logger.info(f"🔍 DEBUG COMPARATIVE: extracted baseline_nps: {baseline_nps}")
                    
                # Calculate difference if both values are available
                if current_nps != "N/A" and baseline_nps != "N/A":
                    try:
                        current_val = float(current_nps)
                        baseline_val = float(baseline_nps)
                        nps_difference = f"{current_val - baseline_val:.1f}"
                        self.current_nps_value = current_val
                        self.baseline_nps_value = baseline_val
                        self.nps_difference_value = current_val - baseline_val
                    except ValueError:
                        nps_difference = "N/A"
                else:
                    self.logger.info(f"🔍 DEBUG COMPARATIVE: nps_context is EMPTY - will use N/A values")
            
            self.logger.info(f"🔍 DEBUG COMPARATIVE: Final values for template - current_nps: {current_nps}, baseline_nps: {baseline_nps}, nps_difference: {nps_difference}")
            
            # Get baseline description for display and store it for final synthesis
            from dashboard_analyzer.deep_research_period import determine_anomaly_mode_for_vslast
            
            # Pass comparison dates if available for "vs Sel. Period"
            comparison_start_str = None
            comparison_end_str = None
            if hasattr(self, 'comparison_start_date') and hasattr(self, 'comparison_end_date'):
                if self.comparison_start_date:
                    comparison_start_str = self.comparison_start_date.strftime('%Y-%m-%d') if hasattr(self.comparison_start_date, 'strftime') else str(self.comparison_start_date)
                if self.comparison_end_date:
                    comparison_end_str = self.comparison_end_date.strftime('%Y-%m-%d') if hasattr(self.comparison_end_date, 'strftime') else str(self.comparison_end_date)
            
            _, baseline_description = determine_anomaly_mode_for_vslast(causal_filter, comparison_start_str, comparison_end_str)
            
            # Store baseline information for final synthesis
            self.baseline_description = baseline_description
            self.causal_filter = causal_filter
            
            user_input = self._get_input_template(mode="comparative").format(
                node_path=node_path,
                start_date=start_date,
                end_date=end_date,
                anomaly_type=anomaly_type,
                anomaly_magnitude=anomaly_magnitude,
                causal_filter=causal_filter,
                baseline_description=baseline_description,
                current_nps=current_nps,
                baseline_nps=baseline_nps,
                nps_difference=nps_difference,
                # New placeholders
                anomaly_detection_mode=anomaly_detection_mode,
                aggregation_days=aggregation_days,
                comparison_context=comparison_context,
                baseline_periods=baseline_periods
            )
            
            self.logger.info(f"🔍 DEBUG COMPARATIVE: Template output preview: {user_input[:200]}...")
            
            message_history.create_and_add_message(
                content=user_input,
                message_type=MessageType.USER
            )
            self.tracker.log_message("USER", user_input)
            
            # FIRST: Let the agent decide which tool to start with based on the initial context
            self.logger.info("🤔 First step: Agent deciding initial tool to execute...")
            
            # Create prompt for initial tool decision
            initial_decision_prompt = f"""
            Based on the investigation parameters and context, decide which tool to execute FIRST.
            
            INVESTIGATION CONTEXT:
            - Node Path: {node_path}
            - Period: {start_date} to {end_date}
            - Anomaly Type: {anomaly_type}
            - Anomaly Magnitude: {anomaly_magnitude}
            - Comparison Filter: {causal_filter}
            - Current NPS: {current_nps}
            - Baseline NPS: {baseline_nps}
            - NPS Difference: {nps_difference}
            
            AVAILABLE TOOLS:
            - explanatory_drivers_tool: Analyze SHAP values and explanatory drivers (for comparative analysis)
            - operative_data_tool: Analyze operational KPIs (OTP, Load Factor, etc.)
            - ncs_tool: Analyze operational incidents and NCS data
            - verbatims_tool: Analyze customer feedback and verbatims
            - routes_tool: Analyze route-specific performance
            - customer_profile_tool: Analyze customer profile impact
            
            DECISION CRITERIA:
            - For comparative analysis: Start with explanatory_drivers_tool to understand SHAP drivers
            - For single period: Start with operative_data_tool to analyze operational metrics
            - Consider the anomaly type and magnitude in your decision
            
            Respond with ONLY the name of the tool to execute first:
            """
            
            # Get initial tool decision from LLM - NO FALLBACKS, AGENT MUST DECIDE
            try:
                from langchain.schema import HumanMessage
                initial_response = await self.llm([HumanMessage(content=initial_decision_prompt)])
                initial_tool = initial_response.content.strip()
                
                # Clean up the response and validate
                valid_tools = [
                    'explanatory_drivers_tool', 'operative_data_tool', 'ncs_tool',
                    'verbatims_tool', 'routes_tool', 'customer_profile_tool'
                ]
                
                if initial_tool in valid_tools:
                    current_tool = initial_tool
                    self.logger.info(f"🎯 Agent decided to start with: {current_tool}")
                else:
                    # ❌ NO FALLBACK - AGENT MUST PROVIDE VALID TOOL
                    error_msg = f"Agent provided invalid tool: '{initial_tool}'. Valid tools are: {valid_tools}"
                    self.logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                    
            except Exception as e:
                # ❌ NO FALLBACK - INVESTIGATION MUST FAIL IF AGENT CANNOT DECIDE
                error_msg = f"Agent failed to decide initial tool: {e}"
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # Now execute tools based on agent decisions
            max_iterations = 10  # Maximum iterations to prevent infinite loops
            iteration = 0
            
            while current_tool and iteration < max_iterations:
                iteration += 1
                self.tracker.next_iteration()
                
                self.logger.info(f"🔧 Iteration {iteration}: Processing {current_tool} (comparative)")
                
                # Execute tool with comparative logic
                tool_result = await self._execute_tool_comparative(
                    current_tool, node_path, start_date, end_date, iteration, baseline_periods, comparison_context
                )
                
                # Store tool context
                self.tracker.set_tool_context(current_tool, tool_result)
                
                # Prepend segment anchor so the LLM never loses track of which segment it's analyzing
                segment_anchor = f"⚠️ SEGMENTO BAJO ANÁLISIS: {node_path} ({self._get_segment_description(node_path)})\n\n"
                tool_result_with_anchor = segment_anchor + tool_result

                # AI MESSAGE: Reflection using clean context + tool results
                reflection_result = await self._get_clean_reflection(
                    system_prompt=system_prompt,
                    tool_name=current_tool,
                    tool_result=tool_result_with_anchor,
                    message_history=message_history
                )
                
                if reflection_result and isinstance(reflection_result, dict):
                    reflection = reflection_result.get("reflection", "")
                    next_tool_code = reflection_result.get("next_tool_code", "")
                
                    if reflection:
                        self.tracker.add_explanation(reflection)
                        message_history.create_and_add_message(
                            content=reflection,
                            message_type=MessageType.AI,
                            agent=AgentName.CONVERSATIONAL
                        )
                        # Include tool_name in metadata so it appears in tools_used
                        self.tracker.log_message("AI", f"REFLECTION: {reflection}", metadata={"tool_name": current_tool})
                        self.logger.info(f"💭 Reflection captured for {current_tool}")

                        # Execute next tool code if provided
                        if next_tool_code:
                            self.logger.info(f"🔄 Executing next tool code: {next_tool_code}")
                            
                            # Execute the tool based on the code from reflection
                            try:
                                if next_tool_code == "explanatory_drivers_tool":
                                    current_tool = "explanatory_drivers_tool"
                                elif next_tool_code == "operative_data_tool":
                                    current_tool = "operative_data_tool"
                                elif next_tool_code == "ncs_tool":
                                    current_tool = "ncs_tool"
                                elif next_tool_code == "verbatims_tool":
                                    current_tool = "verbatims_tool"
                                elif next_tool_code == "routes_tool":
                                    current_tool = "routes_tool"
                                elif next_tool_code == "customer_profile_tool":
                                    current_tool = "customer_profile_tool"
                                elif next_tool_code in ["TERMINAR", "FIN", "END", "STOP"]:
                                    self.logger.info(f"✅ Agent decided to end investigation (code: {next_tool_code})")
                                    current_tool = None
                                    break
                                else:
                                    self.logger.warning(f"⚠️ Unknown tool code: {next_tool_code}")
                                    # Fallback to agent decision
                                    current_tool = await self._determine_next_tool_from_reflection(
                                        reflection, current_tool, iteration, max_iterations
                                    )
                            except Exception as e:
                                self.logger.error(f"❌ Error executing tool code {next_tool_code}: {e}")
                                # Fallback to agent decision
                                current_tool = await self._determine_next_tool_from_reflection(
                                    reflection, current_tool, iteration, max_iterations
                                )
                            
                            # Continue to next iteration with the selected tool
                            # The tool will be executed in the next iteration of the while loop
                        else:
                            # Let the agent decide the next tool based on reflection and helper prompts
                            current_tool = await self._determine_next_tool_from_reflection(
                                reflection, current_tool, iteration, max_iterations
                            )
                            
                            if current_tool:
                                self.logger.info(f"🔄 Agent decided next tool: {current_tool}")
                            else:
                                self.logger.info(f"✅ Agent decided to end investigation")
                                break
                    else:
                        self.logger.warning(f"⚠️ No reflection captured for {current_tool}")
                        # Fallback: end investigation if no reflection
                        current_tool = None

        except Exception as e:
            self.logger.error(f"❌ Error crítico en investigación comparativa: {type(e).__name__}: {e}")
            try:
                await self.export_conversation(node_path=node_path, start_date=start_date, end_date=end_date)
            except Exception:
                pass
            return self._build_collected_data_summary()
            
        # --- FOCUS TOUCHPOINT ENRICHMENT: collect BEFORE final synthesis so it reaches the prompt ---
        if self.focus_touchpoint:
            enrichment_comp_start = comparison_start_date or self.comparison_start_date
            enrichment_comp_end = comparison_end_date or self.comparison_end_date
            if (enrichment_comp_start is None or enrichment_comp_end is None) and self.causal_filter and self.causal_filter != "vs Sel. Period":
                enrichment_comp_start, enrichment_comp_end = self.calculate_dynamic_comparison_dates(start_date, end_date)
                self.logger.info(f"🔄 Focus enrichment (pre-synthesis): dynamically resolved comparison dates: {enrichment_comp_start.strftime('%Y-%m-%d')} → {enrichment_comp_end.strftime('%Y-%m-%d')}")
            try:
                enrichment = await self._collect_focus_touchpoint_enrichment(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date,
                    comparison_start_date=enrichment_comp_start,
                    comparison_end_date=enrichment_comp_end,
                )
                if enrichment:
                    self.collected_data['focus_touchpoint_enrichment'] = enrichment
                    self.logger.info("✅ Focus touchpoint enrichment stored in collected_data before synthesis")
            except Exception as enrich_err:
                self.logger.warning(f"⚠️ Could not collect focus touchpoint enrichment: {enrich_err}")

        # Final synthesis for comparative analysis
        try:
                self.logger.info("🎯 Iniciando síntesis final para análisis comparativo...")
                final_response = await self._generate_final_synthesis(
                    message_history, node_path, start_date, end_date, nps_context
                )
                
                self.logger.info("✅ Investigación comparativa completada")
                return final_response
            
        except Exception as e:
            self.logger.error(f"❌ Error crítico en investigación comparativa: {type(e).__name__}: {e}")
            final_response = self._build_collected_data_summary()
            return final_response
        finally:
            try:
                conversation_file = await self.export_conversation(node_path=node_path, start_date=start_date, end_date=end_date)
                if conversation_file:
                    self.logger.info(f"🗂️ Conversación completa guardada: {conversation_file}")
            except Exception as ex:
                self.logger.warning(f"⚠️ No se pudo guardar la conversación: {ex}")
    
    async def _execute_tool_unified(
        self, 
        tool_name: str, 
        node_path: str, 
        start_date: str, 
        end_date: str,
        iteration: int,
        mode: str = "comparative",
        baseline_periods: int = 7,
        comparison_context: str = ""
    ) -> str:
        """Execute a specific tool using unified mode-aware implementations"""
        try:
            if mode == "single":
                return await self._execute_tool_single_period(tool_name, node_path, start_date, end_date, iteration)
            else:
                return await self._execute_tool_comparative(tool_name, node_path, start_date, end_date, iteration, baseline_periods, comparison_context)
        except Exception as e:
            self.logger.error(f"❌ Error executing unified tool {tool_name} in {mode} mode: {type(e).__name__}: {str(e)}")
            return f"ERROR executing {tool_name} in {mode} mode: {type(e).__name__}: {str(e)}"
    
    async def _execute_tool_comparative(
        self, 
        tool_name: str, 
        node_path: str, 
        start_date: str, 
        end_date: str,
        iteration: int,
        baseline_periods: int = 7,
        comparison_context: str = ""
    ) -> str:
        """Execute a specific tool using comparative implementations"""
        try:
            if tool_name == "explanatory_drivers_tool":
                return await self._explanatory_drivers_tool(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date,
                    min_surveys=10
                )
            elif tool_name == "verbatims_tool":
                return await self._verbatims_tool(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date,
                    anomaly_type=getattr(self, 'current_anomaly_type', 'unknown')
                )
            elif tool_name == "ncs_tool":
                return await self._ncs_tool(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date
                )
            elif tool_name == "routes_tool":
                return await self._routes_tool(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date,
                    min_surveys=1,  # Process all routes, just show survey count
                    anomaly_type=getattr(self, 'current_anomaly_type', 'unknown')
                )
            elif tool_name == "operative_data_tool":
                # The detector already handles the transformation to vslast_dynamic when needed
                # So we just pass the detection_mode directly to all tools
                self.logger.info(f"🔍 DEBUG OPERATIVE_MODE: self.detection_mode='{self.detection_mode}', causal_filter='{self.causal_filter}'")
                # Debug: Check if we have comparison dates and should override comparison_context
                if hasattr(self, 'comparison_start_date') and hasattr(self, 'comparison_end_date') and self.comparison_start_date and self.comparison_end_date:
                    custom_comparison_context = f"• **Comparación**: vs período seleccionado ({self.comparison_start_date.strftime('%Y-%m-%d')} a {self.comparison_end_date.strftime('%Y-%m-%d')})"
                    self.logger.info(f"🔍 DEBUG: Using custom comparison_context for vs Sel. Period: {custom_comparison_context}")
                    final_comparison_context = custom_comparison_context
                else:
                    final_comparison_context = comparison_context
                    self.logger.info(f"🔍 DEBUG: Using original comparison_context: {comparison_context}")
                
                return await self._operative_data_tool(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date,
                    comparison_mode=self.detection_mode,
                    baseline_periods=baseline_periods,
                    comparison_context=final_comparison_context
                )
            elif tool_name == "customer_profile_tool":
                return await self._customer_profile_tool(
                    node_path=node_path,
                    start_date=start_date,
                    end_date=end_date,
                    min_surveys=1,  # Process all profiles, just show survey count
                    mode="comparative"
                )
            else:
                return f"Unknown tool: {tool_name}"
            
        except Exception as e:
            self.logger.error(f"❌ Error executing {tool_name}: {type(e).__name__}: {str(e)}")
            self.logger.error(f"❌ Tool parameters: node_path={node_path}, start_date={start_date}, end_date={end_date}, iteration={iteration}")
            
            # Try to extract more details from the error
            if hasattr(e, '__dict__'):
                self.logger.error(f"❌ Error details: {e.__dict__}")
            
            # Check if it's an API error
            if hasattr(e, 'response'):
                self.logger.error(f"❌ API Response: {e.response}")
            if hasattr(e, 'status_code'):
                self.logger.error(f"❌ Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                self.logger.error(f"❌ Error Body: {e.body}")
                
            return f"ERROR executing {tool_name}: {type(e).__name__}: {str(e)}"
    
    async def _determine_next_tool_from_reflection(
        self, 
        reflection: str, 
        current_tool: str, 
        iteration: int, 
        max_iterations: int
    ) -> Optional[str]:
        """
        Let the agent decide the next tool based on reflection and helper prompts.
        This replaces the programmatic sequence with intelligent decision making.
        """
        try:
            self.logger.info(f"🤔 AGENT DECISION: Determining next tool after {current_tool}")
            self.logger.info(f"📊 Decision context: iteration={iteration}/{max_iterations}")
            self.logger.info(f"💭 Reflection length: {len(reflection)} chars")
            
            # Get the helper prompt for the current tool to guide the agent
            helper_prompt = self._get_helper_prompt_for_tool(current_tool)
            self.logger.info(f"🔧 Helper prompt length: {len(helper_prompt) if helper_prompt else 0} chars")
            
            # Create a prompt for the agent to decide the next tool
            decision_prompt = f"""
            Based on your reflection and the helper guidance, decide which tool to use next.
            
            CURRENT TOOL: {current_tool}
            ITERATION: {iteration}/{max_iterations}
            REFLECTION: {reflection}
            
            HELPER GUIDANCE: {helper_prompt}
            
            AVAILABLE TOOLS:
            - explanatory_drivers_tool: Analyze SHAP values and explanatory drivers
            - operative_data_tool: Analyze operational KPIs (OTP, Load Factor, etc.)
            - ncs_tool: Analyze operational incidents and NCS data
            - verbatims_tool: Analyze customer feedback and verbatims
            - routes_tool: Analyze route-specific performance
            - customer_profile_tool: Analyze customer profile impact
            
            Based on your analysis, respond with ONLY the name of the next tool to use, or 'END' if the investigation is complete.
            
            NEXT TOOL:
            """
            
            # Use the LLM to decide the next tool
            from langchain.schema import HumanMessage
            response = await self.llm([HumanMessage(content=decision_prompt)])
            next_tool = response.content.strip()
            
            self.logger.info(f"🤖 Agent response: '{next_tool}'")
            
            # Clean up the response
            if next_tool.lower() in ['end', 'none', 'complete', 'finished', 'terminar', 'fin']:
                self.logger.info(f"✅ Agent decided to END investigation")
                return None
            
            # Validate that it's a valid tool
            valid_tools = [
                'explanatory_drivers_tool', 'operative_data_tool', 'ncs_tool',
                'verbatims_tool', 'routes_tool', 'customer_profile_tool'
            ]
            
            if next_tool in valid_tools:
                self.logger.info(f"✅ Agent decided next tool: {next_tool}")
                return next_tool
            else:
                self.logger.warning(f"⚠️ Agent suggested invalid tool: '{next_tool}', ending investigation")
                self.logger.warning(f"⚠️ Valid tools are: {valid_tools}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error determining next tool: {e}")
            # Fallback: end investigation
            return None
    
    def _get_helper_prompt_for_tool(self, tool_name: str) -> str:
        """
        Get the helper prompt for a specific tool from the YAML configuration.
        The agent will determine the flow type through its analysis and use the appropriate prompt.
        """
        try:
            # Get the tools_prompts from the configuration
            tools_prompts = self.config.get('tools_prompts', {})
            
            if tool_name in tools_prompts:
                tool_config = tools_prompts[tool_name]
                
                # For comparative mode, return the appropriate flow-specific prompt
                # The agent will use its intelligence to determine which flow to follow
                if 'comparative' in tool_config:
                    comparative_prompts = tool_config['comparative']
                    
                    # If it's a dictionary with flow-specific prompts, provide all options
                    # The agent will analyze and choose the appropriate flow
                    if isinstance(comparative_prompts, dict):
                        # For explanatory_drivers_tool, provide all flow options
                        # The agent will analyze the drivers SHAP and choose the appropriate flow
                        if tool_name == 'explanatory_drivers_tool':
                            flow_options = []
                            if 'operative' in comparative_prompts:
                                flow_options.append("OPERATIVE FLOW: " + comparative_prompts['operative'])
                            if 'product' in comparative_prompts:
                                flow_options.append("PRODUCT FLOW: " + comparative_prompts['product'])
                            if 'mixed' in comparative_prompts:
                                flow_options.append("MIXED FLOW: " + comparative_prompts['mixed'])
                            
                            if flow_options:
                                return "\n\n".join(flow_options)
                            else:
                                return str(comparative_prompts)
                        
                        # For other tools, provide the operative prompt as default
                        # The agent will follow the flow based on its previous analysis
                        if 'operative' in comparative_prompts:
                            return comparative_prompts['operative']
                        elif 'mixed' in comparative_prompts:
                            return comparative_prompts['mixed']
                        else:
                            return str(comparative_prompts)
                    else:
                        # If it's a string, return it directly
                        return comparative_prompts
                        
                elif 'single' in tool_config:
                    return tool_config['single']
                else:
                    return tool_config.get('default', 'No specific guidance available.')
            else:
                return f"No helper prompt found for {tool_name}"
                
        except Exception as e:
            self.logger.error(f"❌ Error getting helper prompt for {tool_name}: {e}")
            return "Error retrieving helper guidance."
    
    async def _operative_data_tool(self, node_path: str, start_date, end_date, comparison_days: int = 7, comparison_mode: str = "mean", baseline_periods: int = 7, comparison_context: str = "") -> str:
        """
        Tool for analyzing operational metrics using parametrized comparison logic.
        
        This uses the enhanced OperationalDataAnalyzer with configurable comparison modes:
        - 'vslast': Compare against the previous period 
        - 'vslast_dynamic': Compare against dynamically calculated baseline period
        - 'mean': Compare against 7-day rolling average (default/legacy)
        - 'target': Compare against target values (future implementation)
        
        Args:
            node_path: Node path for analysis
            start_date: Start date for analysis period (str or datetime)
            end_date: End date for analysis period/target_date (str or datetime)
            comparison_days: Number of days for data collection (default: 7)
            comparison_mode: Comparison mode - "vslast", "vslast_dynamic", "mean", or "target" (default: "mean")
        """
        try:
            self.logger.info(f"Collecting operational data for {node_path} on {end_date} with {comparison_days}-day window, comparison_mode={comparison_mode}")
            
            # Handle both string and datetime inputs
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                target_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                target_dt = end_date
            
            # Import the operational analyzer here to avoid circular imports
            from ....anomaly_explanation.data_analyzer import OperationalDataAnalyzer
            
            # Initialize the operational analyzer with specified comparison mode
            # Pass specific comparison dates if available (for "vs Sel. Period" or dynamic calculation)
            comparison_start_date_for_analyzer = None
            comparison_end_date_for_analyzer = None
            if comparison_mode == "vs Sel. Period" or comparison_mode == "vslast_dynamic":
                if (hasattr(self, 'comparison_start_date') and hasattr(self, 'comparison_end_date') and 
                    self.comparison_start_date and self.comparison_end_date):
                    # Use explicitly provided dates (for "vs Sel. Period")
                    comparison_start_date_for_analyzer = self.comparison_start_date
                    comparison_end_date_for_analyzer = self.comparison_end_date
                    self.logger.info(f"🎯 Using explicit comparison dates: {comparison_start_date_for_analyzer.strftime('%Y-%m-%d')} to {comparison_end_date_for_analyzer.strftime('%Y-%m-%d')}")
                elif hasattr(self, 'causal_filter') and self.causal_filter and self.causal_filter != "vs Sel. Period":
                    # Calculate dynamic dates based on causal_filter
                    comparison_start_date_for_analyzer, comparison_end_date_for_analyzer = self.calculate_dynamic_comparison_dates(start_dt, target_dt)
                    self.logger.info(f"🔄 Using dynamically calculated comparison dates: {comparison_start_date_for_analyzer.strftime('%Y-%m-%d')} to {comparison_end_date_for_analyzer.strftime('%Y-%m-%d')}")
            
            operational_analyzer = OperationalDataAnalyzer(
                comparison_mode=comparison_mode,
                comparison_start_date=comparison_start_date_for_analyzer,
                comparison_end_date=comparison_end_date_for_analyzer,
                aggregation_days=comparison_days,
                baseline_periods=baseline_periods
            )
            
            # For vslast modes, we need data for BOTH current and previous periods
            # So we need to collect more days to ensure we have both periods
            if comparison_mode in ["vslast", "vslast_dynamic", "vs Sel. Period"]:
                # Check if we have specific comparison dates (for "vs Sel. Period" or dynamic calculation)
                if ((comparison_mode == "vslast_dynamic" or comparison_mode == "vs Sel. Period") and 
                    comparison_start_date_for_analyzer and comparison_end_date_for_analyzer):
                    # For specific comparison dates, calculate days needed to cover both periods
                    from datetime import timedelta
                    current_period_days = (target_dt - start_dt).days + 1
                    comparison_period_days = (comparison_end_date_for_analyzer - comparison_start_date_for_analyzer).days + 1
                    # Calculate the earliest date we need
                    earliest_date = min(start_dt, comparison_start_date_for_analyzer)
                    extended_days = (target_dt - earliest_date).days + 1
                    self.logger.info(f"VSLAST_DYNAMIC with specific dates: collecting {extended_days} days to cover both current period ({current_period_days} days) and comparison period ({comparison_period_days} days)")
                else:
                    # For regular vslast we need current period + previous period data
                    # If we're doing 7-day analysis, we need 14 days total (7 + 7)
                    extended_days = comparison_days * 2
                    self.logger.info(f"VSLAST mode: collecting {extended_days} days to cover both periods")
            else:
                # For mean mode, keep the original logic
                extended_days = comparison_days
            
            # Collect operational data using wrapper to track DAX queries
            # Pass the calculated comparison dates AND the current period start date
            operational_data = await self._collect_operative_data_with_query_tracking(
                node_path, target_dt, extended_days, 
                comparison_start_date=comparison_start_date_for_analyzer,
                comparison_end_date=comparison_end_date_for_analyzer,
                current_start_date=start_dt  # Pass the actual start date of the current period
            )
            
            if operational_data.empty:
                return f"No operational data available for {end_date}"
            
            # Clean the operational data by replacing empty strings with NaN for numeric columns
            cleaned_data = operational_data.copy()
            numeric_columns = ['Load_Factor', 'OTP15', 'Misconex', 'Mishandling']
            for col in numeric_columns:
                if col in cleaned_data.columns:
                    # Replace empty strings with NaN, then convert to numeric
                    cleaned_data[col] = cleaned_data[col].replace('', pd.NA)
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Convert date columns if this is flexible aggregated data (to match load_operative_data logic)
            if 'Min_Date' in cleaned_data.columns and 'Max_Date' in cleaned_data.columns:
                # Ensure date columns are datetime.date objects for proper comparison
                cleaned_data['Min_Date'] = pd.to_datetime(cleaned_data['Min_Date']).dt.date
                cleaned_data['Max_Date'] = pd.to_datetime(cleaned_data['Max_Date']).dt.date
                self.logger.info(f"🗓️ Converted date columns to datetime.date for flexible comparison compatibility")
            
            # Set the cleaned operational data directly in the analyzer
            operational_analyzer.operative_data[node_path] = cleaned_data
            
            # Get the enhanced specific explanations (OTP and Load Factor)
            target_date_str = target_dt.strftime('%Y-%m-%d')
            # Use the stored anomaly type instead of "unknown"
            anomaly_type_for_analysis = getattr(self, 'current_anomaly_type', 'unknown')
            self.logger.info(f"🔍 DEBUG OPERATIVE: anomaly_type_for_analysis = '{anomaly_type_for_analysis}'")
            
            # Calculate aggregation_days from date range
            aggregation_days = (target_dt - start_dt).days + 1
            self.logger.info(f"📅 Operative date filter: {start_dt.strftime('%Y-%m-%d')} to {target_dt.strftime('%Y-%m-%d')} ({aggregation_days} days for anomaly comparison)")
            
            specific_explanations = operational_analyzer.get_specific_explanations(
                node_path, target_date_str, anomaly_type_for_analysis, aggregation_days
            )
            
            # Get the comprehensive analysis
            analysis = operational_analyzer.analyze_operative_metrics(node_path, target_date_str)
            
            if "error" in analysis:
                return f"Operational analysis error: {analysis['error']}"
            
            # Build enhanced explanation using the parametrized logic
            operative_parts = []
            
            # Add comparison mode info
            comparison_info = f"📊 **OPERATIVE ANALYSIS** (Referencia: {comparison_context if comparison_context else 'período anterior'})"
            operative_parts.append(comparison_info)
            operative_parts.append("")
            
            # Show all metrics with changes vs reference
            metrics = analysis.get("metrics", {})
            if metrics:
                reference_desc = comparison_context if comparison_context else "período anterior"
                operative_parts.append(f"📊 **Métricas Operativas vs {reference_desc}**:")
                
                for metric, data in metrics.items():
                    # Skip metrics with no real data (all zeros or identical values)
                    current_val = data.get('current_value', data.get('current', 0))
                    previous_val = data.get('previous_value', data.get('previous', 0))
                    
                    # Skip Mishandling if both current and previous are 0 (no data for this segment)
                    if metric == 'Mishandling' and current_val == 0 and previous_val == 0:
                        continue
                        
                    direction = "↗️" if data['direction'] == 'higher' else "↘️"
                    
                    if comparison_mode in ["vslast", "vslast_dynamic"]:
                        delta = data.get('delta', data.get('difference', 0))
                        change_pct = data.get('change_pct', 0)
                        
                        # Determine if change supports or contradicts NPS anomaly
                        correlation_status = "❓"
                        self.logger.info(f"🔍 DEBUG CORRELATION: metric='{metric}', anomaly_type='{anomaly_type_for_analysis}', delta={delta}")
                        if anomaly_type_for_analysis == '-':
                            if metric in ['OTP15', 'Otp15', 'OTP15']:  # Direct correlation - worse punctuality = lower NPS
                                correlation_status = "✅ Explica NPS↓" if delta < 0 else "❌ Contradice NPS↓"
                            elif metric in ['Load_Factor', 'Misconex', 'Mishandling']:  # Inverse correlation - more issues = lower NPS
                                correlation_status = "✅ Explica NPS↓" if delta > 0 else "❌ Contradice NPS↓"
                        elif anomaly_type_for_analysis == '+':
                            if metric in ['OTP15', 'Otp15', 'OTP15']:  # Direct correlation - better punctuality = higher NPS
                                correlation_status = "✅ Explica NPS↑" if delta > 0 else "❌ Contradice NPS↑"
                            elif metric in ['Load_Factor', 'Misconex', 'Mishandling']:  # Inverse correlation - fewer issues = higher NPS
                                correlation_status = "✅ Explica NPS↑" if delta < 0 else "❌ Contradice NPS↑"
                        
                        metric_display = metric.replace('_', ' ').replace('adjusted', '').title()
                        operative_parts.append(f"   • **{metric_display}**: {current_val} vs {previous_val} ({reference_desc}) ({direction}{abs(delta):.1f}) {correlation_status}")
                    else:
                        baseline_value = data.get('week_average', data.get('previous_value', 'N/A'))
                        day_val = data.get('day_value', data.get('current_value', 'N/A'))
                        metric_display = metric.replace('_', ' ').title()
                        operative_parts.append(f"   • **{metric_display}**: {day_val} (actual) vs {baseline_value} ({reference_desc})")
            
            # Add correlation summary
            operative_parts.append("")
            correlation_summary = self._generate_correlation_summary(metrics, anomaly_type_for_analysis, comparison_mode)
            operative_parts.append(correlation_summary)
            
            return "\n".join(operative_parts)
            
        except Exception as e:
            self.logger.error(f"Error in operative data tool: {e}")
            return f"Error analyzing operational data: {str(e)}"
    
    def _get_metric_impact_summary(self, metric: str, direction: str) -> str:
        """Get short impact summary for a metric"""
        impact_map = {
            'Misconex': {
                'higher': 'more connection issues',
                'lower': 'fewer connection issues'
            },
            'Mishandling': {
                'higher': 'more baggage problems',  
                'lower': 'fewer baggage problems'
            }
        }
        return impact_map.get(metric, {}).get(direction, 'operational change')
    
    def _generate_correlation_summary(self, metrics: dict, anomaly_type: str, comparison_mode: str) -> str:
        """Generate correlation summary between operative metrics and NPS anomaly"""
        if not metrics:
            return "🤔 **Correlation**: No significant operational changes detected"
        
        supporting_metrics = []
        contradicting_metrics = []
        
        for metric, data in metrics.items():
            if data.get('is_significant', False):
                # Check if metric change supports the NPS anomaly
                if self._metric_supports_anomaly(metric, data['direction'], anomaly_type):
                    supporting_metrics.append(metric.replace('_', ' ').title())
                else:
                    contradicting_metrics.append(metric.replace('_', ' ').title())
        
        if supporting_metrics and not contradicting_metrics:
            return f"✅ **Correlation**: Strong operational explanation - {', '.join(supporting_metrics)} changes explain the {anomaly_type} NPS anomaly"
        elif supporting_metrics and contradicting_metrics:
            return f"🤔 **Correlation**: Mixed signals - {', '.join(supporting_metrics)} support the anomaly, but {', '.join(contradicting_metrics)} show contradictory patterns"
        elif contradicting_metrics:
            return f"❌ **Correlation**: Operational metrics ({', '.join(contradicting_metrics)}) contradict the {anomaly_type} NPS anomaly - other causes likely"
        else:
            return f"🔍 **Correlation**: No significant operational changes detected using {comparison_mode} comparison"
    
    def _metric_supports_anomaly(self, metric: str, direction: str, nps_anomaly_type: str) -> bool:
        """Determine if a metric deviation supports the observed NPS anomaly"""
        
        # For negative NPS anomalies, these directions support the anomaly
        negative_supporting = {
            'Load_Factor': 'higher',      # Higher LF = worse service
            'OTP15': 'lower',    # Lower OTP = worse experience  
            'Misconex': 'higher',         # Higher misconex = worse experience
            'Mishandling': 'higher'       # Higher mishandling = worse experience
        }
        
        # For positive NPS anomalies, opposite directions support
        positive_supporting = {
            'Load_Factor': 'lower',       # Lower LF = better service
            'OTP15': 'higher',   # Higher OTP = better experience
            'Misconex': 'lower',          # Lower misconex = better experience  
            'Mishandling': 'lower'        # Lower mishandling = better experience
        }
        
        if nps_anomaly_type == "negative":
            return negative_supporting.get(metric) == direction
        elif nps_anomaly_type == "positive":
            return positive_supporting.get(metric) == direction
        else:
            return False
    
    # =========================================================================
    # VERBATIMS TOOL - Refactored clean implementation
    # =========================================================================

    def _format_smart_verbatims(self, df: pd.DataFrame, limit: int = 30) -> str:
        """Format smart verbatims for LLM consumption"""
        if df.empty:
            return "No hay verbatims disponibles."
            
        # Clean column names (remove brackets if they exist)
        df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
        
        formatted_list = []
        # Ensure we don't exceed limit (though query should handle it)
        df_subset = df.head(limit)
        
        for idx, row in df_subset.iterrows():
            # Support both query aliases and raw names for robustness
            nps = row.get('NPS_Score', row.get('surveys_maritz[nps_all]', row.get('nps_all', 'N/A')))
            route = str(row.get('Route', row.get('Route_Master[route]', row.get('route', 'Unknown')))).strip().strip("'\"")
            date_val = row.get('Date', row.get('Date_Master[Date]', ''))
            verbatim = row.get('Verbatim', '').strip() if isinstance(row.get('Verbatim'), str) else ""
            
            # Fix date format
            date_str = "N/A"
            if date_val and date_val != "":
                date_str = str(date_val)
                if hasattr(date_val, 'strftime'):
                    date_str = date_val.strftime('%Y-%m-%d')
                elif ' ' in date_str:
                    date_str = date_str.split(' ')[0]
                
            if verbatim:
                formatted_list.append(f"- [{date_str}] [NPS {nps}] ({route}): \"{verbatim}\"")
            
        if not formatted_list:
            return "No hay comentarios con texto disponibles."
            
        return "\n".join(formatted_list)

    def _extract_routes_from_verbatims(self, df: pd.DataFrame) -> List[str]:
        """Extract unique routes from verbatims dataframe"""
        if df.empty:
            return []
            
        # Clean column names
        df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
            
        # Try clean alias first, then fallback
        route_col = 'Route' if 'Route' in df.columns else None
        if not route_col:
             # Try alternatives
             for col in ['Route_Master[route]', 'route', 'OD', 'Flight_Master[OD_Show]']:
                 if col in df.columns:
                     route_col = col
                     break
        
        if not route_col:
            return []

        raw_routes = df[route_col].dropna().unique().tolist()
        cleaned = []
        for r in raw_routes:
            r = str(r).strip().strip("'\"")  # strip leading/trailing quotes and whitespace
            if r and r != 'Unknown' and re.match(r'^[A-Z]{3}-[A-Z]{3}$', r):
                cleaned.append(r)
        return cleaned
    
    async def _get_focus_touchpoint_routes(
        self,
        node_path: str,
        start_dt: datetime,
        end_dt: datetime,
        top_n: int = 5
    ) -> List[str]:
        """
        Obtiene las rutas con peores CSATs del focus_touchpoint.
        En modo comparative: peor CSAT diff (mayor caída).
        En modo single: peor CSAT absoluto.
        Retorna lista de rutas IATA (e.g., ["MAD-UIO", "MAD-BOG"]).
        """
        if not self.focus_touchpoint:
            return []
        try:
            self.logger.info(f"🔍 Getting focus_touchpoint routes for '{self.focus_touchpoint}'...")

            # Get routes data from PBI
            df_routes = await self.pbi_collector.collect_routes_for_date_range(
                node_path=node_path,
                start_date=start_dt,
                end_date=end_dt,
                comparison_filter=self.causal_filter,
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date
            )

            if df_routes is None or df_routes.empty:
                self.logger.warning(f"⚠️ No routes data available for focus_touchpoint '{self.focus_touchpoint}'")
                return []

            # Find the column for this touchpoint's CSAT
            # Mapping from touchpoint name variants to routes DataFrame column names
            TOUCHPOINT_COLUMN_MAP = {
                'cabin crew': 'Crew',
                'crew': 'Crew',
                'check-in': 'Check_in',
                'check in': 'Check_in',
                'boarding': 'Boarding',
                'aircraft interior': 'Aircraft_interior',
                'aircraft_interior': 'Aircraft_interior',
                'wi-fi': 'Wi-Fi',
                'wifi': 'Wi-Fi',
                'ife': 'IFE',
                'f&b': 'F&B',
                'food & beverage': 'F&B',
                'food and beverage': 'F&B',
                'arrivals': 'Arrivals',
                'connections': 'Connections',
                'lounge': 'Lounge',
            }

            tp_key = self.focus_touchpoint.lower().strip()
            base_col = TOUCHPOINT_COLUMN_MAP.get(tp_key)

            # Fallback: try substring match if not in map
            if not base_col:
                tp_lower = tp_key.replace(' ', '_')
                for col in df_routes.columns:
                    if tp_lower in col.lower() and 'diff' not in col.lower():
                        base_col = col
                        break

            diff_col = f"{base_col}_diff" if base_col and f"{base_col}_diff" in df_routes.columns else None
            abs_col = base_col if base_col and base_col in df_routes.columns else None

            # Choose sort column based on study_mode
            if self.study_mode == "comparative" and diff_col:
                sort_col = diff_col
                ascending = True  # worst diff (most negative) first
            elif abs_col:
                sort_col = abs_col
                ascending = True  # worst absolute CSAT first
            else:
                self.logger.warning(
                    f"⚠️ No CSAT column found for touchpoint '{self.focus_touchpoint}' in routes data. "
                    f"Columns: {list(df_routes.columns)}"
                )
                return []

            # Get route column name
            route_col = None
            for col in df_routes.columns:
                if 'route' in col.lower():
                    route_col = col
                    break

            if not route_col:
                self.logger.warning(f"⚠️ No route column found in routes data")
                return []

            # Sort and get top N worst routes
            df_sorted = df_routes.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending)
            top_routes = df_sorted[route_col].head(top_n).tolist()

            self.logger.info(
                f"✅ Focus touchpoint '{self.focus_touchpoint}' worst routes "
                f"(sorted by {sort_col}): {top_routes}"
            )
            return [r for r in top_routes if r and r != 'Unknown' and re.match(r'^[A-Z]{3}-[A-Z]{3}$', str(r))]

        except Exception as e:
            self.logger.warning(f"⚠️ _get_focus_touchpoint_routes failed: {e}")
            return []

    async def _verbatims_tool(self, node_path: str, start_date: str, end_date: str, anomaly_type: str = "neutral") -> str:
        """
        Verbatims tool for COMPARATIVE mode
        Analyzes verbatims from TWO periods and compares the changes
        
        Args:
            node_path: Node path for filtering  
            start_date: Target period start date (YYYY-MM-DD)
            end_date: Target period end date (YYYY-MM-DD)
            anomaly_type: Type of anomaly (positive/negative/neutral) to filter verbatims
            
        Returns:
            Comparative analysis showing what changed between periods
        """
        from dashboard_analyzer.data_collection.chatbot_verbatims_collector import DateValidationError
        try:
            self.logger.info(f"🤖 VERBATIMS_TOOL COMPARATIVE MODE")
            self.logger.info(f"📅 Input dates — start_date: {start_date}, end_date: {end_date}")
            self.logger.info(f"🔍 causal_filter: {self.causal_filter}")

            # Validate dates before any query
            self._validate_analysis_dates(start_date, end_date, context="verbatims_tool comparative")

            # Calculate comparison dates
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
            
            comparison_start_dt, comparison_end_dt = self.calculate_dynamic_comparison_dates(start_dt, end_dt)
            comparison_start = comparison_start_dt.strftime('%Y-%m-%d')
            comparison_end = comparison_end_dt.strftime('%Y-%m-%d')
            
            self.logger.info(f"📅 Date flow: analysis [{start_date} → {end_date}] | comparison [{comparison_start} → {comparison_end}] | sending to PBI: target=[{start_date}, {end_date}], comparison=[{comparison_start}, {comparison_end}]")
            
            # --- Route and touchpoint orchestration ---
            # Build all_routes: union of identified_routes + focus_touchpoint_routes
            focus_tp_routes = []
            if self.focus_touchpoint:
                focus_tp_routes = await self._get_focus_touchpoint_routes(node_path, start_dt, end_dt)
                self.logger.info(f"🎯 Focus touchpoint '{self.focus_touchpoint}' routes: {focus_tp_routes}")

            identified_routes = getattr(self.tracker, 'identified_routes', [])
            all_routes = list(dict.fromkeys(identified_routes + focus_tp_routes))  # deduplicated, order preserved
            self.logger.info(f"🗺️ All routes to investigate: {all_routes}")

            # Collect verbatims per route
            route_verbatims = []  # list of (route, df_positive, df_negative)
            if all_routes:
                self.logger.info(f"📍 Collecting verbatims for {len(all_routes)} routes...")
                for route in all_routes:
                    self.logger.info(f"  → Route {route}: collecting positive and negative verbatims")
                    df_pos = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="positive", route_filter=route)
                    df_neg = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="negative", route_filter=route)
                    route_verbatims.append((route, df_pos, df_neg))

            # Collect verbatims filtered by focus_touchpoint
            tp_verbatims = None  # (df_positive, df_negative) or None
            if self.focus_touchpoint:
                self.logger.info(f"🎯 Collecting verbatims filtered by touchpoint '{self.focus_touchpoint}'...")
                df_tp_pos = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="positive", touchpoint_filter=self.focus_touchpoint)
                df_tp_neg = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="negative", touchpoint_filter=self.focus_touchpoint)
                tp_verbatims = (df_tp_pos, df_tp_neg)

            # Extract filters from node_path
            filters = self._get_chatbot_filters_from_node_path(node_path)
            
            # In PROD environment, go directly to PBI (skip chatbot)
            if self.environment == "prod":
                self.logger.info("📊 PROD environment: Using PBI directly for comparative verbatims analysis...")
            else:
                # OPTION 1: Try chatbot (preferred) - only in non-prod environments
                if self.chatbot_collector:
                    self.logger.info("🤖 Using chatbot for comparative verbatims analysis...")
                    
                    # Test connection
                    connection_success, connection_message = self.chatbot_collector.test_connection()
                    self.logger.info(f"🔍 Connection: {connection_message}")
                    
                    if connection_success:
                        try:
                            result = await self._analyze_verbatims_comparative_chatbot(
                                node_path=node_path,
                                target_start=start_date,
                                target_end=end_date,
                                comparison_start=comparison_start,
                                comparison_end=comparison_end,
                                filters=filters
                            )
                            
                            if result:
                                self.collected_data['verbatims'] = result
                                self.logger.info("✅ Chatbot comparative analysis completed")
                                return result
                            else:
                                self.logger.warning("⚠️ Chatbot analysis returned empty, falling back to PBI")
                                
                        except Exception as e:
                            self.logger.warning(f"⚠️ Chatbot analysis failed: {e}, falling back to PBI")
            
            # OPTION 2: PBI (default for prod, fallback for other environments)
            self.logger.info("📊 Using PBI for comparative verbatims analysis...")
            
            result = await self._analyze_verbatims_comparative_pbi(
                node_path=node_path,
                target_start=start_date,
                target_end=end_date,
                comparison_start=comparison_start,
                comparison_end=comparison_end,
                anomaly_type=anomaly_type,
                route_verbatims=route_verbatims,
                tp_verbatims=tp_verbatims,
                all_routes=all_routes,
                focus_tp_routes=focus_tp_routes
            )
            
            if result:
                self.collected_data['verbatims'] = result
                self.logger.info("✅ PBI comparative analysis completed")
                return result
            else:
                return f"📝 No verbatims data available for {node_path} in the specified periods"
                
        except DateValidationError as e:
            self.logger.error(f"❌ DATE VALIDATION FAILED in comparative verbatims: {e}")
            return f"ERROR: Fechas inválidas en verbatims_tool (comparative) — {str(e)}"
        except Exception as e:
            self.logger.error(f"❌ Error in comparative verbatims analysis: {e}")
            return f"ERROR in verbatims analysis: {str(e)}"
    
    async def _verbatims_tool_single_period(self, node_path: str, start_date: str, end_date: str, anomaly_type: str = "neutral") -> str:
        """
        Verbatims tool for SINGLE PERIOD mode
        Analyzes verbatims from ONLY the target period (no comparison)
        
        Args:
            node_path: Node path for filtering
            start_date: Period start date (YYYY-MM-DD)
            end_date: Period end date (YYYY-MM-DD)
            anomaly_type: Type of anomaly (positive/negative/neutral)
            
        Returns:
            Analysis of the single period
        """
        from dashboard_analyzer.data_collection.chatbot_verbatims_collector import DateValidationError
        try:
            self.logger.info(f"🤖 VERBATIMS_TOOL SINGLE PERIOD MODE ({anomaly_type})")
            self.logger.info(f"📅 Input dates — start_date: {start_date}, end_date: {end_date}")
            self.logger.info(f"🔍 causal_filter: {self.causal_filter}")

            # Validate dates before any query
            self._validate_analysis_dates(start_date, end_date, context="verbatims_tool single_period")

            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date

            self.logger.info(f"📅 Date flow: analysis [{start_date} → {end_date}] | sending to PBI: [{start_date}, {end_date}]")

            # --- Route and touchpoint orchestration ---
            focus_tp_routes = []
            if self.focus_touchpoint:
                focus_tp_routes = await self._get_focus_touchpoint_routes(node_path, start_dt, end_dt)
                self.logger.info(f"🎯 Focus touchpoint '{self.focus_touchpoint}' routes: {focus_tp_routes}")

            identified_routes = getattr(self.tracker, 'identified_routes', [])
            all_routes = list(dict.fromkeys(identified_routes + focus_tp_routes))
            self.logger.info(f"🗺️ All routes to investigate: {all_routes}")

            route_verbatims = []
            if all_routes:
                self.logger.info(f"📍 Collecting verbatims for {len(all_routes)} routes...")
                for route in all_routes:
                    self.logger.info(f"  → Route {route}: collecting positive and negative verbatims")
                    df_pos = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="positive", route_filter=route)
                    df_neg = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="negative", route_filter=route)
                    route_verbatims.append((route, df_pos, df_neg))

            tp_verbatims = None
            if self.focus_touchpoint:
                self.logger.info(f"🎯 Collecting verbatims filtered by touchpoint '{self.focus_touchpoint}'...")
                df_tp_pos = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="positive", touchpoint_filter=self.focus_touchpoint)
                df_tp_neg = self.pbi_collector.collect_smart_verbatims(node_path, start_dt, end_dt, anomaly_type="negative", touchpoint_filter=self.focus_touchpoint)
                tp_verbatims = (df_tp_pos, df_tp_neg)

            # Extract filters from node_path
            filters = self._get_chatbot_filters_from_node_path(node_path)
            
            # In PROD environment, go directly to PBI (skip chatbot)
            if self.environment == "prod":
                self.logger.info("📊 PROD environment: Using PBI directly for single period verbatims analysis...")
            else:
                # OPTION 1: Try chatbot (preferred) - only in non-prod environments
                if self.chatbot_collector:
                    self.logger.info("🤖 Using chatbot for single period verbatims analysis...")
                    
                    # Test connection
                    connection_success, connection_message = self.chatbot_collector.test_connection()
                    self.logger.info(f"🔍 Connection: {connection_message}")
                    
                    if connection_success:
                        try:
                            result = await self._analyze_verbatims_single_chatbot(
                                node_path=node_path,
                                start_date=start_date,
                                end_date=end_date,
                                filters=filters
                            )
                            
                            if result:
                                self.collected_data['verbatims'] = result
                                self.logger.info("✅ Chatbot single period analysis completed")
                                return result
                            else:
                                self.logger.warning("⚠️ Chatbot analysis returned empty, falling back to PBI")
                                
                        except Exception as e:
                            self.logger.warning(f"⚠️ Chatbot analysis failed: {e}, falling back to PBI")
            
            # OPTION 2: PBI (default for prod, fallback for other environments)
            self.logger.info("📊 Using PBI for single period verbatims analysis...")
            
            result = await self._analyze_verbatims_single_pbi(
                node_path=node_path,
                start_date=start_date,
                end_date=end_date,
                anomaly_type=anomaly_type,
                route_verbatims=route_verbatims,
                tp_verbatims=tp_verbatims,
                all_routes=all_routes,
                focus_tp_routes=focus_tp_routes
            )
            
            if result:
                self.collected_data['verbatims'] = result
                self.logger.info("✅ PBI single period analysis completed")
                return result
            else:
                return f"📝 No verbatims data available for {node_path} in {start_date} to {end_date}"
                
        except DateValidationError as e:
            self.logger.error(f"❌ DATE VALIDATION FAILED in single period verbatims: {e}")
            return f"ERROR: Fechas inválidas en verbatims_tool (single_period) — {str(e)}"
        except Exception as e:
            self.logger.error(f"❌ Error in single period verbatims analysis: {e}")
            return f"ERROR in verbatims analysis: {str(e)}"
    
    async def _analyze_verbatims_comparative_chatbot(
        self,
        node_path: str,
        target_start: str,
        target_end: str,
        comparison_start: str,
        comparison_end: str,
        filters: dict
    ) -> str:
        """Analyze verbatims using chatbot for TWO periods and compare"""
        try:
            # Extract segment info for better questions
            segment_desc = self._get_segment_description(node_path)
            
            self.logger.info("🤖 Question 1: Main problems in TARGET period...")
            
            # Question 1: Target period - Main problems (short and direct)
            question_target_problems = "¿Cuáles son los principales problemas?"
            
            answer_target_problems = self.chatbot_collector.ask_chatbot_question(
                question=question_target_problems,
                start_date=target_start,
                end_date=target_end,
                node_path=node_path,
                filters=filters,
                max_wait_time=120
            )
            
            if not answer_target_problems or not answer_target_problems.get('answer'):
                self.logger.warning("⚠️ No answer received for target problems")
                return None
            
            target_problems = answer_target_problems.get('answer', '')
            self.logger.info(f"✅ Target problems answer ({len(target_problems)} chars)")
            
            # Question 2: Target period - Affected routes
            self.logger.info("🤖 Question 2: Affected routes in TARGET period...")
            
            question_target_routes = "¿Cuáles son las rutas más mencionadas en comentarios negativos?"
            
            answer_target_routes = self.chatbot_collector.ask_chatbot_question(
                question=question_target_routes,
                start_date=target_start,
                end_date=target_end,
                node_path=node_path,
                filters=filters,
                max_wait_time=120
            )
            
            target_routes = answer_target_routes.get('answer', 'No se identificaron rutas específicas') if answer_target_routes else 'No disponible'
            self.logger.info(f"✅ Target routes answer ({len(target_routes)} chars)")
            
            # Question 3: Comparison period - Main problems
            self.logger.info("🤖 Question 3: Main problems in COMPARISON period...")
            
            question_comparison_problems = f"""¿Cuáles son los principales problemas?"""
            
            answer_comparison_problems = self.chatbot_collector.ask_chatbot_question(
                question=question_comparison_problems,
                start_date=comparison_start,
                end_date=comparison_end,
                node_path=node_path,
                filters=filters,
                max_wait_time=120
            )
            
            comparison_problems = answer_comparison_problems.get('answer', 'No hay datos disponibles para el período de comparación') if answer_comparison_problems else 'No disponible'
            self.logger.info(f"✅ Comparison problems answer ({len(comparison_problems)} chars)")
            
            # Question 4: Comparison period - Affected routes
            self.logger.info("🤖 Question 4: Affected routes in COMPARISON period...")
            
            question_comparison_routes = f"""¿Cuáles son las rutas más mencionadas en comentarios negativos?"""
            
            answer_comparison_routes = self.chatbot_collector.ask_chatbot_question(
                question=question_comparison_routes,
                start_date=comparison_start,
                end_date=comparison_end,
                node_path=node_path,
                filters=filters,
                max_wait_time=120
            )
            
            comparison_routes = answer_comparison_routes.get('answer', 'No se identificaron rutas específicas') if answer_comparison_routes else 'No disponible'
            self.logger.info(f"✅ Comparison routes answer ({len(comparison_routes)} chars)")
            
            # Question 5: COMPARATIVE ANALYSIS - Skip this for now (too long and complex)
            # Will rely on Questions 1-4 and let the agent synthesize
            self.logger.info("🤖 Skipping Question 5 (comparative synthesis) - using Q1-Q4 data instead")
            
            # Skip comparative question - too long and causing 500/502 errors
            comparative_analysis = "Análisis comparativo basado en las diferencias observadas entre períodos."
            
            # Build comprehensive result
            result = f"""🤖 ANÁLISIS COMPARATIVO DE VERBATIMS (Chatbot)

🎯 SEGMENTO ANALIZADO: {segment_desc}

📅 PERÍODO ANALIZADO: {target_start} a {target_end}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PRINCIPALES PROBLEMAS:
{target_problems}

🛫 RUTAS MÁS AFECTADAS:
{target_routes}

📅 PERÍODO DE COMPARACIÓN: {comparison_start} a {comparison_end}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PROBLEMAS EN ESE MOMENTO:
{comparison_problems}

🛫 RUTAS AFECTADAS ENTONCES:
{comparison_routes}

🔄 ANÁLISIS COMPARATIVO (Chatbot):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{comparative_analysis}

💡 INSIGHT CLAVE: El chatbot ha analizado las diferencias entre períodos para identificar qué cambió en la percepción del cliente y cómo esto explica la variación en el NPS.
"""
            
            # Store conversation data
            self.collected_data['verbatims_conversation'] = {
                'mode': 'comparative',
                'source': 'chatbot',
                'target_period': f"{target_start} to {target_end}",
                'comparison_period': f"{comparison_start} to {comparison_end}",
                'target_problems': target_problems,
                'target_routes': target_routes,
                'comparison_problems': comparison_problems,
                'comparison_routes': comparison_routes,
                'comparative_analysis': comparative_analysis,
                'node_path': node_path,
                'segment': segment_desc
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in chatbot comparative analysis: {e}")
            return None
    
    async def _analyze_verbatims_comparative_pbi(
        self,
        node_path: str,
        target_start: str,
        target_end: str,
        comparison_start: str,
        comparison_end: str,
        anomaly_type: str = "neutral",
        route_verbatims=None,
        tp_verbatims=None,
        all_routes=None,
        focus_tp_routes=None
    ) -> str:
        """Analyze verbatims using PBI for TWO periods and compare (SMART MODE)"""
        try:
            from datetime import datetime
            
            route_verbatims = route_verbatims or []
            all_routes = all_routes or []
            focus_tp_routes = focus_tp_routes or []

            self.logger.info(f"📊 Collecting SMART verbatims from PBI for target period ({anomaly_type})...")
            
            # Get target period verbatims
            target_start_dt = target_start if isinstance(target_start, datetime) else datetime.strptime(target_start, '%Y-%m-%d')
            target_end_dt = target_end if isinstance(target_end, datetime) else datetime.strptime(target_end, '%Y-%m-%d')
            
            # Get filters for tracking
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Track target period query
            query_target = self.pbi_collector._get_smart_verbatims_query(cabins, companies, hauls, target_start_dt, target_end_dt, anomaly_type)
            parameters_target = {
                "node_path": node_path,
                "period": "target",
                "start_date": target_start_dt.strftime('%Y-%m-%d'),
                "end_date": target_end_dt.strftime('%Y-%m-%d'),
                "anomaly_type": anomaly_type,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("verbatims_tool_smart_target", query_target, parameters_target)
            
            # Use SMART collection
            df_target = self.pbi_collector.collect_smart_verbatims(
                node_path, target_start_dt, target_end_dt, anomaly_type
            )
            
            # Extract routes from target period (crucial for triangulation)
            target_routes = self._extract_routes_from_verbatims(df_target)
            if target_routes:
                self.logger.info(f"🛣️ Extracted {len(target_routes)} routes from target verbatims: {target_routes}")
                # Store in collected_data for later triangulation
                if 'affected_routes' not in self.collected_data:
                    self.collected_data['affected_routes'] = {}
                self.collected_data['affected_routes']['verbatims'] = target_routes
            
            # Get comparison period verbatims
            self.logger.info("📊 Collecting SMART verbatims from PBI for comparison period...")
            
            comparison_start_dt = comparison_start if isinstance(comparison_start, datetime) else datetime.strptime(comparison_start, '%Y-%m-%d')
            comparison_end_dt = comparison_end if isinstance(comparison_end, datetime) else datetime.strptime(comparison_end, '%Y-%m-%d')
            
            # Track comparison period query
            query_comparison = self.pbi_collector._get_smart_verbatims_query(cabins, companies, hauls, comparison_start_dt, comparison_end_dt, anomaly_type)
            parameters_comparison = {
                "node_path": node_path,
                "period": "comparison",
                "start_date": comparison_start_dt.strftime('%Y-%m-%d'),
                "end_date": comparison_end_dt.strftime('%Y-%m-%d'),
                "anomaly_type": anomaly_type,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("verbatims_tool_smart_comparison", query_comparison, parameters_comparison)
            
            df_comparison = self.pbi_collector.collect_smart_verbatims(
                node_path, comparison_start_dt, comparison_end_dt, anomaly_type
            )
            
            if df_target.empty and df_comparison.empty:
                return None
            
            # Format using smart formatter
            target_text = self._format_smart_verbatims(df_target)
            comparison_text = self._format_smart_verbatims(df_comparison)
            
            segment_desc = self._get_segment_description(node_path)

            # Audit header
            date_source = 'pre-configured' if self.causal_filter == 'vs Sel. Period' else 'dynamic'
            audit_line = (
                f"🔍 AUDITORÍA: target=[{target_start} → {target_end}] | "
                f"comparison=[{comparison_start} → {comparison_end}] | "
                f"causal_filter={self.causal_filter} | date_source={date_source}"
            )
            
            result = f"""📊 ANÁLISIS COMPARATIVO DE VERBATIMS (SMART - {anomaly_type.upper()})

{audit_line}

🎯 SEGMENTO: {segment_desc}

📅 PERÍODO ANALIZADO ({target_start} a {target_end}):
Se muestran los comentarios más relevantes (Top 30 por longitud) filtrados por tipo de anomalía ({anomaly_type}).
{target_text}

📅 PERÍODO DE COMPARACIÓN ({comparison_start} a {comparison_end}):
{comparison_text}"""

            # Per-route sections
            if route_verbatims:
                result += "\n\n🗺️ VERBATIMS POR RUTA:"
                for route, df_pos, df_neg in route_verbatims:
                    pos_text = self._format_smart_verbatims(df_pos) if df_pos is not None and not df_pos.empty else "Sin comentarios positivos."
                    neg_text = self._format_smart_verbatims(df_neg) if df_neg is not None and not df_neg.empty else "Sin comentarios negativos."
                    result += f"\n\n  📍 Ruta {route}:\n  ✅ Positivos:\n{pos_text}\n  ❌ Negativos:\n{neg_text}"

            # Focus touchpoint section
            if tp_verbatims is not None and self.focus_touchpoint:
                df_tp_pos, df_tp_neg = tp_verbatims
                tp_pos_text = self._format_smart_verbatims(df_tp_pos) if df_tp_pos is not None and not df_tp_pos.empty else "Sin comentarios positivos."
                tp_neg_text = self._format_smart_verbatims(df_tp_neg) if df_tp_neg is not None and not df_tp_neg.empty else "Sin comentarios negativos."
                result += (
                    f"\n\n🎯 VERBATIMS FOCUS TOUCHPOINT — {self.focus_touchpoint}:\n"
                    f"  ✅ Positivos:\n{tp_pos_text}\n"
                    f"  ❌ Negativos:\n{tp_neg_text}"
                )
                if focus_tp_routes:
                    result += f"\n  🛫 Rutas con peor CSAT en '{self.focus_touchpoint}': {', '.join(focus_tp_routes)}"

            result += f"""

🔄 INSTRUCCIÓN PARA EL AGENTE:
Analiza los comentarios del período analizado para encontrar patrones repetitivos.
Busca coincidencias con las rutas extraídas: {', '.join(target_routes)}
Compara si estos temas aparecían en el período anterior."""

            if self.focus_touchpoint:
                result += f"\nPrioriza el análisis del touchpoint '{self.focus_touchpoint}' y sus rutas más afectadas: {', '.join(focus_tp_routes)}."

            # Store traceability data
            self.collected_data['verbatims_conversation'] = {
                'mode': 'comparative',
                'source': 'pbi',
                'target_period': f"{target_start} to {target_end}",
                'comparison_period': f"{comparison_start} to {comparison_end}",
                # Traceability fields
                'target_start': target_start,
                'target_end': target_end,
                'comparison_start': comparison_start,
                'comparison_end': comparison_end,
                'date_source': date_source,
                'causal_filter': self.causal_filter,
                # Filter fields
                'routes_investigated': all_routes,
                'focus_touchpoint_routes': focus_tp_routes,
                'touchpoint_filter': self.focus_touchpoint,
            }

            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in PBI comparative analysis: {e}")
            return None
    
    async def _analyze_verbatims_single_chatbot(
        self,
        node_path: str,
        start_date: str,
        end_date: str,
        filters: dict
    ) -> str:
        """Analyze verbatims using chatbot for a single period"""
        try:
            segment_desc = self._get_segment_description(node_path)
            
            self.logger.info("🤖 Question 1: Main problems...")
            
            # Question 1: Main problems (short and direct - filters already applied)
            question_problems = f"""¿Cuáles son los principales problemas?"""
            
            answer_problems = self.chatbot_collector.ask_chatbot_question(
                question=question_problems,
                start_date=start_date,
                end_date=end_date,
                node_path=node_path,
                filters=filters,
                max_wait_time=120
            )
            
            if not answer_problems or not answer_problems.get('answer'):
                self.logger.warning("⚠️ No answer received for problems")
                return None
            
            problems = answer_problems.get('answer', '')
            
            # Question 2: Affected routes
            self.logger.info("🤖 Question 2: Affected routes...")
            
            question_routes = f"""¿Cuáles son las rutas más mencionadas en comentarios negativos?"""
            
            answer_routes = self.chatbot_collector.ask_chatbot_question(
                question=question_routes,
                start_date=start_date,
                end_date=end_date,
                node_path=node_path,
                filters=filters,
                max_wait_time=120
            )
            
            routes = answer_routes.get('answer', 'No se identificaron rutas específicas') if answer_routes else 'No disponible'
            
            # Build result
            result = f"""🤖 ANÁLISIS DE VERBATIMS (Chatbot)

🎯 SEGMENTO: {segment_desc}
📅 PERÍODO: {start_date} a {end_date}

📊 PRINCIPALES PROBLEMAS Y QUEJAS:
{problems}

🛫 RUTAS MÁS AFECTADAS:
{routes}

💡 INSIGHT: Este análisis proporciona la perspectiva cualitativa directa de los clientes durante el período analizado.
"""
            
            # Store data
            self.collected_data['verbatims_single'] = {
                'mode': 'single',
                'source': 'chatbot',
                'period': f"{start_date} to {end_date}",
                'problems': problems,
                'routes': routes,
                'node_path': node_path,
                'segment': segment_desc
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in chatbot single period analysis: {e}")
            return None
    
    async def _analyze_verbatims_single_pbi(
        self,
        node_path: str,
        start_date: str,
        end_date: str,
        anomaly_type: str = "neutral",
        route_verbatims=None,
        tp_verbatims=None,
        all_routes=None,
        focus_tp_routes=None
    ) -> str:
        """Analyze verbatims using PBI for a single period (SMART MODE)"""
        try:
            from datetime import datetime

            route_verbatims = route_verbatims or []
            all_routes = all_routes or []
            focus_tp_routes = focus_tp_routes or []
            
            self.logger.info(f"📊 Collecting SMART verbatims from PBI ({anomaly_type})...")
            
            # Convert to datetime if needed
            start_dt = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get filters for tracking
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Track the DAX query
            query = self.pbi_collector._get_smart_verbatims_query(cabins, companies, hauls, start_dt, end_dt, anomaly_type)
            parameters = {
                "node_path": node_path,
                "start_date": start_dt.strftime('%Y-%m-%d'),
                "end_date": end_dt.strftime('%Y-%m-%d'),
                "anomaly_type": anomaly_type,
                "cabins": cabins,
                "companies": companies,
                "hauls": hauls
            }
            self.tracker.add_dax_query("verbatims_tool_smart", query, parameters)
            
            # Use SMART collection
            df = self.pbi_collector.collect_smart_verbatims(
                node_path, start_dt, end_dt, anomaly_type
            )
            
            if df.empty:
                return None
            
            # Extract routes
            target_routes = self._extract_routes_from_verbatims(df)
            if target_routes:
                self.logger.info(f"🛣️ Extracted {len(target_routes)} routes: {target_routes}")
                if 'affected_routes' not in self.collected_data:
                    self.collected_data['affected_routes'] = {}
                self.collected_data['affected_routes']['verbatims'] = target_routes
            
            # Format
            text = self._format_smart_verbatims(df)
            
            segment_desc = self._get_segment_description(node_path)

            # Audit header
            date_source = 'pre-configured' if self.causal_filter == 'vs Sel. Period' else 'dynamic'
            audit_line = (
                f"🔍 AUDITORÍA: target=[{start_date} → {end_date}] | "
                f"causal_filter={self.causal_filter} | date_source={date_source}"
            )
            
            result = f"""📊 ANÁLISIS DE VERBATIMS (SMART - {anomaly_type.upper()})

{audit_line}

🎯 SEGMENTO: {segment_desc}
📅 PERÍODO: {start_date} a {end_date}

COMENTARIOS RELEVANTES (Top 30, {anomaly_type}):
{text}"""

            # Per-route sections
            if route_verbatims:
                result += "\n\n🗺️ VERBATIMS POR RUTA:"
                for route, df_pos, df_neg in route_verbatims:
                    pos_text = self._format_smart_verbatims(df_pos) if df_pos is not None and not df_pos.empty else "Sin comentarios positivos."
                    neg_text = self._format_smart_verbatims(df_neg) if df_neg is not None and not df_neg.empty else "Sin comentarios negativos."
                    result += f"\n\n  📍 Ruta {route}:\n  ✅ Positivos:\n{pos_text}\n  ❌ Negativos:\n{neg_text}"

            # Focus touchpoint section
            if tp_verbatims is not None and self.focus_touchpoint:
                df_tp_pos, df_tp_neg = tp_verbatims
                tp_pos_text = self._format_smart_verbatims(df_tp_pos) if df_tp_pos is not None and not df_tp_pos.empty else "Sin comentarios positivos."
                tp_neg_text = self._format_smart_verbatims(df_tp_neg) if df_tp_neg is not None and not df_tp_neg.empty else "Sin comentarios negativos."
                result += (
                    f"\n\n🎯 VERBATIMS FOCUS TOUCHPOINT — {self.focus_touchpoint}:\n"
                    f"  ✅ Positivos:\n{tp_pos_text}\n"
                    f"  ❌ Negativos:\n{tp_neg_text}"
                )
                if focus_tp_routes:
                    result += f"\n  🛫 Rutas con peor CSAT en '{self.focus_touchpoint}': {', '.join(focus_tp_routes)}"

            result += f"""

🔄 INSTRUCCIÓN PARA EL AGENTE:
Analiza los problemas recurrentes y su relación con las rutas: {', '.join(target_routes)}"""

            if self.focus_touchpoint:
                result += f"\nPrioriza el análisis del touchpoint '{self.focus_touchpoint}' y sus rutas más afectadas: {', '.join(focus_tp_routes)}."

            # Store traceability data
            self.collected_data['verbatims_conversation'] = {
                'mode': 'single',
                'source': 'pbi',
                'target_period': f"{start_date} to {end_date}",
                # Traceability fields
                'target_start': start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d'),
                'target_end': end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d'),
                'comparison_start': None,
                'comparison_end': None,
                'date_source': date_source,
                'causal_filter': self.causal_filter,
                # Filter fields
                'routes_investigated': all_routes,
                'focus_touchpoint_routes': focus_tp_routes,
                'touchpoint_filter': self.focus_touchpoint,
            }

            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in PBI single period analysis: {e}")
            return None
    
    def _summarize_verbatims_period(self, df: pd.DataFrame, period_label: str) -> str:
        """Summarize verbatims for a single period"""
        import pandas as pd
        
        if df.empty:
            return f"No hay datos de verbatims para el período {period_label}"
        
        total_count = len(df)
        
        # Try to get sentiment if available
        sentiment_summary = ""
        if 'sentiment' in df.columns or 'sentiment_category' in df.columns:
            sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'sentiment_category'
            sentiment_dist = df[sentiment_col].value_counts()
            sentiment_summary = f"\n  Sentimiento: {sentiment_dist.to_dict()}"
        
        # Get sample verbatims
        text_col = None
        for col in ['verbatim_text', 'Verbatim', 'text', 'comment', '[Verbatim]']:
            if col in df.columns:
                text_col = col
                break
        
        samples = ""
        if text_col and not df[text_col].empty:
            sample_texts = df[text_col].dropna().head(3)
            samples = "\n  Ejemplos:\n" + "\n".join([f"    - {str(text)[:150]}..." for text in sample_texts])
        
        return f"""  Total: {total_count} comentarios{sentiment_summary}{samples}"""
    
    def _compare_verbatims_periods(self, df_target: pd.DataFrame, df_comparison: pd.DataFrame) -> str:
        """Compare verbatims between two periods"""
        
        if df_target.empty and df_comparison.empty:
            return "No hay datos para comparar"
        
        target_count = len(df_target) if not df_target.empty else 0
        comparison_count = len(df_comparison) if not df_comparison.empty else 0
        
        volume_change = target_count - comparison_count
        volume_pct = (volume_change / comparison_count * 100) if comparison_count > 0 else 0
        
        changes = [f"  • Volumen de comentarios: {target_count} vs {comparison_count} ({volume_pct:+.1f}%)"]
        
        # Compare sentiment if available
        if not df_target.empty and not df_comparison.empty:
            sentiment_col = None
            for col in ['sentiment', 'sentiment_category', 'verbatim_global_sentiment']:
                if col in df_target.columns and col in df_comparison.columns:
                    sentiment_col = col
                    break
            
            if sentiment_col:
                target_negative = (df_target[sentiment_col].str.lower() == 'negative').sum() if sentiment_col in df_target.columns else 0
                comp_negative = (df_comparison[sentiment_col].str.lower() == 'negative').sum() if sentiment_col in df_comparison.columns else 0
                
                target_neg_pct = target_negative / len(df_target) * 100 if len(df_target) > 0 else 0
                comp_neg_pct = comp_negative / len(df_comparison) * 100 if len(df_comparison) > 0 else 0
                
                changes.append(f"  • Comentarios negativos: {target_neg_pct:.1f}% vs {comp_neg_pct:.1f}% ({target_neg_pct - comp_neg_pct:+.1f}pp)")
        
        return "\n".join(changes)
    
    def _get_chatbot_filters_from_node_path(self, node_path: str) -> Dict:
        """Extract chatbot filters from node_path for the chatbot API"""
        try:
            filters = {}
            
            # Extract cabin information
            if '/Business' in node_path:
                filters['cabin'] = ['Business']
            elif '/Premium' in node_path:
                filters['cabin'] = ['Premium']
            elif '/Economy' in node_path:
                filters['cabin'] = ['Economy']
            
            # Extract haul information (use LH/SH as per API spec)
            if '/LH' in node_path:
                filters['haul'] = ['LH']
            elif '/SH' in node_path:
                filters['haul'] = ['SH']
            
            # Extract company information (use 'company' not 'fleet')
            if '/IB' in node_path:
                filters['company'] = ['IB']
            elif '/YW' in node_path:
                filters['company'] = ['YW']
            
            return filters
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting filters from node_path: {e}")
            return {}
    
    def _get_segment_description(self, node_path: str) -> str:
        """Get human-readable description of the segment from node_path"""
        try:
            parts = node_path.split('/')
            descriptions = []
            
            for part in parts:
                if part == 'Global':
                    descriptions.append('todos los segmentos')
                elif part == 'LH':
                    descriptions.append('Long Haul')
                elif part == 'SH':
                    descriptions.append('Short Haul')
                elif part == 'Business':
                    descriptions.append('Business Class')
                elif part == 'Premium':
                    descriptions.append('Premium Economy')
                elif part == 'Economy':
                    descriptions.append('Economy Class')
                elif part == 'IB':
                    descriptions.append('Iberia')
                elif part == 'YW':
                    descriptions.append('Iberia Express')
                else:
                    descriptions.append(part)
            
            return ' - '.join(descriptions) if descriptions else node_path
            
        except:
            return node_path
    
    # =========================================================================
    # END OF VERBATIMS TOOL
    # =========================================================================
    
    async def _ncs_tool(self, node_path: str, start_date: str, end_date: str, analysis_focus: str = "flights", temporal_comparison: bool = True) -> str:
        """
        Enhanced NCS tool for causal analysis that extracts:
        1. DISRUPTION CAUSES: Identifies root causes from incident patterns
        2. AFFECTED ROUTES: Maps specific routes impacted by incidents
        3. TEMPORAL COMPARISON: Compares current period vs previous period (NEW)
        
        For multi-day analysis (aggregation-days > 1), processes each day individually
        and consolidates results for comprehensive period analysis.
        
        Args:
            temporal_comparison: If True, compares current period with previous period of same length
        """
        try:
            print(f"🔧 DEBUG: NCS tool called for {node_path} from {start_date} to {end_date}")
            print(f"🔧 DEBUG: Temporal comparison: {'ENABLED' if temporal_comparison else 'DISABLED'}")
            self.logger.info(f"Collecting NCS operational incidents for {node_path} from {start_date} to {end_date}")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            # Calculate number of days in the range
            total_days = (end_dt - start_dt).days + 1
            print(f"📅 DEBUG: Processing {total_days} days of NCS data (from {start_date} to {end_date})")
            
            # Import NCS collector
            from ....data_collection.ncs_collector import NCSDataCollector
            
            # Initialize NCS collector with correct environment
            ncs_collector = NCSDataCollector(environment=self.environment)
            
            # Calculate comparison period dates if temporal comparison is enabled
            comparison_data = None
            comparison_start_dt = None
            comparison_end_dt = None
            
            if temporal_comparison:
                # Use specific comparison dates if available (for "vs Sel. Period"), otherwise calculate previous period
                if hasattr(self, 'comparison_start_date') and hasattr(self, 'comparison_end_date') and self.comparison_start_date and self.comparison_end_date:
                    # Use the specified comparison period dates
                    if isinstance(self.comparison_start_date, datetime):
                        comparison_start_dt = self.comparison_start_date
                        comparison_end_dt = self.comparison_end_date
                    else:
                        comparison_start_dt = datetime.strptime(self.comparison_start_date, '%Y-%m-%d')
                        comparison_end_dt = datetime.strptime(self.comparison_end_date, '%Y-%m-%d')
                    
                    comparison_start_date = comparison_start_dt.strftime('%Y-%m-%d')
                    comparison_end_date = comparison_end_dt.strftime('%Y-%m-%d')
                    
                    print(f"📅 DEBUG: Using specified comparison period: {comparison_start_date} to {comparison_end_date}")
                    self.logger.info(f"Temporal comparison enabled - comparing with specified period {comparison_start_date} to {comparison_end_date}")
                else:
                    # Calculate previous period of same length (default behavior)
                    comparison_end_dt = start_dt - timedelta(days=1)  # Day before current period
                    comparison_start_dt = comparison_end_dt - timedelta(days=total_days - 1)  # Same length backwards
                    
                    comparison_start_date = comparison_start_dt.strftime('%Y-%m-%d')
                    comparison_end_date = comparison_end_dt.strftime('%Y-%m-%d')
                    
                    print(f"📅 DEBUG: Calculated comparison period: {comparison_start_date} to {comparison_end_date}")
                    self.logger.info(f"Temporal comparison enabled - comparing with calculated period {comparison_start_date} to {comparison_end_date}")
            
            # Collect NCS data for the current period (processes each day individually)
            print(f"🔍 DEBUG: Starting day-by-day NCS data collection for CURRENT period...")
            
            try:
                ncs_data = ncs_collector.collect_ncs_data_for_date_range(start_dt, end_dt)
                print(f"✅ DEBUG: Current period data collected: {len(ncs_data)} rows")
                # 🔄 Filter current period data by segment immediately
                ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path, collect_unattributed_key="current")
                print(f"🔧 DEBUG: After segment filter ({node_path}) rows: {len(ncs_data)}")
                if ncs_data.empty:
                    print(f"⚠️ WARNING: Current period data is EMPTY - no incidents found for period {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
                else:
                    print(f"✅ DEBUG: Current period data sample columns: {list(ncs_data.columns)}")
                access_error = False
                error_msg = None
            except Exception as e:
                error_msg = str(e)
                print(f"❌ ERROR: NCS data collection failed: {error_msg}")
                
                # Check if it's an AWS access error
                if any(aws_error in error_msg.lower() for aws_error in ['invalidaccesskeyid', 'access denied', 'credentials', 'token']):
                    access_error = True
                    error_type = "AWS_ACCESS_ERROR"
                    print(f"🔑 Detected AWS access error - credentials may be expired")
                else:
                    access_error = True
                    error_type = "GENERAL_ERROR"
                
                # Create empty DataFrame for error case
                ncs_data = pd.DataFrame()
            
            # Collect comparison period data if temporal comparison is enabled and no access error
            if temporal_comparison and not access_error and comparison_start_dt is not None and comparison_end_dt is not None:
                print(f"🔍 DEBUG: Starting day-by-day NCS data collection for COMPARISON period...")
                print(f"🔍 DEBUG: Comparison period dates: {comparison_start_dt.strftime('%Y-%m-%d')} to {comparison_end_dt.strftime('%Y-%m-%d')}")
                try:
                    comparison_data = ncs_collector.collect_ncs_data_for_date_range(comparison_start_dt, comparison_end_dt)
                    print(f"✅ DEBUG: Comparison data collected: {len(comparison_data)} rows")
                    # 🔄 Filter comparison data by segment immediately
                    comparison_data = await self._filter_ncs_by_segment(comparison_data, node_path, collect_unattributed_key="comparison")
                    print(f"🔧 DEBUG: After segment filter ({node_path}) rows: {len(comparison_data)}")
                    if comparison_data.empty:
                        print(f"⚠️ WARNING: Comparison data is EMPTY - no incidents found for period {comparison_start_dt.strftime('%Y-%m-%d')} to {comparison_end_dt.strftime('%Y-%m-%d')}")
                    else:
                        print(f"✅ DEBUG: Comparison data sample columns: {list(comparison_data.columns)}")
                except Exception as e:
                    print(f"⚠️ ERROR: Comparison period data collection failed with exception: {str(e)}")
                    print(f"⚠️ ERROR: Exception type: {type(e).__name__}")
                    import traceback
                    print(f"⚠️ ERROR: Full traceback:\n{traceback.format_exc()}")
                    comparison_data = pd.DataFrame()  # Continue with current period only
            else:
                comparison_data = pd.DataFrame()
                print(f"🔍 DEBUG: Comparison data skipped - temporal_comparison={temporal_comparison}, access_error={access_error}")
                print(f"🔍 DEBUG: comparison_start_dt={comparison_start_dt}, comparison_end_dt={comparison_end_dt}")
                
                # Show why comparison was skipped
                if not temporal_comparison:
                    print(f"⚠️ Comparison skipped: temporal_comparison is False")
                elif access_error:
                    print(f"⚠️ Comparison skipped: access_error is True (current period failed)")
                elif comparison_start_dt is None or comparison_end_dt is None:
                    print(f"⚠️ Comparison skipped: comparison dates are None")
            
            # CASE 2: Access error - cannot interpret
            if access_error:
                self.collected_data['ncs_data'] = {
                    'analysis_summary': f"❌ NCS data source unavailable for {start_date} to {end_date} ({total_days} days)",
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_details': {},
                    'touchpoint_correlations': {},
                    'causal_confidence': 'no_data_source',
                    'explanation': f'Fuente de datos NCS no accesible - credenciales temporales pueden haber expirado o error de conectividad para período de {total_days} días',
                    'data_source_status': 'unavailable',
                    'error_details': error_msg[:200] if error_msg else 'Unknown error',
                    'days_analyzed': total_days,
                    'date_range': f"{start_date} to {end_date}",
                    'temporal_comparison_enabled': temporal_comparison,
                    'temporal_comparison_successful': False
                }
                return f"❌ NCS data source unavailable for the {total_days}-day period {start_date} to {end_date}. Access error prevents analysis of operational incidents. Technical details: {error_msg[:100] if error_msg else 'Unknown error'}"
            
            # CASE 1: No NCS data found for the entire period - could be good operational performance  
            if ncs_data.empty:
                # Store empty results with explanation - this is CASE 1: File found but no incidents
                self.collected_data['ncs_data'] = {
                    'analysis_summary': f"📅 No NCS operational incidents found for {start_date} to {end_date} ({total_days} days analyzed)",
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_details': {},
                    'touchpoint_correlations': {},
                    'causal_confidence': 'file_found_no_incidents',
                    'explanation': f'Archivos NCS revisados para {total_days} días pero sin incidentes operacionales - puede interpretarse como operación estable durante todo el período',
                    'data_source_status': 'available_but_empty',
                    'days_analyzed': total_days,
                    'date_range': f"{start_date} to {end_date}"
                }
                return f"📅 No NCS operational incidents found for the {total_days}-day period {start_date} to {end_date}. This could indicate good operational performance during the anomaly period."
            
            # TEMPORAL COMPARISON ANALYSIS (NEW)
            temporal_analysis = None
            comparison_successful = False
            
            if temporal_comparison and comparison_data is not None and not comparison_data.empty and comparison_start_dt is not None and comparison_end_dt is not None:
                print(f"📊 DEBUG: Starting temporal comparison analysis...")
                # Filter both current and comparison data by segment for consistent analysis
                filtered_current_data_for_temporal = await self._filter_ncs_by_segment(ncs_data, node_path)
                filtered_comparison_data_for_temporal = await self._filter_ncs_by_segment(comparison_data, node_path)
                print(f"🔧 DEBUG: Temporal analysis using filtered data - Current: {len(filtered_current_data_for_temporal)}, Comparison: {len(filtered_comparison_data_for_temporal)}")
                
                temporal_analysis = self._create_temporal_ncs_comparison(
                    filtered_current_data_for_temporal, filtered_comparison_data_for_temporal, node_path, 
                    start_date, end_date, 
                    comparison_start_dt.strftime('%Y-%m-%d'), comparison_end_dt.strftime('%Y-%m-%d')
                )
                comparison_successful = True
                print(f"✅ DEBUG: Temporal comparison completed successfully")
            elif temporal_comparison:
                print(f"⚠️ DEBUG: Temporal comparison enabled but comparison data unavailable")
            
            # Filter NCS data by segment routes using Routes Dictionary
            filtered_ncs_data = await self._filter_ncs_by_segment(ncs_data, node_path)
            
            if filtered_ncs_data.empty:
                # SUBCASO 3A: Incidentes existen globalmente pero no en el segmento específico
                total_global_incidents = len(ncs_data)
                
                self.collected_data['ncs_data'] = {
                    'analysis_summary': f"📊 Incidentes operacionales detectados en otras rutas durante {total_days} días - {node_path} no afectado",
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_details': {},
                    'touchpoint_correlations': {},
                    'causal_confidence': 'file_found_no_segment_incidents',
                    'explanation': f'Se encontraron {total_global_incidents} incidentes operacionales pero ninguno en rutas del segmento {node_path}',
                    'data_source_status': 'available_but_empty_for_segment',
                    'incidents_in_other_segments': True,
                    'non_correlated_incidents': total_global_incidents,
                    'days_analyzed': total_days,
                    'date_range': f"{start_date} to {end_date}"
                }
                return f"📊 Se encontraron {total_global_incidents} incidentes operacionales durante el período de {total_days} días ({start_date} a {end_date}), pero ninguno afectó las rutas del segmento {node_path}. El segmento analizado no se vio impactado por estos incidentes."
            
            # EXTRACT STRUCTURED NCS DATA for multi-day aggregation analysis
            incident_col = self._find_column(filtered_ncs_data, ['incident', 'incidents', ''])
            if incident_col is None:
                incident_col = filtered_ncs_data.columns[0] if not filtered_ncs_data.empty else ''
            
            all_incidents_text = "\n".join(filtered_ncs_data[incident_col].astype(str).tolist()) if not filtered_ncs_data.empty else ""
            structured_ncs_data = self._extract_structured_ncs_data(all_incidents_text)
            
            # ENHANCED CAUSAL ANALYSIS - Try new agent reflection approach first
            # Get anomaly type from stored attribute or default to unknown
            current_anomaly_type = getattr(self, 'current_anomaly_type', 'unknown')
            
            # NPS variation for the node (comes from investigation context, not from NCS temporal_analysis)
            nps_variation = getattr(self, 'nps_difference_value', None)
            temporal_incident_changes = {}
            if temporal_analysis:
                # Extract incident type deltas from temporal comparison
                incident_type_deltas = temporal_analysis.get('incident_type_deltas', {})
                if incident_type_deltas:
                    temporal_incident_changes = incident_type_deltas
            
            # Prepare comparison data for dark horses detection
            comparison_data_for_reflection = None
            comparison_start_str = None
            comparison_end_str = None
            if temporal_comparison and comparison_data is not None and not comparison_data.empty:
                # Use the already filtered comparison data if it exists
                if 'filtered_comparison_data_for_temporal' in locals():
                    comparison_data_for_reflection = filtered_comparison_data_for_temporal
                else:
                    # Fall back to filtering again if needed
                    comparison_data_for_reflection = await self._filter_ncs_by_segment(comparison_data, node_path)
                if comparison_start_dt:
                    comparison_start_str = comparison_start_dt.strftime('%Y-%m-%d')
                if comparison_end_dt:
                    comparison_end_str = comparison_end_dt.strftime('%Y-%m-%d')
            
            try:
                # First, try the new agent reflection approach
                self.logger.info(f"🤖 Trying new NCS agent reflection approach...")
                self.logger.info(f"📊 MAIN: About to send to agent reflection - data shape: {filtered_ncs_data.shape}")
                causal_analysis = await self._ncs_reflection_with_agent(
                    filtered_ncs_data=filtered_ncs_data,
                    node_path=node_path,
                    anomaly_type=current_anomaly_type,
                    total_days=total_days,
                    # New dark horses parameters
                    comparison_data=comparison_data_for_reflection,
                    current_start_date=start_date,
                    current_end_date=end_date,
                    comparison_start_date=comparison_start_str,
                    comparison_end_date=comparison_end_str,
                    nps_variation=nps_variation,
                    temporal_incident_changes=temporal_incident_changes
                )
                
                # INTEGRATE STRUCTURED NCS DATA into causal analysis
                causal_analysis['structured_ncs_data'] = structured_ncs_data
                
                # If agent reflection fails or returns no results, fall back to pattern matching
                if (not causal_analysis.get('identified_causes') and 
                    not causal_analysis.get('affected_routes') and
                    causal_analysis.get('confidence_level') != 'high_agent_analysis'):
                    self.logger.info(f"🔄 Agent reflection returned empty results, falling back to pattern matching...")
                    causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(filtered_ncs_data, node_path, current_anomaly_type)
                else:
                    self.logger.info(f"✅ Agent reflection successful: {len(causal_analysis.get('identified_causes', []))} causes, {len(causal_analysis.get('affected_routes', []))} routes")
                    
            except Exception as e:
                self.logger.error(f"Error in NCS agent reflection, falling back to pattern matching: {str(e)}")
                causal_analysis = await self._extract_ncs_causal_insights_workflow_aware(filtered_ncs_data, node_path, current_anomaly_type)
            
            # Format results for agent understanding with WORKFLOW-AWARE FOCUS
            analysis_result = []
            analysis_result.append(f"🎯 NCS SEGMENT ANALYSIS: {node_path}")
            analysis_result.append(f"📅 Period: {start_date} to {end_date}")
            
            # Show workflow match status
            if causal_analysis.get('workflow_match'):
                incident_nature = causal_analysis.get('incident_nature', 'unknown')
                analysis_result.append(f"✅ WORKFLOW MATCH: {incident_nature.upper()} incidents found matching {incident_nature} analysis workflow")
            elif causal_analysis.get('workflow_mismatch'):
                current_workflow = causal_analysis.get('current_workflow', 'unknown')
                available_types = causal_analysis.get('available_incident_types', [])
                analysis_result.append(f"⚠️ WORKFLOW MISMATCH: Current workflow is {current_workflow.upper()}, but available incidents are {', '.join(available_types)}")
                analysis_result.append(f"🔄 Consider switching to appropriate workflow or incidents may not be relevant to current analysis")
            
            # PRIORITY 1: DISRUPTION CAUSES IDENTIFICADAS
            if causal_analysis['identified_causes']:
                analysis_result.append(f"🚨 DISRUPTION CAUSES IDENTIFICADAS:")
                for i, cause in enumerate(causal_analysis['identified_causes'][:3], 1):
                    analysis_result.append(f"  {i}. {cause}")
            
            # PRIORITY 2: AFFECTED ROUTES MAPPED
            if causal_analysis['affected_routes']:
                analysis_result.append(f"🛤️ AFFECTED ROUTES ({len(causal_analysis['affected_routes'])} routes):")
                route_impacts = causal_analysis['route_impact_summary']
                for route, impact in list(route_impacts.items())[:4]:  # Show top 4
                    analysis_result.append(f"  • {route}: {impact}")
            
            # STRUCTURED NCS BREAKDOWN - Information aggregated from actual fields
            structured_breakdown = causal_analysis.get('structured_ncs_data', {})
            if structured_breakdown and structured_breakdown.get('summary', {}).get('total_incidents', 0) > 0:
                summary = structured_breakdown['summary']
                categories = structured_breakdown.get('categories', {})
                motives = structured_breakdown.get('motives_breakdown', {})
                passengers = structured_breakdown.get('passenger_impact', {})
                delays = structured_breakdown.get('delay_statistics', {})
                
                analysis_result.append(f"📊 NCS STRUCTURED BREAKDOWN:")
                
                # Categories breakdown
                if categories:
                    cat_summary = [f"{cat}: {count}" for cat, count in categories.items() if count > 0]
                    if cat_summary:
                        analysis_result.append(f"   📋 Categorías: {', '.join(cat_summary)}")
                
                # Top motives 
                if motives:
                    top_motives = sorted(motives.items(), key=lambda x: x[1], reverse=True)[:3]
                    motive_summary = [f"{motive}: {count}" for motive, count in top_motives]
                    analysis_result.append(f"   🔧 Motivos principales: {', '.join(motive_summary)}")
                
                # Passenger impact
                if passengers.get('total', 0) > 0:
                    pax_breakdown = f"J:{passengers.get('j_class', 0)}, W:{passengers.get('w_class', 0)}, Y:{passengers.get('y_class', 0)} = {passengers['total']} pax"
                    analysis_result.append(f"   👥 Pasajeros afectados: {pax_breakdown}")
                
                # Delay statistics  
                if delays.get('count', 0) > 0:
                    analysis_result.append(f"   ⏱️ Retrasos: {delays['count']} vuelos, promedio {delays.get('avg_delay', 0):.1f} min, total {delays.get('total_minutes', 0)} min")
                
                # Most affected route from structured data
                if summary.get('most_affected_route'):
                    route, count = summary['most_affected_route']
                    analysis_result.append(f"   🛫 Ruta más afectada: {route} ({count} incidentes)")
                
                # ENHANCED: ALL affected routes with detailed incident breakdown by type
                route_incident_breakdown = structured_breakdown.get('route_incident_breakdown', {})
                route_disruptions = structured_breakdown.get('route_disruptions', {})
                
                if route_incident_breakdown:
                    # Intelligent sorting: by total incidents (desc), then by severity of incident types
                    def route_sort_key(item):
                        route, breakdown = item
                        total = breakdown['total']
                        # Weight by incident severity: cancelaciones > desvios > limitacion > retrasos > otras
                        severity_weight = (breakdown['cancelaciones'] * 5 + 
                                         breakdown['desvios'] * 4 + 
                                         breakdown['limitacion_aeronave'] * 3 + 
                                         breakdown['retrasos'] * 2 + 
                                         breakdown['otras_incidencias'] * 1)
                        return (total, severity_weight)
                    
                    sorted_routes = sorted(route_incident_breakdown.items(), key=route_sort_key, reverse=True)
                    
                    # Build detailed route list with incident type breakdown
                    routes_details = []
                    for route, breakdown in sorted_routes:
                        # Create compact breakdown showing only non-zero incident types
                        incident_parts = []
                        if breakdown['cancelaciones'] > 0:
                            incident_parts.append(f"C:{breakdown['cancelaciones']}")
                        if breakdown['desvios'] > 0:
                            incident_parts.append(f"D:{breakdown['desvios']}")
                        if breakdown['retrasos'] > 0:
                            incident_parts.append(f"R:{breakdown['retrasos']}")
                        if breakdown['otras_incidencias'] > 0:
                            incident_parts.append(f"O:{breakdown['otras_incidencias']}")
                        if breakdown['limitacion_aeronave'] > 0:
                            incident_parts.append(f"L:{breakdown['limitacion_aeronave']}")
                        
                        incident_detail = '|'.join(incident_parts) if incident_parts else f"{breakdown['total']}"
                        routes_details.append(f"{route}({incident_detail})")
                    
                    analysis_result.append(f"   🛫 RUTAS AFECTADAS: {', '.join(routes_details)}")
                    analysis_result.append(f"   📝 Leyenda: C=Cancelaciones, D=Desvíos, R=Retrasos, O=Otras, L=Limitación aeronave")
                    analysis_result.append(f"   📊 TOTAL: {len(sorted_routes)} rutas, {sum([b['total'] for b in route_incident_breakdown.values()])} incidentes")
                    
                elif route_disruptions:
                    # Fallback to simple route count if detailed breakdown not available
                    sorted_routes = sorted(route_disruptions.items(), key=lambda x: x[1], reverse=True)
                    routes_text = ', '.join([f"{route}({count})" for route, count in sorted_routes])
                    analysis_result.append(f"   🛫 RUTAS AFECTADAS: {routes_text}")
                    analysis_result.append(f"   📊 TOTAL: {len(sorted_routes)} rutas, {sum(route_disruptions.values())} incidentes")
            
            # Show incident count for segment with multi-day context
            segment_incident_count = len(filtered_ncs_data)
            # Store total for narrative use
            self._ncs_total_incidents = segment_incident_count
            
            if total_days > 1:
                analysis_result.append(f"📊 SEGMENT INCIDENTS: {segment_incident_count} incidents found on {node_path} routes across {total_days} days ({start_date} to {end_date})")
            else:
                analysis_result.append(f"📊 SEGMENT INCIDENTS: {segment_incident_count} incidents found on {node_path} routes")
            
            # Add temporal comparison to stored data if available
            if temporal_analysis is not None:
                causal_analysis['temporal_comparison'] = temporal_analysis
                causal_analysis['temporal_comparison_successful'] = comparison_successful
                
                # Add temporal comparison to analysis result - make it more prominent
                if temporal_analysis.get('analysis_summary'):
                    # Replace the pipe separators with newlines for better readability
                    temporal_summary = temporal_analysis['analysis_summary'].replace(' | ', '\n')
                    analysis_result.append(f"\n{temporal_summary}")
                    
                # Skip redundant temporal insights as they're now included in the comprehensive summary
                
                # Add combined comments summary for operational narrative
                if temporal_comparison and comparison_start_dt and comparison_end_dt and comparison_data is not None:
                    # Filter comparison data by segment to ensure consistent filtering
                    filtered_comparison_data = await self._filter_ncs_by_segment(comparison_data, node_path)
                    print(f"🔧 DEBUG: Filtered comparison data for segment {node_path}: {len(filtered_comparison_data)}/{len(comparison_data)} incidents")
                    
                    combined_comments = self._combine_and_summarize_ncs_comments(
                        filtered_ncs_data, filtered_comparison_data,
                        start_date, end_date,
                        comparison_start_dt.strftime('%Y-%m-%d'), comparison_end_dt.strftime('%Y-%m-%d')
                    )
                    if combined_comments and "Error" not in combined_comments:
                        analysis_result.append(f"\n📋 **NARRATIVA OPERATIVA COMBINADA:**")
                        analysis_result.append(combined_comments)
            
            # Store comprehensive data
            self.collected_data['ncs_data'] = causal_analysis
            
            result = " | ".join(analysis_result)
            return result
            
        except Exception as e:
            import traceback
            self.logger.error(f"💥 Error in NCS analysis: {str(e)}")
            self.logger.error(f"💥 Full traceback: {traceback.format_exc()}")
            return f"Error in NCS analysis: {str(e)}"

    async def _routes_tool(self, node_path: str, start_date: str, end_date: str, min_surveys: int = 2, anomaly_type: str = "unknown") -> str:
        """
        Enhanced tool for analyzing route-specific NPS performance from ALL sources:
        1. Routes from explanatory drivers (ordered by NPS and touchpoint satisfactions)
        2. Routes from NCS data (routes mentioned in operational incidents)
        3. Routes from verbatims (routes mentioned in customer feedback)
        """
        try:
            self.logger.info(f"🛫 ROUTES_TOOL CALLED - Enhanced comprehensive routes analysis")
            self.logger.debug(f"📋 Parameters: node_path={node_path}, start_date={start_date}, end_date={end_date}, anomaly_type={anomaly_type}")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            # Collect routes from all three sources
            all_routes_analysis = await self._consolidate_routes_from_all_sources(
                node_path, start_dt, end_dt, anomaly_type, min_surveys
            )
            
            # Store comprehensive data for cross-references
            self.collected_data['routes_data'] = all_routes_analysis
            
            return all_routes_analysis['analysis_summary']
                
        except Exception as e:
            self.logger.error(f"💥 Error in comprehensive routes analysis: {str(e)}")
            return f"Error in comprehensive routes analysis: {str(e)}"

    async def _consolidate_routes_from_all_sources(self, node_path: str, start_dt, end_dt, anomaly_type: str, min_surveys: int) -> dict:
        """
        Consolidate and analyze routes from all three sources:
        1. Explanatory drivers routes (based on main touchpoint drivers)
        2. NCS routes (from operational incidents)  
        3. Verbatims routes (from customer feedback)
        """
        
        cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        
        # 1. EXPLANATORY DRIVERS ROUTES - Get routes ordered by main touchpoint drivers
        exp_drivers_routes = await self._get_explanatory_drivers_routes(
            node_path, cabins, companies, hauls, start_dt, end_dt, anomaly_type, min_surveys
        )
        
        # 2. NCS ROUTES - Get routes from operational incidents
        ncs_routes = await self._get_ncs_routes(
            node_path, cabins, companies, hauls, start_dt, end_dt
        )
        
        # 3. VERBATIMS ROUTES - Get routes from customer feedback
        verbatims_routes = await self._get_verbatims_routes(
            node_path, cabins, companies, hauls, start_dt, end_dt
        )
        
        # 4. CONSOLIDATE ALL ROUTES
        consolidated_analysis = self._create_consolidated_routes_analysis(
            exp_drivers_routes, ncs_routes, verbatims_routes, 
            node_path, start_date, end_date, anomaly_type
        )
        
        return consolidated_analysis

    async def _get_explanatory_drivers_routes(self, node_path: str, cabins: List[str], companies: List[str], 
                                            hauls: List[str], start_dt, end_dt, anomaly_type: str, min_surveys: int) -> dict:
        """
        Get routes analysis based on explanatory drivers touchpoints.
        For each significant driver, identify the top 5 routes with biggest CSAT changes vs comparison period.
        For negative anomalies: worst CSAT routes first (biggest drops)
        For positive anomalies: best CSAT routes first (biggest improvements)
        """
        try:
            # Get main touchpoint drivers from collected explanatory drivers data
            main_touchpoints = []
            if 'explanatory_drivers' in self.collected_data:
                significant_drivers = self.collected_data['explanatory_drivers'].get('significant_drivers', [])
                # significant_drivers is a list of strings (touchpoints), not dictionaries
                main_touchpoints = significant_drivers[:3]  # Top 3 drivers
            
            if not main_touchpoints:
                self.logger.info("No main touchpoints found from explanatory drivers - using general routes query")
                return await self._get_general_routes_with_touchpoints(cabins, companies, hauls, start_dt, end_dt, anomaly_type, min_surveys)
            
            self.logger.info(f"🎯 Analyzing routes based on main touchpoint drivers: {main_touchpoints}")
            
            # Debug logging
            self.logger.info(f"🔍 DEBUG ROUTES: node_path={node_path}, start_date={start_dt}, end_date={end_dt}")
            self.logger.info(f"🔍 DEBUG ROUTES: causal_filter={self.causal_filter}, comparison_start_date={self.comparison_start_date}, comparison_end_date={self.comparison_end_date}")
            self.logger.info(f"🔍 DEBUG ROUTES: cabins={cabins}, companies={companies}, hauls={hauls}")
            
            # Use the pbi_collector method that properly handles comparison_filter
            df = await self._collect_routes_with_query_tracking(
                node_path=node_path,
                start_date=start_dt,
                end_date=end_dt,
                comparison_filter=self.causal_filter,  # Use the causal_filter from the agent
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date
            )
            
            # Debug logging for returned data
            self.logger.info(f"🔍 DEBUG ROUTES: DataFrame shape={df.shape}, columns={list(df.columns) if not df.empty else 'empty'}")
            if not df.empty:
                self.logger.info(f"🔍 DEBUG ROUTES: First few rows: {df.head(3).to_dict('records')}")
            
            if df.empty:
                return {"routes": [], "analysis": "❌ No explanatory drivers routes data found", "source": "explanatory_drivers"}
                
            # Clean column names
            df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
            
            # Find key columns
            route_col = self._find_column(df, ['route', 'Route', 'ROUTE'])
            nps_col = self._find_column(df, ['nps', 'NPS', 'Nps'])
            nps_diff_col = self._find_column(df, ['NPS diff', 'nps diff', 'nps_diff', 'vs'])
            pax_col = self._find_column(df, ['pax', 'n (route)', 'Pax', 'PAX'])
            
            if not route_col:
                return {"routes": [], "analysis": f"❌ Missing route column. Available: {list(df.columns)}", "source": "explanatory_drivers"}
            
            # Filter by minimum surveys
            if pax_col:
                df = df[df[pax_col].fillna(0) >= min_surveys]
            
            # Find touchpoint satisfaction columns for each driver (use CSAT diff for ordering)
            touchpoint_cols = {}
            touchpoint_abs_cols = {}  # Store absolute CSAT columns for display
            for touchpoint in main_touchpoints:
                # Map touchpoint names to CSAT diff column names (for ordering)
                touchpoint_diff_mapping = {
                    'check-in': 'Check_in_diff',
                    'check_in': 'Check_in_diff',
                    'lounge': 'Lounge_diff',
                    'boarding': 'Boarding_diff',
                    'aircraft interior': 'Aircraft_interior_diff',
                    'aircraft_interior': 'Aircraft_interior_diff',
                    'wi-fi': 'Wi-Fi_diff',
                    'wifi': 'Wi-Fi_diff',
                    'ife': 'IFE_diff',
                    'f&b': 'F&B_diff',
                    'fb': 'F&B_diff',
                    'crew': 'Crew_diff',
                    'cabin crew': 'Crew_diff',
                    'arrivals': 'Arrivals_diff',
                    'arrivals experience': 'Arrivals_diff',
                    'connections': 'Connections_diff',
                    'connections experience': 'Connections_diff',
                    'punctuality': 'Operative_diff'
                    # Note: Other touchpoints not available in routes data
                }
                
                # Map touchpoint names to absolute CSAT column names (for display)
                touchpoint_abs_mapping = {
                    'check-in': 'Check_in',
                    'check_in': 'Check_in',
                    'lounge': 'Lounge',
                    'boarding': 'Boarding',
                    'aircraft interior': 'Aircraft_interior',
                    'aircraft_interior': 'Aircraft_interior',
                    'wi-fi': 'Wi-Fi',
                    'wifi': 'Wi-Fi',
                    'ife': 'IFE',
                    'f&b': 'F&B',
                    'fb': 'F&B',
                    'crew': 'Crew',
                    'cabin crew': 'Crew',
                    'arrivals': 'Arrivals',
                    'arrivals experience': 'Arrivals',
                    'connections': 'Connections',
                    'connections experience': 'Connections',
                    'punctuality': 'Operative'
                    # Note: Other touchpoints not available in routes data
                }
                
                # Try to find the CSAT diff column (for ordering)
                exact_diff_col = touchpoint_diff_mapping.get(touchpoint.lower())
                if exact_diff_col and exact_diff_col in df.columns:
                    touchpoint_cols[touchpoint] = exact_diff_col
                    self.logger.info(f"✅ Found CSAT diff column for {touchpoint}: {exact_diff_col}")
                else:
                    self.logger.warning(f"⚠️ No CSAT diff column found for touchpoint: {touchpoint}")
                    continue
                
                # Try to find the absolute CSAT column (for display)
                exact_abs_col = touchpoint_abs_mapping.get(touchpoint.lower())
                if exact_abs_col and exact_abs_col in df.columns:
                    touchpoint_abs_cols[touchpoint] = exact_abs_col
                    self.logger.info(f"✅ Found absolute CSAT column for {touchpoint}: {exact_abs_col}")
                else:
                    self.logger.warning(f"⚠️ No absolute CSAT column found for touchpoint: {touchpoint}")
            
            if not touchpoint_cols:
                return {"routes": [], "analysis": f"❌ No touchpoint columns found for drivers: {main_touchpoints}. Available: {list(df.columns)}", "source": "explanatory_drivers"}
            
            # Prepare route analysis for each driver
            routes_analysis = []
            analysis_summary = f"🎯 EXPLANATORY DRIVERS ROUTES:\n"
            analysis_summary += f"   📊 Total routes analyzed: {len(df)}\n"
            analysis_summary += f"   🎯 Main touchpoint drivers: {', '.join(main_touchpoints)}\n"
            
            for touchpoint, col in touchpoint_cols.items():
                # Determine sorting direction based on SHAP value and anomaly type
                # Get SHAP value for this touchpoint from explanatory drivers data
                shap_value = 0.0
                if 'explanatory_drivers' in self.collected_data:
                    drivers_data = self.collected_data['explanatory_drivers'].get('all_drivers', [])
                    self.logger.info(f"🔍 DEBUG SHAP: Looking for touchpoint '{touchpoint}' in {len(drivers_data)} drivers")
                    for driver in drivers_data:
                        driver_touchpoint = driver.get('touchpoint', '')
                        # More robust comparison - check for partial matches
                        if (driver_touchpoint.lower() == touchpoint.lower() or 
                            touchpoint.lower() in driver_touchpoint.lower() or 
                            driver_touchpoint.lower() in touchpoint.lower()):
                            shap_value = driver.get('shap_value', 0.0)
                            self.logger.info(f"✅ DEBUG SHAP: Found match! SHAP value for {touchpoint}: {shap_value}")
                            break
                    if shap_value == 0.0:
                        self.logger.warning(f"⚠️ DEBUG SHAP: No SHAP value found for touchpoint '{touchpoint}'")
                
                # Sort routes by this touchpoint's CSAT difference (not absolute CSAT)
                # For negative SHAP values (negative impact): sort by worst CSAT diff first (biggest drops)
                # For positive SHAP values (positive impact): sort by best CSAT diff first (biggest improvements)
                
                # Handle None values before sorting to avoid comparison errors
                df_sorted = df.copy()
                df_sorted[col] = df_sorted[col].fillna(0)  # Replace None with 0 for sorting
                
                if shap_value < 0:
                    # Negative SHAP: sort by worst CSAT diff first (biggest drops)
                    sorted_df = df_sorted.sort_values(by=col, ascending=True)
                    sort_desc = f"worst {touchpoint} CSAT change first (negative SHAP: {shap_value:.3f})"
                else:
                    # Positive SHAP: sort by best CSAT diff first (biggest improvements)
                    sorted_df = df_sorted.sort_values(by=col, ascending=False)
                    sort_desc = f"best {touchpoint} CSAT change first (positive SHAP: {shap_value:.3f})"
            
                # Get top 5 routes for this driver
                top_routes = sorted_df.head(5)
                
                analysis_summary += f"\n   🔝 Top 5 routes by {touchpoint} ({sort_desc}):\n"
                
                for _, route in top_routes.iterrows():
                    # Get absolute CSAT value for display
                    abs_col = touchpoint_abs_cols.get(touchpoint)
                    csat_abs = round(route[abs_col], 1) if abs_col and pd.notna(route[abs_col]) else None
                    
                    route_info = {
                        "route": route[route_col],
                        "driver": touchpoint,
                        "driver_score": csat_abs,  # Absolute CSAT for display
                        "driver_diff": round(route[col], 1) if pd.notna(route[col]) else None,  # CSAT diff (used for ordering)
                        "nps": round(route[nps_col], 1) if nps_col and pd.notna(route[nps_col]) else None,
                        "nps_diff": round(route[nps_diff_col], 1) if nps_diff_col and pd.notna(route[nps_diff_col]) else None,
                        "pax": int(route[pax_col]) if pax_col and pd.notna(route[pax_col]) else 0,
                        "source": "explanatory_drivers"
                    }
                    
                    routes_analysis.append(route_info)
                    
                    # Add to analysis summary
                    driver_score_str = f"{route_info['driver_score']}" if route_info['driver_score'] is not None else "N/A"
                    nps_str = f", NPS {route_info['nps']}" if route_info['nps'] is not None else ""
                    nps_diff_str = f", vs {self.baseline_description or 'período previo'}: {route_info['nps_diff']:+.1f}" if route_info['nps_diff'] is not None else ""
                    analysis_summary += f"     • {route_info['route']}: {touchpoint} {driver_score_str}{nps_str}{nps_diff_str}, Pax {route_info['pax']}\n"
            
            return {
                "routes": routes_analysis,
                "analysis": analysis_summary,
                "source": "explanatory_drivers",
                "main_touchpoints": main_touchpoints
            }
            
        except Exception as e:
            self.logger.error(f"Error in explanatory drivers routes analysis: {e}")
            return {"routes": [], "analysis": f"❌ Explanatory drivers routes error: {str(e)[:100]}", "source": "explanatory_drivers"}

    async def _get_general_routes_with_touchpoints(self, cabins: List[str], companies: List[str], 
                                                 hauls: List[str], start_dt, end_dt, anomaly_type: str, min_surveys: int) -> dict:
        """Fallback method for general routes analysis when no specific touchpoints are available"""
        try:
            # Use the pbi_collector method with comparison filter
            df = await self._collect_routes_with_query_tracking(
                "Global", start_dt, end_dt,
                comparison_filter=self.causal_filter,
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date
            )
            
            if df.empty:
                return {"routes": [], "analysis": "❌ No general routes data found", "source": "explanatory_drivers"}
            
            # Clean column names safely
            df = self._safe_clean_columns(df, method="strip")
            route_col = self._find_column(df, ['route'])
            nps_col = self._find_column(df, ['nps'])
            pax_col = self._find_column(df, ['pax', 'n (route)'])
            
            if pax_col:
                df = df[df[pax_col].fillna(0) >= min_surveys]
            
            # Sort by NPS and NPS diff if available
            nps_diff_col = self._find_column(df, ['NPS diff', 'nps diff'])
            
            if anomaly_type in ['-', 'negative', 'neg']:
                sort_cols = [nps_col]
                sort_ascending = [True]
                if nps_diff_col:
                    sort_cols.append(nps_diff_col)
                    sort_ascending.append(True)  # Most negative diff first
                df = df.sort_values(by=sort_cols, ascending=sort_ascending)
            else:
                sort_cols = [nps_col]
                sort_ascending = [False]
                if nps_diff_col:
                    sort_cols.append(nps_diff_col)
                    sort_ascending.append(False)  # Most positive diff first
                df = df.sort_values(by=sort_cols, ascending=sort_ascending)
            
            routes_analysis = []
            for _, route in df.head(5).iterrows():
                routes_analysis.append({
                    "route": route[route_col],
                    "nps": round(route[nps_col], 1),
                    "pax": int(route[pax_col]) if pax_col else 0,
                    "touchpoint_scores": {}
                })
            
            return {
                "routes": routes_analysis,
                "analysis": f"📊 General routes analysis: {len(routes_analysis)} routes",
                "source": "explanatory_drivers"
            }
            
        except Exception as e:
            return {"routes": [], "analysis": f"❌ General routes error: {str(e)[:100]}", "source": "explanatory_drivers"}

    async def _get_ncs_routes(self, node_path: str, cabins: List[str], companies: List[str], hauls: List[str], start_dt, end_dt) -> dict:
        """Get routes from NCS operational incidents data"""
        try:
            # Get identified routes from NCS analysis
            identified_routes = getattr(self.tracker, 'identified_routes', [])
            ncs_routes_from_data = []
            if hasattr(self, 'collected_data') and 'ncs_data' in self.collected_data:
                ncs_routes_from_data = self.collected_data['ncs_data'].get('affected_routes', [])
            
            all_ncs_routes = list(set(identified_routes + ncs_routes_from_data))
            
            if not all_ncs_routes:
                return {"routes": [], "analysis": "📊 No routes identified from NCS incidents", "source": "ncs"}
            
            self.logger.info(f"🎯 Analyzing NCS-specific routes: {all_ncs_routes}")
            
            # Get data for these specific routes using the same method as explanatory drivers
            # to ensure we get the correct NPS_diff values
            df_ncs = await self.pbi_collector.collect_routes_for_date_range(
                node_path=node_path,
                start_date=start_dt,
                end_date=end_dt,
                comparison_filter=self.causal_filter,
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date
            )
            
            if df_ncs.empty:
                return {"routes": [], "analysis": f"❌ No data found for NCS routes: {', '.join(all_ncs_routes)}", "source": "ncs"}
            
            # Clean column names safely
            df_ncs = self._safe_clean_columns(df_ncs, method="replace")
            
            # Filter to only include the NCS routes
            route_col = self._find_column(df_ncs, ['route', 'Route', 'ROUTE'])
            if route_col:
                df_ncs = df_ncs[df_ncs[route_col].isin(all_ncs_routes)]
            
            if df_ncs.empty:
                return {"routes": [], "analysis": f"❌ No data found for NCS routes after filtering: {', '.join(all_ncs_routes)}", "source": "ncs"}
            
            nps_col = self._find_column(df_ncs, ['nps', 'NPS', 'Nps'])
            nps_diff_col = self._find_column(df_ncs, ['NPS diff', 'nps diff', 'nps_diff', 'vs'])
            pax_col = self._find_column(df_ncs, ['pax', 'n (route)', 'Pax', 'PAX'])
            
            routes_analysis = []
            for _, route in df_ncs.iterrows():
                route_info = {
                    "route": route[route_col] if route_col else "Unknown",
                    "nps": round(route[nps_col], 1) if nps_col and pd.notna(route[nps_col]) else None,
                    "pax": int(route[pax_col]) if pax_col and pd.notna(route[pax_col]) else 0,
                    "nps_diff": round(route[nps_diff_col], 1) if nps_diff_col and pd.notna(route[nps_diff_col]) else None
                }
                routes_analysis.append(route_info)
            
            analysis_summary = f"🚨 NCS OPERATIONAL INCIDENTS ROUTES:\n"
            analysis_summary += f"   📊 Routes found: {len(routes_analysis)}/{len(all_ncs_routes)}\n"
            analysis_summary += f"   🎯 Identified from operational incidents: {', '.join(all_ncs_routes)}\n"
            
            if routes_analysis:
                analysis_summary += f"   📈 Route details:\n"
                for route_info in routes_analysis:
                    nps_str = f"NPS {route_info['nps']}" if route_info['nps'] is not None else "NPS N/A"
                    nps_diff_str = f", vs {self.baseline_description or 'período previo'}: {route_info['nps_diff']:+.1f}" if route_info['nps_diff'] is not None else ""
                    analysis_summary += f"     • {route_info['route']}: {nps_str}{nps_diff_str}, Pax {route_info['pax']}\n"
            
            return {
                "routes": routes_analysis,
                "analysis": analysis_summary,
                "source": "ncs",
                "identified_routes": all_ncs_routes
            }
            
        except Exception as e:
            self.logger.error(f"Error in NCS routes analysis: {e}")
            return {"routes": [], "analysis": f"❌ NCS routes error: {str(e)[:100]}", "source": "ncs"}

    async def _get_verbatims_routes(self, node_path: str, cabins: List[str], companies: List[str], hauls: List[str], start_dt, end_dt) -> dict:
        """Get routes mentioned in verbatims/customer feedback from all sources"""
        try:
            # Source 1: Smart Verbatims (PBI) - NEW
            verbatims_routes = []
            if 'affected_routes' in self.collected_data and 'verbatims' in self.collected_data['affected_routes']:
                verbatims_routes.extend(self.collected_data['affected_routes']['verbatims'])
            
            # Source 2: Chatbot Conversation (Legacy/Non-prod)
            if 'verbatims_conversation' in self.collected_data:
                conversation_data = self.collected_data['verbatims_conversation']
                conversation_log = conversation_data.get('conversation_log', [])
                
                # Extract routes mentioned in conversation responses
                for exchange in conversation_log:
                    response = exchange.get('response', '')
                    routes_in_response = self.tracker._extract_routes_from_result(response)
                    verbatims_routes.extend(routes_in_response)
            
            # Remove duplicates
            verbatims_routes = list(set(verbatims_routes))
            
            if not verbatims_routes:
                return {"routes": [], "analysis": "📊 No routes mentioned in customer verbatims", "source": "verbatims"}
            
            self.logger.info(f"🎯 Analyzing verbatims-mentioned routes from all sources: {verbatims_routes}")
            
            # Get data for these routes using the same method as explanatory drivers
            # to ensure we get the correct NPS_diff values
            df_verbatims = await self.pbi_collector.collect_routes_for_date_range(
                node_path=node_path,
                start_date=start_dt,
                end_date=end_dt,
                comparison_filter=self.causal_filter,
                comparison_start_date=self.comparison_start_date,
                comparison_end_date=self.comparison_end_date
            )
            
            if df_verbatims.empty:
                return {"routes": [], "analysis": f"❌ No data found for verbatims routes: {', '.join(verbatims_routes)}", "source": "verbatims"}
            
            # Clean column names
            df_verbatims.columns = [col.replace('[', '').replace(']', '') for col in df_verbatims.columns]
            
            # Filter to only include the verbatims routes
            route_col = self._find_column(df_verbatims, ['route', 'Route', 'ROUTE'])
            if route_col:
                df_verbatims = df_verbatims[df_verbatims[route_col].isin(verbatims_routes)]
            
            if df_verbatims.empty:
                return {"routes": [], "analysis": f"❌ No data found for verbatims routes after filtering: {', '.join(verbatims_routes)}", "source": "verbatims"}
            
            nps_col = self._find_column(df_verbatims, ['nps', 'NPS', 'Nps'])
            nps_diff_col = self._find_column(df_verbatims, ['NPS diff', 'nps diff', 'nps_diff', 'vs'])
            pax_col = self._find_column(df_verbatims, ['pax', 'n (route)', 'Pax', 'PAX'])
            
            routes_analysis = []
            for _, route in df_verbatims.iterrows():
                route_info = {
                    "route": route[route_col] if route_col else "Unknown",
                    "nps": round(route[nps_col], 1) if nps_col and pd.notna(route[nps_col]) else None,
                    "pax": int(route[pax_col]) if pax_col and pd.notna(route[pax_col]) else 0,
                    "nps_diff": round(route[nps_diff_col], 1) if nps_diff_col and pd.notna(route[nps_diff_col]) else None
                }
                routes_analysis.append(route_info)
            
            analysis_summary = f"💬 CUSTOMER VERBATIMS ROUTES:\n"
            analysis_summary += f"   📊 Routes found: {len(routes_analysis)}/{len(verbatims_routes)}\n"
            analysis_summary += f"   🎯 Mentioned in customer feedback: {', '.join(verbatims_routes)}\n"
            
            if routes_analysis:
                analysis_summary += f"   📈 Route details:\n"
                for route_info in routes_analysis:
                    nps_str = f"NPS {route_info['nps']}" if route_info['nps'] is not None else "NPS N/A"
                    nps_diff_str = f", vs {self.baseline_description or 'período previo'}: {route_info['nps_diff']:+.1f}" if route_info['nps_diff'] is not None else ""
                    analysis_summary += f"     • {route_info['route']}: {nps_str}{nps_diff_str}, Pax {route_info['pax']}\n"
            
            return {
                "routes": routes_analysis,
                "analysis": analysis_summary,
                "source": "verbatims",
                "identified_routes": verbatims_routes
            }
            
        except Exception as e:
            self.logger.error(f"Error in verbatims routes analysis: {e}")
            return {"routes": [], "analysis": f"❌ Verbatims routes error: {str(e)[:100]}", "source": "verbatims"}

    def _create_consolidated_routes_analysis(self, exp_drivers_routes: dict, ncs_routes: dict, 
                                           verbatims_routes: dict, node_path: str, 
                                           start_date: str, end_date: str, anomaly_type: str) -> dict:
        """Create comprehensive analysis consolidating all route sources"""
        
        # Collect all unique routes
        all_routes = {}
        
        # Add explanatory drivers routes
        for route_info in exp_drivers_routes.get('routes', []):
            route_name = route_info['route']
            all_routes[route_name] = {
                'route': route_name,
                'nps': route_info['nps'],
                'pax': route_info['pax'],
                'vs': None,
                'touchpoint_scores': route_info.get('touchpoint_scores', {}),
                'sources': ['explanatory_drivers']
            }
        
        # Add NCS routes
        for route_info in ncs_routes.get('routes', []):
            route_name = route_info['route']
            if route_name in all_routes:
                all_routes[route_name]['sources'].append('ncs')
                if route_info.get('vs') is not None:
                    all_routes[route_name]['vs'] = route_info['vs']
            else:
                all_routes[route_name] = {
                    'route': route_name,
                    'nps': route_info['nps'],
                    'pax': route_info['pax'],
                    'vs': route_info.get('vs'),
                    'touchpoint_scores': {},
                    'sources': ['ncs']
                }
        
        # Add verbatims routes
        for route_info in verbatims_routes.get('routes', []):
            route_name = route_info['route']
            if route_name in all_routes:
                all_routes[route_name]['sources'].append('verbatims')
                if route_info.get('vs') is not None and all_routes[route_name]['vs'] is None:
                    all_routes[route_name]['vs'] = route_info['vs']
            else:
                all_routes[route_name] = {
                    'route': route_name,
                    'nps': route_info['nps'],
                    'pax': route_info['pax'],
                    'vs': route_info.get('vs'),
                    'touchpoint_scores': {},
                    'sources': ['verbatims']
                }
        
        # Create similarities analysis
        similarities_analysis = self._analyze_route_similarities(all_routes, exp_drivers_routes, ncs_routes, verbatims_routes)
        
        # Build comprehensive summary
        summary_parts = []
        summary_parts.append(f"🛫 COMPREHENSIVE ROUTES ANALYSIS")
        summary_parts.append(f"📅 Period: {start_date} to {end_date}")
        summary_parts.append(f"🎯 Segment: {node_path}")
        summary_parts.append(f"📊 Anomaly type: {anomaly_type}")
        summary_parts.append("")
        
        # Add individual source analyses
        summary_parts.append(exp_drivers_routes.get('analysis', ''))
        summary_parts.append("")
        summary_parts.append(ncs_routes.get('analysis', ''))
        summary_parts.append("")
        summary_parts.append(verbatims_routes.get('analysis', ''))
        summary_parts.append("")
        
        # Add consolidated routes list
        if all_routes:
            summary_parts.append("🔄 CONSOLIDATED ROUTES FROM ALL SOURCES:")
            summary_parts.append(f"   📊 Total unique routes: {len(all_routes)}")
            summary_parts.append("")
            
            for route_name, route_data in all_routes.items():
                sources_str = ", ".join(route_data['sources'])
                vs_str = f", VS {route_data['vs']}" if route_data['vs'] is not None else ""
                touchpoint_str = ""
                if route_data['touchpoint_scores']:
                    touchpoint_scores = ", ".join([f"{tp}: {score}" for tp, score in route_data['touchpoint_scores'].items()])
                    touchpoint_str = f", Touchpoints: [{touchpoint_scores}]"
                
                summary_parts.append(f"   • {route_name}: NPS {route_data['nps']}, Pax {route_data['pax']}{vs_str}{touchpoint_str}")
                summary_parts.append(f"     Sources: {sources_str}")
                summary_parts.append("")
        
        # Add similarities analysis
        summary_parts.append(similarities_analysis)
        
        consolidated_summary = "\n".join(summary_parts)
        
        return {
            'analysis_summary': consolidated_summary,
            'all_routes': all_routes,
            'exp_drivers_routes': exp_drivers_routes,
            'ncs_routes': ncs_routes,
            'verbatims_routes': verbatims_routes,
            'similarities_analysis': similarities_analysis,
            'total_routes': len(all_routes),
            'date_range': f"{start_date} to {end_date}",
            'node_path': node_path,
            'anomaly_type': anomaly_type
        }

    def _analyze_route_similarities(self, all_routes: dict, exp_drivers_routes: dict, 
                                  ncs_routes: dict, verbatims_routes: dict) -> str:
        """Analyze similarities and patterns among routes from different sources"""
        
        analysis_parts = []
        analysis_parts.append("🔍 ROUTES SIMILARITY ANALYSIS:")
        
        # Find routes mentioned in multiple sources
        multi_source_routes = {name: data for name, data in all_routes.items() if len(data['sources']) > 1}
        
        if multi_source_routes:
            analysis_parts.append(f"   🎯 Routes mentioned in multiple sources: {len(multi_source_routes)}")
            for route_name, route_data in multi_source_routes.items():
                sources_str = " + ".join(route_data['sources'])
                analysis_parts.append(f"     • {route_name}: Found in {sources_str}")
        else:
            analysis_parts.append("   📊 No routes found in multiple sources - each source identifies different routes")
        
        # Geographic patterns
        route_patterns = self._identify_route_patterns(list(all_routes.keys()))
        if route_patterns:
            analysis_parts.append(f"   🌍 Geographic patterns:")
            for pattern, routes in route_patterns.items():
                analysis_parts.append(f"     • {pattern}: {', '.join(routes)}")
        
        # Performance correlation analysis
        if all_routes:
            # Filter out None values for NPS analysis
            nps_values = [data['nps'] for data in all_routes.values() if data['nps'] is not None]
            
            if nps_values:
                nps_range = max(nps_values) - min(nps_values)
                avg_nps = sum(nps_values) / len(nps_values)
                
                analysis_parts.append(f"   📈 Overall NPS performance:")
                analysis_parts.append(f"     • Range: {min(nps_values):.1f} to {max(nps_values):.1f} (spread: {nps_range:.1f} pts)")
                analysis_parts.append(f"     • Average: {avg_nps:.1f}")
            else:
                analysis_parts.append(f"   📈 Overall NPS performance: N/A (no valid NPS data found for identified routes)")
        
        # Source-specific insights
        exp_count = len(exp_drivers_routes.get('routes', []))
        ncs_count = len(ncs_routes.get('routes', []))
        verbatims_count = len(verbatims_routes.get('routes', []))
        
        analysis_parts.append(f"   📊 Source distribution:")
        analysis_parts.append(f"     • Explanatory drivers: {exp_count} routes")
        analysis_parts.append(f"     • NCS incidents: {ncs_count} routes")  
        analysis_parts.append(f"     • Customer verbatims: {verbatims_count} routes")
        
        return "\n".join(analysis_parts)

    def _identify_route_patterns(self, routes: List[str]) -> dict:
        """Identify geographic patterns in routes by destination regions"""
        patterns = {}
        
        # América del Norte (North America)
        north_america_destinations = ['JFK', 'LGA', 'EWR', 'MIA', 'ORD', 'DFW', 'LAX', 'BOS', 'IAD', 'YUL', 'YYZ']
        north_america_routes = [r for r in routes if any(dest in r.upper() for dest in north_america_destinations)]
        if north_america_routes:
            patterns['América del Norte'] = north_america_routes
        
        # Latinoamérica (Latin America) 
        latin_america_destinations = ['BOG', 'LIM', 'SCL', 'EZE', 'AEP', 'GRU', 'CGH', 'GIG', 'MEX', 'CCS', 'MVD', 'UIO', 'ASU', 'LPB']
        latin_america_routes = [r for r in routes if any(dest in r.upper() for dest in latin_america_destinations)]
        if latin_america_routes:
            patterns['Latinoamérica'] = latin_america_routes
        
        # Caribe (Caribbean)
        caribbean_destinations = ['SDQ', 'HAV', 'PUJ', 'STI', 'CUN', 'SJO', 'PTY', 'GUA', 'SAL', 'TGU', 'MGA']
        caribbean_routes = [r for r in routes if any(dest in r.upper() for dest in caribbean_destinations)]
        if caribbean_routes:
            patterns['Caribe'] = caribbean_routes
        
        # Europa (Europe)
        europe_destinations = ['LHR', 'CDG', 'FRA', 'AMS', 'FCO', 'MXP', 'ZUR', 'VIE', 'MUC', 'DUS', 'BRU', 'LIS', 'OPO', 
                             'DBV', 'ZAG', 'ATH', 'SKG', 'IST', 'WAW', 'PRG', 'BUD', 'OTP', 'SOF', 'TLV', 'CAI', 'CMN', 'ALG', 'TUN']
        europe_routes = [r for r in routes if any(dest in r.upper() for dest in europe_destinations)]
        if europe_routes:
            patterns['Europa'] = europe_routes
        
        return patterns

    def _find_column(self, df, possible_names: List[str]) -> str:
        """Find a column by trying multiple possible names (case-insensitive)"""
        for col in df.columns:
            for name in possible_names:
                # Handle empty string case explicitly
                if name == '' and col == '':
                    return col
                elif name != '' and name.lower() in col.lower():
                    return col
        return None
    
    async def _customer_profile_tool(self, node_path: str, start_date: str, end_date: str, min_surveys: int = 3, profile_dimension: str = "Multiple", mode: str = "comparative") -> str:
        """
        Tool for analyzing customer profile segments.
        - Comparative mode: NPS impact vs comparison filter (L7d, LM, etc.)
        - Single mode: Absolute NPS values for the period
        """
        try:
            self.logger.info(f"👥 CUSTOMER_PROFILE_TOOL CALLED - Mode: {mode}")
            self.logger.debug(f"📋 Parameters: node_path={node_path}, start_date={start_date}, end_date={end_date}, mode={mode}")
            
            # Convert dates to datetime (handle both string and datetime inputs)
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = start_date
                
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = end_date
            
            # Define key dimensions for NPS impact analysis
            # Available dimensions: Tier, Bound, Travel hour, Group age, Residence Region, 
            # Choosing reason, Business/Leisure, Travel reason, Fleet, CodeShare, Region, 
            # Channel, Demand spaces, Corporate, Connection, Fare Family, Issue Type, Travelling with
            if profile_dimension == "Multiple":
                dimensions = ["Business/Leisure", "Fleet", "Residence Region", "CodeShare"]
            else:
                dimensions = [profile_dimension]
            
            results = []
            nps_impact_summary = []
            
            for dimension in dimensions:
                try:
                    self.logger.info(f"Analyzing NPS impact for dimension: {dimension}")
                    
                    # collect_customer_profile_for_date_range IS async
                    if mode == "comparative":
                        # Pass comparison dates as datetime objects (the method expects datetime, not strings)
                        df = await self._collect_customer_profile_with_query_tracking(
                            node_path, start_dt, end_dt, dimension,
                            comparison_filter=self.causal_filter,
                            comparison_start_date=self.comparison_start_date,
                            comparison_end_date=self.comparison_end_date
                        )
                    else:  # single mode
                        df = await self._collect_customer_profile_with_query_tracking(
                            node_path, start_dt, end_dt, dimension,
                            comparison_filter=None,  # No comparison in single mode
                            comparison_start_date=None,
                            comparison_end_date=None
                    )
                    
                    if df.empty:
                        results.append(f"❌ {dimension}: No data found")
                        continue
                    
                    # Clean column names (remove brackets) - same as routes_tool
                    df = self._safe_clean_columns(df, method="replace")
                    
                    # Filter by minimum surveys
                    if 'Pax' in df.columns:
                        df_filtered = df[df['Pax'].fillna(0) >= min_surveys]
                    else:
                        df_filtered = df
                    
                    if df_filtered.empty:
                        results.append(f"❌ {dimension}: No data available for this period")
                        continue
                    
                    # Find category column for profile names (after cleaning)
                    category_col = self._find_column(df_filtered, ['category', 'customer_profile'])
                    
                    if mode == "comparative":
                        # COMPARATIVE MODE: Focus on NPS impact vs comparison filter
                        nps_diff_col = self._find_column(df_filtered, ['nps diff', 'nps_diff', 'diff'])
                        if nps_diff_col:
                        # Sort by NPS diff to show impact order (highest to lowest)
                            df_sorted = df_filtered.sort_values(nps_diff_col, ascending=False)
                        
                        results.append(f"📊 {dimension}: {len(df_filtered)} segments analyzed")
                        
                            # Show EACH individual profile with its NPS_diff vs comparison filter  
                        for _, segment in df_sorted.iterrows():
                            # Safely get segment name
                            if category_col and category_col in segment:
                                segment_name = str(segment[category_col]) if segment[category_col] is not None else 'Unknown'
                            else:
                                segment_name = 'Unknown'
                            
                                # Safely convert NPS current, NPS diff and Pax count
                                try:
                                    nps_diff_raw = segment.get(nps_diff_col, 0)
                                    nps_diff = float(nps_diff_raw) if nps_diff_raw is not None and str(nps_diff_raw).lower() not in ['nan', 'none', ''] else 0.0
                                except (ValueError, TypeError):
                                    nps_diff = 0.0
                                
                                # Also get current NPS value
                                nps_current_col = self._find_column(df_filtered, ['nps'])
                                try:
                                    nps_current_raw = segment.get(nps_current_col, 0) if nps_current_col else 0
                                    nps_current = float(nps_current_raw) if nps_current_raw is not None and str(nps_current_raw).lower() not in ['nan', 'none', ''] else 0.0
                                except (ValueError, TypeError):
                                    nps_current = 0.0
                                
                                # Calculate baseline NPS (current - diff)
                                nps_baseline = nps_current - nps_diff
                                
                                try:
                                    pax_count = int(segment.get('Pax', 0))
                                except (ValueError, TypeError):
                                    pax_count = 0
                            
                                # Format with both NPS values and difference
                                nps_diff_str = f"{nps_diff:+.1f}" if nps_diff != 0 else "0.0"
                                filter_text = self.causal_filter if self.causal_filter else "período de referencia"
                                results.append(f"   • {segment_name}: NPS {nps_baseline:.1f} → {nps_current:.1f} (diff {nps_diff_str} vs {filter_text}) ({pax_count} surveys)")
                        
                            # Store impact data for summary
                        if len(df_filtered) >= 2:
                            max_impact = df_filtered[nps_diff_col].max()
                            min_impact = df_filtered[nps_diff_col].min()
                            impact_spread = max_impact - min_impact
                            
                            nps_impact_summary.append({
                                'dimension': dimension,
                                'impact_spread': impact_spread,
                                'avg_impact': df_filtered[nps_diff_col].mean(),
                                'max_impact': max_impact,
                                'min_impact': min_impact,
                                'segments_analyzed': len(df_filtered)
                            })
                        elif len(df_filtered) == 1:
                            # Single segment case
                            single_impact = df_filtered[nps_diff_col].iloc[0]
                            nps_impact_summary.append({
                                'dimension': dimension,
                                'impact_spread': 0.0,
                                'avg_impact': single_impact,
                                'max_impact': single_impact,
                                'min_impact': single_impact,
                                'segments_analyzed': 1
                            })
                        else:
                            # Fallback: show current NPS spread if diff not available
                            nps_col_fallback = self._find_column(df_filtered, ['nps'])
                            if nps_col_fallback:
                                nps_spread = df_filtered[nps_col_fallback].max() - df_filtered[nps_col_fallback].min()
                                avg_nps = df_filtered[nps_col_fallback].mean()
                                
                                results.append(f"✅ {dimension}: Current NPS Analysis (comparison not available)")
                                results.append(f"   📊 NPS spread: {nps_spread:.1f} pts, Average: {avg_nps:.1f}")
                            else:
                                results.append(f"❌ {dimension}: No NPS data available")
                                continue
                    
                    else:  # SINGLE MODE
                        # SINGLE MODE: Show absolute NPS values for the period (no comparison)
                        nps_col = self._find_column(df_filtered, ['nps'])
                        if nps_col:
                            # Sort by NPS value (highest to lowest) 
                            df_sorted = df_filtered.sort_values(nps_col, ascending=False)
                            
                            results.append(f"📊 {dimension}: {len(df_filtered)} segments analyzed")
                            
                            # Show EACH individual profile with its absolute NPS value
                            for _, segment in df_sorted.iterrows():
                                # Safely get segment name
                                if category_col and category_col in segment:
                                    segment_name = str(segment[category_col]) if segment[category_col] is not None else 'Unknown'
                                else:
                                    segment_name = 'Unknown'
                                
                                # Safely convert NPS value and Pax count
                                try:
                                    nps_raw = segment.get(nps_col, 0)
                                    nps_value = float(nps_raw) if nps_raw is not None and str(nps_raw).lower() not in ['nan', 'none', ''] else 0.0
                                except (ValueError, TypeError):
                                    nps_value = 0.0
                                
                                try:
                                    pax_count = int(segment.get('Pax', 0))
                                except (ValueError, TypeError):
                                    pax_count = 0
                                
                                results.append(f"   • {segment_name}: NPS {nps_value:.1f} ({pax_count} surveys)")
                            
                            # Calculate NPS spread for summary
                            if len(df_filtered) >= 2:
                                nps_spread = df_filtered[nps_col].max() - df_filtered[nps_col].min()
                                avg_nps = df_filtered[nps_col].mean()
                                
                                nps_impact_summary.append({
                                    'dimension': dimension,
                                    'impact_spread': nps_spread,
                                    'avg_impact': avg_nps,
                                    'max_impact': df_filtered[nps_col].max(),
                                    'min_impact': df_filtered[nps_col].min(),
                                    'segments_analyzed': len(df_filtered)
                                })
                            elif len(df_filtered) == 1:
                                single_nps = df_filtered[nps_col].iloc[0]
                                nps_impact_summary.append({
                                    'dimension': dimension,
                                    'impact_spread': 0.0,
                                    'avg_impact': single_nps,
                                    'max_impact': single_nps,
                                    'min_impact': single_nps,
                                    'segments_analyzed': 1
                                })
                        else:
                            results.append(f"❌ {dimension}: No NPS data available")
                            continue
                    
                except Exception as e:
                    results.append(f"❌ {dimension}: Error - {str(e)[:50]}")
                    continue
            
            # Build focused result summary
            if mode == "comparative":
                filter_text = self.causal_filter if self.causal_filter else "período de referencia"
                result = f"👥 CUSTOMER PROFILE - NPS IMPACT vs {filter_text.upper()}:\n"
            else:
                result = f"👥 CUSTOMER PROFILE - ABSOLUTE NPS VALUES:\n"
            result += f"📅 Period: {start_date} to {end_date}\n"
            result += f"🎯 Segment: {node_path}\n"
            result += f"📊 Dimensions analyzed: {len(nps_impact_summary)}/{len(dimensions)}\n\n"
            
            # Add dimension details
            result += "\n".join(results)
            
            # Add overall impact summary - CATEGORY COMPARISON
            if nps_impact_summary:
                if mode == "comparative":
                    filter_text = self.causal_filter if self.causal_filter else "período de referencia"
                    result += f"\n\n📈 REACTIVIDAD POR CATEGORÍA (NPS_diff vs {filter_text}):\n"
                else:
                    result += f"\n\n📈 DISPERSIÓN NPS POR CATEGORÍA (valores absolutos):\n"
                
                # Sort dimensions by impact spread (most reactive first)
                sorted_dimensions = sorted(nps_impact_summary, key=lambda x: x['impact_spread'], reverse=True)
                
                if mode == "comparative":
                    result += f"   🏆 RANKING DE REACTIVIDAD:\n"
                else:
                    result += f"   🏆 RANKING DE DISPERSIÓN NPS:\n"
                for i, dim_data in enumerate(sorted_dimensions, 1):
                    dimension = dim_data['dimension']
                    spread = dim_data['impact_spread']
                    max_impact = dim_data['max_impact']
                    min_impact = dim_data['min_impact']
                    segments = dim_data['segments_analyzed']
                    
                    # Show spread and range for context
                    result += f"   {i}. {dimension}: Spread {spread:.1f} pts "
                    result += f"(rango: {min_impact:+.1f} a {max_impact:+.1f}, {segments} perfiles)\n"
                
                # Overall summary
                total_avg_spread = sum([d['impact_spread'] for d in nps_impact_summary]) / len(nps_impact_summary)
                result += f"\n   📊 Reactividad promedio: {total_avg_spread:.1f} pts entre categorías"
            
            # Store data for cross-references
            self.collected_data['customer_profile'] = {
                'analysis_summary': result,
                'nps_impact_data': nps_impact_summary,
                'focus': 'nps_impact_vs_last_days'
            }
            return result
            
        except Exception as e:
            self.logger.error(f"💥 Error in customer profile NPS impact analysis: {str(e)}")
            return f"Error in customer profile NPS impact analysis: {str(e)}"
    
    def _safe_clean_columns(self, df: pd.DataFrame, method: str = "strip") -> pd.DataFrame:
        """
        Safely clean column names by removing brackets, handling None columns
        
        Args:
            df: DataFrame to clean
            method: "strip" for .strip('[]') or "replace" for .replace('[', '').replace(']', '')
        """
        if df.empty:
            return df
            
        cleaned_columns = []
        for col in df.columns:
            if col is not None and isinstance(col, str):
                if method == "strip":
                    cleaned_columns.append(col.strip('[]'))
                elif method == "replace":
                    cleaned_columns.append(col.replace('[', '').replace(']', ''))
                else:
                    cleaned_columns.append(col)
            else:
                # Log problematic columns and use a safe default
                self.logger.warning(f"⚠️ Found None/invalid column name: {col}, using default")
                cleaned_columns.append(f"Column_{len(cleaned_columns)}")
        df.columns = cleaned_columns
        return df

    def _create_temporal_ncs_comparison(self, current_data: pd.DataFrame, comparison_data: pd.DataFrame, 
                                      node_path: str, current_start: str, current_end: str, 
                                      comparison_start: str, comparison_end: str) -> dict:
        """
        Create temporal comparison analysis of NCS data between two periods.
        Generates the Route × Incident Type matrix with deltas as requested.
        
        Returns:
            Dict with temporal comparison analysis including route-incident matrix and deltas
        """
        try:
            print(f"🔄 Creating temporal NCS comparison for {node_path}")
            
            # Extract structured data for both periods
            current_structured = self._extract_structured_ncs_data_for_comparison(current_data, "current")
            comparison_structured = self._extract_structured_ncs_data_for_comparison(comparison_data, "comparison")
            
            # Create route × incident type matrix
            route_incident_matrix = self._create_route_incident_matrix(current_structured, comparison_structured)
            
            # Calculate global deltas by incident type
            incident_type_deltas = self._calculate_incident_type_deltas(current_structured, comparison_structured)
            
            # Identify patterns and trends
            improvement_patterns = self._identify_improvement_patterns(route_incident_matrix, incident_type_deltas)
            
            # Create comprehensive analysis
            temporal_analysis = {
                'current_period': f"{current_start} to {current_end}",
                'comparison_period': f"{comparison_start} to {comparison_end}",
                'route_incident_matrix': route_incident_matrix,
                'incident_type_deltas': incident_type_deltas,
                'improvement_patterns': improvement_patterns,
                'current_structured': current_structured,
                'comparison_structured': comparison_structured,
                'analysis_summary': self._generate_temporal_summary(route_incident_matrix, incident_type_deltas, improvement_patterns)
            }
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.error(f"Error creating temporal NCS comparison: {str(e)}")
            return {
                'error': str(e),
                'current_period': f"{current_start} to {current_end}",
                'comparison_period': f"{comparison_start} to {comparison_end}",
                'analysis_summary': f"❌ Error in temporal comparison: {str(e)}"
                         }
    
    def _extract_structured_ncs_data_for_comparison(self, data: pd.DataFrame, period_label: str) -> dict:
        """Extract structured NCS data optimized for temporal comparison"""
        if data.empty:
            return {
                'period': period_label,
                'total_incidents': 0,
                'route_incidents': {},
                'incident_types': {
                    'cancelaciones': 0,
                    'retrasos': 0,
                    'desvios': 0,
                    'limitacion_aeronave': 0,
                    'otras_incidencias': 0
                },
                'routes': []
            }
        
        # Process incident types directly from DataFrame columns
        incident_types = {
            'cancelaciones': 0,
            'retrasos': 0, 
            'desvios': 0,
            'limitacion_aeronave': 0,
            'otras_incidencias': 0
        }
        
        # Map DataFrame columns to incident types
        column_mapping = {
            'Cancelaciones': 'cancelaciones',
            'Retrasos': 'retrasos', 
            'Desvíos': 'desvios',
            'Desvios': 'desvios',
            'Otras incidencias': 'otras_incidencias',
            'Limitación de la aeronave': 'limitacion_aeronave'
        }
        
        # Count incidents by type from DataFrame columns
        for col in data.columns:
            if col in column_mapping:
                incident_type = column_mapping[col]
                # Sum non-null values in the column
                count = data[col].notna().sum()
                incident_types[incident_type] = count
        
        # Extract routes from incident text
        route_incidents = {}
        incident_col = self._find_column(data, ['incident', 'incidents', ''])
        if incident_col and incident_col in data.columns:
            # Extract routes from incident descriptions
            route_pattern = r'\b([A-Z]{3})\s*[-–]\s*([A-Z]{3})\b'
            all_routes = set()
            
            for incident_text in data[incident_col].dropna():
                if isinstance(incident_text, str):
                    routes = re.findall(route_pattern, incident_text, re.IGNORECASE)
                    for origin, dest in routes:
                        route_code = f"{origin.upper()}-{dest.upper()}"
                        all_routes.add(route_code)
                        
                        if route_code not in route_incidents:
                            route_incidents[route_code] = {
                                'cancelaciones': 0, 'desvios': 0, 'retrasos': 0,
                                'otras_incidencias': 0, 'limitacion_aeronave': 0, 'total': 0
                            }
                        
                        # Simple classification based on keywords in incident text
                        incident_lower = incident_text.lower()
                        if any(word in incident_lower for word in ['cancel', 'cancelación']):
                            route_incidents[route_code]['cancelaciones'] += 1
                        elif any(word in incident_lower for word in ['retraso', 'delay']):
                            route_incidents[route_code]['retrasos'] += 1
                        elif any(word in incident_lower for word in ['desvío', 'desvio', 'divert']):
                            route_incidents[route_code]['desvios'] += 1
                        elif any(word in incident_lower for word in ['limitación', 'limitacion']):
                            route_incidents[route_code]['limitacion_aeronave'] += 1
                        else:
                            route_incidents[route_code]['otras_incidencias'] += 1
                        
                        route_incidents[route_code]['total'] += 1
        
        return {
            'period': period_label,
            'total_incidents': len(data),
            'route_incidents': route_incidents,
            'incident_types': incident_types,
            'routes': list(route_incidents.keys())
        }
    
    def _create_route_incident_matrix(self, current: dict, comparison: dict) -> dict:
        """Create the Route × Incident Type matrix with deltas"""
        matrix = {}
        
        # Get all unique routes from both periods
        all_routes = set(current['routes'] + comparison['routes'])
        
        for route in all_routes:
            current_breakdown = current['route_incidents'].get(route, {
                'cancelaciones': 0, 'retrasos': 0, 'desvios': 0, 
                'limitacion_aeronave': 0, 'otras_incidencias': 0, 'total': 0
            })
            comparison_breakdown = comparison['route_incidents'].get(route, {
                'cancelaciones': 0, 'retrasos': 0, 'desvios': 0,
                'limitacion_aeronave': 0, 'otras_incidencias': 0, 'total': 0
            })
            
            # Calculate deltas for each incident type
            deltas = {}
            for incident_type in ['cancelaciones', 'retrasos', 'desvios', 'limitacion_aeronave', 'otras_incidencias']:
                current_count = current_breakdown.get(incident_type, 0)
                comparison_count = comparison_breakdown.get(incident_type, 0)
                delta = current_count - comparison_count
                
                # Calculate percentage change
                if comparison_count > 0:
                    pct_change = (delta / comparison_count) * 100
                elif current_count > 0:
                    pct_change = float('inf')  # New incidents
                else:
                    pct_change = 0
                
                deltas[incident_type] = {
                    'current': current_count,
                    'previous': comparison_count,
                    'delta': delta,
                    'pct_change': pct_change
                }
            
            # Calculate total delta
            current_total = current_breakdown.get('total', 0)
            comparison_total = comparison_breakdown.get('total', 0)
            total_delta = current_total - comparison_total
            total_pct_change = (total_delta / comparison_total * 100) if comparison_total > 0 else (float('inf') if current_total > 0 else 0)
            
            matrix[route] = {
                'deltas': deltas,
                'total_delta': {
                    'current': current_total,
                    'previous': comparison_total,
                    'delta': total_delta,
                    'pct_change': total_pct_change
                }
            }
        
        return matrix
    
    def _calculate_incident_type_deltas(self, current: dict, comparison: dict) -> dict:
        """Calculate global deltas by incident type"""
        deltas = {}
        
        for incident_type in ['cancelaciones', 'retrasos', 'desvios', 'limitacion_aeronave', 'otras_incidencias']:
            current_count = current['incident_types'].get(incident_type, 0)
            comparison_count = comparison['incident_types'].get(incident_type, 0)
            delta = current_count - comparison_count
            
            if comparison_count > 0:
                pct_change = (delta / comparison_count) * 100
            elif current_count > 0:
                pct_change = float('inf')
            else:
                pct_change = 0
            
            deltas[incident_type] = {
                'current': current_count,
                'previous': comparison_count,
                'delta': delta,
                'pct_change': pct_change
            }
        
        return deltas
    
    def _identify_improvement_patterns(self, route_matrix: dict, type_deltas: dict) -> dict:
        """Identify improvement and deterioration patterns"""
        patterns = {
            'major_improvements': [],  # Significant reductions in incidents
            'moderate_improvements': [],
            'deteriorations': [],  # Increases in incidents
            'new_problems': [],  # Routes with new incidents
            'resolved_problems': []  # Routes that had incidents but now don't
        }
        
        for route, data in route_matrix.items():
            # Safety check for data structure
            if not isinstance(data, dict) or 'total_delta' not in data:
                self.logger.warning(f"⚠️ Invalid route data structure for {route}: {data}")
                continue
                
            total_delta_obj = data['total_delta']
            if not isinstance(total_delta_obj, dict) or 'delta' not in total_delta_obj:
                self.logger.warning(f"⚠️ Invalid total_delta structure for {route}: {total_delta_obj}")
                continue
                
            total_delta = total_delta_obj['delta']
            if total_delta is None:
                self.logger.warning(f"⚠️ None delta value for route {route}")
                continue
            
            if total_delta < -5:  # Major improvement
                patterns['major_improvements'].append({
                    'route': route,
                    'delta': total_delta,
                    'current': data['total_delta']['current'],
                    'previous': data['total_delta']['previous']
                })
            elif total_delta < 0:  # Moderate improvement
                patterns['moderate_improvements'].append({
                    'route': route,
                    'delta': total_delta,
                    'current': data['total_delta']['current'],
                    'previous': data['total_delta']['previous']
                })
            elif total_delta > 0:  # Deterioration
                patterns['deteriorations'].append({
                    'route': route,
                    'delta': total_delta,
                    'current': data['total_delta']['current'],
                    'previous': data['total_delta']['previous']
                })
        
        return patterns
    
    def _generate_temporal_summary(self, route_matrix: dict, type_deltas: dict, patterns: dict) -> str:
        """Generate comprehensive VS temporal comparison analysis for NCS data"""
        summary_parts = []
        
        # HEADER
        summary_parts.append("📊 **ANÁLISIS COMPARATIVO NCS - PERÍODO ACTUAL vs ANTERIOR**")
        summary_parts.append("")
        
        # SECTION 1: GLOBAL COMPARISON BY INCIDENT TYPE
        summary_parts.append("📈 **RESUMEN GLOBAL POR TIPO DE INCIDENTE:**")
        
        if type_deltas:
            total_current = sum(delta_info.get('current', 0) for delta_info in type_deltas.values())
            total_previous = sum(delta_info.get('previous', 0) for delta_info in type_deltas.values())
            total_delta = total_current - total_previous
            
            summary_parts.append(f"   🎯 **TOTAL**: {total_previous} → {total_current} ({total_delta:+d}, {((total_current/total_previous-1)*100 if total_previous > 0 else 0):+.1f}%)")
            summary_parts.append("")
            
            # Detailed breakdown by type with NPS impact validation
            for incident_type, delta_info in sorted(type_deltas.items(), key=lambda x: abs(x[1].get('delta', 0)), reverse=True):
                current = delta_info.get('current', 0)
                previous = delta_info.get('previous', 0)
                delta = delta_info.get('delta', 0)
                pct_change = delta_info.get('pct_change', 0)
                
                if current > 0 or previous > 0:  # Only show types that had incidents
                    status_icon = "🔴" if delta > 0 else ("🟢" if delta < 0 else "⚪")
                    pct_str = f"{pct_change:+.1f}%" if pct_change != float('inf') and abs(pct_change) > 0 else "N/A"
                    
                    # Add NPS impact validation
                    nps_impact = self._get_nps_impact_validation(incident_type, delta)
                    
                    summary_parts.append(f"   {status_icon} **{incident_type.upper()}**: {previous} → {current} ({delta:+d}, {pct_str}) {nps_impact}")
        
        summary_parts.append("")
        
        # SECTION 2: ROUTE-LEVEL ANALYSIS
        summary_parts.append("🛣️ **RUTAS MÁS AFECTADAS:**")
        
        if route_matrix:
            # Sort routes by total impact
            route_impacts = {}
            for route, incidents in route_matrix.items():
                total_delta = sum(abs(inc.get('delta', 0)) for inc in incidents.values())
                route_impacts[route] = total_delta
            
            sorted_routes = sorted(route_impacts.items(), key=lambda x: x[1], reverse=True)
            
            shown_routes = 0
            for route, impact in sorted_routes:
                if impact > 0 and shown_routes < 5:  # Show top 5 affected routes
                    incidents = route_matrix[route]
                    route_changes = []
                    for inc_type, inc_data in incidents.items():
                        if inc_data.get('delta', 0) != 0:
                            delta = inc_data.get('delta', 0)
                            route_changes.append(f"{inc_type}({delta:+d})")
                    
                    summary_parts.append(f"   🛫 **{route}**: {', '.join(route_changes)}")
                    shown_routes += 1
        
        # SECTION 3: KEY INSIGHTS
        summary_parts.append("")
        summary_parts.append("🔍 **INSIGHTS:**")
        
        if patterns.get('major_improvements'):
            improvements = patterns['major_improvements'][:2]
            routes_improved = [f"{item['route']} ({item['delta']:+d})" for item in improvements]
            summary_parts.append(f"   ✅ Mejoras: {', '.join(routes_improved)}")
        
        if patterns.get('deteriorations'):
            deteriorations = patterns['deteriorations'][:2] 
            routes_worse = [f"{item['route']} ({item['delta']:+d})" for item in deteriorations]
            summary_parts.append(f"   ⚠️ Deterioros: {', '.join(routes_worse)}")
        
        return " | ".join(summary_parts)
    
    def _get_nps_impact_validation(self, incident_type: str, delta: int) -> str:
        """Get NPS impact validation for incident type changes"""
        
        # Simple validation: more incidents = lower NPS, fewer incidents = higher NPS
        if delta > 0:
            return "📉 Valida una bajada de NPS"
        elif delta < 0:
            return "📈 Valida una subida de NPS"
        else:
            return "⚪ Sin impacto en NPS"
    
    async def _get_clean_reflection(self, system_prompt: str, tool_name: str, tool_result: str, message_history: MessageHistory, mode: str = "comparative") -> Optional[Dict[str, str]]:
        """Get AI reflection using clean context + tool results"""
        try:
            self.logger.info(f"🤔 Starting reflection for {tool_name}...")
            
            # Create clean message history for reflection
            clean_messages = self.tracker.get_clean_context(system_prompt)
            
            # Add current tool context with helper prompt and results using new mode-specific template
            # Try to determine flow type from context for comparative mode
            flow_type = None
            if mode == "comparative" and tool_name == "explanatory_drivers_tool":
                # For explanatory drivers, we need to analyze the result to determine flow
                flow_type = self._determine_flow_type_from_drivers(tool_result)
            
            reflection_template = self._get_reflection_prompt(mode, tool_name, flow_type)
            self.logger.info(f"🔍 DEBUG LLM: reflection_template length={len(reflection_template) if reflection_template else 0}")
            
            if reflection_template:
                # Ensure causal_filter has a safe value for template formatting
                safe_causal_filter = self.causal_filter if self.causal_filter else "período de referencia"
                reflection_prompt = reflection_template.format(
                tool_name=tool_name.upper(),
                tool_result=tool_result,
                    causal_filter=safe_causal_filter,
                    comparison_filter=safe_causal_filter  # For backwards compatibility
            )
                self.logger.info(f"🔍 DEBUG LLM: reflection_prompt formatted successfully, length={len(reflection_prompt)}")
            else:
                reflection_prompt = f"Analiza los resultados de {tool_name.upper()}\n\n{tool_result}"
                self.logger.warning(f"🔍 DEBUG LLM: Using fallback reflection_prompt, length={len(reflection_prompt)}")
            
            # Check reflection prompt size
            prompt_size = len(reflection_prompt)
            self.logger.info(f"🔍 DEBUG LLM: Final reflection_prompt size: {prompt_size} chars")
            self.logger.info(f"🔍 DEBUG LLM: reflection_prompt preview: {reflection_prompt[:500]}...")
            
            # Claude Sonnet 4.5 can handle larger contexts - increase limits
            if prompt_size > 200000:  # 200KB limit for reflection (was 60KB)
                self.logger.warning(f"⚠️ Reflection prompt very large ({prompt_size} chars) - truncating tool result")
                truncated_result = tool_result[:100000] + "\n\n[... TOOL RESULT TRUNCATED DUE TO SIZE ...]"
                reflection_template = self._get_reflection_prompt(mode, tool_name, flow_type)
                if reflection_template:
                    # Ensure causal_filter has a safe value for template formatting
                    safe_causal_filter = self.causal_filter if self.causal_filter else "período de referencia"
                    reflection_prompt = reflection_template.format(
                    tool_name=tool_name.upper(),
                    tool_result=truncated_result,
                        causal_filter=safe_causal_filter,
                        comparison_filter=safe_causal_filter  # For backwards compatibility
                )
                else:
                    reflection_prompt = f"Analiza los resultados de {tool_name.upper()}\n\n{truncated_result}"
                self.logger.debug(f"📏 Truncated reflection prompt size: {len(reflection_prompt)} chars")
            
            clean_messages.append({
                "role": "user",
                "content": reflection_prompt
            })
            
            # Add timeout to prevent hanging reflections
            self.logger.debug(f"🕒 Starting LLM call for {tool_name} reflection with 60s timeout")
            
            response, _, _ = await asyncio.wait_for(
                self.agent.invoke(
                    messages=clean_messages,
                    tools=[],  # No tools for reflection
                    structured_output=None  # No structured output for reflections
                ),
                timeout=300.0  # 5 minute timeout for reflections
            )
            
            self.logger.debug(f"✅ Reflection response received for {tool_name}")
            
            # Check if response content is valid
            if response.content and isinstance(response.content, str) and response.content.strip():
                self.logger.info(f"🤔 Reflection completed for {tool_name}: {len(response.content)} chars")
                self.logger.info(f"🔍 DEBUG LLM OUTPUT: Raw response preview: {response.content[:500]}...")
                
                # Parse the structured response to extract reflection and next tool code
                reflection, next_tool_code = self._parse_reflection_response(response.content)
                self.logger.info(f"🔍 DEBUG LLM OUTPUT: Parsed reflection length={len(reflection) if reflection else 0}")
                self.logger.info(f"🔍 DEBUG LLM OUTPUT: Parsed next_tool_code length={len(next_tool_code) if next_tool_code else 0}")
                
                # Store reflection in message history
                if reflection:
                    message_history.create_and_add_message(
                        content=f"REFLECTION: {reflection}",
                        message_type=MessageType.AI,
                        agent=AgentName.CONVERSATIONAL
                    )
                
                # Return both reflection and next tool code
                return {
                    "reflection": reflection,
                    "next_tool_code": next_tool_code
                }
            else:
                self.logger.warning(f"⚠️ No valid reflection content for {tool_name}")
                return None
                
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Reflection for {tool_name} timed out after 120 seconds")
            return f"Reflection timed out for {tool_name} - tool result too complex"
        except Exception as e:
            self.logger.error(f"❌ Error getting clean reflection for {tool_name}: {type(e).__name__}: {e}")
            return None
    
    async def _generate_final_synthesis(
        self, 
        message_history: MessageHistory,
        node_path: str,
        start_date: str, 
        end_date: str,
        nps_context: str = ""
    ) -> str:
        """Generate final synthesis with collected data"""
        
        # Build comprehensive data summary from collected_data
        try:
            data_summary = self._build_collected_data_summary()
            self.logger.info(f"📊 Data summary built successfully: {len(data_summary)} chars")
        except Exception as e:
            self.logger.error(f"❌ Error building data summary: {e}")
            data_summary = "Error building data summary - using minimal fallback"
        
        # FINAL USER MESSAGE: Summary request with collected data
        enhanced_final_request = f"""
{self.config.get('final_synthesis_prompt', 'Genera un informe causal consolidado basado en los datos recolectados.')}

📊 **DATOS RECOLECTADOS DURANTE LA INVESTIGACIÓN:**

{data_summary}

**INSTRUCCIONES ESPECÍFICAS:**
- Usa TODOS los datos numéricos recolectados arriba
- Incluye valores SHAP exactos, rutas específicas, números de incidentes, etc.
- NO hagas afirmaciones sin datos concretos
- Integra los hallazgos de todas las herramientas ejecutadas
- **IMPORTANTE**: Menciona los NPS específicos del período analizado y el período de comparación{nps_context}
"""
        
        # Check if request is too large
        # Claude Sonnet 4.5 has 200K tokens (~600K chars) context window
        # We use 400K as limit to leave room for response
        MAX_REQUEST_SIZE = 400000  # 400KB - safe for Claude's 200K token context
        MAX_TRUNCATED_SIZE = 150000  # 150KB - still preserves most data if truncation needed
        
        request_size = len(enhanced_final_request)
        self.logger.info(f"📏 Final request size: {request_size} chars")
        
        if request_size > MAX_REQUEST_SIZE:
            self.logger.warning(f"⚠️ Final request very large ({request_size} chars) - truncating data summary")
            # Extract enrichment section BEFORE truncating so it always survives
            enrichment_section = ""
            enrichment_marker = "\n🎯 **FOCUS TOUCHPOINT ENRICHMENT DATA:**"
            if enrichment_marker in data_summary:
                enrichment_idx = data_summary.index(enrichment_marker)
                enrichment_section = data_summary[enrichment_idx:]
                data_summary_without_enrichment = data_summary[:enrichment_idx]
            else:
                data_summary_without_enrichment = data_summary

            truncated_summary = data_summary_without_enrichment[:MAX_TRUNCATED_SIZE] + "\n\n[... DATA TRUNCATED DUE TO SIZE ...]"
            # Always re-append enrichment after truncation so it reaches the LLM
            if enrichment_section:
                truncated_summary += f"\n{enrichment_section}"
                self.logger.info(f"✅ Focus touchpoint enrichment preserved after truncation ({len(enrichment_section)} chars)")

            enhanced_final_request = f"""
{self.config.get('final_synthesis_prompt', 'Genera un informe causal consolidado basado en los datos recolectados.')}

📊 **DATOS RECOLECTADOS DURANTE LA INVESTIGACIÓN:**

{truncated_summary}

**INSTRUCCIONES ESPECÍFICAS:**
- Usa TODOS los datos numéricos recolectados arriba
- Incluye valores SHAP exactos, rutas específicas, números de incidentes, etc.
- NO hagas afirmaciones sin datos concretos
- Integra los hallazgos de todas las herramientas ejecutadas
- **IMPORTANTE**: Menciona los NPS específicos del período analizado y el período de comparación{nps_context}
"""
            self.logger.info(f"📏 Truncated request size: {len(enhanced_final_request)} chars")
        
        message_history.create_and_add_message(
            content=enhanced_final_request,
            message_type=MessageType.USER
        )
        
        self.tracker.log_message("USER", enhanced_final_request)
        
        # FINAL AI MESSAGE: Synthesis
        try:
            self.logger.info("🎯 Starting final synthesis with LLM...")
            self.logger.debug(f"📊 Enhanced final request length: {len(enhanced_final_request)} chars")
            self.logger.debug(f"📊 Message history length: {len(message_history.get_messages())} messages")
            
            # Add timeout to prevent hanging
            import asyncio
            final_response, final_structured_response, _ = await asyncio.wait_for(
                self.agent.invoke(
                    messages=message_history.get_messages(),
                    tools=[],
                    structured_output=None  # Remove structured output to get narrative content
                ),
                timeout=300.0  # 5 minute timeout for final synthesis
            )
            
            self.logger.info("✅ LLM response received")
            
            # Log the full response for debugging
            self.logger.info(f"🔍 LLM Response Type: {type(final_response)}")
            self.logger.info(f"🔍 LLM Response Content: {final_response.content}")
            self.logger.info(f"🔍 LLM Response Length: {len(final_response.content) if final_response.content else 0}")
            
            if final_response.content:
                self.tracker.log_message("AI", f"FINAL_SYNTHESIS: {final_response.content}")
                self.logger.info("🎯 Final synthesis completed successfully")
                self.logger.debug(f"📊 Final response length: {len(final_response.content)} chars")
                return final_response.content
            else:
                self.logger.warning("⚠️ Empty response content from LLM")
                self.logger.error(f"❌ LLM returned empty content. Full response object: {final_response}")
                return "No final synthesis generated - empty LLM response"
                
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Final synthesis timed out after 15 minutes")
            return f"Final synthesis timed out - request too complex for LLM processing"
        except Exception as e:
            self.logger.error(f"❌ Error generating final synthesis: {type(e).__name__}: {str(e)}")
            self.logger.error(f"❌ Enhanced request preview: {enhanced_final_request[:200]}...")
            self.logger.error(f"❌ Request size: {len(enhanced_final_request)} chars")
            self.logger.error(f"❌ Message history length: {len(message_history.get_messages())} messages")
            
            # Try to extract more details from the error
            if hasattr(e, '__dict__'):
                self.logger.error(f"❌ Error details: {e.__dict__}")
            
            # Check if it's an API error
            if hasattr(e, 'response'):
                self.logger.error(f"❌ API Response: {e.response}")
            if hasattr(e, 'status_code'):
                self.logger.error(f"❌ Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                self.logger.error(f"❌ Error Body: {e.body}")
                
            return f"Error in final synthesis: {type(e).__name__}: {str(e)}"
    
    def _build_collected_data_summary(self) -> str:
        """Build a comprehensive summary of all collected data with specific details"""
        summary_parts = []
        
        # Add baseline information at the beginning
        if hasattr(self, 'baseline_description') and hasattr(self, 'causal_filter'):
            self.logger.info(f"🔍 DEBUG DATA_SUMMARY: Adding baseline info - causal_filter: {self.causal_filter}, baseline_description: {self.baseline_description}")
            summary_parts.append(f"📊 **INFORMACIÓN DE COMPARACIÓN:**")
            summary_parts.append(f"   • Filtro de comparación: {self.causal_filter}")
            summary_parts.append(f"   • Descripción de la comparativa: {self.baseline_description}")
            summary_parts.append("")  # Empty line for separation
        else:
            self.logger.warning(f"🔍 DEBUG DATA_SUMMARY: Baseline info NOT added - hasattr baseline_description: {hasattr(self, 'baseline_description')}, hasattr causal_filter: {hasattr(self, 'causal_filter')}")
        
        # Explanatory Drivers Data
        if 'explanatory_drivers' in self.collected_data:
            drivers_data = self.collected_data['explanatory_drivers']
            summary_parts.append(f"🔍 **EXPLANATORY DRIVERS EJECUTADO:**")
            if isinstance(drivers_data, dict):
                summary_parts.append(f"   📊 Encuestas analizadas: {drivers_data.get('survey_count', 'N/A')}")
                
                # Get all drivers with SHAP values
                all_drivers = drivers_data.get('all_drivers', [])
                operational_drivers = drivers_data.get('operational_drivers', [])
                product_drivers = drivers_data.get('product_drivers', [])
                
                # Show operational drivers with SHAP values
                if operational_drivers and all_drivers:
                    summary_parts.append(f"   ⚙️ **Drivers OPERATIVOS (con valores SHAP):**")
                    for driver_name in operational_drivers:
                        driver_data = next((d for d in all_drivers if d.get('touchpoint') == driver_name), None)
                        if driver_data:
                            shap = driver_data.get('shap_value', 0)
                            sat_diff = driver_data.get('satisfaction_diff', 'N/A')
                            sig = "⭐" if abs(shap) > 0.1 else "⚪"
                            summary_parts.append(f"      {sig} {driver_name}: SHAP={shap:.3f}, Sat_diff={sat_diff}")
                        else:
                            summary_parts.append(f"      • {driver_name}: (sin datos SHAP)")
                
                # Show product drivers with SHAP values
                if product_drivers and all_drivers:
                    summary_parts.append(f"   🛎️ **Drivers de PRODUCTO (con valores SHAP):**")
                    for driver_name in product_drivers:
                        driver_data = next((d for d in all_drivers if d.get('touchpoint') == driver_name), None)
                        if driver_data:
                            shap = driver_data.get('shap_value', 0)
                            sat_diff = driver_data.get('satisfaction_diff', 'N/A')
                            sig = "⭐" if abs(shap) > 0.1 else "⚪"
                            summary_parts.append(f"      {sig} {driver_name}: SHAP={shap:.3f}, Sat_diff={sat_diff}")
                        else:
                            summary_parts.append(f"      • {driver_name}: (sin datos SHAP)")
            else:
                summary_parts.append(f"   {drivers_data}")
        
        # Verbatims Data
        if 'verbatims' in self.collected_data:
            verbatims_data = self.collected_data['verbatims']
            summary_parts.append(f"💬 **VERBATIMS EJECUTADO:**")
            summary_parts.append(f"   {verbatims_data}")
        elif 'verbatims_conversation' in self.collected_data:
            conversation_data = self.collected_data['verbatims_conversation']
            summary_parts.append(f"💬 **CONVERSACIÓN CHATBOT EJECUTADA:**")
            summary_parts.append(f"   📡 Fuente: {conversation_data.get('source', 'Unknown')}")
            summary_parts.append(f"   🔄 Total intercambios: {conversation_data.get('total_exchanges', 0)}")
            summary_parts.append(f"   📊 Total verbatims analizados: {conversation_data.get('total_data_points', 0)}")
            summary_parts.append(f"   🎯 Tipo de verbatim: {conversation_data.get('verbatim_type', 'N/A')}")
            # Include key insights from conversation
            conversation_log = conversation_data.get('conversation_log', [])
            if conversation_log:
                summary_parts.append(f"   💡 Principales hallazgos del chatbot:")
                for i, exchange in enumerate(conversation_log[:3], 1):  # First 3 exchanges
                    response = exchange.get('response', '')[:200] + '...' if len(exchange.get('response', '')) > 200 else exchange.get('response', '')
                    summary_parts.append(f"      Q{i}: {response}")
        
        # NCS Data
        if 'ncs_data' in self.collected_data:
            ncs_data = self.collected_data['ncs_data']
            summary_parts.append(f"🚨 **NCS EJECUTADO:**")
            if isinstance(ncs_data, dict):
                summary_parts.append(f"   🔍 Causas identificadas: {ncs_data.get('identified_causes', [])}")
                summary_parts.append(f"   🛫 Rutas afectadas: {ncs_data.get('affected_routes', [])}")
                summary_parts.append(f"   📅 Días analizados: {ncs_data.get('days_analyzed', 'N/A')}")
                summary_parts.append(f"   ⚙️ Estado fuente datos: {ncs_data.get('data_source_status', 'N/A')}")
                analysis_summary = ncs_data.get('analysis_summary', 'No disponible')
                # If the analysis contains a structured NCS_REFLEXION block, extract and preserve
                # the key sections (INCIDENTES_DELTA, DARK_HORSES, HIPOTESIS_DE_RELACION) so they
                # reach the final synthesis intact even if the full text is long.
                if "NCS_REFLEXION" in analysis_summary:
                    parsed = self._parse_ncs_reflexion(analysis_summary)
                    inc_delta = parsed.get("incidentes_delta_compact", "")
                    dark_horses = parsed.get("dark_horses_compact", "")
                    structured_block = "NCS_REFLEXION:\n"
                    if inc_delta:
                        structured_block += f"- INCIDENTES_DELTA: {inc_delta}\n"
                    if dark_horses:
                        structured_block += f"- DARK_HORSES: {dark_horses}\n"
                    # Append remaining free text (truncated) after the structured block
                    free_text = analysis_summary[:800] + '...' if len(analysis_summary) > 800 else analysis_summary
                    analysis_summary = structured_block + free_text
                elif len(analysis_summary) > 1200:
                    analysis_summary = analysis_summary[:1200] + '...'
                summary_parts.append(f"   📊 Análisis: {analysis_summary}")
            else:
                summary_parts.append(f"   {ncs_data}")
        
        # Routes Data - Enhanced comprehensive analysis
        if 'routes_data' in self.collected_data:
            routes_data = self.collected_data['routes_data']
            summary_parts.append(f"🛫 **ROUTES EJECUTADO - ANÁLISIS CONSOLIDADO:**")
            if isinstance(routes_data, dict):
                summary_parts.append(f"   📊 Total rutas únicas analizadas: {routes_data.get('total_routes', 0)}")
                
                # Summary by source
                exp_drivers_count = len(routes_data.get('exp_drivers_routes', {}).get('routes', []))
                ncs_count = len(routes_data.get('ncs_routes', {}).get('routes', []))
                verbatims_count = len(routes_data.get('verbatims_routes', {}).get('routes', []))
                
                summary_parts.append(f"   🎯 Distribución por fuente:")
                summary_parts.append(f"      • Explanatory Drivers: {exp_drivers_count} rutas")
                summary_parts.append(f"      • NCS Incidentes: {ncs_count} rutas")
                summary_parts.append(f"      • Verbatims Clientes: {verbatims_count} rutas")
                
                # All routes consolidated
                all_routes = routes_data.get('all_routes', {})
                if all_routes:
                    summary_parts.append(f"   📋 Rutas consolidadas:")
                    for route_name, route_data in list(all_routes.items())[:5]:  # Show first 5
                        sources_str = ", ".join(route_data.get('sources', []))
                        vs_str = f", VS {route_data['vs']}" if route_data.get('vs') is not None else ""
                        touchpoint_str = ""
                        if route_data.get('touchpoint_scores'):
                            touchpoint_scores = ", ".join([f"{tp}: {score}" for tp, score in route_data['touchpoint_scores'].items()])
                            touchpoint_str = f", [{touchpoint_scores}]"
                        summary_parts.append(f"      • {route_name}: NPS {route_data['nps']}, Pax {route_data['pax']}{vs_str}{touchpoint_str} (Fuentes: {sources_str})")
                
                # Similarities analysis
                similarities = routes_data.get('similarities_analysis', '')
                if similarities:
                    similarities_summary = similarities[:200] + '...' if len(similarities) > 200 else similarities
                    summary_parts.append(f"   🔍 Análisis similitudes: {similarities_summary}")
            else:
                summary_parts.append(f"   {routes_data}")
        
        # Customer Profile Data
        if 'customer_profile' in self.collected_data:
            profile_data = self.collected_data['customer_profile']
            summary_parts.append(f"👥 **CUSTOMER PROFILE EJECUTADO:**")
            if len(str(profile_data)) > 400:
                profile_summary = str(profile_data)[:400] + '...'
            else:
                profile_summary = str(profile_data)
            summary_parts.append(f"   {profile_summary}")
        
        # Focus Touchpoint Enrichment Data (verbatims + % issues)
        if 'focus_touchpoint_enrichment' in self.collected_data:
            enrichment_data = self.collected_data['focus_touchpoint_enrichment']
            summary_parts.append(f"\n🎯 **FOCUS TOUCHPOINT ENRICHMENT DATA:**")
            summary_parts.append(enrichment_data)
        
        if not summary_parts:
            return "❌ No se recolectaron datos durante la investigación"
        
        return "\n".join(summary_parts)
    
    def get_conversation_log(self) -> List[Dict]:
        """Get the conversation log"""
        return self.tracker.conversation_log

    def get_investigation_log(self) -> List[Dict]:
        """Get the structured investigation log (tool executions)"""
        return self.tracker.get_tool_executions()
    
    async def export_conversation(self, filename: Optional[str] = None, node_path: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, execution_id: Optional[str] = None, execution_duration_ms: Optional[float] = None) -> str:
        """Export the conversation log to the logging directory and DynamoDB."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            period_range = format_period_range(start_date=start_date, end_date=end_date)
            safe_node = node_path.replace("/", "_") if node_path else "unknown"

            if filename is None:
                filename = f"causal_{timestamp}.json"

            full_path = get_logging_path(
                self.report_group, "causal", period_range, filename, node_path=safe_node,
            )

            start_date_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date) if start_date else None
            end_date_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date) if end_date else None

            comp_start = self.comparison_start_date.strftime("%Y-%m-%d") if hasattr(self.comparison_start_date, "strftime") else str(self.comparison_start_date) if self.comparison_start_date else None
            comp_end = self.comparison_end_date.strftime("%Y-%m-%d") if hasattr(self.comparison_end_date, "strftime") else str(self.comparison_end_date) if self.comparison_end_date else None

            conversation_data = {
                "metadata": build_execution_metadata(
                    model=self.llm_type.value,
                    study_mode=self.study_mode,
                    anomaly_detection_mode=self.detection_mode,
                    causal_filter=self.causal_filter,
                    comparison_start_date=comp_start,
                    comparison_end_date=comp_end,
                    segment=safe_node,
                    focus_touchpoint=self.focus_touchpoint,
                    agent_type="causal_explanation",
                    analysis_type="separated_workflow_investigation",
                    node_path=node_path,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    total_iterations=self.tracker.iteration_count,
                    anomaly_type=self.current_anomaly_type,
                    total_messages=len(self.tracker.conversation_log),
                    tools_used=list(set([
                        msg.get("metadata", {}).get("tool_name")
                        for msg in self.tracker.conversation_log
                        if msg.get("metadata", {}).get("tool_name")
                    ])),
                    investigation_success=len(self.tracker.previous_explanations) > 0,
                ),
                "conversation_log": self.tracker.conversation_log,
                "clean_explanations": self.tracker.previous_explanations,
                "collected_data_summary": self._build_collected_data_summary() if hasattr(self, "collected_data") else {},
                "conversation_summary": self._get_conversation_summary(),
                "dax_queries": self.tracker.dax_queries,
            }

            save_pretty_json(full_path, conversation_data)
            self.logger.info(f"📝 Conversation exported to: {full_path}")

            s3_key = None
            # Upload to S3 in production
            if self.environment == "prod":
                try:
                    s3_key = get_s3_logging_key(
                        self.s3_uploader.base_prefix, self.report_group,
                        "causal", period_range, filename, node_path=safe_node,
                    )
                    self.s3_uploader.s3_client.put_object(
                        Bucket=self.s3_uploader.bucket_name,
                        Key=s3_key,
                        Body=json.dumps(conversation_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                        ContentType="application/json",
                    )
                    self.logger.info(f"📤 Causal conversation uploaded to S3: {s3_key}")
                except Exception as s3_err:
                    self.logger.warning(f"⚠️ Failed to upload causal conversation to S3: {s3_err}")

            # Persist to DynamoDB
            if execution_id:
                try:
                    from ..utils.dynamodb_report_persistence import DynamoDBReportPersistence
                    dynamo = DynamoDBReportPersistence(environment=self.environment)
                    final_synth = self.tracker.previous_explanations[-1] if self.tracker.previous_explanations else None
                    report_id = dynamo.save_causal_report(
                        execution_id=execution_id,
                        node_path=node_path or "unknown",
                        agent=self,
                        final_synthesis=final_synth,
                        execution_duration_ms=execution_duration_ms,
                        status="completed",
                        s3_report_key=s3_key,
                    )
                    if report_id:
                        self.logger.info(f"📊 Causal report persisted to DynamoDB: {report_id}")
                        self._last_dynamo_report_id = report_id
                except Exception as ddb_err:
                    self.logger.warning(f"⚠️ Failed to persist causal report to DynamoDB: {ddb_err}")

            return str(full_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to export conversation: {e}")
            return ""
    
    def _get_conversation_summary(self) -> Dict[str, int]:
        """Get summary of conversation message types"""
        summary = {}
        for entry in self.tracker.conversation_log:
            msg_type = entry['type']
            summary[msg_type] = summary.get(msg_type, 0) + 1
        return summary

    async def _filter_ncs_by_segment(
        self,
        ncs_data: pd.DataFrame,
        node_path: str,
        collect_unattributed_key: str | None = None
    ) -> pd.DataFrame:
        """
        Simple NCS filtering by segment characteristics without external dependencies.
        Focuses on the specific incidents for the analyzed segment.
        """
        try:
            # Check if we have NCS data to process
            if ncs_data.empty:
                return ncs_data
            
            # Get the first column (usually contains incident text)
            incident_col = ncs_data.columns[0] if len(ncs_data.columns) > 0 else None
            if incident_col is None:
                return ncs_data
            
            self.logger.info(f"Starting NCS filtering for segment: {node_path}")
            self.logger.info(f"Total NCS incidents to filter: {len(ncs_data)}")
            
            # Get segment filters to understand what routes/characteristics are relevant
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            # Apply contextual filtering based on segment characteristics
            filtered_ncs = await self._apply_contextual_filtering(
                ncs_data,
                node_path,
                collect_unattributed_key=collect_unattributed_key
            )
            
            # Log filtering results
            if len(filtered_ncs) < len(ncs_data):
                reduction_rate = (1 - len(filtered_ncs) / len(ncs_data)) * 100
                self.logger.info(f"Filtered NCS data for {node_path}: {len(filtered_ncs)}/{len(ncs_data)} incidents ({reduction_rate:.1f}% reduction)")
            else:
                self.logger.info(f"No filtering applied - returning all {len(ncs_data)} incidents for analysis")
            
            return filtered_ncs
            
        except Exception as e:
            self.logger.error(f"Error in NCS segment filtering: {str(e)}")
            # Return original data on error to ensure analysis continues
            return ncs_data
    
    async def _apply_contextual_filtering(
        self,
        ncs_data: pd.DataFrame,
        node_path: str,
        collect_unattributed_key: str | None = None
    ) -> pd.DataFrame:
        """
        Apply contextual filtering with the correct NCS logic:
        1. ALWAYS filter by haul (Global, LH, SH) based on routes/airports
        2. ONLY filter by cabin when we're at cabin level AND the incident specifies it affects ONLY another cabin
        3. NEVER discard incidents just because they don't mention cabin
        """
        try:
            # Get segment filters
            cabins, companies, hauls = self.pbi_collector._get_node_filters(node_path)
            
            if ncs_data.empty or len(ncs_data.columns) == 0:
                return ncs_data
            
            incident_col = ncs_data.columns[0]
            filtered_ncs = ncs_data.copy()
            
            # STEP 1: ALWAYS apply haul-based filtering using Routes Dictionary (Global, LH, SH)
            if hauls and len(hauls) == 1:
                haul_type = hauls[0]
                
                # Use routes dictionary instead of hardcoded keywords
                # IMPORTANT: for LH/SH nodes we keep ONLY attributable incidents in the quantitative pipeline,
                # but we optionally collect the "unattributed" ones (no route / no MAD-IATA inference) to pass
                # to the LLM for step (3) using the Iberia heuristic.
                is_global_root = node_path.strip() == "Global"
                if is_global_root:
                    filtered_ncs = await self._filter_using_routes_dictionary(
                        filtered_ncs,
                        haul_type,
                        incident_col,
                        allow_unknown_route_incidents=True
                    )
                else:
                    # A) strictly attributable to this haul (metrics / counts)
                    target_only = await self._filter_using_routes_dictionary(
                        filtered_ncs,
                        haul_type,
                        incident_col,
                        allow_unknown_route_incidents=False
                    )
                    # B) attributable + unknown (for extracting the unknown slice only)
                    if collect_unattributed_key:
                        target_plus_unknown = await self._filter_using_routes_dictionary(
                            filtered_ncs,
                            haul_type,
                            incident_col,
                            allow_unknown_route_incidents=True
                        )
                        unknown_only = target_plus_unknown[~target_plus_unknown.index.isin(target_only.index)]
                        if not hasattr(self, "_ncs_unattributed"):
                            self._ncs_unattributed = {}
                        self._ncs_unattributed[collect_unattributed_key] = unknown_only
                    filtered_ncs = target_only
            
            # STEP 2: Apply cabin filtering ONLY when we're at cabin level AND need to exclude incidents that affect ONLY other cabins
            if cabins and len(cabins) == 1:
                cabin_type = cabins[0].lower()
                
                # Define keywords for each cabin type
                cabin_keywords = {
                    'business': ['business', 'ejecutiva', 'premium', 'preferente', 'clase.*business'],
                    'economy': ['economy', 'turista', 'económica', 'clase.*turista'],
                    'premium': ['premium', 'preferente', 'plus', 'premium.*economy']
                }
                
                if cabin_type in cabin_keywords:
                    # Find incidents that mention OTHER cabins (to potentially exclude them)
                    other_cabin_patterns = []
                    for other_cabin, keywords in cabin_keywords.items():
                        if other_cabin != cabin_type:
                            other_cabin_patterns.extend(keywords)
                    
                    if other_cabin_patterns:
                        # Pattern to find incidents that mention ONLY other cabins (exclusive language)
                        exclusive_keywords = ['solo', 'solamente', 'únicamente', 'exclusivamente', 'solo.*clase', 'solo.*cabina']
                        exclusive_pattern = '|'.join([f'\\b{kw}\\b' for kw in exclusive_keywords])
                        
                        # Combine patterns to find incidents that mention other cabins with exclusive language
                        other_cabin_pattern = '|'.join([f'\\b{kw}\\b' for kw in other_cabin_patterns])
                        combined_pattern = f"({exclusive_pattern}).*({other_cabin_pattern})|({other_cabin_pattern}).*({exclusive_pattern})"
                        
                        # Find incidents to potentially exclude
                        exclusive_mask = filtered_ncs[incident_col].str.contains(combined_pattern, na=False, regex=True, case=False)
                        incidents_to_exclude = filtered_ncs[exclusive_mask]
                        
                        if len(incidents_to_exclude) > 0:
                            self.logger.info(f"Cabin exclusion filtering: {len(incidents_to_exclude)} incidents mention ONLY other cabins, excluding them")
                            # Remove these incidents from our filtered data
                            filtered_ncs = filtered_ncs[~exclusive_mask]
                        
                        self.logger.info(f"After cabin filtering: {len(filtered_ncs)} incidents remain (keeping incidents that don't specify cabin or affect our cabin)")
            
            # STEP 3: If no incidents remain after filtering, return original data for Global analysis
            if len(filtered_ncs) == 0 and node_path.strip() == "Global":
                self.logger.info("No incidents remain after filtering, returning all NCS data for Global analysis")
                return ncs_data
            
            self.logger.info(f"Final NCS filtering result: {len(filtered_ncs)}/{len(ncs_data)} incidents after contextual filtering")
            return filtered_ncs
            
        except Exception as e:
            self.logger.error(f"Error in contextual filtering: {str(e)}")
            return ncs_data

    async def _extract_ncs_causal_insights_workflow_aware(self, filtered_ncs_data: pd.DataFrame, node_path: str, anomaly_type: str) -> dict:
        """
        Extract causal insights from NCS data with workflow awareness:
        1. Classify incidents as operative vs product-related
        2. Filter by current workflow type
        3. Interpret based on anomaly direction
        4. Handle workflow mismatches by finding operational causes of product issues
        """
        try:
            if filtered_ncs_data.empty:
                return {
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_summary': {},
                    'touchpoint_correlations': {},
                    'confidence_level': 'low'
                }
            
            incident_col = filtered_ncs_data.columns[0] if len(filtered_ncs_data.columns) > 0 else None
            if incident_col is None:
                return {
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_summary': {},
                    'touchpoint_correlations': {},
                    'confidence_level': 'low'
                }
            
            # Combine all incident text for analysis
            all_incidents_text = "\n".join(filtered_ncs_data[incident_col].astype(str).tolist())
            
            # 1. EXTRACT STRUCTURED NCS DATA (categorías, motivos, rutas, vuelos)
            structured_ncs_data = self._extract_structured_ncs_data(all_incidents_text)
            
            # 2. CLASSIFY INCIDENTS by nature (operative vs product)
            incident_classification = self._classify_ncs_incidents_by_nature(all_incidents_text)
            
            # 3. LOG STRUCTURED NCS DATA for visibility
            self.logger.info(f"📊 NCS STRUCTURED DATA EXTRACTED:")
            self.logger.info(f"   - Categories: {structured_ncs_data.get('categories', {})}")
            self.logger.info(f"   - Total incidents: {structured_ncs_data.get('summary', {}).get('total_incidents', 0)}")
            self.logger.info(f"   - Most affected route: {structured_ncs_data.get('summary', {}).get('most_affected_route', 'None')}")
            self.logger.info(f"   - Main motive: {structured_ncs_data.get('summary', {}).get('main_motive', 'None')}")
            self.logger.info(f"   - Total passengers affected: {structured_ncs_data.get('passenger_impact', {}).get('total', 0)}")
            self.logger.info(f"   - Average delay: {structured_ncs_data.get('delay_statistics', {}).get('avg_delay', 0)} min")
            
            # 4. DETERMINE CURRENT WORKFLOW TYPE from explanatory drivers
            current_workflow = self._determine_current_workflow_type()
            
            # 3. HANDLE WORKFLOW MATCHING WITH SMART CORRELATION
            workflow_match = False
            relevant_incidents_text = ""
            incident_nature = "unknown"
            analysis_approach = "direct"
            
            if current_workflow == "operative" and incident_classification['operative_incidents']:
                # Direct match: operative workflow + operative incidents
                relevant_incidents_text = incident_classification['operative_incidents_text']
                incident_nature = "operative"
                workflow_match = True
                analysis_approach = "direct"
            elif current_workflow == "product" and incident_classification['product_incidents']:
                # Direct match: product workflow + product incidents
                relevant_incidents_text = incident_classification['product_incidents_text']
                incident_nature = "product"
                workflow_match = True
                analysis_approach = "direct"
            elif current_workflow == "product" and incident_classification['operative_incidents']:
                # SMART CORRELATION: product issues caused by operative incidents
                relevant_incidents_text = incident_classification['operative_incidents_text']
                incident_nature = "operative_causing_product"
                workflow_match = True  # This is actually a valuable correlation
                analysis_approach = "causal_correlation"
                self.logger.info(f"🔗 SMART CORRELATION: Found operational incidents that likely caused the detected product/service issues")
            elif current_workflow == "operative" and incident_classification['product_incidents']:
                # Less common but possible: operative issues manifest as product improvements
                relevant_incidents_text = incident_classification['product_incidents_text']
                incident_nature = "product_affecting_operative"
                workflow_match = True
                analysis_approach = "reverse_correlation"
            else:
                # Try to use any available incidents with lower confidence
                if incident_classification['operative_incidents']:
                    relevant_incidents_text = incident_classification['operative_incidents_text']
                    incident_nature = "operative"
                    analysis_approach = "fallback"
                elif incident_classification['product_incidents']:
                    relevant_incidents_text = incident_classification['product_incidents_text']
                    incident_nature = "product"
                    analysis_approach = "fallback"
                else:
                    # Truly no usable incidents
                    return {
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_summary': {},
                    'touchpoint_correlations': {},
                        'confidence_level': 'no_relevant_incidents',
                    'workflow_mismatch': True,
                    'current_workflow': current_workflow,
                    'available_incident_types': list(incident_classification.keys())
                }
            
            # 4. EXTRACT INSIGHTS from relevant incidents
            identified_causes = self._identify_disruption_causes(relevant_incidents_text)
            route_analysis = self._extract_route_impacts(relevant_incidents_text)
            touchpoint_correlations = self._correlate_ncs_with_touchpoints(relevant_incidents_text)
            
            # 5. ENHANCE INTERPRETATIONS FOR CAUSAL CORRELATIONS
            if analysis_approach == "causal_correlation":
                # Enhance causes to show operational → product causation
                enhanced_causes = []
                for cause in identified_causes:
                    enhanced_causes.append(f"🔗 OPERATIONAL ROOT CAUSE of product issues: {cause}")
                identified_causes = enhanced_causes
                
                # Add correlation insight
                touchpoint_correlations['causal_insight'] = f"Operational incidents ({len(filtered_ncs_data)} incidents) likely caused the detected product/service touchpoint issues. This is a deeper root cause analysis."
            
            # 6. INTERPRET BASED ON ANOMALY DIRECTION
            interpreted_causes = self._interpret_causes_by_anomaly_direction(
                identified_causes, incident_nature, anomaly_type
            )
            
            # 7. ASSESS CONFIDENCE LEVEL (boost confidence for smart correlations)
            base_confidence = self._assess_ncs_confidence(
                len(filtered_ncs_data), 
                len(interpreted_causes), 
                len(route_analysis['affected_routes'])
            )
            
            # Adjust confidence based on analysis approach
            if analysis_approach == "causal_correlation":
                if "low" in base_confidence:
                    confidence_level = base_confidence.replace("low", "medium")
                elif "medium" in base_confidence:
                    confidence_level = base_confidence.replace("medium", "high")
                else:
                    confidence_level = base_confidence
                confidence_level += " (enhanced by operational-product correlation)"
            else:
                confidence_level = base_confidence
            
            return {
                'identified_causes': interpreted_causes,
                'affected_routes': route_analysis['affected_routes'],
                'route_impact_summary': route_analysis['route_impact_summary'],
                'touchpoint_correlations': touchpoint_correlations,
                'confidence_level': confidence_level,
                'incident_nature': incident_nature,
                'workflow_match': workflow_match,
                'analysis_approach': analysis_approach,
                'current_workflow': current_workflow
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting workflow-aware NCS causal insights: {str(e)}")
            return {
                'identified_causes': [],
                'affected_routes': [],
                'route_impact_summary': {},
                'touchpoint_correlations': {},
                'confidence_level': 'low'
            }

    # Simplified stub methods for NCS analysis - these can be enhanced later
    def _classify_ncs_incidents_by_nature(self, incidents_text: str) -> dict:
        """Classify NCS incidents as operative vs product-related."""
        return {
            'operative_incidents': True,
            'product_incidents': False,
            'operative_incidents_text': incidents_text,
            'product_incidents_text': ""
        }
    
    def _determine_current_workflow_type(self) -> str:
        """Determine workflow type from explanatory drivers."""
        explanatory_data = str(self.collected_data.get('explanatory_drivers', ''))
        if 'punctuality' in explanatory_data.lower() or 'otp' in explanatory_data.lower():
            return 'operative'
        elif 'crew' in explanatory_data.lower() or 'food' in explanatory_data.lower():
            return 'product'
        else:
            return 'unknown'
    
    def _identify_disruption_causes(self, incidents_text: str) -> list:
        """Extract disruption causes from incident text."""
        causes = []
        if 'delay' in incidents_text.lower():
            causes.append('Flight delays')
        if 'cancel' in incidents_text.lower():
            causes.append('Flight cancellations')
        if 'weather' in incidents_text.lower():
            causes.append('Weather disruptions')
        if 'maintenance' in incidents_text.lower():
            causes.append('Maintenance issues')
        return causes or ['Operational incidents detected']
    
    def _extract_route_impacts(self, incidents_text: str) -> dict:
        """Extract affected routes from incident text with enhanced patterns for LH routes."""
        import re
        
        routes = []
        route_impact_summary = {}
        
        # Enhanced route patterns for long-haul routes
        # Pattern 1: Standard XXX-YYY format
        route_pattern = r'\b([A-Z]{3})\s*[-–]\s*([A-Z]{3})\b'
        route_matches = re.findall(route_pattern, incidents_text, re.IGNORECASE)
        
        for origin, dest in route_matches:
            route_code = f"{origin.upper()}-{dest.upper()}"
            routes.append(route_code)
            route_impact_summary[route_code] = 'Route-specific operational incidents'
        
        # Pattern 2: Individual airport codes mentioned (for LH segment)
        lh_airports = {
            # Americas
            'JFK': 'New York', 'MIA': 'Miami', 'EZE': 'Buenos Aires', 'BOG': 'Bogotá',
            'LIM': 'Lima', 'SCL': 'Santiago', 'GRU': 'São Paulo', 'MEX': 'Mexico City',
            'ORD': 'Chicago', 'DFW': 'Dallas', 'LAX': 'Los Angeles', 'CCS': 'Caracas',
            'SDQ': 'Santo Domingo', 'HAV': 'Havana', 'BOS': 'Boston', 'YUL': 'Montreal',
            # Africa/Middle East  
            'CAI': 'Cairo', 'JNB': 'Johannesburg', 'CMN': 'Casablanca', 'ALG': 'Algiers',
            'TUN': 'Tunis', 'DAR': 'Dar es Salaam', 'ADD': 'Addis Ababa',
            # Asia/Pacific
            'NRT': 'Tokyo', 'PVG': 'Shanghai', 'ICN': 'Seoul', 'BKK': 'Bangkok',
            'SIN': 'Singapore', 'HKG': 'Hong Kong', 'DEL': 'Delhi', 'BOM': 'Mumbai'
        }
        
        # Look for specific LH airport mentions
        for airport_code, city_name in lh_airports.items():
            if re.search(rf'\b{airport_code}\b', incidents_text, re.IGNORECASE):
                # Create route as MAD-XXX (assuming MAD as hub)
                route_code = f"MAD-{airport_code}"
                if route_code not in routes:
                    routes.append(route_code)
                    route_impact_summary[route_code] = f'Incidents affecting {city_name} route'
        
        # Pattern 3: City/country names to airport mapping
        city_to_airport = {
            'new york': 'JFK', 'miami': 'MIA', 'buenos aires': 'EZE', 'bogota': 'BOG',
            'lima': 'LIM', 'santiago': 'SCL', 'sao paulo': 'GRU', 'mexico': 'MEX',
            'chicago': 'ORD', 'dallas': 'DFW', 'los angeles': 'LAX', 'caracas': 'CCS',
            'tokyo': 'NRT', 'shanghai': 'PVG', 'seoul': 'ICN', 'bangkok': 'BKK'
        }
        
        for city_name, airport_code in city_to_airport.items():
            if re.search(rf'\b{city_name}\b', incidents_text, re.IGNORECASE):
                route_code = f"MAD-{airport_code}"
                if route_code not in routes:
                    routes.append(route_code)
                    route_impact_summary[route_code] = f'Incidents mentioned for {city_name.title()}'
        
        # Deduplicate routes
        unique_routes = list(dict.fromkeys(routes))
        
        return {
            'affected_routes': unique_routes,
            'route_impact_summary': route_impact_summary
        }
    
    def _correlate_ncs_with_touchpoints(self, incidents_text: str) -> dict:
        """Correlate NCS incidents with known touchpoints."""
        correlations = {}
        if 'punctuality' in incidents_text.lower():
            correlations['Punctuality'] = 'NCS incidents affecting punctuality'
        if 'boarding' in incidents_text.lower():
            correlations['Boarding'] = 'NCS incidents affecting boarding'
        return correlations
    
    def _interpret_causes_by_anomaly_direction(self, identified_causes: list, incident_nature: str, anomaly_type: str) -> list:
        """Interpret causes based on anomaly direction."""
        return identified_causes  # Simple passthrough for now
    
    def _get_routes_query_for_date_range(self, cabins: List[str], companies: List[str], hauls: List[str], start_date: datetime, end_date: datetime) -> str:
        """Generate DAX query for routes data using date range and Rutas.txt template."""
        try:
            # Load the routes template from PBI collector
            template = self.pbi_collector._load_query_template("Rutas.txt")
            
            # Replace placeholders with actual values
            cabins_str = '", "'.join(cabins)
            companies_str = '", "'.join(companies)
            hauls_str = '", "'.join(hauls)
            
            # Convert datetime to date components
            if hasattr(start_date, 'year'):
                start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
                end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
            else:
                # Fallback for string dates
                from datetime import datetime
                start_dt = datetime.strptime(str(start_date)[:10], '%Y-%m-%d')
                end_dt = datetime.strptime(str(end_date)[:10], '%Y-%m-%d')
                start_year, start_month, start_day = start_dt.year, start_dt.month, start_dt.day
                end_year, end_month, end_day = end_dt.year, end_dt.month, end_dt.day
            
            # Replace the template placeholders
            query = template.replace(
                'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
                f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
            ).replace(
                'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
                f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
            ).replace(
                'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
                f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
            ).replace(
                '\'Date_Master\'[Date] =date(2025,05,12)',
                f'\'Date_Master\'[Date] >= date({start_year},{start_month},{start_day}) && \'Date_Master\'[Date] <= date({end_year},{end_month},{end_day})'
            )
            
            return query
            
        except Exception as e:
            self.logger.error(f"Error generating routes query: {e}")
            # Fallback: return a basic query structure if template fails
            return f"""
            EVALUATE
            SUMMARIZECOLUMNS(
                'Route_Master'[route],
                'Measure'[NPS (Route)],
                'Measure'[n (Route)]
            )
            """
    
    def _get_ncs_routes_query(self, ncs_routes: List[str], cabins: List[str], companies: List[str], hauls: List[str], start_date, end_date) -> str:
        """
        Generate simplified DAX query for NCS-specific routes using date range filter.
        This is different from the general routes query and filters for specific routes.
        """
        try:
            # Create route filter for specific NCS routes
            routes_str = '", "'.join(ncs_routes)
            cabins_str = '", "'.join(cabins)
            companies_str = '", "'.join(companies)
            hauls_str = '", "'.join(hauls)
            
            # Convert datetime to date components
            if hasattr(start_date, 'year'):
                start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
                end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
            else:
                # Fallback for string dates
                from datetime import datetime
                start_dt = datetime.strptime(str(start_date)[:10], '%Y-%m-%d')
                end_dt = datetime.strptime(str(end_date)[:10], '%Y-%m-%d')
                start_year, start_month, start_day = start_dt.year, start_dt.month, start_dt.day
                end_year, end_month, end_day = end_dt.year, end_dt.month, end_dt.day
            
            # Simplified DAX query with date range filter (for NCS and verbatims routes)
            query = f"""
DEFINE
    VAR __DS0FilterTable =
        TREATAS({{"{cabins_str}"}}, 'Cabin_Master'[Cabin_Show])
    
    VAR __DS0FilterTable2 =
        TREATAS({{"{companies_str}"}}, 'Company_Master'[Company])
    
    VAR __DS0FilterTable3 =
        TREATAS({{"{hauls_str}"}}, 'Haul_Master'[Haul_Aggr])
    
    VAR __DS0FilterTableRoutes =
        TREATAS({{"{routes_str}"}}, 'Route_Master'[route])

EVALUATE
SUMMARIZECOLUMNS(
    'Route_Master'[route],
    __DS0FilterTable,
    __DS0FilterTable2,
    __DS0FilterTable3,
    __DS0FilterTableRoutes,
    FILTER('Date_Master', 'Date_Master'[Date] >= date({start_year},{start_month},{start_day}) && 'Date_Master'[Date] <= date({end_year},{end_month},{end_day})),
    "NPS", 'Measure'[NPS (Route)],
    "VS", [Monthly_Satisfaction],
    "Pax", 'Measure'[n (Route)]
)
ORDER BY 'Route_Master'[route]
"""
            return query
            
        except Exception as e:
            self.logger.error(f"Error generating NCS routes query: {e}")
            # Fallback: simplified query with route filter
            routes_str = '", "'.join(ncs_routes)
            return f"""
EVALUATE
SUMMARIZECOLUMNS(
    'Route_Master'[route],
    TREATAS({{"{routes_str}"}}, 'Route_Master'[route]),
    "NPS", 'Measure'[NPS (Route)],
    "VS", [Monthly_Satisfaction],
    "Pax", 'Measure'[n (Route)]
)
ORDER BY 'Route_Master'[route]
            """
    
    def _assess_ncs_confidence(self, incident_count: int, cause_count: int, route_count: int) -> str:
        """Assess confidence level of NCS analysis."""
        if incident_count > 5 and cause_count > 0:
            return "high"
        elif incident_count > 2:
            return "medium"
        else:
            return "low"

    def _extract_multiple_causal_touchpoints_from_history(self) -> list:
        """Extract multiple causal touchpoints from analysis history."""
        # Simple placeholder - return empty list for now
        return []

    def _aggregate_ncs_incidents(self, incidents_list: List[Dict[str, str]]) -> List[str]:
        """
        Group identical incidents and add count: '[Date] (xCount) Text'
        Reduces volume before sending to LLM.
        """
        grouped = {} # (date, text) -> count
        
        for inc in incidents_list:
            text_normalized = inc['text'].strip()
            date_normalized = inc['date'].strip()
            
            # Use tuple key for grouping
            key = (date_normalized, text_normalized)
            grouped[key] = grouped.get(key, 0) + 1
            
        # Format output
        aggregated = []
        # Sort by date
        sorted_keys = sorted(grouped.keys(), key=lambda x: x[0])
        
        for (date, text) in sorted_keys:
            count = grouped[(date, text)]
            if count > 1:
                aggregated.append(f"[{date}] (x{count}) {text}")
            else:
                aggregated.append(f"[{date}] {text}")
                
        return aggregated

    async def _filter_incidents_sequentially(self, incidents: List[str], max_final: int = 30, batch_size: int = 150) -> List[str]:
        """
        Sequential tournament to select top dark horses using LLM.
        batch_size increased to 150 to reduce LLM round-trips and prevent timeouts.
        """
        if len(incidents) <= max_final:
            return incidents
            
        current_top = []
        
        # Split into batches
        batches = [incidents[i:i + batch_size] for i in range(0, len(incidents), batch_size)]
        
        self.logger.info(f"🐎 Starting sequential Dark Horse filtering: {len(incidents)} candidates in {len(batches)} batches")
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        for i, batch in enumerate(batches):
            # Combine current winners with new challengers
            candidates = current_top + batch
            
            # If total candidates fit in max_final, just keep them all (no need to filter yet)
            if len(candidates) <= max_final:
                current_top = candidates
                continue
                
            # Otherwise, ask LLM to select the best
            self.logger.info(f"🐎 Filtering batch {i+1}/{len(batches)}: {len(candidates)} candidates -> top {max_final}")
            
            candidates_str = "\n".join(candidates)
            
            prompt_content = f"""
            ACT AS: Operational Analyst for an airline.
            TASK: Select the top {max_final} most significant "Dark Horse" events from the list below.
            
            CRITERIA FOR "DARK HORSE" (Prioritize in this order):
            1. High Impact/Severity: Strikes, Weather storms, System failures (IT/Baggage), Security alerts, Geopolitics.
            2. Volume/Repetition: Higher incident counts (e.g. "(x50)") indicate higher impact.
            3. Specificity: Prefer specific events (e.g. "Storm in EAS") over generic codes (e.g. "W0").
            4. REJECT: Routine operational delays (crew rotation, cleaning, late arrival), single isolated minor delays, generic codes without context.
            
            CANDIDATES LIST:
            {candidates_str}
            
            OUTPUT FORMAT:
            Return ONLY the selected lines from the list, exactly as they appear. One per line. Do not add bullets or comments.
            """
            
            try:
                # Use wrapper __call__ method instead of ainvoke directly on wrapper
                response = await self.llm(
                    prompt=[
                        SystemMessage(content="You are an expert Airline Operations Analyst. You filter noise to find critical disruptions."),
                        HumanMessage(content=prompt_content)
                    ]
                )
                
                selection = response.content.strip().split('\n')
                # Clean up selection
                cleaned_selection = [s.strip().replace('•', '').replace('-', '').strip() for s in selection if s.strip()]
                # Keep only valid lines that were in candidates
                current_top = []
                
                # Helper for fuzzy matching
                def clean_line_for_match(line):
                    # Remove [Date] and (xCount) patterns to compare core text content
                    # This handles cases where LLM strips metadata or changes format
                    return re.sub(r'\[.*?\]|\(x\d+\)', '', line).strip()

                for s in cleaned_selection:
                    # 1. Direct containment (most reliable)
                    matched = next((c for c in candidates if s in c), None)
                    
                    # 2. Fuzzy match based on core text content
                    if not matched:
                        s_clean = clean_line_for_match(s)
                        # Only try fuzzy match if we have enough content (>5 chars) to avoid false positives
                        if len(s_clean) > 5:
                            # Check if core text of selection matches core text of any candidate
                            matched = next((c for c in candidates if s_clean in clean_line_for_match(c)), None)
                    
                    if matched and matched not in current_top:
                        current_top.append(matched)
                
                # If LLM returned nothing or garbage, fallback to top N by length/count?
                if not current_top:
                    self.logger.warning(f"🐎 LLM returned empty selection. Raw response preview: {str(response.content)[:200]}")
                    current_top = candidates[:max_final]
                    
                # Hard limit
                current_top = current_top[:max_final]
                
            except Exception as e:
                self.logger.error(f"🐎 Error in sequential filtering: {e}. Keeping first {max_final}.")
                current_top = candidates[:max_final]
                
        return current_top

    async def _ncs_reflection_with_agent(
        self, 
        filtered_ncs_data: pd.DataFrame, 
        node_path: str, 
        anomaly_type: str, 
        total_days: int,
        # New parameters for dark horses analysis
        comparison_data: pd.DataFrame = None,
        current_start_date: str = None,
        current_end_date: str = None,
        comparison_start_date: str = None,
        comparison_end_date: str = None,
        nps_variation: float = None,
        temporal_incident_changes: dict = None
    ) -> dict:
        """
        Send filtered NCS data to agent with helper prompt for reflection.
        Enhanced to include dark horses detection by comparing comments between periods.
        
        Args:
            filtered_ncs_data: Current period NCS data filtered by segment
            node_path: Segment path being analyzed
            anomaly_type: Type of anomaly (positive/negative)
            total_days: Number of days in period
            comparison_data: Comparison period NCS data filtered by segment (optional)
            current_start_date: Start date of current period
            current_end_date: End date of current period
            comparison_start_date: Start date of comparison period
            comparison_end_date: End date of comparison period
            nps_variation: NPS variation between periods (e.g., -10.0)
            temporal_incident_changes: Dict with incident type changes (e.g., {'cancelaciones': +12})
        """
        try:
            # DEBUG: Log actual data structure
            self.logger.info(f"🔍 DEBUG: NCS data shape: {filtered_ncs_data.shape}")
            self.logger.info(f"🔍 DEBUG: NCS columns: {list(filtered_ncs_data.columns) if not filtered_ncs_data.empty else 'EMPTY'}")
            if not filtered_ncs_data.empty:
                self.logger.info(f"🔍 DEBUG: First few rows:\n{filtered_ncs_data.head(2).to_string()}")
            
            if filtered_ncs_data.empty:
                self.logger.info("❌ DEBUG: NCS data is completely empty after filtering")
                return {
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_summary': {},
                    'touchpoint_correlations': {},
                    'confidence_level': 'no_data',
                    'analysis_summary': 'No NCS incidents found for this segment'
                }
            
            # Get incident column (first column, which may have no name)
            if len(filtered_ncs_data.columns) == 0:
                self.logger.error("❌ DEBUG: No columns found in NCS data")
                return {
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_summary': {},
                    'touchpoint_correlations': {},
                    'confidence_level': 'no_data',
                    'analysis_summary': 'NCS data structure invalid - no columns found'
                }
            
            # Get incident text from first column (handles empty column names)
            incident_col = filtered_ncs_data.columns[0]
            self.logger.info(f"🔍 DEBUG: Using incident column: '{incident_col}' (empty name is normal)")
            
            # Prepare incidents for dark horse detection (ALL incidents -> Aggregated -> Filtered)
            
            # 1. Extract ALL incidents (with dates) from current period
            current_incidents_raw = self._extract_incidents_with_dates(filtered_ncs_data, max_incidents=None)
            incident_count = len(current_incidents_raw)
            
            # 2. Aggregate identical incidents (add counts)
            current_aggregated = self._aggregate_ncs_incidents(current_incidents_raw)
            self.logger.info(f"🐎 Current incidents aggregated: {len(current_incidents_raw)} raw -> {len(current_aggregated)} unique events")
            
            # 3. Sequential Filtering Tournament (LLM)
            current_final_list = await self._filter_incidents_sequentially(current_aggregated, max_final=30)
            
            # Comparison Period
            comparison_final_list = []
            if comparison_data is not None and not comparison_data.empty:
                comp_raw = self._extract_incidents_with_dates(comparison_data, max_incidents=None)
                comp_agg = self._aggregate_ncs_incidents(comp_raw)
                self.logger.info(f"🐎 Comparison incidents aggregated: {len(comp_raw)} raw -> {len(comp_agg)} unique events")
                comparison_final_list = await self._filter_incidents_sequentially(comp_agg, max_final=30)
            
            self.logger.info(f"🔍 DEBUG: Filtered down to {len(current_final_list)} current dark horses, {len(comparison_final_list)} comparison dark horses")
            
            # ENHANCEMENT: Pre-analyze routes for disruption counting
            # Use ALL raw incidents text for accurate route statistics
            all_raw_texts = [inc['text'] for inc in current_incidents_raw]
            all_incidents_text = "\n".join(all_raw_texts[:3000])
            route_disruption_summary = self._extract_route_disruption_counts(all_incidents_text)
            
            # Build NPS variation context
            nps_context = ""
            nps_variation_str = "N/A"  # String version for prompt
            if nps_variation is not None:
                direction = "subió" if nps_variation > 0 else "bajó"
                nps_variation_str = f"{nps_variation:+.1f}"
                nps_context = f"📈 **VARIACIÓN DE NPS (del nodo analizado):** El NPS {direction} **{nps_variation_str} pts** respecto al baseline del análisis"
            
            # Build incident changes context
            incident_changes_context = ""
            if temporal_incident_changes:
                incident_changes_context = "📊 **VARIACIÓN DE INCIDENTES:**\n"
                for incident_type, change_data in temporal_incident_changes.items():
                    # change_data is a dict with {current, previous, delta, pct_change}
                    delta = change_data.get('delta', 0) if isinstance(change_data, dict) else change_data
                    if delta != 0:
                        direction = "+" if delta > 0 else ""
                        incident_changes_context += f"   • {incident_type.capitalize()}: {direction}{delta}\n"
            
            # Build radio heuristic section — Global nodes must NOT filter by haul
            is_global_node = node_path.strip().lower() == "global"
            if is_global_node:
                radio_section = (
                    "📌 **RADIO DEL SEGMENTO: GLOBAL**\n"
                    "- Este análisis cubre TODA la red (LH + SH). NO apliques filtros de radio.\n"
                    "- Todos los incidentes son relevantes independientemente de si la ruta es LH o SH.\n"
                    "- La heurística de radio NO aplica aquí. No descartes ningún incidente por su destino."
                )
            else:
                radio_section = (
                    "📌 **HEURÍSTICA DE RADIO (solo si NO aparece la ruta completa):**\n"
                    "- Asume que Iberia vuela **en** o **desde** la península ibérica.\n"
                    "- Si el comentario menciona destinos en **América, Asia u Oriente Medio**, interprétalo como **LH**.\n"
                    "- En caso contrario (principalmente **Europa**, **norte de África**, etc.), interprétalo como **SH**.\n"
                    "- Si no hay suficiente información para inferir el radio con confianza, indícalo como **\"no concluyente\"** (no inventes rutas)."
                )

            # Create enhanced NCS helper prompt with DARK HORSES analysis
            ncs_helper_prompt = f"""
🚨 **ANÁLISIS NCS - REFLEXIÓN CON DETECCIÓN DE DARK HORSES**

📊 **CONTEXTO DEL ANÁLISIS:**
- Segmento: {node_path}
- Segmento (repr): {repr(node_path)}
- is_global_node: {is_global_node}
- Período actual: {current_start_date or 'N/A'} a {current_end_date or 'N/A'} ({total_days} días)
- Período comparativo: {comparison_start_date or 'N/A'} a {comparison_end_date or 'N/A'}
- Incidentes período actual: {incident_count}
- Incidentes período comparativo: {len(comparison_data) if comparison_data is not None else 'N/A'}
- Tipo de anomalía NPS: {anomaly_type}

{nps_context}

{incident_changes_context}

📈 **CONTEO DE DISRUPCIONES POR RUTA (período actual):**"""
            
            if route_disruption_summary:
                ncs_helper_prompt += "\n"
                for route, count in list(route_disruption_summary.items())[:10]:
                    ncs_helper_prompt += f"\n   • {route}: {count} disrupciones"
                if len(route_disruption_summary) > 10:
                    ncs_helper_prompt += f"\n   ... y {len(route_disruption_summary) - 10} rutas adicionales"
            else:
                ncs_helper_prompt += "\n   • No se encontraron patrones de ruta claros"

            # Add current period incidents WITH DATES (FILTERED DARK HORSES)
            ncs_helper_prompt += f"""

📋 **COMENTARIOS PERÍODO ACTUAL ({current_start_date or 'N/A'} a {current_end_date or 'N/A'}):**"""
            
            if current_final_list:
                for inc_str in current_final_list:
                    ncs_helper_prompt += f"\n• {inc_str}"
            else:
                ncs_helper_prompt += "\n• No se encontraron eventos significativos tras el filtrado."
            
            ncs_helper_prompt += f"\n(Selección inteligente de eventos significativos de un total de {incident_count} incidentes)"
            
            # Add comparison period incidents WITH DATES (if available)
            if comparison_data is not None and not comparison_data.empty:
                ncs_helper_prompt += f"""

📋 **COMENTARIOS PERÍODO COMPARATIVO ({comparison_start_date or 'N/A'} a {comparison_end_date or 'N/A'}):**"""
                
                if comparison_final_list:
                    for inc_str in comparison_final_list:
                        ncs_helper_prompt += f"\n• {inc_str}"
                else:
                    ncs_helper_prompt += "\n• No se encontraron eventos significativos tras el filtrado."
                
                ncs_helper_prompt += f"\n(Selección inteligente de eventos significativos de un total de {len(comparison_data)} incidentes)"

            # Add unattributed incidents (no route / no MAD-IATA inference). These are intentionally excluded from
            # LH/SH quantitative counts but should be classified by the LLM using the Iberia heuristic.
            unattributed_current = getattr(self, "_ncs_unattributed", {}).get("current") if hasattr(self, "_ncs_unattributed") else None
            unattributed_comparison = getattr(self, "_ncs_unattributed", {}).get("comparison") if hasattr(self, "_ncs_unattributed") else None

            def _format_unattributed(df, label: str) -> str:
                if df is None or df.empty:
                    return f"\n📌 **INCIDENTES NO ATRIBUIDOS ({label}):** No disponible"
                
                # Use standard extraction (with aggregation) logic
                # Extract all to aggregate properly
                try:
                    raw_incidents = self._extract_incidents_with_dates(df, max_incidents=None)
                    aggregated = self._aggregate_ncs_incidents(raw_incidents)
                    
                    rows = []
                    max_rows = 50
                    for inc_str in aggregated[:max_rows]:
                        rows.append(f"• {inc_str}")
                    
                    more = f"\n... y {len(aggregated) - max_rows} grupos adicionales" if len(aggregated) > max_rows else ""
                    
                    return (
                        f"\n📌 **INCIDENTES NO ATRIBUIDOS ({label})** (sin ruta/IATA → clasificar con heurística):\n"
                        + "\n".join(rows)
                        + more
                    )
                except Exception as e:
                    self.logger.error(f"Error formating unattributed incidents: {e}")
                    return f"\n📌 **INCIDENTES NO ATRIBUIDOS ({label}):** Error al procesar ({str(e)})"

            ncs_helper_prompt += _format_unattributed(unattributed_current, "PERÍODO ACTUAL")
            if comparison_data is not None:
                ncs_helper_prompt += _format_unattributed(unattributed_comparison, "PERÍODO COMPARATIVO")
            
            ncs_helper_prompt += f"""

🎯 **SOLICITUD DE ANÁLISIS (sin asumir correlaciones):**

Genera una **REFLEXIÓN NCS** que pueda persistirse y reutilizarse en la síntesis final. Debe integrar SIEMPRE:

1) **CUANTITATIVO (incidentes):**
   - Variación por tipo (cancelaciones, desvíos, retrasos, etc.) y lectura de rutas más afectadas.

2) **DARK HORSES (texto libre, ambos períodos):**
   - Identifica eventos excepcionales mencionados y cita **fechas** y, si están, **rutas/vuelos**.
   - Si no hay eventos, dilo explícitamente.

3) **RELACIÓN NPS ↔ INCIDENTES ↔ DARK HORSES (con incertidumbre):**
   - Explica **cómo podría** conectarse la variación de NPS ({nps_variation_str} pts) con:
     (a) la variación cuantitativa de incidentes,
     (b) la presencia/ausencia de dark horses.
   - Si un dark horse podría explicar parte del cambio cuantitativo (p.ej. más desvíos/retrasos), dilo.
   - Si el cuantitativo parece NO alinearse con el NPS, dilo y sugiere hipótesis alternativas.
   - **No asumas causalidad**: etiqueta como “posible”, “consistente con”, “no concluyente”.

{radio_section}

📌 **REGLA DE USO DE INCIDENTES NO ATRIBUIDOS (OBLIGATORIA):**
- Los bloques "INCIDENTES NO ATRIBUIDOS" NO están en los conteos cuantitativos por radio/cabina.
- Clasíficalos tú en: **RELEVANTES para este radio**, **RELEVANTES para el radio opuesto**, o **NO CONCLUYENTE (solo Global)**.
- Solo usa los **RELEVANTES para este radio** para explicar NPS/incidentes del nodo.

🧩 **FORMATO DE SALIDA (OBLIGATORIO, para persistencia):**
Empieza EXACTAMENTE con:
NCS_REFLEXION:

Luego incluye estas secciones (en este orden):
- NPS_DELTA: {nps_variation_str} pts
- INCIDENTES_DELTA: [resumen breve por tipo]
- DARK_HORSES: [lista con fechas; o “No detectados”]
- HIPOTESIS_DE_RELACION: [2-5 líneas]
- NIVEL_DE_CONFIANZA: [bajo/medio/alto] + por qué
"""
            
            self.logger.info(f"🤖 Enviando {incident_count} incidentes NCS al agente para reflexión con datos de disrupciones por ruta")
            
            # Check if we have valid incidents
            if not current_incidents_raw:
                self.logger.error("❌ DEBUG: No valid incidents found after extraction and filtering")
                return {
                    'identified_causes': [],
                    'affected_routes': [],
                    'route_impact_summary': {},
                    'touchpoint_correlations': {},
                    'confidence_level': 'no_data',
                    'analysis_summary': 'No valid NCS incidents found after extraction'
                }
            
            # Use the existing LLM to analyze the incidents
            from ..message_history import MessageHistory  
            from ..utils.enums import MessageType
            
            message_history = MessageHistory(logger=self.logger)
            
            # System prompt for NCS analysis with dark horses detection
            system_prompt = """Eres un experto analista de incidentes operacionales de aerolíneas. 
Tu tarea es analizar incidentes NCS (Network Control Center) y extraer insights causales para explicar variaciones de NPS.

**CAPACIDADES DE ANÁLISIS:**
1. Identificar causas operacionales cuantificables (cancelaciones, retrasos, desvíos)
2. Detectar "DARK HORSES" - eventos excepcionales en comentarios:
   - Huelgas (ATC, handling, pilotos)
   - Condiciones meteorológicas extremas
   - Fallos de sistemas IT
   - Problemas de tripulación
   - Cierres de aeropuertos
   - Overbooking masivo
   - Cualquier evento inusual mencionado en texto libre

**ENFOQUE CRÍTICO:**
- NO asumas correlaciones automáticas entre incidentes y NPS
- REFLEXIONA sobre la posible influencia de cada factor
- DISTINGUE entre causas recurrentes (métricas) vs excepcionales (dark horses)
- MENCIONA fechas específicas cuando los comentarios las indiquen
- CONECTA los dark horses con las variaciones de incidentes cuantificados

Proporciona análisis estructurado, específico y basado en evidencia de los datos proporcionados."""
            
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            
            message_history.create_and_add_message(
                content=ncs_helper_prompt,
                message_type=MessageType.USER
            )
            self.tracker.log_message("USER", ncs_helper_prompt)
            
            # Get LLM analysis
            self.logger.info(f"🔄 DEBUG: About to call LLM with {len(current_final_list)} filtered current + {len(comparison_final_list)} filtered comparison dark horses...")
            
            response, _, _ = await self.agent.invoke(
                messages=message_history.get_messages(),
                tools=[],  # No tools needed for this reflection
                structured_output=None
            )
            
            llm_analysis = str(response.content) if response.content is not None else "No analysis provided"
            
            self.logger.info(f"✅ Reflexión NCS completada: {len(llm_analysis)} caracteres")
            self.logger.info(f"🔍 DEBUG: LLM response preview: {llm_analysis[:200]}...")
            
            # Extract structured data from LLM response (simple extraction)
            identified_causes = self._extract_causes_from_llm_response(llm_analysis)
            affected_routes = self._extract_routes_from_llm_response(llm_analysis)
            
            self.logger.info(f"🔍 DEBUG: Extracted {len(identified_causes)} causes and {len(affected_routes)} routes")
            
            return {
                'identified_causes': identified_causes,
                'affected_routes': affected_routes,
                'route_impact_summary': {route: f'Incident reported in NCS analysis - {route_disruption_summary.get(route, 0)} disruptions' for route in affected_routes},
                'touchpoint_correlations': self._extract_touchpoints_from_llm_response(llm_analysis),
                'confidence_level': 'high_agent_analysis',
                'analysis_summary': llm_analysis,
                'incident_count': incident_count,
                'sample_size': incident_count,
                'route_disruption_counts': route_disruption_summary,
                'method': 'agent_reflection_ncs',
                'dark_horses_analysis_included': comparison_data is not None and not comparison_data.empty,
                'comparison_incidents_analyzed': len(comparison_final_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error in NCS agent reflection: {str(e)}")
            return {
                'identified_causes': [],
                'affected_routes': [],
                'route_impact_summary': {},
                'touchpoint_correlations': {},
                'confidence_level': 'error',
                'analysis_summary': f'Error en reflexión NCS: {str(e)}',
                'method': 'agent_reflection_failed'
            }

    def _extract_incidents_with_dates(self, ncs_data: pd.DataFrame, max_incidents: Optional[int] = 30) -> List[Dict[str, str]]:
        """
        Extract incidents from NCS data with their associated dates.
        
        Args:
            ncs_data: DataFrame with NCS incident data
            max_incidents: Maximum number of incidents to extract (None for all)
            
        Returns:
            List of dicts with 'date' and 'text' keys
        """
        incidents_with_dates = []
        
        if ncs_data.empty:
            return incidents_with_dates
        
        try:
            # Get incident text column (first column)
            incident_col = ncs_data.columns[0] if len(ncs_data.columns) > 0 else None
            if incident_col is None:
                return incidents_with_dates
            
            # Try to find date column
            date_col = None
            date_column_candidates = ['collection_date', 'email_date', 'date', 'fecha', 'Date', 'Fecha']
            for col in date_column_candidates:
                if col in ncs_data.columns:
                    date_col = col
                    break
            
            # If no date column found, try to extract date from source_file
            if date_col is None and 'source_file' in ncs_data.columns:
                date_col = 'source_file'  # Will extract date from filename
            
            # Select data to process: distributed sampling if we have more than max_incidents
            # This ensures we cover the entire period range instead of just the first N incidents (which might be all on day 1)
            if max_incidents is not None and len(ncs_data) > max_incidents:
                # Use linspace to get evenly spaced indices
                import numpy as np
                indices = np.linspace(0, len(ncs_data) - 1, max_incidents, dtype=int)
                # Remove duplicates if any (though unlikely with large ncs_data) and sort
                indices = sorted(list(set(indices)))
                data_to_process = ncs_data.iloc[indices]
            else:
                data_to_process = ncs_data

            # Extract incidents with dates
            for idx, row in data_to_process.iterrows():
                incident_text = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                
                if not incident_text or incident_text == 'nan':
                    continue
                
                # Extract date
                date_str = "N/A"
                if date_col:
                    date_value = row.get(date_col, None)
                    if date_value is not None and pd.notna(date_value):
                        if date_col == 'source_file':
                            # Extract date from filename (format: ndc-YYYY-MM-DD...)
                            import re
                            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', str(date_value))
                            if date_match:
                                date_str = date_match.group(1)
                        elif hasattr(date_value, 'strftime'):
                            date_str = date_value.strftime('%Y-%m-%d')
                        else:
                            # Try to parse string date
                            try:
                                from datetime import datetime
                                parsed_date = datetime.strptime(str(date_value)[:10], '%Y-%m-%d')
                                date_str = parsed_date.strftime('%d-%b')  # Format: 06-dic
                            except:
                                date_str = str(date_value)[:10]
                
                incidents_with_dates.append({
                    'date': date_str,
                    'text': incident_text
                })
            
            self.logger.info(f"🔍 Extracted {len(incidents_with_dates)} incidents with dates")
            return incidents_with_dates
            
        except Exception as e:
            self.logger.error(f"Error extracting incidents with dates: {str(e)}")
            return incidents_with_dates

    def _extract_structured_ncs_data(self, incidents_text: str) -> dict:
        """Extract structured NCS data with categories, motives, routes, and aggregated counts"""
        import re
        
        structured_data = {
            "categories": {},
            "route_disruptions": {},
            "motives_breakdown": {},
            "operator_breakdown": {},
            "passenger_impact": {"j_class": 0, "w_class": 0, "y_class": 0, "total": 0},
            "delay_statistics": {"total_minutes": 0, "count": 0, "avg_delay": 0},
            "summary": {}
        }
        
        # Extract category totals (Cancelaciones, Retrasos, Desvíos, etc.)
        category_patterns = {
            "cancelaciones": r'Cancelaciones\s*Total\s*(\d+)',
            "retrasos": r'Retrasos\s*Total\s*(\d+)', 
            "desvios": r'Desvíos\s*Total\s*(\d+)',
            "otras_incidencias": r'Otras incidencias\s*Total\s*(\d+)',
            "limitacion_aeronave": r'Limitación de la aeronave\s*Total\s*(\d+)'
        }
        
        for category, pattern in category_patterns.items():
            match = re.search(pattern, incidents_text, re.IGNORECASE)
            if match:
                structured_data["categories"][category] = int(match.group(1))
        
        # Extract route patterns with incident type breakdown
        route_pattern = r'\b([A-Z]{3})\s*[-–]\s*([A-Z]{3})\b'
        
        # Enhanced route extraction with incident type classification
        route_incident_breakdown = {}
        
        # Split text into sections by incident type for better classification
        sections = {
            'cancelaciones': re.findall(r'Cancelaciones.*?(?=Desvíos|Retrasos|Otras incidencias|Limitación|\Z)', incidents_text, re.DOTALL | re.IGNORECASE),
            'desvios': re.findall(r'Desvíos.*?(?=Cancelaciones|Retrasos|Otras incidencias|Limitación|\Z)', incidents_text, re.DOTALL | re.IGNORECASE),
            'retrasos': re.findall(r'Retrasos.*?(?=Cancelaciones|Desvíos|Otras incidencias|Limitación|\Z)', incidents_text, re.DOTALL | re.IGNORECASE),
            'otras_incidencias': re.findall(r'Otras incidencias.*?(?=Cancelaciones|Desvíos|Retrasos|Limitación|\Z)', incidents_text, re.DOTALL | re.IGNORECASE),
            'limitacion_aeronave': re.findall(r'Limitación de la aeronave.*?(?=Cancelaciones|Desvíos|Retrasos|Otras incidencias|\Z)', incidents_text, re.DOTALL | re.IGNORECASE)
        }
        
        # Process each section to extract routes by incident type
        for incident_type, sections_text in sections.items():
            for section in sections_text:
                routes_in_section = re.findall(route_pattern, section, re.IGNORECASE)
                for origin, dest in routes_in_section:
                    route_code = f"{origin.upper()}-{dest.upper()}"
                    
                    if route_code not in route_incident_breakdown:
                        route_incident_breakdown[route_code] = {
                            'cancelaciones': 0, 'desvios': 0, 'retrasos': 0, 
                            'otras_incidencias': 0, 'limitacion_aeronave': 0, 'total': 0
                        }
                    
                    route_incident_breakdown[route_code][incident_type] += 1
                    route_incident_breakdown[route_code]['total'] += 1
        
        # Fallback: if no structured sections found, do general route extraction
        if not route_incident_breakdown:
            route_matches = re.findall(route_pattern, incidents_text, re.IGNORECASE)
            for origin, dest in route_matches:
                route_code = f"{origin.upper()}-{dest.upper()}"
                structured_data["route_disruptions"][route_code] = structured_data["route_disruptions"].get(route_code, 0) + 1
        else:
            # Use the detailed breakdown
            for route, breakdown in route_incident_breakdown.items():
                structured_data["route_disruptions"][route] = breakdown['total']
            
            # Store detailed breakdown separately
            structured_data["route_incident_breakdown"] = route_incident_breakdown
        
        # Extract passenger impact by class
        passenger_patterns = {
            "j_class": r'J\s*(\d+)',
            "w_class": r'W\s*(\d+)', 
            "y_class": r'Y\s*(\d+)'
        }
        
        for class_type, pattern in passenger_patterns.items():
            matches = re.findall(pattern, incidents_text, re.IGNORECASE)
            if matches:
                structured_data["passenger_impact"][class_type] = sum(int(match) for match in matches)
        
        # Calculate total passengers
        structured_data["passenger_impact"]["total"] = (
            structured_data["passenger_impact"]["j_class"] + 
            structured_data["passenger_impact"]["w_class"] + 
            structured_data["passenger_impact"]["y_class"]
        )
        
        # Extract delay minutes for statistics
        delay_pattern = r'(\d+)\s*minutos'
        delay_matches = re.findall(delay_pattern, incidents_text, re.IGNORECASE)
        if delay_matches:
            delays = [int(match) for match in delay_matches]
            structured_data["delay_statistics"]["total_minutes"] = sum(delays)
            structured_data["delay_statistics"]["count"] = len(delays)
            structured_data["delay_statistics"]["avg_delay"] = round(sum(delays) / len(delays), 1) if delays else 0
        
        # Extract common motives
        motive_patterns = {
            "tecnicas": r'causas técnicas|técnicos|mantenimiento',
            "operativas": r'causas operativas|operativo',
            "meteorologia": r'meteorología|meteorológic',
            "rotacion_avion": r'rotación de avión|rotación',
            "atc": r'\bATC\b',
            "equipaje": r'equipaje|maletas'
        }
        
        for motive, pattern in motive_patterns.items():
            count = len(re.findall(pattern, incidents_text, re.IGNORECASE))
            if count > 0:
                structured_data["motives_breakdown"][motive] = count
        
        # Extract operators (Privilege, Iberojet, etc.)
        operator_patterns = {
            "privilege": r'Privilege',
            "iberojet": r'Iberojet', 
            "iberia": r'IB\d+'  # Regular Iberia flights
        }
        
        for operator, pattern in operator_patterns.items():
            count = len(re.findall(pattern, incidents_text, re.IGNORECASE))
            if count > 0:
                structured_data["operator_breakdown"][operator] = count
        
        # Create summary with enhanced multi-day aggregation
        total_incidents = sum(structured_data["categories"].values())
        most_affected_route = max(structured_data["route_disruptions"].items(), key=lambda x: x[1]) if structured_data["route_disruptions"] else None
        main_motive = max(structured_data["motives_breakdown"].items(), key=lambda x: x[1]) if structured_data["motives_breakdown"] else None
        
        # ENHANCED: Multi-day route prioritization for routes_tool
        priority_routes_for_investigation = []
        if structured_data["route_disruptions"]:
            # Sort routes by frequency (most disrupted first)
            sorted_routes = sorted(structured_data["route_disruptions"].items(), key=lambda x: x[1], reverse=True)
            
            # Categorize routes by disruption frequency for investigation priority
            for route, count in sorted_routes:
                if count >= 5:  # High priority: 5+ incidents
                    priority_routes_for_investigation.append({"route": route, "incidents": count, "priority": "HIGH"})
                elif count >= 3:  # Medium priority: 3-4 incidents  
                    priority_routes_for_investigation.append({"route": route, "incidents": count, "priority": "MEDIUM"})
                elif count >= 2:  # Low priority: 2 incidents
                    priority_routes_for_investigation.append({"route": route, "incidents": count, "priority": "LOW"})
        
        structured_data["summary"] = {
            "total_incidents": total_incidents,
            "most_affected_route": most_affected_route,
            "main_motive": main_motive,
            "total_passengers_affected": structured_data["passenger_impact"]["total"],
            "avg_delay_minutes": structured_data["delay_statistics"]["avg_delay"],
            "priority_routes_for_investigation": priority_routes_for_investigation,
            "routes_requiring_nps_analysis": [r["route"] for r in priority_routes_for_investigation if r["priority"] in ["HIGH", "MEDIUM"]]
        }
        
        return structured_data

    def _extract_route_disruption_counts(self, incidents_text: str) -> dict:
        """Extract route disruption counts from incident text for enhanced analysis"""
        import re
        
        route_disruptions = {}
        
        # Enhanced route pattern to capture XXX-YYY format
        route_pattern = r'\b([A-Z]{3})\s*[-–]\s*([A-Z]{3})\b'
        route_matches = re.findall(route_pattern, incidents_text, re.IGNORECASE)
        
        for origin, dest in route_matches:
            route_code = f"{origin.upper()}-{dest.upper()}"
            route_disruptions[route_code] = route_disruptions.get(route_code, 0) + 1
        
        # Enhanced patterns for hub routes (MAD as origin/destination)
        mad_patterns = [
            r'MAD[O-]\s*([A-Z]{3})',  # MAD-XXX or MADO XXX
            r'([A-Z]{3})\s*[-O]\s*MAD',  # XXX-MAD or XXX O MAD  
            r'Madrid\s*[-–]\s*([A-Z]{3})',  # Madrid-XXX
            r'([A-Z]{3})\s*[-–]\s*Madrid',  # XXX-Madrid
        ]
        
        for pattern in mad_patterns:
            matches = re.findall(pattern, incidents_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple matches from multiple capture groups
                    for airport in match:
                        if airport and airport != 'MAD':
                            route_code = f"MAD-{airport.upper()}"
                            route_disruptions[route_code] = route_disruptions.get(route_code, 0) + 1
                else:
                    # Single airport code
                    if match and match != 'MAD':
                        route_code = f"MAD-{match.upper()}"
                        route_disruptions[route_code] = route_disruptions.get(route_code, 0) + 1
        
        # Sort by disruption count (highest first)
        sorted_disruptions = dict(sorted(route_disruptions.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_disruptions

    def _extract_causes_from_llm_response(self, llm_response: str) -> list:
        """Extract identified causes from LLM analysis response"""
        import re
        # If the model produced our structured NCS_REFLEXION format, parse that instead of keyword-snippets
        if llm_response and "NCS_REFLEXION" in llm_response:
            parsed = self._parse_ncs_reflexion(llm_response)
            incident_delta = parsed.get("incidentes_delta_compact")
            dark_horses = parsed.get("dark_horses_compact")
            out = []
            if incident_delta:
                out.append(f"INCIDENTES_DELTA: {incident_delta}")
            if dark_horses:
                out.append(f"DARK_HORSES: {dark_horses}")
            return out[:5]

        causes = []
        
        # Look for common cause indicators in the response
        cause_patterns = [
            r'retraso[s]?|delay[s]?',
            r'cancelaci[oó]n[es]?|cancellation[s]?',  
            r'problema[s]?\s+t[eé]cnico[s]?|technical\s+issue[s]?',
            r'equipaje|baggage',
            r'tripulaci[oó]n|crew',
            r'embarque|boarding',
            r'mantenimiento|maintenance',
            r'clima|weather'
        ]
        
        for pattern in cause_patterns:
            if re.search(pattern, llm_response, re.IGNORECASE):
                # Extract context around the match
                matches = re.finditer(pattern, llm_response, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(llm_response), match.end() + 30)
                    context = llm_response[start:end].strip()
                    causes.append(context[:100])  # Limit length
                    break  # One example per pattern
        
        return causes[:5]  # Return top 5

    def _parse_ncs_reflexion(self, llm_response: str) -> dict:
        """
        Parse the structured NCS_REFLEXION block emitted by the LLM.
        Returns compact strings that are safe to show in collected_data_summary.
        """
        import re
        text = llm_response or ""
        # Normalize
        text = text.replace("\r\n", "\n")

        # Extract INCIDENTES_DELTA section (until DARK_HORSES or HIPOTESIS)
        inc = ""
        m_inc = re.search(r"INCIDENTES_DELTA\s*:\s*(.*?)(?:\n\s*DARK_HORSES\s*:|\n\s*HIPOTESIS_DE_RELACION\s*:|\n\s*NIVEL_DE_CONFIANZA\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m_inc:
            inc = m_inc.group(1).strip()
            inc = re.sub(r"\s+", " ", inc)
            # Compact bullet-ish content
            inc = inc.replace("•", "").strip()

        # Extract DARK_HORSES section similarly
        dh = ""
        m_dh = re.search(r"DARK_HORSES\s*:\s*(.*?)(?:\n\s*HIPOTESIS_DE_RELACION\s*:|\n\s*NIVEL_DE_CONFIANZA\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m_dh:
            dh = m_dh.group(1).strip()
            dh = re.sub(r"\s+", " ", dh)
            dh = dh.replace("•", "").strip()

        # Tighten length so it doesn't pollute summaries
        def clip(s: str, n: int = 180) -> str:
            s = s.strip()
            return s if len(s) <= n else s[: n - 3] + "..."

        return {
            "incidentes_delta_compact": clip(inc, 220) if inc else "",
            "dark_horses_compact": clip(dh, 220) if dh else "",
        }

    def _extract_routes_from_llm_response(self, llm_response: str) -> list:
        """Extract affected routes from LLM analysis response"""
        import re
        
        # Extract routes in XXX-YYY format
        route_pattern = r'\b([A-Z]{3})-([A-Z]{3})\b'
        routes = re.findall(route_pattern, llm_response)
        
        # Convert to route format and deduplicate
        route_list = []
        for origin, dest in routes:
            route = f"{origin}-{dest}"
            if route not in route_list:
                route_list.append(route)
        
        return route_list[:10]  # Return top 10
    
    def _extract_touchpoints_from_llm_response(self, llm_response: str) -> dict:
        """Extract touchpoint correlations from LLM analysis response"""
        import re
        touchpoints = {}
        
        touchpoint_patterns = {
            'Punctuality': r'puntualidad|punctuality|retraso|delay',
            'Baggage': r'equipaje|baggage|maleta',
            'Boarding': r'embarque|boarding|puerta|gate',
            'Crew': r'tripulaci[oó]n|crew|personal',
            'Aircraft': r'avi[oó]n|aircraft|aeronave|t[eé]cnico'
        }
        
        for touchpoint, pattern in touchpoint_patterns.items():
            if re.search(pattern, llm_response, re.IGNORECASE):
                touchpoints[touchpoint] = f'Mentioned in NCS analysis'
        
        return touchpoints

    async def _filter_using_routes_dictionary(
        self,
        ncs_data: pd.DataFrame,
        target_haul: str,
        incident_col: str,
        allow_unknown_route_incidents: bool = True
    ) -> pd.DataFrame:
        """
        Filtra NCS usando el diccionario de rutas de PBI.
        
        LÓGICA: Extrae rutas completas (XXX-YYY) del texto del incidente y las busca
        en el diccionario de PBI para determinar su haul_aggr. Si el incidente menciona
        una ruta del haul opuesto, se excluye.
        
        Ejemplo: Si target_haul='LH' y el incidente menciona 'MAD-BLQ' (que es SH según PBI),
        el incidente se EXCLUYE del análisis de LH.
        
        Args:
            ncs_data: DataFrame con incidentes NCS
            target_haul: Haul objetivo ("LH" o "SH")
            incident_col: Nombre de la columna con texto de incidentes
            
        Returns:
            DataFrame filtrado por rutas del haul objetivo
        """
        import re
        
        try:
            self.logger.info(f"🗺️ Filtering NCS using routes dictionary for haul: {target_haul}")
            
            # 1. Obtener diccionario completo de rutas de PBI
            routes_dict = await self.pbi_collector.collect_routes_dictionary()
            
            if routes_dict.empty:
                self.logger.warning("❌ Routes dictionary is empty, falling back to original data")
                return ncs_data
            
            # 2. Crear lookup de ruta → haul_aggr (ambas direcciones: MAD-JFK y JFK-MAD)
            route_to_haul = {}
            for _, row in routes_dict.iterrows():
                route = str(row.get('route', '')).upper().strip()
                haul = str(row.get('haul_aggr', '')).upper().strip()
                if route and haul and '-' in route:
                    route_to_haul[route] = haul
                    # También añadir la ruta inversa (JFK-MAD si tenemos MAD-JFK)
                    parts = route.split('-')
                    if len(parts) == 2:
                        reverse_route = f"{parts[1]}-{parts[0]}"
                        route_to_haul[reverse_route] = haul
            
            self.logger.info(f"📊 Built route lookup with {len(route_to_haul)} entries (including reverse routes)")
            
            # 3. Determinar el haul opuesto
            opposite_haul = 'SH' if target_haul == 'LH' else 'LH'
            
            # 4. Patrón regex para extraer rutas del texto (formato: 3 letras - 3 letras)
            route_pattern = re.compile(r'\b([A-Z]{3})\s*[-–]\s*([A-Z]{3})\b', re.IGNORECASE)
            
            def incident_belongs_to_target_haul(text):
                """
                Determina si un incidente pertenece al haul objetivo.
                
                Reglas:
                1. Extraer todas las rutas mencionadas en el texto
                2. Buscar cada ruta en el diccionario de PBI
                3. Si ALGUNA ruta es del haul OPUESTO → EXCLUIR (return False)
                4. Si ALGUNA ruta es del haul TARGET → INCLUIR (return True)
                5. Si no se encuentra ninguna ruta en el diccionario:
                   - Si allow_unknown_route_incidents=True → INCLUIR por defecto
                   - Si allow_unknown_route_incidents=False → EXCLUIR (no atribuible al haul)
                """
                if pd.isna(text):
                    return True if allow_unknown_route_incidents else False
                
                text_upper = str(text).upper()
                
                # Extraer todas las rutas del texto
                matches = route_pattern.findall(text_upper)
            
                if not matches:
                    # No se encontraron rutas explícitas (XXX-YYY). Intentar fallback:
                    # si aparece un IATA suelto, asumir ruta MAD-IATA (y su inversa) para atribuir haul.
                    # Esto permite filtrar casos tipo: "por meteorología en GRX" => MAD-GRX.
                    iata_candidates = re.findall(r'\b[A-Z]{3}\b', text_upper)
                    # Eliminar tokens comunes/no-aeropuertos
                    stop = {
                        'IB', 'YW', 'NPS', 'OTP', 'UTC', 'MAD', 'NOC', 'PAX', 'CTO', 'ETA', 'ETD',
                        'ATC', 'IT', 'AOG', 'PIR'
                    }
                    iata_candidates = [c for c in iata_candidates if c not in stop]

                    # Solo considerar IATAs que existan en el diccionario (como destino u origen)
                    known_iatas = set()
                    for r in route_to_haul.keys():
                        if '-' in r:
                            a, b = r.split('-', 1)
                            known_iatas.add(a)
                            known_iatas.add(b)
                    iata_candidates = [c for c in iata_candidates if c in known_iatas]

                    found_target_route = False
                    found_opposite_route = False
                    for code in iata_candidates:
                        assumed_routes = [f"MAD-{code}", f"{code}-MAD"]
                        for route in assumed_routes:
                            route_haul = route_to_haul.get(route)
                            if not route_haul:
                                continue
                            if route_haul == target_haul:
                                found_target_route = True
                            elif route_haul == opposite_haul:
                                found_opposite_route = True

                    if found_opposite_route and not found_target_route:
                        return False
                    if found_target_route:
                        return True

                    # Si no se puede inferir nada, aplicar política por defecto
                    return True if allow_unknown_route_incidents else False
                
                found_target_route = False
                found_opposite_route = False
                
                for origin, dest in matches:
                    route = f"{origin}-{dest}"
                    
                    if route in route_to_haul:
                        route_haul = route_to_haul[route]
                        if route_haul == target_haul:
                            found_target_route = True
                        elif route_haul == opposite_haul:
                            found_opposite_route = True
                            self.logger.debug(f"🚫 Excluding incident: route {route} is {route_haul}, not {target_haul}")
                
                # Si encontramos una ruta del haul opuesto → EXCLUIR
                if found_opposite_route and not found_target_route:
                    return False
                
                # Si encontramos una ruta del haul objetivo → INCLUIR
                if found_target_route:
                    return True
                
                # Si no encontramos rutas conocidas → según configuración
                return True if allow_unknown_route_incidents else False
            
            # 5. Aplicar filtro a cada incidente
            mask = ncs_data[incident_col].apply(incident_belongs_to_target_haul)
            filtered_data = ncs_data[mask]
            
            # 6. Log resultados
            excluded_count = len(ncs_data) - len(filtered_data)
            if excluded_count > 0:
                self.logger.info(f"✅ {target_haul} route filtering: kept {len(filtered_data)}/{len(ncs_data)} incidents (excluded {excluded_count} with {opposite_haul} routes)")
            else:
                self.logger.info(f"✅ {target_haul} route filtering: kept all {len(filtered_data)} incidents")
            
            return filtered_data
                
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            self.logger.warning(f"⚠️ Timeout/Cancelled in routes filtering: {e}, using unfiltered NCS data")
            return ncs_data
        except Exception as e:
            self.logger.error(f"❌ Error filtering with routes dictionary: {e}")
            # Fallback: devolver datos originales para que el análisis continúe
            return ncs_data

    def _combine_and_summarize_ncs_comments(self, current_data: pd.DataFrame, comparison_data: pd.DataFrame, 
                                          current_start: str, current_end: str, 
                                          comparison_start: str, comparison_end: str) -> str:
        """
        Combine and summarize NCS comments from both current and comparison periods.
        This provides operational narrative support for the temporal comparison.
        
        Args:
            current_data: DataFrame with current period NCS data
            comparison_data: DataFrame with comparison period NCS data
            current_start/end: Current period date range
            comparison_start/end: Comparison period date range
            
        Returns:
            String with combined and summarized comments narrative
        """
        try:
            # Extract incident text from both periods
            current_incidents = []
            comparison_incidents = []
            
            if not current_data.empty:
                self.logger.info(f"🔍 DEBUG NCS NARRATIVE: current_data columns: {list(current_data.columns)}")
                incident_col = self._find_column(current_data, ['incident', 'incidents', ''])
                self.logger.info(f"🔍 DEBUG NCS NARRATIVE: incident_col found: {incident_col}")
                if incident_col is not None:
                    current_incidents = current_data[incident_col].astype(str).tolist()
                    self.logger.info(f"🔍 DEBUG NCS NARRATIVE: current_incidents sample: {current_incidents[:3] if current_incidents else 'Empty'}")
            
            if not comparison_data.empty:
                self.logger.info(f"🔍 DEBUG NCS NARRATIVE: comparison_data columns: {list(comparison_data.columns)}")
                incident_col = self._find_column(comparison_data, ['incident', 'incidents', ''])
                self.logger.info(f"🔍 DEBUG NCS NARRATIVE: comparison incident_col found: {incident_col}")
                if incident_col is not None:
                    comparison_incidents = comparison_data[incident_col].astype(str).tolist()
                    self.logger.info(f"🔍 DEBUG NCS NARRATIVE: comparison_incidents sample: {comparison_incidents[:3] if comparison_incidents else 'Empty'}")
            
            # Debug: Check incident data
            self.logger.info(f"🔍 DEBUG NCS NARRATIVE: current_incidents count: {len(current_incidents)}")
            self.logger.info(f"🔍 DEBUG NCS NARRATIVE: comparison_incidents count: {len(comparison_incidents)}")
            
            # Combine all incidents for summary
            all_current_incidents = "\n".join([inc for inc in current_incidents if inc and inc.strip() and inc != 'nan'])
            all_comparison_incidents = "\n".join([inc for inc in comparison_incidents if inc and inc.strip() and inc != 'nan'])
            
            self.logger.info(f"🔍 DEBUG NCS NARRATIVE: all_current_incidents length: {len(all_current_incidents)}")
            self.logger.info(f"🔍 DEBUG NCS NARRATIVE: all_comparison_incidents length: {len(all_comparison_incidents)}")
            
            # Create summary narrative
            summary_parts = []
            
            # Current period summary
            if all_current_incidents:
                current_count = len([inc for inc in current_incidents if inc and inc.strip() and inc != 'nan'])
                summary_parts.append(f"📅 **Período Actual ({current_start} a {current_end}):** {current_count} incidentes operativos")
                
                # Extract key themes from current period
                current_themes = self._extract_operational_themes(all_current_incidents)
                if current_themes:
                    summary_parts.append(f"🔍 **Temas principales:** {', '.join(current_themes)}")
            else:
                # If no detailed incidents available, but we have total count from analysis
                if hasattr(self, '_ncs_total_incidents') and self._ncs_total_incidents > 0:
                    summary_parts.append(f"📅 **Período Actual ({current_start} a {current_end}):** {self._ncs_total_incidents} incidentes registrados (sin detalles narrativos disponibles)")
                else:
                    summary_parts.append(f"📅 **Período Actual ({current_start} a {current_end}):** Sin incidentes registrados")
            
            # Comparison period summary
            if all_comparison_incidents:
                comparison_count = len([inc for inc in comparison_incidents if inc and inc.strip() and inc != 'nan'])
                summary_parts.append(f"📅 **Período Comparativo ({comparison_start} a {comparison_end}):** {comparison_count} incidentes operativos")
                
                # Extract key themes from comparison period
                comparison_themes = self._extract_operational_themes(all_comparison_incidents)
                if comparison_themes:
                    summary_parts.append(f"🔍 **Temas principales:** {', '.join(comparison_themes)}")
            else:
                summary_parts.append(f"📅 **Período Comparativo ({comparison_start} a {comparison_end}):** Sin incidentes registrados")
            
            # Combined narrative
            if all_current_incidents or all_comparison_incidents:
                summary_parts.append("\n📋 **Narrativa Operativa Combinada:**")
                
                # Create combined narrative using LLM if available
                if hasattr(self, 'llm') and self.llm:
                    combined_narrative = self._generate_combined_narrative(
                        all_current_incidents, all_comparison_incidents,
                        current_start, current_end, comparison_start, comparison_end
                    )
                    summary_parts.append(combined_narrative)
                else:
                    # Fallback to simple summary
                    total_current = len([inc for inc in current_incidents if inc and inc.strip() and inc != 'nan'])
                    total_comparison = len([inc for inc in comparison_incidents if inc and inc.strip() and inc != 'nan'])
                    
                    if total_current > 0 or total_comparison > 0:
                        summary_parts.append(f"📊 **Resumen:** {total_current} incidentes en período actual vs {total_comparison} en período comparativo")
                        
                        if total_current > total_comparison:
                            summary_parts.append("📈 **Tendencia:** Aumento en incidentes operativos")
                        elif total_current < total_comparison:
                            summary_parts.append("📉 **Tendencia:** Reducción en incidentes operativos")
                        else:
                            summary_parts.append("➡️ **Tendencia:** Nivel similar de incidentes operativos")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error combining NCS comments: {str(e)}")
            return f"❌ Error al combinar comentarios NCS: {str(e)}"
    
    def _extract_operational_themes(self, incidents_text: str) -> List[str]:
        """Extract key operational themes from NCS incidents text"""
        if not incidents_text or not incidents_text.strip():
            return []
        
        themes = []
        text_lower = incidents_text.lower()
        
        # Define theme keywords
        theme_keywords = {
            'cancelaciones': ['cancelado', 'cancelación', 'cancelados'],
            'retrasos': ['retraso', 'retrasos', 'demora', 'demoras'],
            'desvios': ['desvío', 'desvios', 'desviado'],
            'equipos': ['a330', 'a350', 'a359', 'equipo', 'aircraft'],
            'conexiones': ['conexión', 'conexiones', 'conecta'],
            'técnicos': ['técnico', 'técnica', 'técnicas', 'técnicos'],
            'operativos': ['operativo', 'operativa', 'operacional']
        }
        
        # Count theme occurrences
        theme_counts = {}
        for theme, keywords in theme_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                theme_counts[theme] = count
        
        # Return top 3 themes by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:3]]
    
    def _generate_combined_narrative(self, current_incidents: str, comparison_incidents: str,
                                   current_start: str, current_end: str, 
                                   comparison_start: str, comparison_end: str) -> str:
        """Generate combined narrative using LLM"""
        try:
            from langchain.schema import HumanMessage
            
            prompt_text = f"""
            Analiza los incidentes operativos de dos períodos y genera una narrativa operativa combinada.
            
            Período Actual ({current_start} a {current_end}):
            {current_incidents[:2000] if current_incidents else "Sin incidentes"}
            
            Período Comparativo ({comparison_start} a {comparison_end}):
            {comparison_incidents[:2000] if comparison_incidents else "Sin incidentes"}
            
            Genera una narrativa operativa que:
            1. Identifique los principales tipos de incidentes en cada período
            2. Compare las tendencias entre períodos
            3. Destaque cambios significativos en operaciones
            4. Proporcione contexto operativo relevante
            
            Responde en español de manera concisa y profesional.
            """
            
            # Use LangChain invoke method with proper message format
            messages = [HumanMessage(content=prompt_text)]
            # self.llm is the LLM wrapper, self.llm.llm is the actual LangChain ChatBedrock instance
            response = self.llm.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating combined narrative: {str(e)}")
            return "📝 Narrativa operativa no disponible debido a error en procesamiento"



    def _get_system_prompt(self, mode: str = "comparative") -> str:
        """Get system prompt for specified mode, injecting focus_touchpoint block if active"""
        if mode == "single":
            prompt = self.config.get('single_prompts', {}).get('system_prompt', '')
        else:
            prompt = self.config.get('comparative_prompts', {}).get('system_prompt', '')

        # Inject focus_touchpoint section if active
        if self.focus_touchpoint:
            focus_block = self.config.get('focus_touchpoint_prompt', '')
            if focus_block:
                prompt = prompt + "\n\n" + focus_block.replace('{focus_touchpoint}', self.focus_touchpoint)

        return prompt
    
    def _get_input_template(self, mode: str = "comparative") -> str:
        """Get input template for specified mode"""
        if mode == "single":
            return self.config.get('single_prompts', {}).get('input_template', '')
        else:
            return self.config.get('comparative_prompts', {}).get('input_template', '')
    
    def _get_tool_result_message(self, mode: str = "comparative") -> str:
        """Get tool result message template for specified mode"""
        if mode == "single":
            return self.config.get('single_prompts', {}).get('tool_result_message', '📊 RESULTADO {tool_name}:\n{tool_result}')
        else:
            return self.config.get('comparative_prompts', {}).get('tool_result_message', '📊 RESULTADO DE HERRAMIENTA: {tool_name}\n{tool_result}')
    
    def _get_reflection_prompt(self, mode: str = "comparative", tool_name: str = None, flow_type: str = None) -> str:
        """Get reflection prompt for specified mode with dynamic tool-specific guidance"""
        base_prompt = ""
        if mode == "single":
            base_prompt = self.config.get('single_prompts', {}).get('reflection_prompt', '')
        else:
            base_prompt = self.config.get('comparative_prompts', {}).get('reflection_prompt', '')
        
        # DEBUG: Log the base prompt
        self.logger.info(f"🔍 DEBUG REFLECTION: mode={mode}, tool_name={tool_name}, flow_type={flow_type}")
        self.logger.info(f"🔍 DEBUG REFLECTION: base_prompt length={len(base_prompt)}")
        
        # If tool_name is provided, add dynamic tool-specific guidance
        if tool_name and base_prompt:
            tool_prompt = self._get_tool_prompt(tool_name, mode, flow_type)
            self.logger.info(f"🔍 DEBUG REFLECTION: tool_prompt length={len(tool_prompt) if tool_prompt else 0}")
            
            if tool_prompt:
                # Add the tool-specific guidance to the base reflection prompt
                enhanced_prompt = f"{base_prompt}\n\n🎯 GUÍA ESPECÍFICA DE LA HERRAMIENTA:\n{tool_prompt}"
                self.logger.info(f"🔍 DEBUG REFLECTION: enhanced_prompt length={len(enhanced_prompt)}")
                return enhanced_prompt
            else:
                self.logger.warning(f"🔍 DEBUG REFLECTION: No tool_prompt found for {tool_name}")
        
        self.logger.info(f"🔍 DEBUG REFLECTION: Returning base_prompt (no enhancement)")
        return base_prompt
    
    def _get_synthesis_prompt(self, mode: str = "comparative") -> str:
        """Get synthesis prompt for specified mode"""
        if mode == "single":
            return self.config.get('single_prompts', {}).get('synthesis_prompt', '')
        else:
            return self.config.get('comparative_prompts', {}).get('synthesis_prompt', '')
    
    def _determine_flow_type_from_drivers(self, tool_result: str) -> str:
        """Determine flow type from explanatory drivers tool result"""
        try:
            result_lower = tool_result.lower()
            self.logger.info(f"🔍 DEBUG FLOW: Analyzing tool_result of length {len(tool_result)}")
            
            # Check for operational drivers
            operational_keywords = ['punctuality', 'otp', 'delay', 'baggage', 'mishandling', 'misconex', 'load factor', 'connections', 'arrivals']
            operational_count = sum(1 for keyword in operational_keywords if keyword in result_lower)
            self.logger.info(f"🔍 DEBUG FLOW: Found {operational_count} operational keywords")
            
            # Check for product drivers
            product_keywords = ['crew', 'food', 'comfort', 'entertainment', 'service', 'cleanliness', 'seating', 'amenities']
            product_count = sum(1 for keyword in product_keywords if keyword in result_lower)
            self.logger.info(f"🔍 DEBUG FLOW: Found {product_count} product keywords")
            
            # Determine flow based on driver types
            if operational_count > 0 and product_count > 0:
                flow_type = 'mixed'
            elif operational_count > 0:
                flow_type = 'operative'
            elif product_count > 0:
                flow_type = 'product'
            else:
                # Default to operative if unclear
                flow_type = 'operative'
            
            self.logger.info(f"🔍 DEBUG FLOW: Determined flow_type: {flow_type}")
            return flow_type
                
        except Exception as e:
            self.logger.warning(f"Could not determine flow type from drivers: {e}")
            return 'operative'  # Default fallback
    
    def _get_tool_prompt(self, tool_name: str, mode: str = "comparative", flow_type: str = None) -> str:
        """Get tool-specific prompt for specified mode and flow type"""
        self.logger.info(f"🔍 DEBUG TOOL_PROMPT: tool_name={tool_name}, mode={mode}, flow_type={flow_type}")
        
        tool_prompts = self.config.get('tools_prompts', {}).get(tool_name, {})
        self.logger.info(f"🔍 DEBUG TOOL_PROMPT: Found tool_prompts for {tool_name}: {bool(tool_prompts)}")
        
        if mode not in tool_prompts:
            self.logger.warning(f"🔍 DEBUG TOOL_PROMPT: Mode {mode} not found in tool_prompts for {tool_name}")
            return ''
        
        mode_prompts = tool_prompts[mode]
        self.logger.info(f"🔍 DEBUG TOOL_PROMPT: mode_prompts type={type(mode_prompts)}, content preview={str(mode_prompts)[:100]}...")
        
        # If flow_type is specified and exists, return that specific flow
        if flow_type and isinstance(mode_prompts, dict) and flow_type in mode_prompts:
            self.logger.info(f"🔍 DEBUG TOOL_PROMPT: Returning specific flow_type={flow_type}")
            return mode_prompts[flow_type]
        
        # If no flow_type specified, return the mode prompt (could be string or dict)
        if isinstance(mode_prompts, str):
            self.logger.info(f"🔍 DEBUG TOOL_PROMPT: Returning string mode_prompts")
            return mode_prompts
        elif isinstance(mode_prompts, dict):
            # For comparative mode with flows, provide guidance on how to choose
            if tool_name == 'explanatory_drivers_tool':
                # For explanatory drivers, provide all flow options
                flow_options = []
                for flow, prompt in mode_prompts.items():
                    flow_options.append(f"**{flow.upper()} FLOW:**\n{prompt}")
                self.logger.info(f"🔍 DEBUG TOOL_PROMPT: explanatory_drivers_tool - returning {len(flow_options)} flow options")
                return "\n\n".join(flow_options)
            else:
                # For other tools, return the operative flow as default
                operative_prompt = mode_prompts.get('operative', str(mode_prompts))
                self.logger.info(f"🔍 DEBUG TOOL_PROMPT: Returning operative flow as default")
                return operative_prompt
        
        self.logger.info(f"🔍 DEBUG TOOL_PROMPT: Returning str(mode_prompts)")
        return str(mode_prompts)

    def _parse_reflection_response(self, response: str) -> Tuple[str, str]:
        """Parse reflection response to extract reflection and next tool code"""
        import re
        
        reflection = ""
        next_tool_code = ""
        
        # DEBUG: Log the raw response
        self.logger.info(f"🔍 DEBUG PARSE: Parsing response of length {len(response)}")
        
        # Extract reflection from ```reflection``` block
        reflection_match = re.search(r'```reflection\s*\n(.*?)\n```', response, re.DOTALL)
        if reflection_match:
            reflection = reflection_match.group(1).strip()
            self.logger.info(f"🔍 DEBUG PARSE: Found reflection block, length={len(reflection)}")
        else:
            self.logger.warning(f"🔍 DEBUG PARSE: No reflection block found in response")
        
        # Extract next tool code from ```next_tool``` block
        next_tool_code_match = re.search(r'```next_tool\s*\n(.*?)\n```', response, re.DOTALL)
        if next_tool_code_match:
            next_tool_code = next_tool_code_match.group(1).strip()
            self.logger.info(f"🔍 DEBUG PARSE: Found next_tool block, length={len(next_tool_code)}")
            self.logger.info(f"🔍 DEBUG PARSE: next_tool_code content: {next_tool_code}")
        else:
            self.logger.warning(f"🔍 DEBUG PARSE: No next_tool block found in response")
        
        self.logger.info(f"🔍 DEBUG PARSE: Final result - reflection: {len(reflection)}, next_tool: {len(next_tool_code)}")
        return reflection, next_tool_code


# Convenience function
async def investigate_anomaly_causally(
    node_path: str,
    start_date: str,
    end_date: str,
    anomaly_type: str,
    anomaly_magnitude: float,
    llm_type: Optional[LLMType] = None,
    custom_helper_prompts: Optional[Dict[str, Any]] = None
) -> str:
    """
    Clean causal investigation with separated workflow
    """
    agent = CausalExplanationAgent(
        llm_type=llm_type,
        custom_helper_prompts=custom_helper_prompts
    )
    
    result = await agent.investigate_anomaly(
        node_path=node_path,
        start_date=start_date,
        end_date=end_date,
        anomaly_type=anomaly_type,
        anomaly_magnitude=anomaly_magnitude
    )
    
    return result


# Example usage
if __name__ == "__main__":
    async def main():
        # Test the clean causal investigation
        result = await investigate_anomaly_causally(
            node_path="Global/LH/Business",
            start_date="2025-05-15",
            end_date="2025-05-15", 
            anomaly_type="negative",
            anomaly_magnitude=-15.0,
            llm_type=get_default_llm_type()
        )
        
        print("🎯 Investigation Result:")
        print(result)
    
    asyncio.run(main()) 