import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging
from botocore.exceptions import ClientError, NoCredentialsError

from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_agent_conversations_folder


from dashboard_analyzer.anomaly_explanation.genai_core.utils.aws_session import get_aws_session

class S3ReportUploader:
    """
    Handles uploading comprehensive analysis reports to S3 bucket.
    
    Features:
    - AWS S3 integration with proper error handling
    - JSON report generation with metadata
    - Automatic file naming with timestamps and date ranges
    - Credential management (environment variables or temp files)
    - Comprehensive logging
    """
    
    def __init__(self, environment: str = "local"):
        """
        Initialize S3 Report Uploader
        
        Args:
            environment: Environment type ("local" or "prod")
        """
        self.logger = logging.getLogger(__name__)
        
        # S3 configuration
        self.bucket_name = "ibdata-sbx-ew1-s3-customer"
        self.base_prefix = "customer/catia/reports/raw/"
        
        # Environment configuration
        self.environment = environment
        
        # Initialize AWS session using unified resolver - Reports use standard keys (not sandbox)
        self.session = get_aws_session(environment=self.environment, use_sandbox=False)
        self.s3_client = self.session.client('s3')
    
    def _setup_aws_credentials(self):
        # Legacy method kept for backward compatibility but functionality moved to __init__
        pass
    
    def _generate_filename(self, execution_date: datetime, analysis_date: str, 
                          comparison_start_date: Optional[str] = None, 
                          comparison_end_date: Optional[str] = None) -> str:
        """
        Generate S3 filename based on execution date and analysis date range
        
        Args:
            execution_date: When the analysis was executed
            analysis_date: Main analysis date
            comparison_start_date: Start of comparison period (optional)
            comparison_end_date: End of comparison period (optional)
            
        Returns:
            Formatted filename string
        """
        # Format execution timestamp
        exec_timestamp = execution_date.strftime('%Y-%m-%dT%H-%M-%S')
        
        # Determine date range for filename
        if comparison_start_date and comparison_end_date:
            date_range = f"{comparison_start_date}_to_{analysis_date}"
        else:
            date_range = analysis_date
        
        return f"{exec_timestamp}_{date_range}.json"
    
    def _build_report_json(self, 
                          execution_metadata: Dict[str, Any],
                          weekly_analysis_params: Dict[str, Any],
                          daily_analysis_params: Dict[str, Any],
                          date_ranges: Dict[str, Any],
                          final_synthesis: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build the complete report JSON structure
        
        Args:
            execution_metadata: Basic execution information
            weekly_analysis_params: Parameters for weekly analysis
            daily_analysis_params: Parameters for daily analysis
            date_ranges: Date range information
            final_synthesis: The final synthesis - can be string (HTML) or dict (Adaptive Card JSON)
            
        Returns:
            Complete report dictionary
        """
        return {
            "execution_metadata": execution_metadata,
            "weekly_analysis_params": weekly_analysis_params,
            "daily_analysis_params": daily_analysis_params,
            "date_ranges": date_ranges,
            "final_synthesis": final_synthesis,
            "report_generated_at": datetime.now().isoformat() + "Z"
        }
    
    async def upload_comprehensive_report(self,
                                        execution_date: datetime,
                                        analysis_date: str,
                                        segment: str,
                                        explanation_mode: str,
                                        causal_filter: str,
                                        weekly_analysis_params: Dict[str, Any],
                                        daily_analysis_params: Dict[str, Any],
                                        date_ranges: Dict[str, Any],
                                        final_synthesis: Union[str, Dict[str, Any]],
                                        comparison_start_date: Optional[str] = None,
                                        comparison_end_date: Optional[str] = None) -> Optional[str]:
        """
        Upload comprehensive analysis report to S3
        
        Args:
            execution_date: When the analysis was executed
            analysis_date: Main analysis date
            segment: Analysis segment (e.g., "Global")
            explanation_mode: Explanation mode used
            causal_filter: Causal filter applied
            weekly_analysis_params: Weekly analysis parameters
            daily_analysis_params: Daily analysis parameters
            date_ranges: Date range information
            final_synthesis: Final synthesis - can be string (HTML) or dict (Adaptive Card JSON)
            comparison_start_date: Start of comparison period
            comparison_end_date: End of comparison period
            
        Returns:
            S3 key of uploaded file if successful, None if failed
        """
        try:
            # Validate inputs
            if not final_synthesis:
                self.logger.warning("⚠️ Final synthesis is empty, skipping S3 upload")
                return None
            # For string synthesis, also check if it's empty after stripping
            if isinstance(final_synthesis, str) and not final_synthesis.strip():
                self.logger.warning("⚠️ Final synthesis is empty, skipping S3 upload")
                return None
            
            # Build execution metadata
            execution_metadata = {
                "execution_date": execution_date.isoformat() + "Z",
                "analysis_date": analysis_date,
                "segment": segment,
                "explanation_mode": explanation_mode,
                "causal_filter": causal_filter
            }
            
            # Build complete report
            report_data = self._build_report_json(
                execution_metadata=execution_metadata,
                weekly_analysis_params=weekly_analysis_params,
                daily_analysis_params=daily_analysis_params,
                date_ranges=date_ranges,
                final_synthesis=final_synthesis
            )
            
            # Generate filename and S3 key
            filename = self._generate_filename(
                execution_date=execution_date,
                analysis_date=analysis_date,
                comparison_start_date=comparison_start_date,
                comparison_end_date=comparison_end_date
            )
            s3_key = f"{self.base_prefix}{filename}"
            
            # Convert to JSON string
            json_content = json.dumps(report_data, indent=2, ensure_ascii=False)
            
            # Upload to S3
            self.logger.info(f"📤 Uploading comprehensive report to S3: s3://{self.bucket_name}/{s3_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'execution_date': execution_date.isoformat(),
                    'analysis_date': analysis_date,
                    'segment': segment,
                    'report_type': 'comprehensive_analysis'
                }
            )
            
            self.logger.info(f"✅ Successfully uploaded comprehensive report: s3://{self.bucket_name}/{s3_key}")
            self.logger.info(f"📊 Report size: {len(json_content)} characters")
            
            return s3_key
            
        except NoCredentialsError:
            self.logger.error("❌ AWS credentials not found. Cannot upload to S3.")
            return None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                self.logger.error(f"❌ S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                self.logger.error(f"❌ Access denied to S3 bucket '{self.bucket_name}'")
            else:
                self.logger.error(f"❌ S3 client error: {error_code} - {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error uploading to S3: {str(e)}")
            return None
    
    async def upload_agent_conversation(self, 
                                      conversation_data: Dict[str, Any],
                                      agent_type: str,
                                      filename: str) -> Optional[str]:
        """
        Upload agent conversation to S3
        
        Args:
            conversation_data: The conversation data dictionary
            agent_type: Type of agent ("causal_explanation", "anomaly_interpreter", "anomaly_summary")
            filename: The filename to use for the upload
            
        Returns:
            S3 key of uploaded file if successful, None if failed
        """
        try:
            # Only upload in production environment
            if self.environment != "prod":
                self.logger.info(f"🔧 Local environment: Skipping S3 upload for {agent_type} conversation")
                return None
            
            # Validate inputs
            if not conversation_data or not filename:
                self.logger.warning("⚠️ Invalid conversation data or filename, skipping S3 upload")
                return None
            
            # Generate S3 key (includes LLM type prefix in folder name)
            s3_key = f"{self.base_prefix}{get_agent_conversations_folder()}/{agent_type}/{filename}"
            
            # Convert to JSON string
            json_content = json.dumps(conversation_data, indent=2, ensure_ascii=False)
            
            # Upload to S3
            self.logger.info(f"📤 Uploading {agent_type} conversation to S3: s3://{self.bucket_name}/{s3_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'agent_type': agent_type,
                    'upload_timestamp': datetime.now().isoformat(),
                    'conversation_type': 'agent_conversation'
                }
            )
            
            self.logger.info(f"✅ Successfully uploaded {agent_type} conversation: s3://{self.bucket_name}/{s3_key}")
            self.logger.info(f"📊 Conversation size: {len(json_content)} characters")
            
            return s3_key
            
        except NoCredentialsError:
            self.logger.error("❌ AWS credentials not found. Cannot upload conversation to S3.")
            return None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                self.logger.error(f"❌ S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                self.logger.error(f"❌ Access denied to S3 bucket '{self.bucket_name}'")
            else:
                self.logger.error(f"❌ S3 client error: {error_code} - {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error uploading conversation to S3: {str(e)}")
            return None
    
    async def upload_causal_conversation(self, conversation_data: Dict[str, Any], filename: str) -> Optional[str]:
        """Upload causal explanation agent conversation to S3"""
        return await self.upload_agent_conversation(conversation_data, "causal_explanation", filename)
    
    async def upload_interpreter_conversation(self, conversation_data: Dict[str, Any], filename: str) -> Optional[str]:
        """Upload anomaly interpreter agent conversation to S3"""
        return await self.upload_agent_conversation(conversation_data, "interpreter", filename)
    
    async def upload_summary_conversation(self, conversation_data: Dict[str, Any], filename: str) -> Optional[str]:
        """Upload anomaly summary agent conversation to S3"""
        return await self.upload_agent_conversation(conversation_data, "anomaly_summary", filename)

    async def upload_mapped_info(self, data: Dict[str, Any], filename: str) -> Optional[str]:
        """
        Upload mapped info (detailed tree analysis) to S3
        
        Args:
            data: The JSON data dictionary
            filename: The filename to use
            
        Returns:
            S3 key of uploaded file if successful, None if failed
        """
        try:
            # Only upload in production environment
            if self.environment != "prod":
                self.logger.info(f"🔧 Local environment: Skipping S3 upload for mapped info")
                return None
            
            # Validate inputs
            if not data or not filename:
                self.logger.warning("⚠️ Invalid data or filename, skipping S3 upload")
                return None
            
            # Generate S3 key
            s3_key = f"{self.base_prefix}mapped_info/{filename}"
            
            # Convert to JSON string
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Upload to S3
            self.logger.info(f"📤 Uploading mapped info to S3: s3://{self.bucket_name}/{s3_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'upload_timestamp': datetime.now().isoformat(),
                    'content_type': 'mapped_info_tree'
                }
            )
            
            self.logger.info(f"✅ Successfully uploaded mapped info: s3://{self.bucket_name}/{s3_key}")
            
            return s3_key
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error uploading mapped info to S3: {str(e)}")
            return None
