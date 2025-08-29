import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from botocore.exceptions import ClientError, NoCredentialsError


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
    
    def __init__(self, temp_env_file: str = None):
        """
        Initialize S3 Report Uploader
        
        Args:
            temp_env_file: Path to temporary credentials file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # S3 configuration
        self.bucket_name = "ibdata-sbx-ew1-s3-customer"
        self.base_prefix = "customer/catia/reports/raw/"
        
        # Initialize AWS session
        self.temp_env_file = temp_env_file
        self._setup_aws_credentials()
    
    def _setup_aws_credentials(self):
        """Setup AWS credentials from temporary file or environment"""
        try:
            if self.temp_env_file and os.path.exists(self.temp_env_file):
                # Read credentials from temp file
                credentials = {}
                with open(self.temp_env_file, 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            credentials[key.strip()] = value.strip()
                
                # Set up boto3 session with credentials
                session = boto3.Session(
                    aws_access_key_id=credentials.get('aws_access_key_id'),
                    aws_secret_access_key=credentials.get('aws_secret_access_key'),
                    aws_session_token=credentials.get('aws_session_token'),
                    region_name='eu-west-1'
                )
                self.s3_client = session.client('s3')
                self.logger.info("✅ Successfully configured AWS credentials from temp file")
                
            else:
                # Fallback to environment variables
                self.s3_client = boto3.client('s3', region_name='eu-west-1')
                self.logger.info("✅ Using AWS credentials from environment")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup AWS credentials: {str(e)}")
            raise
    
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
                          final_synthesis: str) -> Dict[str, Any]:
        """
        Build the complete report JSON structure
        
        Args:
            execution_metadata: Basic execution information
            weekly_analysis_params: Parameters for weekly analysis
            daily_analysis_params: Parameters for daily analysis
            date_ranges: Date range information
            final_synthesis: The final executive summary text
            
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
                                        final_synthesis: str,
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
            final_synthesis: Final executive summary
            comparison_start_date: Start of comparison period
            comparison_end_date: End of comparison period
            
        Returns:
            S3 key of uploaded file if successful, None if failed
        """
        try:
            # Validate inputs
            if not final_synthesis or not final_synthesis.strip():
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
    
    def test_connection(self) -> bool:
        """
        Test S3 connection and permissions
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to list objects in the bucket (minimal permission test)
            self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.base_prefix,
                MaxKeys=1
            )
            self.logger.info("✅ S3 connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ S3 connection test failed: {str(e)}")
            return False
