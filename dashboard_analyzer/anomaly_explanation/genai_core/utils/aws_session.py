import os
import boto3
import logging
from pathlib import Path
from typing import Optional
from .enums import load_aws_credentials_from_temp_file

logger = logging.getLogger(__name__)

def get_aws_session(environment: str = "prod", region_name: str = "eu-west-1", use_sandbox: bool = True) -> boto3.Session:
    """
    Unified AWS session resolver for both local and production environments.
    
    Order of priority:
    1. If environment is 'local', try loading from temp_aws_credentials.env.
    2. If environment is 'prod' or local file missing, use native boto3 session (IAM roles/Env vars).
    
    Args:
        environment: 'local' or 'prod'
        region_name: AWS region (default: eu-west-1)
        use_sandbox: Whether to use sbx_* keys (Bedrock) or standard keys (S3).
        
    Returns:
        boto3.Session object
    """
    creds = get_aws_credentials(environment, region_name, use_sandbox=use_sandbox)
    
    if creds.get('aws_access_key_id') and creds.get('aws_secret_access_key'):
        logger.info(f"✅ Resolving AWS Session: Using explicit credentials (mode: {environment}, sandbox: {use_sandbox})")
        return boto3.Session(
            aws_access_key_id=creds['aws_access_key_id'],
            aws_secret_access_key=creds['aws_secret_access_key'],
            aws_session_token=creds['aws_session_token'],
            region_name=creds.get('region_name', region_name)
        )

    # Production or Fallback: Let boto3 handle it (IAM Roles, Environment Variables, etc.)
    logger.info(f"✅ Resolving AWS Session: Using NATIVE mode (environment={environment})")
    return boto3.Session(region_name=region_name)

def get_aws_credentials(environment: str = "prod", default_region: str = "eu-west-1", use_sandbox: bool = True) -> dict:
    """
    Unified AWS credentials resolver.
    
    Args:
        environment: 'local' or 'prod'
        default_region: Default region if not found
        use_sandbox: Whether to use sbx_* keys (Bedrock) or standard keys (S3).
        
    Returns:
        Dictionary with keys: aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
    """
    if environment == "local":
        return load_aws_credentials_from_temp_file(use_sandbox=use_sandbox)
    
    # In production, we return empty/env-based credentials so boto3 uses IAM roles
    return {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'aws_session_token': os.getenv('AWS_SESSION_TOKEN'),
        'region_name': os.getenv('AWS_REGION', default_region)
    }

