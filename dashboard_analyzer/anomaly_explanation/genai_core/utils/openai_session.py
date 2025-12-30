import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def get_openai_credentials(environment: str = "prod") -> Dict[str, Optional[str]]:
    """
    Unified OpenAI credentials resolver for both OpenAI Platform and Azure OpenAI.
    
    Args:
        environment: 'local' or 'prod' (currently handled similarly via os.getenv)
        
    Returns:
        Dictionary with credentials for both providers.
    """
    creds = {
        # OpenAI Platform (Direct)
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'openai_project_id': os.getenv('OPENAI_PROJECT_ID'),
        'openai_project_name': os.getenv('OPENAI_PROJECT_NAME'),
        'openai_service_account_name': os.getenv('OPENAI_SERVICE_ACCOUNT_NAME'),
        'openai_service_account_id': os.getenv('OPENAI_SERVICE_ACCOUNT_ID'),
        
        # Azure OpenAI
        'azure_api_key': os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        'azure_endpoint': os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT"),
        'azure_api_version': os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        'azure_deployment': os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }
    
    return creds

def is_openai_platform_configured(creds: Dict[str, Optional[str]]) -> bool:
    """Check if OpenAI Platform credentials are sufficiently provided."""
    return bool(creds.get('openai_api_key'))

def is_azure_openai_configured(creds: Dict[str, Optional[str]]) -> bool:
    """Check if Azure OpenAI credentials are sufficiently provided."""
    return all([
        creds.get('azure_api_key'),
        creds.get('azure_endpoint'),
        creds.get('azure_deployment')
    ])

