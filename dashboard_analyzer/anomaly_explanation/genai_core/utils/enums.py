from enum import Enum
from pathlib import Path
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Role-specific LLM configuration (Bedrock)
# - CausalExplanationAgent (multi-tool causal investigations): Sonnet 4.5
# - AnomalyInterpreterAgent (tree interpretation): Sonnet 4.5 (temporarily; Opus 4.6 pending Marketplace subscription in prod)
# - AnomalySummaryAgent (executive summarizer): Sonnet 4.5 (temporarily; Opus 4.6 pending Marketplace subscription in prod)
CAUSAL_LLM_TYPE = "CLAUDE_SONNET_4_5"
INTERPRETER_LLM_TYPE = "CLAUDE_SONNET_4_5"  # TODO: restore to CLAUDE_OPUS_4_6 once AWS Marketplace subscription confirmed in prod
SUMMARIZER_LLM_TYPE = "CLAUDE_SONNET_4_5"  # TODO: restore to CLAUDE_OPUS_4_6 once AWS Marketplace subscription confirmed in prod

# Legacy: conversation folder prefix + get_default_llm_type() (interpreter-aligned)
DEFAULT_LLM_TYPE = INTERPRETER_LLM_TYPE


def load_aws_credentials_from_temp_file(temp_env_file: str = None, use_sandbox: bool = True) -> dict:
    """
    Load AWS credentials from temp_aws_credentials.env file.
    
    Args:
        temp_env_file: Optional explicit path to credentials file. 
                       If not provided, uses Path.cwd() / 'temp_aws_credentials.env'
        use_sandbox: If True, uses sbx_* keys (Bedrock). If False, uses standard keys (S3/NCS).
    
    Returns:
        dict with keys: aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
    """
    creds = {
        'aws_access_key_id': None,
        'aws_secret_access_key': None,
        'aws_session_token': None,
        'region_name': 'eu-west-1'  # Default region for inference profiles
    }
    
    # Use provided path or default to cwd (same pattern as NCSDataCollector)
    if temp_env_file:
        creds_path = Path(temp_env_file)
    else:
        creds_path = Path.cwd() / 'temp_aws_credentials.env'
    
    if not creds_path.exists():
        # Fallback to environment variables
        logger.warning(f"⚠️ {creds_path} not found! Falling back to environment variables.")
        
        creds['aws_access_key_id'] = os.getenv('AWS_ACCESS_KEY_ID')
        creds['aws_secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY')
        creds['aws_session_token'] = os.getenv('AWS_SESSION_TOKEN')
        creds['region_name'] = os.getenv('AWS_REGION', 'eu-west-1')
        
        return creds
    
    # Read and parse the credentials file
    mode_str = "Bedrock (sbx_*)" if use_sandbox else "Standard (NCS/S3)"
    logger.info(f"✅ Loading {mode_str} credentials from: {creds_path}")
    with open(creds_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if use_sandbox:
                # Use sbx_* credentials (sandbox) - these have access to inference profiles
                if key == 'sbx_aws_access_key_id':
                    creds['aws_access_key_id'] = value
                elif key == 'sbx_aws_secret_access_key':
                    creds['aws_secret_access_key'] = value
                elif key == 'sbx_aws_session_token':
                    creds['aws_session_token'] = value
            else:
                # Use standard credentials (no prefix)
                if key == 'aws_access_key_id':
                    creds['aws_access_key_id'] = value
                elif key == 'aws_secret_access_key':
                    creds['aws_secret_access_key'] = value
                elif key == 'aws_session_token':
                    creds['aws_session_token'] = value
    
    return creds

class MessageType(Enum):
    """
    Enumerated type representing the possible message types.
    It distinguishes between messages from the user, the AI model, and system-level messages.
    """
    USER = 'USER'
    AI = 'AI'
    SYSTEM = 'SYSTEM'
    TOOL = 'TOOL'

class LLMType(Enum):
    """
    Enumerated type representing the Large Language Models (LLM) types.
    This could be used to choose between different versions or types of AI models for different tasks.
    """
    CLAUDE_V2 = 'CLAUDE_V2'
    CLAUDE_INSTANT = 'CLAUDE_INSTANT'
    CLAUDE_3_HAIKU = 'CLAUDE_3_HAIKU'
    CLAUDE_3_5_HAIKU = 'CLAUDE_3_5_HAIKU'
    CLAUDE_3_OPUS = 'CLAUDE_3_OPUS'
    CLAUDE_3_5_SONNET = 'CLAUDE_3_5_SONNET'
    CLAUDE_3_5_SONNET_V2 = 'CLAUDE_3_5_SONNET_V2'
    CLAUDE_SONNET_4 = 'CLAUDE_SONNET_4'
    CLAUDE_OPUS_4_5 = 'CLAUDE_OPUS_4_5'
    GPT3_5 = 'GPT3_5'
    GPT4 = 'GPT4'
    GPT4o = 'GPT4o'
    GPT4o_MINI = 'GPT4o_MINI'
    O1_MINI = 'O1_MINI'  # o1-mini in Azure OpenAI
    O3_MINI = 'O3_MINI'  # o3-mini in Azure OpenAI
    O3 = 'O3'            # o3 in Azure OpenAI
    O4_MINI = 'O4_MINI'  # o4-mini in Azure OpenAI
    LLAMA3_70 = 'LLAMA3_70'
    LLAMA3_1_70 = 'LLAMA3_1_70'
    LLAMA3_1_405 = 'LLAMA3_1_405'
    CLAUDE_3_7_SONNET = 'CLAUDE_3_7_SONNET'
    
    # New models
    AMAZON_NOVA_2_LITE = 'AMAZON_NOVA_2_LITE'
    AMAZON_NOVA_PRO = 'AMAZON_NOVA_PRO'
    AMAZON_TITAN_EMBED_TEXT_V2 = 'AMAZON_TITAN_EMBED_TEXT_V2'
    CLAUDE_HAIKU_4_5 = 'CLAUDE_HAIKU_4_5'
    CLAUDE_SONNET_4_5 = 'CLAUDE_SONNET_4_5'
    CLAUDE_OPUS_4_6 = 'CLAUDE_OPUS_4_6'  # eu.anthropic.claude-opus-4-6-v1
    GPT_OSS_120B = 'GPT_OSS_120B'
    GPT_5_2 = 'GPT_5_2'

def _resolve_llm_type(name: str, fallback: LLMType) -> LLMType:
    try:
        return LLMType[name]
    except KeyError:
        logger.warning("Unknown LLM type %r, using %s", name, fallback.name)
        return fallback


def get_causal_llm_type() -> LLMType:
    """LLM for CausalExplanationAgent (tool-heavy causal investigations)."""
    return _resolve_llm_type(CAUSAL_LLM_TYPE, LLMType.CLAUDE_SONNET_4_5)


def get_interpreter_llm_type() -> LLMType:
    """LLM for AnomalyInterpreterAgent (NPS tree interpretation)."""
    return _resolve_llm_type(INTERPRETER_LLM_TYPE, LLMType.CLAUDE_OPUS_4_6)


def get_summarizer_llm_type() -> LLMType:
    """LLM for AnomalySummaryAgent (executive / stratified summaries)."""
    return _resolve_llm_type(SUMMARIZER_LLM_TYPE, LLMType.CLAUDE_OPUS_4_6)


def get_default_llm_type() -> LLMType:
    """Backward-compatible default; same as interpreter (Opus) unless DEFAULT_LLM_TYPE is changed."""
    return _resolve_llm_type(DEFAULT_LLM_TYPE, LLMType.O4_MINI)


def get_agent_conversations_folder() -> str:
    """
    Get the agent conversations folder name with the LLM type prefix.
    
    Returns:
        Folder name in format: {DEFAULT_LLM_TYPE}_agent_conversations
        Example: O4_MINI_agent_conversations, CLAUDE_SONNET_4_agent_conversations
    """
    return f"{DEFAULT_LLM_TYPE}_agent_conversations"


class AgentName(Enum):
    """
    Enumerated type representing the various agent names.
    This could be used to categorize or distinguish between different functionalities
    or responsibilities of agents within the system.
    """
    CONVERSATIONAL = 'CONVERSATIONAL'
