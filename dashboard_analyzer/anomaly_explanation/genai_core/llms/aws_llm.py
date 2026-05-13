import json
import os

from langchain_aws import ChatBedrock, ChatBedrockConverse
from botocore.config import Config
import boto3

from ..utils.enums import LLMType
from .llm import LLM

# Make BEDROCK_MODELS optional to avoid errors when not using AWS
try:
    BEDROCK_MODELS = json.loads(os.environ.get("BEDROCK_MODELS", "{}"))
except (json.JSONDecodeError, KeyError):
    BEDROCK_MODELS = {}

# Model ARNs by environment (local / sbx / prod)
# - local: developer machine (uses sbx account credentials)
# - sbx:   AWS Sandbox account (856897973040)
# - prod:  AWS Production account (320714865578)
# Structure: { LLMType.value: { "local": "arn...", "sbx": "arn...", "prod": "arn..." } }
MODEL_ARNS_BY_ENV = {
    LLMType.CLAUDE_HAIKU_4_5.value: {
        "local": "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/78486g7eitdv",
        "sbx":   "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/deh8x2wc9ohx",
        "prod":  "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/deh8x2wc9ohx",  # TODO: replace with prod account ARN
    },
    LLMType.CLAUDE_SONNET_4_5.value: {
        "local": "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/j9l4fod1sker",
        "sbx":   "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/4o3iago8pudu",
        "prod":  "arn:aws:bedrock:eu-west-1:320714865578:application-inference-profile/1asrudnp23md",
    },
    LLMType.GPT_OSS_120B.value: {
        "local": "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/01q61xjcup73",
        "sbx":   "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/1a1vdoh2kwzv",
        "prod":  "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/1a1vdoh2kwzv",  # TODO: replace with prod account ARN
    },
    LLMType.CLAUDE_OPUS_4_6.value: {
        "local": "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/dfsaqs9f3a5v",
        "sbx":   "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/dfsaqs9f3a5v",
        "prod":  "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/dfsaqs9f3a5v",  # TODO: replace with prod account ARN
    },
}


class AWSLLM(LLM):
    """
    AWSLLM class that uses LangChain's ChatBedrock or ChatBedrockConverse integration
    """

    def __init__(self, llm_type: LLMType, region_name, aws_access_key_id=None, aws_secret_access_key=None, 
                 aws_session_token=None, profile_name=None, environment: str = None,
                 token_input_price: float = 11.02 / 1000000, token_output_price: float = 32.68 / 1000000):
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.profile_name = profile_name
        self.model_id = None
        # "local" = developer machine; anything else = cloud (Airflow).
        # Inference profile account (sbx vs prod) is resolved separately via ENV env var.
        self.environment = environment or os.getenv("ENVIRONMENT", "prod").lower()

        super().__init__(llm_type, token_input_price, token_output_price)

    def create_llm(self):
        """Create the LangChain ChatBedrock or ChatBedrockConverse client for the specified model"""
        self._set_model_id()
        
        # Configure credentials for boto3 session (if needed)
        credentials = {}
        if self.aws_access_key_id and self.aws_secret_access_key:
            credentials["aws_access_key_id"] = self.aws_access_key_id
            credentials["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            credentials["aws_session_token"] = self.aws_session_token
        if self.profile_name:
            credentials["profile_name"] = self.profile_name
        
        # Determine provider based on model ID
        provider = self._get_provider()
        
        # Build a boto3 client with extended timeouts and retries to handle long generations
        # Increased to 1 hour (3600s) to handle large JSON outputs like Adaptive Cards
        read_timeout = int(os.getenv("BEDROCK_READ_TIMEOUT", "3600"))
        connect_timeout = int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "60"))
        max_attempts = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
        cfg = Config(read_timeout=read_timeout, connect_timeout=connect_timeout, retries={"max_attempts": max_attempts, "mode": "standard"})

        session = boto3.Session(
            aws_access_key_id=credentials.get("aws_access_key_id"),
            aws_secret_access_key=credentials.get("aws_secret_access_key"),
            aws_session_token=credentials.get("aws_session_token"),
            profile_name=credentials.get("profile_name"),
            region_name=self.region_name,
        )
        bedrock_client = session.client("bedrock-runtime", config=cfg)

        # Create ChatBedrock client with custom boto3 client and tunable generation params
        # Increased default to 64000 to avoid truncation of large JSON outputs (Adaptive Cards)
        max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "64000"))
        
        # Adjust max_tokens for models with lower limits
        # Nova Pro limit is 5k-10k depending on region/version (error said 10000)
        if self.llm_type in [LLMType.AMAZON_NOVA_PRO, LLMType.AMAZON_NOVA_2_LITE]:
            max_tokens = min(max_tokens, 5000)  # Safe limit
        # Claude Sonnet 4.5 supports up to 64k output tokens
        elif self.llm_type == LLMType.CLAUDE_SONNET_4_5:
            max_tokens = min(max_tokens, 64000)
            
        temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.7"))
        
        # Use ChatBedrockConverse for all chat models (it's the new standard and supports all modern models)
        # Note: Embedding models like Titan Embed are not supported here, they should use BedrockEmbeddings
        if self.llm_type == LLMType.AMAZON_TITAN_EMBED_TEXT_V2:
             raise ValueError("AWSLLM is for Chat models only. Use BedrockEmbeddings for Titan Embeddings.")

        return ChatBedrockConverse(
            model_id=self.model_id,
            client=bedrock_client,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def _get_model_arn_by_env(self, llm_type_value: str) -> str:
        """Get the model ARN based on the environment (local / sbx / prod).

        - local:  developer machine → uses 'local' ARNs (sbx account)
        - cloud:  reads ENV env var set by Airflow ('sbx' or 'prod') to pick
                  the right AWS account inference profile.
        """
        env_arns = MODEL_ARNS_BY_ENV.get(llm_type_value, {})
        if self.environment == "local":
            env_key = "local"
        else:
            # In cloud (Airflow), ENV=sbx or ENV=prod distinguishes accounts
            env_key = os.getenv("ENV", "sbx").lower()
            if env_key not in ("sbx", "prod"):
                env_key = "sbx"
        return env_arns.get(env_key, env_arns.get("sbx", env_arns.get("local", "")))

    def _get_provider(self):
        """Get the provider name based on the model type"""
        if self.llm_type.value in [
            LLMType.CLAUDE_3_HAIKU.value, 
            LLMType.CLAUDE_3_5_HAIKU.value,
            LLMType.CLAUDE_3_OPUS.value,
            LLMType.CLAUDE_3_5_SONNET.value,
            LLMType.CLAUDE_3_5_SONNET_V2.value,
            LLMType.CLAUDE_SONNET_4.value,
            LLMType.CLAUDE_3_7_SONNET.value,
            LLMType.CLAUDE_OPUS_4_5.value,
            LLMType.CLAUDE_HAIKU_4_5.value,
            LLMType.CLAUDE_SONNET_4_5.value,
            LLMType.CLAUDE_OPUS_4_6.value
        ]:
            return "anthropic"
        elif self.llm_type.value in [
            LLMType.LLAMA3_70.value,
            LLMType.LLAMA3_1_70.value,
            LLMType.LLAMA3_1_405.value,
            LLMType.GPT_OSS_120B.value  # Assuming standard/meta-like interface for OSS model
        ]:
            return "meta"
        elif self.llm_type.value in [
            LLMType.AMAZON_NOVA_2_LITE.value,
            LLMType.AMAZON_NOVA_PRO.value,
            LLMType.AMAZON_TITAN_EMBED_TEXT_V2.value
        ]:
            return "amazon"
        else:
            raise ValueError(f"Unknown provider for model: {self.llm_type}")
    
    def _set_model_id(self):
        """Set the appropriate model ID based on the LLM type"""
        if self.llm_type.value == LLMType.CLAUDE_3_HAIKU.value:
            self.model_id = BEDROCK_MODELS.get("anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-haiku-20240307-v1:0")
        elif self.llm_type.value == LLMType.CLAUDE_3_5_HAIKU.value:
            self.model_id = 'anthropic.claude-3-5-haiku-20241022-v1:0'
        elif self.llm_type.value == LLMType.CLAUDE_3_OPUS.value:
            self.model_id = 'anthropic.claude-3-opus-20240229-v1:0'
        elif self.llm_type.value == LLMType.LLAMA3_70.value:
            self.model_id = "meta.llama3-70b-instruct-v1:0"
        elif self.llm_type.value == LLMType.LLAMA3_1_70.value:
            self.model_id = "meta.llama3-1-70b-instruct-v1:0"
        elif self.llm_type.value == LLMType.LLAMA3_1_405.value:
            self.model_id = "meta.llama3-1-405b-instruct-v1:0"
        elif self.llm_type.value == LLMType.CLAUDE_3_5_SONNET.value:
            self.model_id = BEDROCK_MODELS.get("anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        elif self.llm_type.value == LLMType.CLAUDE_3_5_SONNET_V2.value:
            self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        elif self.llm_type.value == LLMType.CLAUDE_SONNET_4.value:
            self.model_id = "arn:aws:bedrock:us-east-1:737192913161:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
        elif self.llm_type.value == LLMType.CLAUDE_OPUS_4_5.value:
            self.model_id = "arn:aws:bedrock:eu-west-1:856897973040:inference-profile/eu.anthropic.claude-opus-4-5-20251101-v1:0"
        elif self.llm_type.value == LLMType.CLAUDE_3_7_SONNET.value:
            # Use MODEL_ARN from environment for inference profile
            model_arn = os.getenv('MODEL_ARN')
            if model_arn:
                self.model_id = model_arn
            else:
                self.model_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"
        
        # New models added
        elif self.llm_type.value == LLMType.AMAZON_NOVA_2_LITE.value:
            self.model_id = "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/0g94pa86q97o"
        elif self.llm_type.value == LLMType.AMAZON_NOVA_PRO.value:
            self.model_id = "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/uh15i5cakpji"
        elif self.llm_type.value == LLMType.AMAZON_TITAN_EMBED_TEXT_V2.value:
            self.model_id = "arn:aws:bedrock:eu-west-1:856897973040:application-inference-profile/hprm0w08u8k8"
        elif self.llm_type.value == LLMType.CLAUDE_HAIKU_4_5.value:
            self.model_id = self._get_model_arn_by_env(LLMType.CLAUDE_HAIKU_4_5.value)
        elif self.llm_type.value == LLMType.CLAUDE_SONNET_4_5.value:
            self.model_id = self._get_model_arn_by_env(LLMType.CLAUDE_SONNET_4_5.value)
        elif self.llm_type.value == LLMType.GPT_OSS_120B.value:
            self.model_id = self._get_model_arn_by_env(LLMType.GPT_OSS_120B.value)
        elif self.llm_type.value == LLMType.CLAUDE_OPUS_4_6.value:
            self.model_id = self._get_model_arn_by_env(LLMType.CLAUDE_OPUS_4_6.value)
            
        else:
            raise ValueError(f"Invalid model: {self.llm_type}")