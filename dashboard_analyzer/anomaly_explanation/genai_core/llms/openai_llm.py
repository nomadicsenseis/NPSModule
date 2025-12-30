from langchain_openai import AzureChatOpenAI, ChatOpenAI
from ..utils.enums import LLMType
from .llm import LLM
import os


class OpenAiLLM(LLM):
    """
    OpenAiLLM represents a Large Language Model hosted on OpenAI Platform or Azure OpenAI.
    """

    def __init__(self, llm_type: LLMType, api_key: str, api_base: str = None, api_version: str = None,
                 api_dep_gpt: str = None, project_id: str = None,
                 token_input_price: float = 3.75 / 1000000, token_output_price: float = 15 / 1000000,
                 temperature: float = 0.1, size: int = 4):
        """
        Initialize the OpenAiLLM instance.

        :param llm_type: Specifies the type of LLM.
        :param api_key: Configuration parameters for OpenAI API.
        :param api_base: Configuration parameters for OpenAI API (endpoint).
        :param api_version: Configuration parameters for OpenAI API (Azure only).
        :param api_dep_gpt: Deployment name (Azure only).
        :param project_id: Project ID for OpenAI Platform.
        :param token_input_price: Cost per input token (default is provided).
        :param token_output_price: Cost per output token (default is provided).
        :param temperature: Sampling temperature for model response.
        :param size: Size configuration for the model.
        """
        # Configure OpenAI API
        self.api_base = api_base
        self.api_key = api_key
        self.api_dep_gpt = api_dep_gpt
        self.project_id = project_id

        self.size = size
        self.api_version = api_version
        self.temperature = temperature

        super().__init__(llm_type, token_input_price, token_output_price)

    def create_llm(self):
        """
        Establish connection with OpenAI and instantiate the appropriate language model.

        :return: Configured LLM from OpenAI or Azure OpenAI.
        """
        # Check if we should use Azure or OpenAI Platform
        if self.api_dep_gpt:
            # Azure OpenAI path (existing behavior)
            return AzureChatOpenAI(
                azure_endpoint=self.api_base,
                openai_api_key=self.api_key,
                deployment_name=self.api_dep_gpt,
                openai_api_version=self.api_version,
                temperature=self.temperature,
                max_completion_tokens=16000
            )
        else:
            # OpenAI Platform path
            model_name = self._get_model_name()
            
            # Use OpenAI Platform credentials
            # Note: project_id can be passed via default_headers if not directly supported
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key,
                base_url=self.api_base,
                temperature=self.temperature,
                max_completion_tokens=16000,
                default_headers={"OpenAI-Project": self.project_id} if self.project_id else None
            )

    def _get_model_name(self) -> str:
        """Map LLMType to OpenAI Platform model names."""
        mapping = {
            LLMType.GPT3_5: "gpt-3.5-turbo",
            LLMType.GPT4: "gpt-4",
            LLMType.GPT4o: "gpt-4o",
            LLMType.GPT4o_MINI: "gpt-4o-mini",
            LLMType.O1_MINI: "o1-mini",
            LLMType.O3_MINI: "o3-mini",
            LLMType.O3: "o3",
            LLMType.O4_MINI: "o4-mini",
            LLMType.GPT_5_2: "gpt-5.2", # From user example
        }
        return mapping.get(self.llm_type, "gpt-4o") # Default to gpt-4o if unknown
