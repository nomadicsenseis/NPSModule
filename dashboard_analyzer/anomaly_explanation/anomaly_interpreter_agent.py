"""
Anomaly Interpreter Agent

A specialized AI agent for interpreting NPS anomaly trees and generating conclusions
about operational performance and customer perception correlations.
"""

import asyncio
import os
import yaml
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import GenAI Core components
from .genai_core.agents.agent import Agent
from .genai_core.llms.openai_llm import OpenAiLLM
from .genai_core.llms.aws_llm import AWSLLM
from .genai_core.utils.enums import LLMType, MessageType
from .genai_core.message_history import MessageHistory


class AnomalyInterpreterAgent:
    """
    Specialized agent for interpreting anomaly trees and generating actionable conclusions.
    
    Features:
    - Multi-LLM support (OpenAI, AWS Bedrock)
    - External prompt configuration via YAML
    - Professional anomaly analysis with operational insights
    """
    
    def __init__(
        self,
        llm_type: LLMType = LLMType.GPT4o_MINI,
        config_path: str = "config/prompts/anomaly_interpreter.yaml",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Anomaly Interpreter Agent.
        
        Args:
            llm_type: Type of LLM to use (supports OpenAI and AWS Bedrock models)
            config_path: Path to YAML configuration file with prompts
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        self.llm_type = llm_type
        
        # Load environment variables from .devcontainer/.env
        dotenv_path = Path(__file__).parent.parent.parent / '.devcontainer' / '.env'
        load_dotenv(dotenv_path)
        
        # Load prompt configuration
        self.config = self._load_prompt_config(config_path)
        
        # Initialize LLM based on type
        self.llm = self._create_llm(llm_type)
        
        # Initialize base agent
        self.agent = Agent(llm=self.llm, logger=self.logger)
        
        self.logger.info(f"AnomalyInterpreterAgent initialized with {llm_type.value}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger for the agent."""
        logger = logging.getLogger("anomaly_interpreter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_prompt_config(self, config_path: str) -> Dict[str, Any]:
        """Load prompt configuration from YAML file."""
        try:
            # Get the directory of this file to resolve relative paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, config_path)
            
            with open(full_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            self.logger.debug(f"Loaded prompt configuration from {full_path}")
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _create_llm(self, llm_type: LLMType):
        """Factory method to create appropriate LLM instance based on type."""
        
        # OpenAI/Azure OpenAI models
        if llm_type in [LLMType.GPT3_5, LLMType.GPT4, LLMType.GPT4o, LLMType.GPT4o_MINI, LLMType.O4_MINI]:
            return self._create_openai_llm(llm_type)
        
        # AWS Bedrock models
        elif llm_type in [
            LLMType.CLAUDE_3_HAIKU, LLMType.CLAUDE_3_5_HAIKU, LLMType.CLAUDE_3_OPUS,
            LLMType.CLAUDE_3_5_SONNET, LLMType.CLAUDE_3_5_SONNET_V2, LLMType.CLAUDE_3_7_SONNET,
            LLMType.CLAUDE_SONNET_4, LLMType.LLAMA3_70, LLMType.LLAMA3_1_70, LLMType.LLAMA3_1_405
        ]:
            return self._create_aws_llm(llm_type)
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def _create_openai_llm(self, llm_type: LLMType) -> OpenAiLLM:
        """Create OpenAI/Azure OpenAI LLM instance."""
        # Get credentials from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, api_base, deployment_name]):
            raise ValueError("Missing required OpenAI/Azure OpenAI environment variables")
        
        return OpenAiLLM(
            llm_type=llm_type,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            api_dep_gpt=deployment_name,
            temperature=0.1  # Low temperature for consistent analysis
        )
    
    def _create_aws_llm(self, llm_type: LLMType) -> AWSLLM:
        """Create AWS Bedrock LLM instance."""
        region_name = os.getenv("AWS_REGION", "us-east-1")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        profile_name = os.getenv("AWS_PROFILE")
        
        return AWSLLM(
            llm_type=llm_type,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            profile_name=profile_name
        )
    
    async def interpret_anomaly_tree(self, tree_data: str, date: str = None) -> str:
        """
        Interpret an anomaly tree and generate actionable conclusions.
        
        Args:
            tree_data: The anomaly tree string with operational explanations
            date: Optional date for context (e.g., "2025-05-24")
            
        Returns:
            Professional interpretation and conclusions about the anomalies
        """
        start_time = datetime.now()
        
        try:
            # Create message history
            message_history = MessageHistory(logger=self.logger)
            
            # Add system prompt
            system_prompt = self.config['system_prompt']
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            
            # Format user input using template
            user_input = self.config['input_template'].format(anomaly_tree=tree_data)
            if date:
                user_input += f"\n\nFecha de análisis: {date}"
            
            message_history.create_and_add_message(
                content=user_input,
                message_type=MessageType.USER
            )
            
            # Get interpretation from LLM
            self.logger.debug("Sending anomaly tree to LLM for interpretation")
            
            response, structured_response, tool_calls = await self.agent.invoke(
                messages=message_history.get_messages()
            )
            
            interpretation = response.content.strip()
            
            # Log performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Interpretation completed in {execution_time:.2f}s "
                f"(Input: {self.agent.input_tokens}, Output: {self.agent.output_tokens} tokens)"
            )
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Error during anomaly interpretation: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "num_calls": self.agent.num_calls,
            "total_time": self.agent.total_time,
            "avg_time": self.agent.avg_time,
            "last_execution_time": self.agent.last_execution_time,
            "input_tokens": self.agent.input_tokens,
            "output_tokens": self.agent.output_tokens,
            "money_spent": self.agent.money_spent,
            "llm_type": self.llm_type.value
        }


# Convenience function for quick usage
async def interpret_anomaly_tree(
    tree_data: str, 
    date: str = None,
    llm_type: LLMType = LLMType.GPT4o_MINI
) -> str:
    """
    Quick function to interpret an anomaly tree without creating an agent instance.
    
    Args:
        tree_data: The anomaly tree string with operational explanations
        date: Optional date for context
        llm_type: LLM type to use for interpretation
        
    Returns:
        Professional interpretation and conclusions
    """
    agent = AnomalyInterpreterAgent(llm_type=llm_type)
    return await agent.interpret_anomaly_tree(tree_data, date)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example anomaly tree data
        example_tree = """
        **Anomaly Tree with Operational Explanations: 2025-05-24**  
        Global [N]  
          All children normal (LH, SH)

          LH [N]  
            Positive nodes (Business, Premium) diluted by normal nodes (Economy)

            Economy [N]

            Business [+]  
              • OTP stable at 91.8% (Δ+0.1 pts vs 7-day avg) – no significant impact  
              • Load Factor stable at 94.68% (Δ+0.8 pts vs 7-day avg) – no significant impact

            Premium [+]  
              • OTP stable at 91.8% (Δ+0.1 pts vs 7-day avg) – no significant impact  
              • Load Factor increased by 3.85 pts (93.24% vs 89.39%) – contradicts NPS anomaly
        """
        
        # Create agent and interpret
        agent = AnomalyInterpreterAgent(llm_type=LLMType.GPT4o_MINI)
        interpretation = await agent.interpret_anomaly_tree(example_tree, "2025-05-24")
        
        print("🔍 ANOMALY INTERPRETATION:")
        print("=" * 50)
        print(interpretation)
        print("=" * 50)
        
        # Show performance metrics
        metrics = agent.get_performance_metrics()
        print(f"\n📊 Performance: {metrics['execution_time']:.2f}s, "
              f"Tokens: {metrics['input_tokens']}+{metrics['output_tokens']}, "
              f"Cost: ${metrics['money_spent']:.4f}")
    
    asyncio.run(main()) 