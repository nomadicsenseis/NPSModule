from enum import Enum

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
    GPT3_5 = 'GPT3_5'
    GPT4 = 'GPT4'
    GPT4o = 'GPT4o'
    GPT4o_MINI = 'GPT4o_MINI'
    O4_MINI = 'O4_MINI'
    LLAMA3_70 = 'LLAMA3_70'
    LLAMA3_1_70 = 'LLAMA3_1_70'
    LLAMA3_1_405 = 'LLAMA3_1_405'
    CLAUDE_3_7_SONNET = 'CLAUDE_3_7_SONNET'

class AgentName(Enum):
    """
    Enumerated type representing the various agent names.
    This could be used to categorize or distinguish between different functionalities
    or responsibilities of agents within the system.
    """
    CONVERSATIONAL = 'CONVERSATIONAL'
