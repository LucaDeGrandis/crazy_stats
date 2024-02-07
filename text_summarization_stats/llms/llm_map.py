from .openai_chat import LlmChatOpenAI
from .openai import LlmOpenAI


MODELS = {
    'gpt-3.5-turbo': LlmChatOpenAI,
    'gpt-3.5-turbo-1106': LlmChatOpenAI,
    'gpt-3.5-turbo-instruct': LlmOpenAI,
}
