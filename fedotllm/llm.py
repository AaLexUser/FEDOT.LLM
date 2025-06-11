import os
from typing import Optional, TypeVar, Type

import tiktoken
from openai import OpenAI
import litellm
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from fedotllm import prompts
from fedotllm.settings.config_loader import get_settings
from fedotllm.agents.utils import parse_json


T = TypeVar("T", bound=BaseModel)

litellm._logging._disable_debugging()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class AIInference:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.get("config.llm_api_base") # Corrected key
        self.model = model or settings.get("config.llm_model")       # Corrected key
        self.api_key = api_key or os.getenv("FEDOTLLM_LLM_API_KEY")

        if not self.api_key:
            raise Exception(
                "LLM API key is not set. Provide it via argument or FEDOTLLM_LLM_API_KEY env variable."
            )

        self.completion_params = { # Base parameters, can be overridden by query_kwargs
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "extra_headers": {"X-Title": "FEDOT.LLM"}
        }

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def create(self, message: str, response_model: Type[T]) -> T: # Renamed messages to message for clarity
        system_prompt = prompts.utils.structured_response(response_model)
        # Pass system_prompt to query method
        response = self.query(user_message=message, system_message=system_prompt)
        json_obj = parse_json(response)
        return response_model.model_validate(json_obj)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def query(self, user_message: str, system_message: Optional[str] = None, **query_kwargs) -> str:
        formatted_messages = []
        if system_message:
            formatted_messages.append({"role": "system", "content": system_message})
        formatted_messages.append({"role": "user", "content": user_message})

        # Merge base completion_params with specific query_kwargs
        # query_kwargs can override temperature, frequency_penalty, etc.
        final_params = {**self.completion_params, **query_kwargs}

        response = litellm.completion(
            messages=formatted_messages,
            **final_params,
        )
        return response.choices[0].message.content


class OpenaiEmbeddings:
    MAX_INPUT = 8191 # Default, can be overridden by settings

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        self.base_url = base_url or settings.get("config.embeddings_api_base") # Corrected key
        self.model = model or settings.get("config.embeddings_model")       # Corrected key
        self.api_key = api_key or os.getenv("FEDOTLLM_EMBEDDINGS_API_KEY")
        self.MAX_INPUT = settings.get("config.max_input_tokens", OpenaiEmbeddings.MAX_INPUT)


        if not self.api_key:
            raise Exception(
                "Embeddings API key is not set. Provide it via argument or FEDOTLLM_EMBEDDINGS_API_KEY env variable."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) # Use self.api_key and self.base_url

    def encode(self, input_text: str): # Renamed input to input_text
        num_tokens = num_tokens_from_string(input_text, self.model if "ada" in self.model else "cl100k_base") # Use model for token count if applicable
        if num_tokens > self.MAX_INPUT:
            raise Exception(f"Input exceeds the limit of {self.MAX_INPUT} tokens for model {self.model}. Given: {num_tokens}")

        try:
            response = self.client.embeddings.create(
                model=self.model, input=[input_text], encoding_format="float" # API expects a list of inputs
            )
            return response.data # Return the list of embedding objects
        except Exception as e:
            # Log the error for more details
            # logger.error(f"Embeddings generation failed for model {self.model}: {e}")
            raise Exception(f"Embeddings generation failed! Original error: {e}")


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

if __name__ == "__main__":
    inference = AIInference(model="deepseek/DeepSeek-V3-0324")
    print(inference.query("Say hello world!"))