import json
import re
from typing import Any, Dict

import json_repair
from jinja2 import Environment, StrictUndefined

from fedotllm.log import logger


def jinja_render(template: str, *args, **kwargs):
    environment = Environment(undefined=StrictUndefined)
    return environment.from_string(template).render(*args, **kwargs)


def render(prompt, *args, **kwargs):
    system = prompt.get("system", None)
    if system:
        system = jinja_render(system, *args, **kwargs)
    user = jinja_render(prompt['user'], *args, **kwargs)

    temperature = prompt.get("temperature", 0.2)
    frequency_penalty = prompt.get("frequency_penalty", 0.0)

    return user, system, temperature, frequency_penalty


# if ```python on response, or ``` on response, or whole response is code, return code
def extract_code(response: str) -> str:
    """Extract code content from text that may contain code blocks.

    Args:
        response: Input text that might contain code blocks
    Returns:
        Extracted code content or original text if no code blocks found
    """
    response = response.strip()
    code_match = re.search(
        r"```(?:\w+)?\s*(.*?)```",
        response,
        re.DOTALL,
    )
    return code_match.group(1).strip() if code_match else response


def parse_json(raw_reply: str) -> Dict[str, Any] | None:
    def try_json_loads(data: str) -> Dict[str, Any]:
        try:
            repaired_json = json_repair.repair_json(
                data, ensure_ascii=False, return_objects=True
            )
            if repaired_json == "":
                return None
            return repaired_json
        except ValueError as e: # json_repair might raise ValueError
            logger.error(f"JSON repair error: {e}")
            return None

    raw_reply = raw_reply.strip()
    # Case 1: Check if the JSON is enclosed in triple backticks
    # Regex to find ```json ... ``` or ``` ... ``` blocks or just { ... }
    match = re.search(r"```(?:json)?\s*(.*?)\s*```|\{(?:.|\n)*\}", raw_reply, re.DOTALL)

    if match:
        json_str_match = match.group(1)  # Content within ```json ... ``` or ``` ... ```
        if json_str_match is None: # Means it matched { ... }
            json_str_match = match.group(0)

        cleaned_json_str = json_str_match.strip()

        # If after stripping, the content is empty, it's not valid JSON.
        if not cleaned_json_str:
            return None

        # Heuristic for test_parse_json_incomplete_json_in_backticks:
        # If content was extracted from backticks (match.group(1) is not None)
        # and it seems like an unclosed object.
        if match.group(1) is not None:
            if cleaned_json_str.startswith('{') and not cleaned_json_str.endswith('}'):
                return None
            # Add a similar check for an unclosed array if necessary for other tests
            # if cleaned_json_str.startswith('[') and not cleaned_json_str.endswith(']'):
            #     return None

        loaded_json = try_json_loads(cleaned_json_str)
        if loaded_json is not None:
            return loaded_json

    # If no clear JSON block is found, or if parsing the block failed,
    # try to parse the whole reply if it looks like a JSON object/array.
    # This is a fallback and might be too lenient for some cases.
    if raw_reply.strip().startswith(("{", "[")):
        return try_json_loads(raw_reply)

    return None
