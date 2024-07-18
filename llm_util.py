import os
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from requests.exceptions import RequestException

_MAX_RETRIES = 6


def run_model_multicall(model, tokenizer, generation_config, prompts):
    """Run all prompts on local model
    """
    
    responses = {}
    for task in prompts:
        messages = [
            {"role": "system", "content": prompts[task]["system"]},
            {"role": "context", "content": prompts[task]["context"]},
            {"role": "user", "content": prompts[task]["task"]},
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            input_ids,
            **generation_config
        )
        response = outputs[0][input_ids.shape[-1]:]
        responses[task] = tokenizer.decode(response, skip_special_tokens=True)

    return responses

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(_MAX_RETRIES),
    retry=(retry_if_exception_type(RequestException) | retry_if_exception_type(RuntimeError)),
    reraise=True
)
def run_web_model_multicall(model, prompts):
    """Run all prompts on web model
    """

    responses = {}
    for task in prompts:

        model.set_sys_prompt(prompts[task]["system"])
        model.add_context(prompts[task]["context"])
        
        response = model(prompts[task]["task"], as_json=True)
        responses[task] = response

    return responses

def process_model_responses(responses):
    responses["categorical_columns"] = responses["categorical_columns"].split("\n")
    responses["task_type"] = responses["task_type"].lower()
    return responses

def save_model_responses(responses, path):
    with open(os.sep.join([path, 'model_responses.json']), 'w') as json_file:
        json.dump(responses, json_file)