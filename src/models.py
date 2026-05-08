import os
import time
import json
import datetime
from google import genai
from dotenv import load_dotenv

from google.genai import types
from google.genai.types import HttpOptions
from openai import AzureOpenAI, OpenAI

load_dotenv(override=True)
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PORT = os.environ.get("VLLM_PORT", "")

if AZURE_OPENAI_KEY != "":
    azure_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-10-21",
    )

if OPENAI_API_KEY != "":
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

if GENAI_API_KEY != "":
    USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
    if USE_VERTEX:
        gen_client = genai.Client(vertexai=True, api_key=GENAI_API_KEY, http_options=HttpOptions(api_version="v1"))
    else:
        gen_client = genai.Client(api_key=GENAI_API_KEY)

time_gap = {"gpt-4": 3, "gemini-2.5-flash": 13, "gemini-2.5-flash-lite": 13, "gemini-flash-lite-latest": 4}


def get_answer(response):
    if hasattr(response, "choices"):
        answer = response.choices[0].message.content
    elif hasattr(response, "text"):
        answer = response.text.strip()
    else:
        raise NotImplementedError(f"Fail to extract answer: {answer}")
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].replace("<think>", "").replace("\n", "").strip()
    return answer


def get_token_log(response):
    token_usage = {}
    if hasattr(response, "usage"):
        token_usage["prompt_tokens"] = response.usage.prompt_tokens
        token_usage["completion_tokens"] = response.usage.completion_tokens
        token_usage["total_tokens"] = response.usage.total_tokens
        if hasattr(response.usage, "completion_tokens_details"):  # for gpt-5 series
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                token_usage["extra_info"] = {"reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens}
    elif hasattr(response, "usage_metadata"):
        token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
        token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count
        token_usage["total_tokens"] = response.usage_metadata.total_token_count
    else:
        raise NotImplementedError(f"Fail to extract usage data: {response}")
    return token_usage
    

def gpt_azure_response(message: list, model="gpt-4o", temperature=0, seed=42, **kwargs):
    time.sleep(time_gap.get(model, 3))
    try:
        return azure_client.chat.completions.create(model=model, messages=message, temperature=temperature, seed=seed, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg:
            if isinstance(message, list) and len(message) > 2:
                message = [message[0]] + message[2:]
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return gpt_azure_response(model=model, messages=message, temperature=temperature, seed=seed, **kwargs)


def openai_response(message: list, model="gpt-4o-mini", temperature=0, seed=42, _retries_left=3, **kwargs):
    """Direct (non-Azure) OpenAI Chat Completions. Bounded retries to avoid infinite loops on auth/quota errors."""
    time.sleep(time_gap.get(model, 1))
    # gpt-5 family does not accept custom temperature/seed (per OpenAI 2025+ behavior)
    safe_kwargs = dict(kwargs)
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        # newer reasoning-style models reject these
        params = {"model": model, "messages": message}
    else:
        params = {"model": model, "messages": message, "temperature": temperature, "seed": seed}
    try:
        return openai_client.chat.completions.create(**params, **safe_kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg or "maximum" in error_msg:
            if isinstance(message, list) and len(message) > 2:
                message = [message[0]] + message[2:]
        if _retries_left <= 0:
            raise
        print(e)
        time.sleep(time_gap.get(model, 1) * 2)
        return openai_response(message, model=model, temperature=temperature, seed=seed,
                               _retries_left=_retries_left - 1, **kwargs)


def gemini_response(message: list, model="gemini-2.0-flash", temperature=0, seed=42, _retries_left=3, **kwargs):
    time.sleep(time_gap.get(model, 3))
    system_prompt = message[0]["content"] if message[0]["role"] == "system" else None
    if system_prompt:
        contents = message[1:]
    else:
        contents = message

    try:
        # Gemini accepts 'user' and 'model' roles only — map 'assistant' -> 'model'.
        def _map_role(r):
            return "model" if r == "assistant" else r
        contents = [{"role": _map_role(item["role"]), "parts": [{"text": item["content"]}]} for item in contents]
    except:
        raise NotImplementedError

    try:
        if model == "gemini-2.5-flash":
            return gen_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    seed=seed,
                    thinking_config=types.ThinkingConfig(thinking_budget=kwargs.get("thinking_budget", 0))
                ),
            )
        
        elif model.startswith("gemini-3"):
            return gen_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    seed=seed,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    labels={
                        "team": "ai612",
                        "environment": f"patientsim-{model}",
                    },
                ),
            )
        
        else:
            return gen_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    seed=seed,
                ),
            )

    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg or 'maximum context length' in error_msg:
            if isinstance(message, list) and len(message) > 2:
                message = [message[0]] + message[2:]
        print(e)
        if _retries_left <= 0:
            raise
        # rate-limit / quota errors: back off longer
        sleep_s = 30 if "429" in error_msg or "quota" in error_msg or "rate" in error_msg else time_gap.get(model, 3) * 2
        time.sleep(sleep_s)
        return gemini_response(message, model, temperature, seed, _retries_left=_retries_left - 1, **kwargs)


def vllm_model_setup(model):
    if model == "vllm-llama3-70b-instruct":
        model = "meta-llama/Llama-3-70B-Instruct"
    elif model == "vllm-llama3-8b-instruct":
        model = "meta-llama/Llama-3-8B-Instruct"
    elif model == "vllm-llama3.1-8b-instruct":
        model = "meta-llama/Llama-3.1-8B-Instruct"
    elif model == "vllm-llama3.1-70b-instruct":
        model = "meta-llama/Llama-3.1-70B-Instruct"
    elif model == "vllm-llama3.3-70b-instruct":
        model = "meta-llama/Llama-3.3-70B-Instruct"
    elif model == "vllm-qwen2.5-72b-instruct":
        model = "Qwen/Qwen2.5-72B-Instruct"
    elif model == "vllm-qwen2.5-7b-instruct":
        model = "Qwen/Qwen2.5-7B-Instruct"
    elif model == "vllm-deepseek-llama-70b":
        model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    else:
        raise ValueError(f"Invalid model: {model}")
    return model


def vllm_response(message: list, model=None, temperature=0, seed=42, **kwargs):
    VLLM_API_KEY = "EMPTY"
    VLLM_API_BASE = f"http://localhost:{PORT}/v1"
    vllm_client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)

    assert model in [
        "meta-llama/Llama-3-70B-Instruct",
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ]
    time.sleep(time_gap.get(model, 3))

    try:
        return vllm_client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            seed=seed,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg or 'maximum context length' in error_msg:
            if isinstance(message, list) and len(message) > 2:
                message = [message[0]] + message[2:]
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return vllm_response(message, model, temperature, seed)


def get_response_method(model):
    response_methods = {
        "gpt_azure": gpt_azure_response,
        "openai": openai_response,
        "vllm": vllm_response,
        "genai": gemini_response,
    }
    return response_methods.get(model.split("-")[0] if "-" in model else model, lambda _: NotImplementedError())