import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

# Load workspace .env if available
load_dotenv()

from .models import JudgmentMetadata

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a judicial data registry officer. Extract only explicit facts from the provided Court of Appeal judgment of Sri Lanka.

Follow these strict rules to ensure data integrity:
1. **Case Number**: Extract the unique court reference/case number (e.g. CA/720/2020, CA (PHC) APN 53/18).
2. **Judges**: Extract the list of Justices on the panel. Keep their titles uniform (e.g. "L.T.B. Dehideniya, J.").
3. **Date of Judgment**: Extract the delivery date. Format as YYYY-MM-DD if parseable (e.g. '2026-06-08'). If the date is missing or ambiguous, return it exactly as raw text from the document—do not invent it.
4. **Parties**: Structure the Appellants/Petitioners and Respondents. Strip away long residential addresses or corporate descriptions (e.g. "John Doe of No. 23, Galle Road..." becomes "John Doe").
5. **Legislation Cited**: Extract explicit statutes, acts, or sections mentioned (e.g. 'Section 68 of the Evidence Ordinance'). Do not assume or guess statutory links.
6. **Keywords**: Generate 5 to 10 highly relevant, standardized legal keywords derived from the text context (e.g. 'Writ of Certiorari', 'Prescriptive Title', 'Wrongful Dismissal') to support consistent search indexing.

**IMPORTANT**: If a field is not explicitly present, return an empty list or null value. Do not hallucinate or guess any metadata."""

def trim_judgment_text(text: str, max_chars: int = 25000) -> str:
    """Trim extremely long judgments to fit within LLM context windows.
    Keeps the head (contains case number, judges, date, parties) and 
    the tail (contains final order, citations, summary of law) while omitting the middle.
    """
    if len(text) <= max_chars:
        return text
    
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len
    
    trimmed = (
        text[:head_len]
        + "\n\n[... lengthy middle portion of judgment text omitted to optimize context ...]\n\n"
        + text[-tail_len:]
    )
    return trimmed

def extract_metadata_openai(text: str, model: str, api_key: str, base_url: Optional[str] = None) -> JudgmentMetadata:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    trimmed_text = trim_judgment_text(text)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"=== JUDGMENT TEXT ===\n{trimmed_text}"}
    ]
    
    # Use official OpenAI structured outputs helper if it is the official OpenAI API and model supports it
    is_official_openai = not base_url or "api.openai.com" in base_url
    
    if is_official_openai:
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=JudgmentMetadata,
                temperature=0.0
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            logger.warning(f"Native OpenAI .parse failed, falling back to JSON Mode: {e}")
            
    # Fallback/General OpenAI compatible JSON Mode (useful for Groq, Ollama, etc.)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    return JudgmentMetadata.model_validate_json(content)

def extract_metadata_anthropic(text: str, model: str, api_key: str) -> JudgmentMetadata:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    trimmed_text = trim_judgment_text(text)
    
    # Convert Pydantic model to Anthropic tools schema
    tools = [
        {
            "name": "extract_judgment_metadata",
            "description": "Extract structured metadata fields from the judicial judgment.",
            "input_schema": JudgmentMetadata.model_json_schema()
        }
    ]
    
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_judgment_metadata"},
        messages=[
            {"role": "user", "content": f"=== JUDGMENT TEXT ===\n{trimmed_text}"}
        ]
    )
    
    tool_use_block = [block for block in response.content if block.type == "tool_use"][0]
    return JudgmentMetadata.model_validate(tool_use_block.input)

import threading

_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_LLAMA_EXECUTION_LOCK = threading.Lock()

def extract_metadata_llamacpp(text: str, model_path: str) -> JudgmentMetadata:
    global _MODEL_CACHE
    
    # Check if we can reuse the prewarmed default singleton from src.analyze
    use_default_singleton = False
    llm = None
    
    try:
        from src.config import settings
        # Compare paths to verify if we are loading the default model
        default_path = os.path.abspath(settings.llamacpp_model_path)
        requested_path = os.path.abspath(model_path)
        if default_path == requested_path:
            use_default_singleton = True
    except Exception as e:
        logger.warning(f"Could not check default GGUF singleton: {e}")
        
    if use_default_singleton:
        logger.info("Reusing prewarmed default GGUF model singleton from src.analyze.")
        print("Reusing prewarmed default GGUF model singleton from src.analyze...", flush=True)
        try:
            from src.analyze import _get_llama
            llm = _get_llama()
        except Exception as e:
            logger.error(f"Failed to fetch prewarmed singleton model: {e}")
            use_default_singleton = False

    # Fallback if not using default singleton or failed to fetch
    if llm is None:
        with _MODEL_CACHE_LOCK:
            if model_path not in _MODEL_CACHE:
                from llama_cpp import Llama
                # Attempt to pull GPU layers from project config
                gpu_layers = 36
                try:
                    from src.config import settings
                    gpu_layers = settings.llamacpp_gpu_layers
                except Exception:
                    pass
                
                logger.info(f"Loading GGUF model from {model_path} (n_gpu_layers={gpu_layers})...")
                print(f"Loading GGUF model from {model_path} (n_gpu_layers={gpu_layers})...", flush=True)
                
                _MODEL_CACHE[model_path] = Llama(
                    model_path=model_path,
                    n_ctx=8192,
                    n_gpu_layers=gpu_layers,
                    verbose=False
                )
                logger.info("GGUF model loaded successfully.")
                print("GGUF model loaded successfully.", flush=True)
                
        llm = _MODEL_CACHE[model_path]
    
    trimmed_text = trim_judgment_text(text, max_chars=12000) # smaller context limit for local CPU/GPU
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"=== JUDGMENT TEXT ===\n{trimmed_text}"}
    ]
    
    # 2. Execute inference (using either the main app lock or custom execution lock)
    if use_default_singleton:
        from src.analyze import _llama_guard
        with _llama_guard():
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=1500,
                temperature=0.0,
                response_format={
                    "type": "json_object",
                    "schema": JudgmentMetadata.model_json_schema()
                }
            )
    else:
        with _LLAMA_EXECUTION_LOCK:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=1500,
                temperature=0.0,
                response_format={
                    "type": "json_object",
                    "schema": JudgmentMetadata.model_json_schema()
                }
            )
    
    content = response["choices"][0]["message"]["content"]
    return JudgmentMetadata.model_validate_json(content)

def run_extraction(text: str, provider: str, model_or_path: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> JudgmentMetadata:
    """Entry point dispatching extraction to the designated provider."""
    provider = provider.lower()
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key and not base_url:
            raise ValueError("OPENAI_API_KEY must be provided or configured in .env")
        return extract_metadata_openai(text, model_or_path, key, base_url)
        
    elif provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or configured in .env")
        return extract_metadata_anthropic(text, model_or_path, key)
        
    elif provider == "llamacpp":
        model_path = model_or_path or os.getenv("LLAMACPP_MODEL_PATH")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Llama GGUF model path not found: {model_path}")
        return extract_metadata_llamacpp(text, model_path)
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")
