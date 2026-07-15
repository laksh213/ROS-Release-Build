"""Phase 5 (Smart version) — Grounded analysis & structured breakdown.

This smart version implements a decomposed, multi-step extraction and analysis pipeline
with Chain-of-Thought reasoning.
"""

from __future__ import annotations

from typing import Callable
import argparse
import contextlib
import json
import re
import sys
import threading
import time

from .config import REPO_ROOT, settings
from .ingest import case_no_from_filename
from .parsing import extract_pages
from .schema import CaseAnalysis
from .retrieve import retrieve

PROMPT_PATH = REPO_ROOT / "prompts" / "system_prompt_smart.md"

_LLAMA = None
_LLAMA_LOCK = threading.RLock()
_LLAMA_LOCK_TIMEOUT = 300  # seconds


@contextlib.contextmanager
def _llama_guard():
    if not _LLAMA_LOCK.acquire(timeout=_LLAMA_LOCK_TIMEOUT):
        raise RuntimeError("Local model is busy (lock timeout). Please retry in a moment.")
    try:
        yield
    finally:
        _LLAMA_LOCK.release()


def _get_llama():
    """Lazy-load a local GGUF via llama-cpp-python (cached for the process)."""
    global _LLAMA
    if _LLAMA is None:
        with _llama_guard():
            if _LLAMA is None:
                from llama_cpp import Llama

                kv_type = settings.llamacpp_kv_type.lower()
                type_k, type_v = None, None
                if kv_type == "q8_0":
                    type_k, type_v = 8, 8  # GGML_TYPE_Q8_0
                elif kv_type == "q4_0":
                    type_k, type_v = 2, 2  # GGML_TYPE_Q4_0
                elif kv_type == "f16":
                    type_k, type_v = 1, 1  # GGML_TYPE_F16

                _LLAMA = Llama(
                    model_path=settings.llamacpp_model_path,
                    n_ctx=settings.ollama_num_ctx,
                    n_gpu_layers=settings.llamacpp_gpu_layers,
                    type_k=type_k,
                    type_v=type_v,
                    flash_attn=True if (type_k is not None or type_v is not None) else False,
                    verbose=False,
                )
    return _LLAMA


def _get_llm_settings() -> tuple[str, str, str | None]:
    """Retrieve LLM settings, checking user storage overrides first if in a UI context,
    falling back to server global settings."""
    try:
        from nicegui import app as nicegui_app
        # app.storage.user can only be accessed if a request context is active
        # (meaning we are inside a NiceGUI user request/connection thread)
        if nicegui_app.storage.user:
            provider = nicegui_app.storage.user.get("llm_provider") or settings.llm_provider
            api_key = nicegui_app.storage.user.get("custom_api_key")
            
            # Resolve model
            model = nicegui_app.storage.user.get("llm_model")
            if not model:
                if provider.lower() == "anthropic":
                    model = settings.anthropic_model
                else:
                    model = settings.llm_model
                    
            # If no custom api key was set, fall back to global
            if not api_key:
                if provider.lower() == "anthropic":
                    api_key = settings.anthropic_api_key
                elif provider.lower() == "openai":
                    api_key = settings.openai_api_key
                    
            return provider, model, api_key
    except Exception:
        # Outside UI context (e.g. CLI indexer, migrations), fall back to global settings
        pass

    provider = settings.llm_provider
    if provider.lower() == "anthropic":
        model = settings.anthropic_model
        api_key = settings.anthropic_api_key
    else:
        model = settings.llm_model
        api_key = settings.openai_api_key if provider.lower() == "openai" else None
    return provider, model, api_key

LLM_METRICS = {
    "query_count": 0,
    "total_tokens": 0,
    "api_cost": 0.0
}

def _update_metrics(sys_text: str, user_text: str, out_text: str):
    try:
        in_tok = (len(sys_text) + len(user_text)) // 4
        out_tok = len(out_text) // 4
        cost = (in_tok * 0.000003 + out_tok * 0.000015)
        
        LLM_METRICS["query_count"] += 1
        LLM_METRICS["total_tokens"] += (in_tok + out_tok)
        LLM_METRICS["api_cost"] += cost
        
        # Get the username from NiceGUI storage context if available
        try:
            from nicegui import app
            if app.storage.user and app.storage.user.get("authenticated", False):
                username = app.storage.user.get("username", "Guest")
            else:
                username = "system"
        except Exception:
            username = "system"
            
        # Update user-specific metrics in SQLite dynamically
        import sqlite3
        con = sqlite3.connect(settings.sqlite_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS user_metrics (
                username TEXT PRIMARY KEY,
                query_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                api_cost REAL DEFAULT 0.0
            )
        """)
        con.execute("""
            INSERT INTO user_metrics (username, query_count, total_tokens, api_cost)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                query_count = query_count + 1,
                total_tokens = total_tokens + excluded.total_tokens,
                api_cost = api_cost + excluded.api_cost
        """, (username, in_tok + out_tok, cost))
        con.commit()
        con.close()
    except Exception:
        pass

def _chat(system_text: str, user_text: str, max_tokens: int = 4096, json_mode: bool = False,
          provider: str | None = None, model: str | None = None, api_key: str | None = None) -> str:
    res = _chat_raw(system_text, user_text, max_tokens, json_mode, provider, model, api_key)
    _update_metrics(system_text, user_text, res)
    return res

def _chat_raw(system_text: str, user_text: str, max_tokens: int = 4096, json_mode: bool = False,
              provider: str | None = None, model: str | None = None, api_key: str | None = None) -> str:
    """Dispatch one chat completion to the configured provider; return raw text."""
    if not provider or not model or not api_key:
        ov_provider, ov_model, ov_api_key = _get_llm_settings()
        provider = provider or ov_provider
        model = model or ov_model
        api_key = api_key or ov_api_key

    provider = provider.lower()

    if provider == "anthropic":
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
        )
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    if provider == "llamacpp":
        llm = _get_llama()
        kwargs: dict = {
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1 if json_mode else 0.5,
            "repeat_penalty": 1.15,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with _llama_guard():
            return llm.create_chat_completion(**kwargs)["choices"][0]["message"]["content"]

    # OpenAI-compatible: Ollama (local) or OpenAI
    from openai import OpenAI

    if provider == "ollama":
        client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama", timeout=600.0)
    elif provider == "openai":
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key, timeout=600.0)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}")

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if provider == "ollama":
        kwargs["extra_body"] = {"options": {"num_ctx": settings.ollama_num_ctx}}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        if provider == "ollama":
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_base_url}. Is it running?\n"
                f"  Try:  ollama serve   (and)   ollama pull {model}"
            ) from e
        raise
    return resp.choices[0].message.content


def stream_chat(system_text: str, user_text: str, max_tokens: int = 4096, json_mode: bool = False,
                provider: str | None = None, model: str | None = None, api_key: str | None = None):
    out_tokens_list = []
    for token in _stream_chat_raw(system_text, user_text, max_tokens, json_mode, provider, model, api_key):
        out_tokens_list.append(token)
        yield token
    _update_metrics(system_text, user_text, "".join(out_tokens_list))


def _stream_chat_raw(system_text: str, user_text: str, max_tokens: int = 4096, json_mode: bool = False,
                    provider: str | None = None, model: str | None = None, api_key: str | None = None):
    """Dispatch a streaming chat completion to the configured provider; yields tokens."""
    if not provider or not model or not api_key:
        ov_provider, ov_model, ov_api_key = _get_llm_settings()
        provider = provider or ov_provider
        model = model or ov_model
        api_key = api_key or ov_api_key

    provider = provider.lower()

    if provider == "anthropic":
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
        ) as stream:
            for text in stream.text_stream:
                yield text
        return

    if provider == "llamacpp":
        llm = _get_llama()
        kwargs: dict = {
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3 if json_mode else 0.5,
            "repeat_penalty": 1.15,
            "stream": True,
        }
        with _llama_guard():
            response = llm.create_chat_completion(**kwargs)
            for chunk in response:
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    yield content
        return

    # OpenAI-compatible: Ollama (local) or OpenAI
    from openai import OpenAI

    if provider == "ollama":
        client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama", timeout=600.0)
    elif provider == "openai":
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key, timeout=600.0)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}")

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "stream": True,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if provider == "ollama":
        kwargs["extra_body"] = {"options": {"num_ctx": settings.ollama_num_ctx}}

    try:
        response = client.chat.completions.create(**kwargs)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        if provider == "ollama":
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_base_url}. Is it running?\n"
                f"  Try:  ollama serve   (and)   ollama pull {model}"
            ) from e
        raise


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _repair_json(s: str) -> str:
    out: list[str] = []
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                out.append(ch)
        else:
            out.append(ch)
    res = "".join(out)
    if in_str:
        res += '"'
    res = res.rstrip()
    res = re.sub(r",\s*$", "", res)
    res = re.sub(r'"[^"]*"\s*:\s*$', "", res).rstrip()
    res = re.sub(r",\s*$", "", res)
    res = _strip_trailing_commas(res)
    for ch in reversed(stack):
        res += "}" if ch == "{" else "]"
    return res


def _extract_json(text: str) -> dict:
    """Parse the model's JSON, ignoring any thinking blocks or markdown code fences."""
    # Strip thinking blocks first
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Strip markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the model response.")
    snippet = text[start:]
    end = snippet.rfind("}")
    base = snippet[: end + 1] if end != -1 else snippet
    for cand in (base, _strip_trailing_commas(base), _repair_json(snippet)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    try:
        from json_repair import repair_json

        obj = repair_json(snippet, return_objects=True)
        if isinstance(obj, dict) and obj:
            return obj
    except Exception:
        pass
    raise ValueError("Could not parse JSON from the model response.")


def _fit_to_context(text: str, max_output: int = 4096, provider: str | None = None) -> str:
    provider = (provider or settings.llm_provider).lower()
    if provider == "anthropic":
        return text
    budget = max(1024, settings.ollama_num_ctx - max_output - 1800)
    if provider == "llamacpp":
        try:
            llm = _get_llama()
            with _llama_guard():
                toks = llm.tokenize(text.encode("utf-8", "ignore"), add_bos=False)
                if len(toks) <= budget:
                    return text
                head = int(budget * 0.6)
                tail = budget - head - 40
                sep = llm.tokenize(b"\n\n[... lengthy middle omitted to fit the model context ...]\n\n", add_bos=False)
                return llm.detokenize(toks[:head] + sep + toks[-tail:]).decode("utf-8", "ignore")
        except Exception:
            pass
    budget_chars = int(budget * 3.5)
    if len(text) <= budget_chars:
        return text
    h = int(budget_chars * 0.6)
    return text[:h] + "\n\n[... lengthy middle omitted to fit context ...]\n\n" + text[-(budget_chars - h):]


def load_smart_prompts() -> list[str]:
    """Parse the system_prompt_smart.md file into distinct steps."""
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Smart prompts file not found at {PROMPT_PATH}")
    text = PROMPT_PATH.read_text(encoding="utf-8")
    parts = text.split("## Prompt ")
    # parts[0] is header
    # parts[1] is Core Extraction
    # parts[2] is Judicial Analysis
    # parts[3] is Academic Synthesis
    return [p.strip() for p in parts[1:]]


def clean_academic_synthesis(text: str) -> str:
    """Strip out bracketed folder/file paths and note references (e.g. [Subject/File|12:3]) from the text."""
    if not text:
        return ""
    # Remove (as per [Notes...]) references
    cleaned = re.sub(r'\(?as per \[.*?\]\)?', '', text)
    # Remove raw [Notes...] references
    cleaned = re.sub(r'\[[^\]]*?(?:/|\||Notes|\.pdf|\.txt|\.docx)[^\]]*?\]', '', cleaned)
    # Remove remaining citations containing |
    cleaned = re.sub(r'\[[^\]]*?\|[^\]]*?\]', '', cleaned)
    # Remove empty brackets
    cleaned = re.sub(r'\[\s*\]', '', cleaned)
    # Collapse runs of spaces/tabs but PRESERVE paragraph breaks (blank lines) —
    # the newlines carry the model's real section structure.
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    # A '### Heading' on its own line -> bold (a real <h3> balloons to title size).
    cleaned = re.sub(r'(?m)^\s*#{1,6}\s+(.+?)\s*$', r'**\1**', cleaned)
    # Any stray inline '#' markers the model dropped mid-sentence -> just remove.
    cleaned = re.sub(r'(?<=\S)\s+#{1,6}\s+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Fix spacing before punctuation
    cleaned = re.sub(r' +([.,;:?!])', r'\1', cleaned)
    return cleaned.strip()


def analyze_text(case_no: str, full_text: str, progress_callback: Callable[[float, str, str | None], None] | None = None,
                 doc_kind: str = "SC judgment", provider: str | None = None, model: str | None = None, api_key: str | None = None) -> CaseAnalysis:
    """Run the decomposed 3-step pipeline with Chain-of-Thought scratchpads."""
    typology = _TYPOLOGY_RULES.get(doc_kind, _TYPOLOGY_RULES["SC judgment"])
    prompts = load_smart_prompts()
    if len(prompts) < 3:
        raise ValueError(f"Expected 3 prompts in system_prompt_smart.md, found {len(prompts)}")

    full_text = _fit_to_context(full_text, max_output=4096)
    
    # ------------------ STEP 1: CORE EXTRACTION ------------------
    print(f"[{case_no}] Step 1: Core Legal Extraction (Metadata, Facts, Issues)...", flush=True)
    if progress_callback:
        progress_callback(0.10, "Step 1 of 3: Extracting facts and key legal issues", "\n\n>>> INITIALISING STEP 1: CORE LEGAL EXTRACTION <<<\n\n")
    step1_sys = prompts[0]
    step1_user = (
        f"Case No: {case_no}\n\n"
        f"{typology}\n\n"
        "Extract core metadata, the factual narrative (flowing prose, per the factual-matrix rules), "
        "and legal issues. Think first in <thinking>.\n\n"
        f"=== JUDGMENT TEXT ===\n{full_text}\n\n"
        "Remember, your final output after </thinking> MUST be a valid JSON object matching the requested schema. Return ONLY the JSON."
    )
    
    step1_raw = ""
    for chunk in stream_chat(step1_sys, step1_user, max_tokens=8000, json_mode=True, provider=provider, model=model, api_key=api_key):
        step1_raw += chunk
        if progress_callback:
            progress_callback(0.10, "Step 1 of 3: Extracting facts and key legal issues", chunk)
            
    step1_data = _extract_json(step1_raw)
    
    # ------------------ STEP 2: JUDICIAL ANALYSIS ------------------
    print(f"[{case_no}] Step 2: Judicial Analysis (Ratio, Precedent, Legislation)...", flush=True)
    if progress_callback:
        progress_callback(0.40, "Step 2 of 3: Analyzing ratio decidendi, deciding factors, and precedents", "\n\n>>> INITIALISING STEP 2: JUDICIAL ANALYSIS <<<\n\n")
    step2_sys = prompts[1]
    step2_user = (
        f"Case No: {case_no}\n\n"
        f"{typology}\n\n"
        f"Facts & Issues Extracted:\n{json.dumps(step1_data, indent=2)}\n\n"
        "Analyze deciding factors, ratio decidendi, final order, legislation, and precedents. Think first in <thinking>.\n\n"
        f"=== JUDGMENT TEXT ===\n{full_text}\n\n"
        "Remember, your final output after </thinking> MUST be a valid JSON object matching the requested schema. Return ONLY the JSON."
    )
    
    step2_raw = ""
    for chunk in stream_chat(step2_sys, step2_user, max_tokens=8000, json_mode=True, provider=provider, model=model, api_key=api_key):
        step2_raw += chunk
        if progress_callback:
            progress_callback(0.40, "Step 2 of 3: Analyzing ratio decidendi, deciding factors, and precedents", chunk)
            
    step2_data = _extract_json(step2_raw)

    # ------------------ STEP 3: ACADEMIC SYNTHESIS ------------------
    print(f"[{case_no}] Step 3: Academic Synthesis (RAG Personal Notes)...", flush=True)
    if progress_callback:
        progress_callback(0.70, "Step 3 of 3: Comparing notes", "\n\n>>> INITIALISING STEP 3: ACADEMIC SYNTHESIS <<<\n\n")
    
    # Use keywords & ratio decidendi to search personal notes
    keywords = step1_data.get("metadata", {}).get("keywords", [])
    ratio = step2_data.get("ratio_decidendi", "")
    query = f"{' '.join(keywords)} {ratio}"[:300]
    
    print(f"[{case_no}] RAG query for personal notes: {query!r}", flush=True)
    note_hits = retrieve(query, k=5, source="personal_repo")
    note_context = "\n\n".join(f"[{h['meta'].get('anchor', '?')}]\n{h['text']}" for h in note_hits)
    
    if not note_context:
        note_context = "No relevant personal repository notes found in the vector index."
        
    step3_sys = prompts[2]
    step3_user = (
        f"Case facts:\n{step1_data.get('factual_matrix', '')}\n\n"
        f"Ratio decidendi:\n{ratio}\n\n"
        f"=== RETRIEVED PERSONAL NOTES ===\n{note_context}\n\n"
        "Analyze how the court's findings compare to the notes. Think first in <thinking>.\n\n"
        "Remember, your final output after </thinking> MUST be a valid JSON object wrapping your synthesis in the key \"academic_synthesis\", e.g. {\"academic_synthesis\": \"your markdown here\"}."
    )
    
    step3_raw = ""
    for chunk in stream_chat(step3_sys, step3_user, max_tokens=3000, json_mode=True, provider=provider, model=model, api_key=api_key):
        step3_raw += chunk
        if progress_callback:
            progress_callback(0.70, "Step 3 of 3: Comparing notes", chunk)
            
    try:
        step3_data = _extract_json(step3_raw)
    except ValueError:
        # Fallback: if the model outputs raw markdown directly instead of wrapping in JSON, capture it directly
        synthesis_text = re.sub(r"<thinking>.*?</thinking>", "", step3_raw, flags=re.DOTALL)
        synthesis_text = re.sub(r"<think>.*?</think>", "", synthesis_text, flags=re.DOTALL)
        synthesis_text = re.sub(r"```json\s*", "", synthesis_text)
        synthesis_text = re.sub(r"```\s*", "", synthesis_text).strip()
        step3_data = {"academic_synthesis": synthesis_text}
    
    # ------------------ STEP 4: COMBINE RESULTS ------------------
    print(f"[{case_no}] Step 4: Assembling final CaseAnalysis object...", flush=True)
    if progress_callback:
        progress_callback(0.95, "Assembling final case breakdown...", "\n\n>>> ASSEMBLING FINAL CASE ANALYSIS BLOCK <<<\n\n")
    
    # The model sometimes nests the whole step-1 schema inside factual_matrix
    # (esp. on unusual PDF layouts like the NLR/SLR report scans) — lift the
    # legal_issues out so they land in their own section, not in the facts text.
    fm = step1_data.get("factual_matrix", "")
    if isinstance(fm, dict):
        if not step1_data.get("legal_issues") and isinstance(fm.get("legal_issues"), list):
            step1_data["legal_issues"] = fm.pop("legal_issues")
        fm = step1_data["factual_matrix"] = {k: v for k, v in fm.items()}
    # Prose guarantee: the 4B model sometimes returns structured notes anyway —
    # one short rewrite pass turns them into the narrative the UI expects.
    if isinstance(fm, (dict, list)) and fm:
        try:
            prose = _chat(
                "You turn structured case-fact notes into 2-3 short paragraphs of flowing "
                "prose for a legal research tool. Chronological, complete sentences; no "
                "bullets, no 'Date:'/'Event:' labels, no headings, and no facts that are "
                "not in the notes. Return only the paragraphs.",
                json.dumps(fm, ensure_ascii=False)[:4000], max_tokens=700,
                provider=provider, model=model, api_key=api_key)
            prose = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", prose, flags=re.DOTALL).strip()
            if len(prose) > 60:
                step1_data["factual_matrix"] = prose
        except Exception:
            pass  # schema's pretty-renderer remains the fallback

    # --- deterministic tidy-up of step-2 structure (small-model quirks) ------
    _statute = re.compile(r"\b(?:Ordinance|Act\b|Code\b|Constitution|Law\s+No\.)", re.I)

    # Ordinances/Acts the model filed as 'precedent cases' belong in legislation.
    prec, legis = step2_data.get("precedent_index") or [], step2_data.get("legislation_cited") or []
    if isinstance(legis, str):
        legis = [legis]
    keep_prec = []
    for p in prec:
        nm = str(p.get("case_name", "") if isinstance(p, dict) else p)
        if _statute.search(nm) and not re.search(r"\bv\.?\s", nm, re.I):
            legis.append(nm)
        else:
            keep_prec.append(p)
    step2_data["precedent_index"] = keep_prec

    # 'Legislation' entries must name an actual instrument. The model sometimes
    # emits case captions, dicts, or 'Not specified' placeholders — extract the
    # instrument when one is present, drop the entry entirely when none is.
    clean_leg = []
    for entry in legis:
        if isinstance(entry, dict):  # e.g. {'statute': 'Not specified in the text'}
            entry = next((v for v in entry.values() if isinstance(v, str) and v.strip()), "")
        s = str(entry).strip()
        if not s or re.search(r"not\s+(?:specified|available)|^n/?a$|^none$", s, re.I):
            continue
        # ALWAYS reduce to '<Instrument>[, s.N]' — never a sentence.
        inst = re.search(r"([A-Z][\w'() ,]*?(?:Ordinance|Act|Code|Law|Constitution)"
                         r"(?:,?\s*No\.\s*\d+\s*of\s*\d{4})?)", s)
        sect = re.search(r"(?:[Ss]ections?|[Ss]\.)\s*([\dA-Z]+(?:\(\d+\))?(?:\s*(?:,|and)\s*[\dA-Z()]+)*)"
                         r"|Article\s+[\w()]+", s)
        if inst:
            base = re.sub(r"\s+", " ", inst.group(1)).strip(" ,")
            s = f"{base} — {sect.group(0).strip()}" if sect else base
        if len(s) > 90 or not _statute.search(s):
            art = re.search(r"Article\s+[\w()]+", s)
            m2 = re.search(r"([A-Z][\w'() ]+?(?:Ordinance|Act|Code|Law)(?:,?\s*No\.\s*\d+\s*of\s*\d{4})?)", s)
            if art and "Constitution" in s:
                s = f"Constitution of Sri Lanka, {art.group(0)}"
            elif m2:
                s = m2.group(1).strip()
            else:
                continue  # caption / procedural text with no identifiable instrument
        if s and s not in clean_leg:
            clean_leg.append(s)
    step2_data["legislation_cited"] = clean_leg

    # Deciding factors must be reasons, not misplaced precedent/citation dicts.
    df = step2_data.get("deciding_factors")
    if isinstance(df, list):
        keep_df = []
        for d in df:
            if isinstance(d, dict):
                if {"case_name", "citation", "treatment"} & set(d):
                    continue
                d = d.get("factor") or d.get("reason") or ""
            s = str(d).strip()
            if s and not re.match(r"case[_ ]?name\s*:", s, re.I):
                keep_df.append(s)
        step2_data["deciding_factors"] = keep_df

    # A citation object ('Case Name: … Page:8, Para:5') is not a final order.
    fo = step2_data.get("final_order")
    if isinstance(fo, dict):
        fo = fo.get("order") or fo.get("text") or ""
        step2_data["final_order"] = fo
    if isinstance(fo, str) and re.match(r"\s*case[_ ]?name\s*:", fo, re.I):
        step2_data["final_order"] = ""

    combined = {
        "metadata": step1_data.get("metadata", {}),
        "topics_discussed": step1_data.get("metadata", {}).get("keywords", []),
        "factual_matrix": step1_data.get("factual_matrix", ""),
        "legal_issues": step1_data.get("legal_issues", []),
        "evidence_weighing": [], # optional field in default schema
        "precedent_index": step2_data.get("precedent_index", []),
        "legislation_cited": step2_data.get("legislation_cited", []),
        "deciding_factors": step2_data.get("deciding_factors", ""),
        "ratio_decidendi": step2_data.get("ratio_decidendi", ""),
        "final_order": step2_data.get("final_order", ""),
        "academic_synthesis": clean_academic_synthesis(step3_data.get("academic_synthesis", ""))
    }
    
    return CaseAnalysis.model_validate(combined)


# Typology-specific extraction protocols, injected into steps 1-2 so an NLR
# headnote, an SLR appellate report, a digest entry, and a direct SC judgment
# each get the reading they deserve instead of one generic frame.
_TYPOLOGY_RULES = {
    "NLR report": (
        "DOCUMENT TYPOLOGY: New Law Reports (NLR) — an old published law report.\n"
        "- The headnote (catchwords + abstract, printed before the opinion) states the points of law: "
        "derive legal_issues from it, one issue per headnote point.\n"
        "- ratio_decidendi = the bench's explicit holding on those points, in the report's own terms; "
        "preserve archaic phrasing where it IS the principle.\n"
        "- Note the era's context (ordinances in force, courts as then constituted) in the facts.\n"
        "- Cite this report and other NLR cases as '[Vol] NLR [Page]'."
    ),
    "SLR report": (
        "DOCUMENT TYPOLOGY: Sri Lanka Law Reports (SLR) — a modern appellate report.\n"
        "- Map the procedural progression (court of first instance → HC/CA → SC) as the closing "
        "paragraph of factual_matrix.\n"
        "- legal_issues = the issues expressly framed for determination.\n"
        "- If there is a dissent or split, record it in deciding_factors ('Majority: … / Dissent (per X, J.): …').\n"
        "- Cite SLR cases as '[Year] [Vol] Sri LR [Page]'."
    ),
    "Digest": (
        "DOCUMENT TYPOLOGY: Case digest entry — an editor's condensed summary, NOT a full judgment.\n"
        "- The text is already a summary: factual_matrix = a tightened version of the core summary; "
        "do NOT invent procedural history or evidence that is not printed.\n"
        "- ratio_decidendi = the digested principle; keywords = the digest's subject rubric headings.\n"
        "- Leave final_order empty unless the digest itself states the disposition."
    ),
    "SC judgment": (
        "DOCUMENT TYPOLOGY: Supreme Court judgment (direct/unreported).\n"
        "- Identify the specific constitutional articles / statutory provisions under review and list "
        "each in legislation_cited with its section number.\n"
        "- final_order = the operative 'Order' / relief paragraph AS PRINTED (verbatim), not a paraphrase.\n"
        "- Cite as 'SC (FR) Application No. X/Year' or 'SC Appeal No. Y/Year' exactly as captioned."
    ),
}


def _doc_kind_for(case_no: str) -> str:
    """Typology from the corpus row: report_cite 'NN NLR' / 'SLR YYYY' / 'Digest', else direct SC."""
    import sqlite3
    try:
        con = sqlite3.connect(settings.sqlite_path)
        row = con.execute(
            "SELECT COALESCE(report_cite,'') FROM judgements WHERE case_no=? OR filename=? LIMIT 1",
            (case_no, case_no)).fetchone()
        con.close()
        cite = (row[0] if row else "") or ""
    except Exception:
        cite = ""
    if cite.endswith("NLR"):
        return "NLR report"
    if cite.startswith("SLR"):
        return "SLR report"
    if cite == "Digest":
        return "Digest"
    return "SC judgment"


def analyze_pdf(pdf_path: str, progress_callback: Callable[[float, str], None] | None = None,
                doc_kind: str = "SC judgment", provider: str | None = None, model: str | None = None, api_key: str | None = None) -> CaseAnalysis:
    pages = extract_pages(pdf_path, ocr_langs=settings.tesseract_langs)
    text = "\n".join(f"===== Page {i} =====\n{t}" for i, t in enumerate(pages, 1))
    return analyze_text(case_no_from_filename(pdf_path), text, progress_callback, doc_kind=doc_kind, provider=provider, model=model, api_key=api_key)


def analyze_case(case_no: str, force: bool = False, progress_callback: Callable[[float, str], None] | None = None,
                 provider: str | None = None, model: str | None = None, api_key: str | None = None) -> CaseAnalysis:
    """Cached or on-demand smart analysis using decomposed pipeline."""
    from . import store

    con = store.init_db()
    if not force:
        # Cache key is smart-pipeline
        cached = store.get_analysis(con, case_no)
        if cached:
            # Check if this was a smart run or the simple one
            # We can re-run if needed, but for simplicity, check if schema conforms
            return CaseAnalysis.model_validate(cached)
            
    row = con.execute(
        "SELECT local_path FROM judgements WHERE case_no=? OR filename=? LIMIT 1",
        (case_no, case_no),
    ).fetchone()
    if not row or not row[0]:
        raise FileNotFoundError(f"No indexed judgement found for {case_no!r}")
        
    pdf_path = row[0]
    import os
    if not os.path.exists(pdf_path):
        from pathlib import Path
        basename = os.path.basename(pdf_path)
        fallback = REPO_ROOT / "data" / "sc_judgements" / basename
        if fallback.exists():
            pdf_path = str(fallback)
        else:
            raise FileNotFoundError(f"Judgement PDF not found: {pdf_path} (tried fallback: {fallback})")

    ca = analyze_pdf(pdf_path, progress_callback, doc_kind=_doc_kind_for(case_no), provider=provider, model=model, api_key=api_key)
    p_provider = provider or settings.llm_provider
    p_model = model or (settings.anthropic_model if p_provider.lower() == "anthropic" else settings.llm_model)
    if not ca.quality()["hollow"]:
        store.save_analysis(con, case_no, ca.model_dump(mode="json"), f"{p_provider}:smart-pipeline-{p_model}")
    return ca


_STATUTE_SYS = (
    "You are a senior Sri Lankan legislative-drafting analyst. You are given the "
    "text of an Act / Ordinance / statute (often OCR of an official gazette). "
    "Analyse it AS LEGISLATION — never as a court judgment; it has no parties, "
    "ratio, or precedents. Ground every field strictly in the text; if something "
    "is not stated, use an empty string/list rather than inventing it.\n\n"
    "Return ONLY a JSON object:\n"
    '{"long_title": "<the Act\'s long title / one-line purpose as enacted>",'
    ' "purpose": "<2-4 sentences: what this law does and why, in plain English>",'
    ' "key_provisions": [{"section": "<s.N or Part>", "effect": "<what it does, <=25 words>"}],'
    ' "definitions": [{"term": "<defined term>", "meaning": "<statutory definition, <=20 words>"}],'
    ' "scope": "<who/what it applies to; commencement and territorial extent if stated>",'
    ' "amendments": "<amendments, repeals, or Acts it amends/is amended by, if stated; else empty>"}\n'
    "Rules: 5-10 key_provisions covering the operative sections in enacted order; "
    "0-8 definitions (only ones the Act expressly defines); quote section numbers "
    "exactly as printed. No commentary outside the JSON."
)


def analyze_statute(statute_id: str, force: bool = False,
                    progress_callback: Callable[[float, str, str | None], None] | None = None,
                    provider: str | None = None, model: str | None = None, api_key: str | None = None) -> dict:
    """Tailored analysis for an Act/statute (NOT the judgment pipeline). Returns a
    dict with doc_kind='statute' and legislation-appropriate sections; cached in
    the analyses table keyed by statute_id."""
    from . import store
    con = store.init_db()
    if not force:
        cached = store.get_analysis(con, statute_id)
        if cached and cached.get("doc_kind") == "statute":
            return cached

    row = con.execute(
        "SELECT local_path, title, act_no, year FROM statutes WHERE statute_id=? LIMIT 1",
        (statute_id,),
    ).fetchone()
    if not row or not row[0]:
        raise FileNotFoundError(f"No indexed statute found for {statute_id!r}")

    if progress_callback:
        progress_callback(0.15, "Reading the Act text…", None)
    pages = extract_pages(row[0], ocr_langs=settings.tesseract_langs)
    text = _fit_to_context("\n".join(pages), max_output=3072)
    if progress_callback:
        progress_callback(0.45, "Analysing the legislation…", None)

    raw = _chat(_STATUTE_SYS,
                f"Act title (from catalogue): {row[1]}\n"
                f"Act No.: {row[2] or '—'}  Year: {row[3] or '—'}\n\n"
                f"=== STATUTE TEXT ===\n{text}\n\nReturn ONLY the JSON object.",
                max_tokens=2000, json_mode=True, provider=provider, model=model, api_key=api_key)
    data = _extract_json(raw)
    data["doc_kind"] = "statute"
    data.setdefault("title", row[1])
    if progress_callback:
        progress_callback(0.95, "Finalising…", None)

    # Save only if it actually extracted something (purpose or provisions present).
    if (data.get("purpose") or data.get("key_provisions")):
        p_provider = provider or settings.llm_provider
        p_model = model or (settings.anthropic_model if p_provider.lower() == "anthropic" else settings.llm_model)
        store.save_analysis(con, statute_id, data, f"{p_provider}:statute:{p_model}")
    return data


def precedent_test(case_x: str, scenario: str, k: int = 8,
                   provider: str | None = None, model: str | None = None, api_key: str | None = None) -> str:
    """Precedent test using smart prompt guidelines."""
    from .retrieve import retrieve

    context = retrieve(f"{case_x} {scenario}", k=k)
    snippets = "\n\n".join(f"{h['meta'].get('anchor', '?')}\n{h['text']}" for h in context)
    user = (
        f"User scenario:\n{scenario}\n\nCandidate precedent: {case_x}\n\n"
        "Apply the Precedent Test. Think first inside <thinking>.\n\n"
        f"=== RETRIEVED CONTEXT ===\n{snippets}"
    )
    # Use Prompt 2 style system instructions (judicial clerk role)
    prompts = load_smart_prompts()
    sys_prompt = prompts[1] if len(prompts) > 1 else "You are a legal assistant."
    return _chat(sys_prompt, user, max_tokens=2048, provider=provider, model=model, api_key=api_key)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Break down a judgment PDF using smart decomposed pipeline.")
    ap.add_argument("pdf")
    args = ap.parse_args(argv)
    try:
        print(analyze_pdf(args.pdf).model_dump_json(indent=2))
    except RuntimeError as e:
        print(f"\n{e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
