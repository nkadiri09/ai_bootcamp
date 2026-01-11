"""Simple LLM client wrapper.

- Tries to use OpenAI if available and environment variables set.
- Otherwise uses a tiny heuristic JSON extractor as a fallback.

Functions:
- extract_financials(text, provider=None, temperature=0.7) -> dict or list
"""
from typing import Optional, Any, Dict
import os
import json

try:
    # optional import
    import openai
except Exception:
    openai = None


PROMPT_TEMPLATE = '''From the news/article below, extract per-period financial values as strict JSON. Return an array of objects, each containing keys: period (e.g., "Q1 2025" or "FY 2024"), revenue_actual, revenue_expected, eps_actual, eps_expected.

- Use numbers for numeric fields (no commas) and null when value missing.
- Only return valid JSON (no explanatory text).

Article:
{article}
'''


def _resp_to_text(resp) -> str:
    """Extract a best-effort text string from different OpenAI response shapes.

    Handles:
    - legacy dict-like ChatCompletion responses (choices -> message/text)
    - new client chat.completions objects (choices -> message -> content)
    - new client responses objects (output -> content)
    - generic fallback to str(resp)
    """
    try:
        # dict-like (older API)
        if isinstance(resp, dict):
            choices = resp.get('choices')
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get('message')
                    if isinstance(msg, dict):
                        # content may be a string
                        return msg.get('content') or first.get('text') or ''
                    return first.get('text') or ''
            # try other keys
            return resp.get('text', '') or str(resp)

        # attribute-style (new client objects)
        # Try chat.completions shape: resp.choices[0].message.content
        choices = getattr(resp, 'choices', None)
        if choices:
            try:
                c0 = choices[0]
                # message could be attribute or dict
                msg = getattr(c0, 'message', None) or (c0.get('message') if isinstance(c0, dict) else None)
                if isinstance(msg, str):
                    return msg
                content = getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # find first string/text
                    for part in content:
                        if isinstance(part, str):
                            return part
                        if isinstance(part, dict):
                            # new client content pieces may be {'type':'output_text','text':'...'}
                            if 'text' in part:
                                return part.get('text')
                    # fallthrough
            except Exception:
                pass

        # Try responses output shape: resp.output or resp.outputs
        output = getattr(resp, 'output', None) or getattr(resp, 'outputs', None)
        if output:
            texts = []
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        cont = item.get('content')
                        if isinstance(cont, list):
                            for c in cont:
                                if isinstance(c, dict) and 'text' in c:
                                    texts.append(c['text'])
                                elif isinstance(c, str):
                                    texts.append(c)
                        elif isinstance(cont, str):
                            texts.append(cont)
                    else:
                        cont = getattr(item, 'content', None)
                        if isinstance(cont, list):
                            for c in cont:
                                t = getattr(c, 'text', None) or (c.get('text') if isinstance(c, dict) else None)
                                if t:
                                    texts.append(t)
                        elif isinstance(cont, str):
                            texts.append(cont)
            if texts:
                return "\n".join(texts)

        # fallback to string representation
        return str(resp)
    except Exception:
        return str(resp)


def _call_openai(prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
    if openai is None:
        raise RuntimeError("openai package not installed")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY or LLM_API_KEY environment variable is required to use OpenAI")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Prefer the new OpenAI client if present (openai.OpenAI)
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=key)
        # Try chat.completions (new client) if available
        try:
            if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=800,
                )
                return resp
        except Exception as e_chat:
            # If chat.completions fails, try responses API as a fallback
            try:
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_tokens=800,
                )
                return resp
            except Exception as e_resp:
                raise RuntimeError(f"OpenAI new-client calls failed (chat error: {e_chat}; responses error: {e_resp})")

    # Fallback to older module-level API if available
    if hasattr(openai, "ChatCompletion"):
        try:
            try:
                openai.api_key = key
            except Exception:
                pass
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=800,
            )
            return resp
        except Exception as e:
            raise RuntimeError(f"OpenAI (old API) call failed: {e}")

    raise RuntimeError("Installed openai package does not expose a usable client interface (OpenAI or ChatCompletion)")


def _fallback_extract(article: str):
    """Very small heuristic parser that extracts numbers for revenue/eps mentions.
    Returns a single-element list with best-effort fields.
    """
    import re

    def find_number_after(keyword):
        # find pattern like 'keyword ... 28.09 B USD' or '0.50 USD'
        m = re.search(fr"{keyword}[^\d\n\r]*([0-9]+(?:[\.,][0-9]+)?)(?:\s*[Bb]?)", article, flags=re.IGNORECASE)
        if not m:
            return None
        val = m.group(1).replace(',', '.')
        try:
            return float(val)
        except Exception:
            return None

    revenue_actual = find_number_after('revenue')
    eps_actual = find_number_after('earnings per share|eps|earnings')
    # expected / forecast - try 'estimated' or 'expected'
    revenue_expected = find_number_after('expected|estimated.*revenue')
    eps_expected = find_number_after('expected|estimated.*eps|estimated.*earnings')

    return [{
        "period": "extracted",
        "revenue_actual": revenue_actual,
        "revenue_expected": revenue_expected,
        "eps_actual": eps_actual,
        "eps_expected": eps_expected,
        "forecast_actual": None,
    }]


def extract_financials(text: str, provider: Optional[str] = None, temperature: float = 0.7):
    """Extract financials using configured provider or fallback.

    Returns a list of dicts on success, or a dict {error: str, raw: str} on failure.
    """
    prompt = PROMPT_TEMPLATE.format(article=text)

    # If provider explicitly openai or env available, try openai
    use_openai = (provider == "openai") or (openai is not None and (os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY')))
    if use_openai:
        try:
            resp = _call_openai(prompt, temperature=temperature)
            # extract text
            raw = _resp_to_text(resp)

            # try parse JSON from raw
            try:
                parsed = json.loads(raw)
                return parsed
            except Exception:
                # attempt to find json substring
                import re
                m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", raw)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        return parsed
                    except Exception:
                        return {"error": "LLM returned invalid JSON", "raw": raw}
                return {"error": "LLM returned non-JSON response", "raw": raw}
        except Exception as e:
            return {"error": f"OpenAI call failed: {e}", "raw": ""}

    # fallback
    try:
        parsed = _fallback_extract(text)
        return parsed
    except Exception as e:
        return {"error": f"Fallback parser failed: {e}", "raw": ""}
