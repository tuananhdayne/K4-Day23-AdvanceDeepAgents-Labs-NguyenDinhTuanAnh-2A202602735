"""Host-side research tools. Network requests and API keys never enter the sandbox."""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_ARXIV_LOCK = threading.Lock()
_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Retry temporary failures with capped exponential backoff and jitter."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0, float(exc.retry_after)))
            else:
                delay = min(cap, base * (2 ** attempt) + random.uniform(0, base))
            time.sleep(delay)
    raise ValueError("attempts must be positive")


def _retry_after(response):
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=30, follow_redirects=True, **kwargs)
        if response.status_code in _RETRY_STATUSES:
            raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
        response.raise_for_status()
        return response
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc


def _compact(value, length=600):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:length]


def _error(exc, secret=None):
    message = f"ERROR: {type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message


def _records_result(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


def _arxiv_request(params):
    global _LAST_ARXIV_CALL
    with _ARXIV_LOCK:
        time.sleep(max(0, 3 - (time.monotonic() - _LAST_ARXIV_CALL)))
        _LAST_ARXIV_CALL = time.monotonic()
        return _request("GET", ARXIV_URL, params=params)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Find recent arXiv papers by keywords. Return JSON records with id, URL, date, title and abstract."""
    try:
        terms = [term for term in re.findall(r"[\w-]+", query, flags=re.UNICODE)
                 if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{term}" for term in terms[:8]),
                  "sortBy": "submittedDate", "sortOrder": "descending",
                  "max_results": max(1, min(int(max_results), 30)), "start": 0}
        response = with_retry(lambda: _arxiv_request(params), attempts=6, cap=60)
        root = ET.fromstring(response.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            url = entry.findtext("a:id", "", ns)
            paper_id = re.sub(r"v\d+$", "", url.rsplit("/", 1)[-1])
            if not paper_id:
                continue
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                            "published": entry.findtext("a:published", "", ns)[:10],
                            "title": _compact(entry.findtext("a:title", "", ns), 300),
                            "summary": _compact(entry.findtext("a:summary", "", ns))})
        return _records_result(records)
    except Exception as exc:
        return _error(exc)


def _hf_record(item, prefer_ai=False):
    paper = item.get("paper", item) if isinstance(item, dict) else {}
    if not isinstance(paper, dict) or not paper.get("id"):
        return None
    paper_id = paper["id"]
    summary = paper.get("ai_summary") if prefer_ai else None
    summary = summary or paper.get("summary") or item.get("summary", "")
    return {"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _compact(paper.get("title") or item.get("title"), 300),
            "summary": _compact(summary), "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "", "stars": paper.get("githubStars") or 0}


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Find trending Hugging Face Daily Papers, optionally by date and local keyword filter. Return compact JSON."""
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        data = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json()
        records = [record for item in data if (record := _hf_record(item))]
        if keyword:
            records = [r for r in records if keyword.casefold() in (r["title"] + " " + r["summary"]).casefold()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return _records_result(records)
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic and return compact JSON metadata and summaries."""
    try:
        if not query.strip():
            return "NO RESULTS"
        params = {"q": query, "limit": max(1, min(int(limit), 50))}
        data = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params)).json()
        return _records_result([record for item in data if (record := _hf_record(item, True))])
    except Exception as exc:
        return _error(exc)


def _exa_call(name, arguments):
    key = os.getenv("EXA_API_KEY", "").strip()
    headers = {"Accept": "application/json, text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}

    def call():
        response = _request("POST", EXA_URL, json=payload, headers=headers)
        raw = response.text
        messages = []
        if "text/event-stream" in response.headers.get("content-type", ""):
            for line in raw.splitlines():
                if line.startswith("data:"):
                    messages.append(json.loads(line[5:].strip()))
        else:
            messages = [response.json()]
        if not messages:
            raise ValueError("empty Exa MCP response")
        message = messages[-1]
        if message.get("error"):
            detail = str(message["error"])
            if "rate limit" in detail.lower():
                raise RetryableError("Exa rate limited", 20)
            raise RuntimeError(detail)
        result = message.get("result", {})
        meta = result.get("_meta") or {}
        content = "\n".join(part.get("text", "") for part in result.get("content", [])
                            if part.get("type") == "text")
        limited = any("ratelimit" in str(k).lower().replace("_", "") and v is True
                      for k, v in meta.items()) if isinstance(meta, dict) else False
        if limited or "rate limit" in content.lower():
            raise RetryableError("Exa rate limited", 20)
        if result.get("isError"):
            raise RuntimeError(content or "Exa tool error")
        return content or "NO RESULTS"

    try:
        return with_retry(call, attempts=6, cap=60)
    except Exception as exc:
        return _error(exc, key)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search web pages through Exa MCP using a natural-language query and objective. Return URLs and snippets."""
    try:
        if not query.strip():
            return "NO RESULTS"
        return _exa_call("web_search_exa", {"query": query,
                         "objective": objective or f"Find reliable sources about {query}",
                         "numResults": max(1, min(int(num_results), 10))})
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY"))


@tool
def web_fetch(url: str) -> str:
    """Fetch one web page through Exa MCP as text. Return up to 12000 characters."""
    try:
        if not url.startswith(("https://", "http://")):
            return "ERROR: invalid URL"
        result = _exa_call("web_fetch_exa", {"urls": [url]})
        return result if result.startswith("ERROR:") else result[:12000]
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY"))


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    for tool_obj, args in [
        (arxiv_search, {"query": "world model", "max_results": 3}),
        (hf_daily_papers, {"limit": 20}),
        (hf_search_papers, {"query": "world model", "limit": 3}),
        (web_search, {"query": "survey paper on world models", "num_results": 2}),
        (web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {tool_obj.name}\n{tool_obj.invoke(args)[:400]}\n")
