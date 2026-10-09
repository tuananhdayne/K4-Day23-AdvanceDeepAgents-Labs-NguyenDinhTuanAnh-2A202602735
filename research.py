"""Run a bounded sandbox research agent for one topic and save its validated report."""
import json
import os
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from langchain_core.callbacks import BaseCallbackHandler

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent, build_repair_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
NORMALIZER_SOURCE = ROOT / "normalize_sources.py"
NORMALIZER_PATH = f"{WORKDIR}/research/normalize_sources.py"


class Progress(BaseCallbackHandler):
    """Print tool names only; never print retrieved text or credentials."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name") or "tool"
        print(f"[research] tool: {name}", flush=True)


def slugify(topic):
    """Make a safe, bounded report stem from user input."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """Ask the lead for one complete, source-grounded survey."""
    source_note = ("The arXiv API is currently rate limited; skip arxiv_search and use "
                   "hf-daily, hf-search and web as the three source families. "
                   if os.getenv("SKIP_ARXIV") == "1" else "")
    return (f"Research topic: {topic.strip()}\n{source_note}Produce the complete English survey in {REPORT_PATH} "
            f"and its source manifest in {SOURCES_PATH}. Follow your planning, delegation, "
            "citation finalization, validation, and spot-check workflow. Do not stop after planning.")


def summarize(messages, elapsed, model_name):
    """Count lead tool calls and lead tokens recorded by LangChain."""
    calls = Counter()
    input_tokens = output_tokens = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call.get("name", "unknown")] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens += usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1),
            "subagent_calls": calls.get("task", 0), "tool_calls": dict(calls),
            "tokens": {"input": input_tokens, "output": output_tokens}}


def audit_hf_sources(backend):
    """Find HF citations whose paper page is absent or has a conflicting title."""
    raw = download(backend, [SOURCES_PATH]).get(SOURCES_PATH)
    if not raw:
        raise RuntimeError("sources.json is missing before source audit")
    sources = json.loads(raw.decode("utf-8"))
    issues = []
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        for source in sources:
            if source.get("source") not in {"hf-daily", "hf-search"}:
                continue
            paper_id = str(source.get("id") or "")
            if not paper_id:
                issues.append(f"[{source.get('n')}] has no Hugging Face paper ID")
                continue
            try:
                response = client.get(f"https://huggingface.co/api/papers/{paper_id}")
            except httpx.TransportError:
                continue  # availability failure is not evidence that a paper is fake
            if response.status_code == 404:
                issues.append(f"[{source.get('n')}] {source.get('title')}: HF paper ID {paper_id} returns 404")
            elif response.status_code == 200:
                actual = response.json().get("title", "")
                expected = str(source.get("title") or "")
                if SequenceMatcher(None, expected.casefold(), actual.casefold()).ratio() < 0.4:
                    issues.append(f"[{source.get('n')}] title {expected!r} conflicts with real title {actual!r}")
    return issues


def finalize_and_validate(backend):
    """Run every deterministic citation step inside the sandbox."""
    for label, path in [("source normalizer", NORMALIZER_PATH),
                        ("citation finalizer", FINALIZER_PATH),
                        ("citation validator", VALIDATOR_PATH)]:
        response = backend.execute(f"python3 {path}")
        if response.exit_code != 0 or (label == "citation validator" and "OK:" not in response.output):
            raise RuntimeError(f"{label} failed: {response.output}")


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download, check, then write one report and its metadata."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not sources_bytes:
        raise RuntimeError("report.md or sources.json is missing or empty")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid report or sources.json: {exc}") from exc
    if not report.strip() or not isinstance(sources, list) or not sources:
        raise RuntimeError("report or source list is empty")
    problems = check(report, sources)
    if problems:
        raise RuntimeError("citation validation failed: " + "; ".join(problems[:5]))
    families = sorted({item.get("source") for item in sources if isinstance(item, dict)
                       and isinstance(item.get("source"), str)})
    valid_families = {"arxiv", "hf-daily", "hf-search", "web"}
    if set(families) - valid_families:
        raise RuntimeError("sources.json contains an unknown source family")
    for item in sources:
        family, url = item["source"], item["url"]
        if family == "arxiv" and not url.startswith("https://arxiv.org/abs/"):
            raise RuntimeError("arxiv source URL does not match its family")
        if family in {"hf-daily", "hf-search"} and not url.startswith("https://huggingface.co/papers/"):
            raise RuntimeError("Hugging Face source URL does not match its family")
    stats = summarize(messages, elapsed, model_name)
    if stats["subagent_calls"] < 3:
        raise RuntimeError("lead delegated fewer than three research tasks")
    if len(families) < 3:
        raise RuntimeError("report cites fewer than three source families")
    meta = {"topic": topic, **stats,
            "n_sources": len(sources), "source_families": families}
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = reports_dir / slugify(topic)
    paths = [stem.with_suffix(".md"), Path(f"{stem}.sources.json"), Path(f"{stem}.meta.json")]
    contents = [report_bytes, sources_bytes, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")]
    staged = []
    try:
        for path, content in zip(paths, contents):
            temp = path.with_name(path.name + ".tmp")
            temp.write_bytes(content)
            staged.append((temp, path))
        for temp, path in staged:
            os.replace(temp, path)
    finally:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)
    return paths[0]


def repair_existing(topic, hints=""):
    """Repair a generated report with a bounded agent, entirely inside a fresh sandbox."""
    stem = REPORTS / slugify(topic)
    report_path = stem.with_suffix(".md")
    sources_path = Path(f"{stem}.sources.json")
    meta_path = Path(f"{stem}.meta.json")
    try:
        original_report = report_path.read_bytes()
        original_sources = sources_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = make_model()
        start = time.monotonic()
        with open_sandbox() as backend:
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if setup.exit_code != 0:
                raise RuntimeError(f"sandbox setup failed: {setup.output}")
            upload(backend, {REPORT_PATH: original_report, SOURCES_PATH: original_sources,
                             VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                             NORMALIZER_PATH: NORMALIZER_SOURCE.read_bytes()})
            finalize_and_validate(backend)
            issues = audit_hf_sources(backend)
            for attempt in range(2):
                if not issues:
                    break
                print(f"[research] repairing {len(issues)} invalid HF sources (attempt {attempt + 1})", flush=True)
                agent = build_repair_agent(backend, model)
                agent.invoke({"messages": [{"role": "user", "content":
                    "Repair the listed citations in the existing report. Fetch and verify every replacement, "
                    "rewrite claims to match evidence, and do not invent or reuse invalid IDs. "
                    "Known candidate URLs are only leads; read each before citing.\n" +
                    "\n".join(issues) + ("\nCandidate URLs: " + hints if hints else "")}]},
                    config={"recursion_limit": 300, "callbacks": [Progress()]})
                finalize_and_validate(backend)
                issues = audit_hf_sources(backend)
            if issues:
                raise RuntimeError("unresolved invalid HF sources: " + "; ".join(issues[:6]))
            files = download(backend, [REPORT_PATH, SOURCES_PATH])
            report_bytes, sources_bytes = files[REPORT_PATH], files[SOURCES_PATH]
        report = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
        problems = check(report, sources)
        if problems:
            raise RuntimeError("repaired report has invalid citations: " + "; ".join(problems[:5]))
        families = sorted({item["source"] for item in sources})
        if len(families) < 3:
            raise RuntimeError("repaired report cites fewer than three source families")
        meta.update(n_sources=len(sources), source_families=families,
                    repair_elapsed_s=round(time.monotonic() - start, 1))
        for path, content in [(report_path, report_bytes), (sources_path, sources_bytes),
                              (meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))]:
            temp = path.with_name(path.name + ".tmp")
            temp.write_bytes(content)
            os.replace(temp, path)
        print(f"Repaired {report_path}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


def main(topic):
    """Run the agent in a disposable sandbox and return a CLI status code."""
    if not topic.strip():
        print('Usage: python research.py "topic"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")
        start = time.monotonic()
        with open_sandbox() as backend:
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if setup.exit_code != 0:
                raise RuntimeError(f"sandbox setup failed: {setup.output}")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                             NORMALIZER_PATH: NORMALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            agent_input = {"messages": [{"role": "user", "content": build_prompt(topic)}]}
            config = {"recursion_limit": 1000, "callbacks": [Progress()]}
            result = agent.invoke(agent_input, config=config)
            report_probe = download(backend, [REPORT_PATH]).get(REPORT_PATH)
            for continuation in range(2):
                if report_probe:
                    break
                print(f"[research] continuing lead after notes (attempt {continuation + 1})", flush=True)
                result = agent.invoke({"messages": [*result["messages"],
                    {"role": "user", "content": (
                        "Continue from the researcher notes already written in the sandbox. "
                        f"Read {WORKDIR}/research/notes, merge {SOURCES_PATH}, write {REPORT_PATH}, "
                        f"run python3 {FINALIZER_PATH} and python3 {VALIDATOR_PATH}. "
                        "Do not repeat completed research unless a source family is missing."
                    )}]}, config=config)
                report_probe = download(backend, [REPORT_PATH]).get(REPORT_PATH)
            if not report_probe:
                calls = summarize(result["messages"], time.monotonic() - start, str(model_name))
                listing = backend.execute(f"find {WORKDIR} -maxdepth 3 -type f")
                last = result["messages"][-5:]
                last_text = [(type(message).__name__, str(getattr(message, "content", ""))[:100].replace("\n", " "),
                              [call.get("name") for call in getattr(message, "tool_calls", [])])
                             for message in last]
                raise RuntimeError(f"lead created no report; calls={calls['tool_calls']}; "
                                   f"files={listing.output[:500]!r}; last_message={last_text!r}")
            finalize_and_validate(backend)
            issues = audit_hf_sources(backend)
            for repair_attempt in range(2):
                if not issues:
                    break
                print(f"[research] repairing {len(issues)} invalid HF sources (attempt {repair_attempt + 1})", flush=True)
                repair_agent = build_repair_agent(backend, model)
                repair_agent.invoke({"messages": [{"role": "user", "content":
                    "Repair these invalid citations in the existing report. Verify replacements from fetched "
                    "source text; rewrite or remove claims that the replacement cannot support. "
                    "Do not reuse an invalid HF ID.\n" + "\n".join(issues)}]},
                    config={"recursion_limit": 300, "callbacks": [Progress()]})
                finalize_and_validate(backend)
                issues = audit_hf_sources(backend)
            if issues:
                raise RuntimeError("unresolved invalid HF sources: " + "; ".join(issues[:6]))
            elapsed = time.monotonic() - start
            path = save_outputs(backend, topic, result["messages"], elapsed, str(model_name))
        print(f"Saved {path}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
