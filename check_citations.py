"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    numbers = set()
    urls = set()
    source_by_number = {}
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source {index} is not an object")
            continue
        number, url = source.get("n"), source.get("url")
        if type(number) is not int:
            problems.append(f"source {index} has invalid n")
        elif number in numbers:
            problems.append(f"duplicate source [{number}]")
        else:
            numbers.add(number)
            source_by_number[number] = source
        if not isinstance(url, str) or not re.match(r"^https?://\S+$", url):
            problems.append(f"source {index} has invalid url")
        elif url in urls:
            problems.append(f"duplicate url: {url}")
        else:
            urls.add(url)

    heading = re.search(r"^## References\s*$", report_text, re.MULTILINE)
    if not heading:
        problems.append("missing ## References")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]
    body = re.sub(r"```.*?```|~~~.*?~~~", "", body, flags=re.DOTALL)
    body = re.sub(r"`[^`\n]*`", "", body)
    body = re.sub(r"\[(\d+)\]\([^)]*\)", "", body)
    cited = set()
    for match in re.finditer(r"\[((?:\d+)(?:\s*[,–-]\s*\d+)*)\]", body):
        value = match.group(1)
        for part in re.split(r"\s*,\s*", value):
            bounds = re.split(r"\s*[–-]\s*", part)
            if len(bounds) == 2:
                start, end = map(int, bounds)
                if end < start or end - start > 1000:
                    problems.append(f"invalid citation range [{part}]")
                else:
                    cited.update(range(start, end + 1))
            else:
                cited.add(int(part))
    for number in sorted(cited - numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = []
    for line in references.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\[(\d+)\]\s+(.+)$", line)
        if match:
            reference_lines.append((int(match.group(1)), match.group(2)))
        else:
            problems.append(f"invalid reference line: {line[:80]}")
    counts = Counter(number for number, _ in reference_lines)
    for number in sorted(numbers | counts.keys()):
        if number not in numbers:
            problems.append(f"reference [{number}] missing from sources.json")
        elif counts[number] != 1:
            problems.append(f"source [{number}] has {counts[number]} reference lines")
    for number, line in reference_lines:
        found = re.findall(r"https?://[^\s<>]+", line)
        found = [url.rstrip(".,;)") for url in found]
        if len(found) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in source_by_number and found[0] != source_by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")
        if re.search(r"(?:;|\band\b).*https?://", line, re.IGNORECASE):
            problems.append(f"reference [{number}] appears to bundle sources")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
