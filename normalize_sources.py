"""Normalize canonical paper URLs inside the sandbox before citation finalization."""
import json
import re
import sys
from pathlib import Path

SOURCES = Path('/tmp/work/research/sources.json')
ARXIV_ID = re.compile(r'^\d{4}\.\d{4,5}(?:v\d+)?$')


def normalize(sources):
    for source in sources:
        family = source.get('source')
        if (family == 'web' and source.get('url') ==
                'https://openai.com/research/video-generation-models-as-world-simulators'):
            source['url'] = 'https://openai.com/index/video-generation-models-as-world-simulators/'
        if family not in {'arxiv', 'hf-daily', 'hf-search'}:
            continue
        paper_id = str(source.get('id') or '').strip()
        if not ARXIV_ID.fullmatch(paper_id):
            url = str(source.get('url') or '')
            match = re.search(r'(\d{4}\.\d{4,5})(?:v\d+)?(?:$|[?#])', url)
            if not match:
                continue
            paper_id = match.group(1)
        paper_id = re.sub(r'v\d+$', '', paper_id)
        source['id'] = paper_id
        source['url'] = ('https://arxiv.org/abs/' if family == 'arxiv' else
                         'https://huggingface.co/papers/') + paper_id
    return sources


def main():
    try:
        sources = json.loads(SOURCES.read_text(encoding='utf-8'))
        if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
            raise ValueError('sources.json must be a list of objects')
        SOURCES.write_text(json.dumps(normalize(sources), ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'NORMALIZED: {len(sources)} source records')
        return 0
    except (OSError, ValueError) as exc:
        print(f'cannot normalize sources: {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
