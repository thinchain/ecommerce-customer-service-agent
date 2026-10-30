"""Import normalized source-attributed JSONL into the configured RAG backend."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.knowledge.store import Document, get_knowledge


def ingest_file(path, store=None):
    store = store or get_knowledge()
    documents = count = 0
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                document = Document.model_validate_json(line)
                count += store.ingest(document)
                documents += 1
            except Exception as exc:
                raise ValueError(f'{path}:{line_number}: 导入失败 ({type(exc).__name__})') from exc
    return {'documents': documents, 'chunks': count, 'backend': store.backend}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    print(json.dumps(ingest_file(parser.parse_args().path), ensure_ascii=False))
