"""Source-preserving RAG: offline lexical retrieval or explicitly configured Chroma vectors."""
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings, get_settings


class Document(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    document_id: str = Field(min_length=1, max_length=200)
    product_id: str = Field(default='', max_length=160)
    kind: str = Field(default='product', pattern='^(product|product_qa|policy)$')
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=100000)
    source: str = Field(min_length=1, max_length=100)
    source_url: str = Field(min_length=1, max_length=1000)
    license: str = Field(min_length=1, max_length=200)
    synthetic: bool = False


def terms(text):
    """Chinese bigrams + Latin words, for the documented offline lexical mode."""
    result = re.findall(r'[a-z0-9]+', text.lower())
    for part in re.findall(r'[\u4e00-\u9fff]+', text):
        result.extend(part[i:i+2] for i in range(max(1, len(part)-1)))
    return list(dict.fromkeys(result))


def chunks(text, length=700, overlap=100):
    for start in range(0, len(text), length-overlap):
        yield text[start:start+length]
        if start+length >= len(text):
            break


class KnowledgeStore:
    def __init__(self, path=None, backend=None, chroma_path=None, embed=None, *, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.path = Path(path if path is not None else self.settings.knowledge_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend or self.settings.rag_backend
        if self.backend not in ('fts', 'chroma'):
            raise ValueError('RAG_BACKEND 只支持 fts 或 chroma')
        self.chroma_path = str(chroma_path if chroma_path is not None else self.settings.chroma_path)
        self.embed = embed or self._embed
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL)')
            db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, tokens)')
        self._collection = None

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _embed(self, texts):
        if not self.settings.embedding_base_url or not self.settings.embedding_model:
            raise ValueError('Chroma 模式需要配置 EMBEDDING_BASE_URL 和 EMBEDDING_MODEL 并重新导入知识')
        # Independent embedding endpoint; the chat model need not support embeddings.
        with httpx.Client(timeout=30) as client:
            response = client.post(self.settings.embedding_base_url + '/embeddings',
                                   headers={'Authorization': 'Bearer ' + self.settings.embedding_api_key.get_secret_value()},
                                   json={'model': self.settings.embedding_model, 'input': texts})
            response.raise_for_status()
            rows = sorted(response.json()['data'], key=lambda row: row['index'])
        if len(rows) != len(texts):
            raise ValueError('Embedding 服务返回数量不一致')
        return [row['embedding'] for row in rows]

    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            # Model identity separates incompatible vector spaces.
            name = 'commerce-' + hashlib.sha256((self.settings.embedding_base_url + self.settings.embedding_model).encode()).hexdigest()[:12]
            self._collection = chromadb.PersistentClient(path=self.chroma_path, settings=Settings(anonymized_telemetry=False)).get_or_create_collection(
                name, embedding_function=None, metadata={'hnsw:space': 'cosine'})
        return self._collection

    def ingest(self, document: Document):
        metadata = document.model_dump(exclude={'text', 'document_id'})
        parts = list(chunks(document.text))
        ids = [hashlib.sha256(f'{document.document_id}:{i}'.encode()).hexdigest() for i in range(len(parts))]
        with self.connection() as db:
            old = [row[0] for row in db.execute('SELECT id FROM chunks WHERE document_id=?', (document.document_id,))]
        if self.backend == 'chroma':
            vectors = self.embed(parts)
            collection = self.collection()
            collection.upsert(ids=ids, documents=parts, metadatas=[metadata] * len(parts), embeddings=vectors)
            obsolete = list(set(old) - set(ids))
            if obsolete:
                collection.delete(ids=obsolete)
        with self.connection() as db:
            db.executemany('DELETE FROM chunks_fts WHERE id=?', [(id,) for id in old])
            db.execute('DELETE FROM chunks WHERE document_id=?', (document.document_id,))
            for id, part in zip(ids, parts):
                db.execute('INSERT INTO chunks VALUES (?,?,?,?)', (id, document.document_id, part, json.dumps(metadata, ensure_ascii=False)))
                db.execute('INSERT INTO chunks_fts VALUES (?,?)', (id, ' '.join(terms(document.title + ' ' + part))))
        return len(parts)

    def search(self, query, product_id='', kind='', limit=5):
        filters = {key: value for key, value in {'product_id': product_id, 'kind': kind}.items() if value}
        if self.backend == 'chroma':
            collection = self.collection()
            if not collection.count():
                return {'backend': self.backend, 'hits': [], 'notice': '向量索引为空，请在 chroma 模式重新导入知识'}
            where = {'$and': [{k: v} for k, v in filters.items()]} if len(filters) > 1 else filters or None
            results = collection.query(query_embeddings=self.embed([query]), n_results=min(limit, collection.count()), where=where,
                                       include=['documents', 'metadatas', 'distances'])
            hits = [self.hit(id, text, meta, float(distance)) for id, text, meta, distance in zip(
                results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])]
        else:
            tokens = terms(query)[:50]
            if not tokens:
                return {'backend': self.backend, 'hits': []}
            match = ' OR '.join('"' + token + '"' for token in tokens)
            clauses, args = ['chunks_fts MATCH ?'], [match]
            for key, value in filters.items():
                clauses.append(f"json_extract(c.metadata, '$.{key}')=?")
                args.append(value)
            args.append(limit)
            with self.connection() as db:
                rows = db.execute('''SELECT c.*,bm25(chunks_fts) AS rank FROM chunks_fts
                    JOIN chunks c ON c.id=chunks_fts.id WHERE ''' + ' AND '.join(clauses) + ' ORDER BY rank LIMIT ?', args).fetchall()
                if not rows and product_id and kind != 'policy':
                    # Known-product evidence can be inspected even when the
                    # question's wording shares no tokens with its documents.
                    fallback = [f"json_extract(metadata, '$.{key}')=?" for key in filters]
                    rows = db.execute('SELECT *,NULL AS rank FROM chunks WHERE ' +
                                      ' AND '.join(fallback) + ' ORDER BY document_id,id LIMIT ?',
                                      [*filters.values(), limit]).fetchall()
            hits = [self.hit(row['id'], row['text'], json.loads(row['metadata']), row['rank']) for row in rows]
        return {'backend': self.backend, 'hits': hits, 'notice': '检索片段是候选证据；不相关时应说明资料不足。rank 不是置信度。'}

    @staticmethod
    def hit(id, text, metadata, rank):
        return {'citation': 'KB-' + id[:12], 'chunk_id': id, 'text': text, 'rank': rank, **metadata}

    def status(self):
        with self.connection() as db:
            count = db.execute('SELECT count(*) FROM chunks').fetchone()[0]
        return {'backend': self.backend, 'stored_chunks': count,
                'embedding_configured': bool(self.settings.embedding_base_url and self.settings.embedding_model)}


def get_knowledge():
    return _get_knowledge(get_settings())


@lru_cache
def _get_knowledge(settings):
    return KnowledgeStore(settings=settings)
