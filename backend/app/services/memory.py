"""SQLite session storage. A turn commits history and memory changes together."""
import json
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.core.message import ConversationMessage


class MemoryStore:
    def __init__(self, path=None, user_id=None, *, settings: Settings | None = None):
        settings = settings or get_settings()
        self.path = Path(path if path is not None else settings.memory_db_path)
        self.user_id = user_id if user_id is not None else settings.demo_user_id
        self.lease_seconds = settings.agent_timeout_seconds + 30
        self.lease = ContextVar('session_lease', default=None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, history TEXT NOT NULL DEFAULT '[]',
                memories TEXT NOT NULL DEFAULT '{}', busy INTEGER NOT NULL DEFAULT 0)''')
            if 'lease_token' not in {row[1] for row in db.execute('PRAGMA table_info(sessions)')}:
                db.execute('ALTER TABLE sessions ADD COLUMN lease_token TEXT')
            if 'user_id' not in {row[1] for row in db.execute('PRAGMA table_info(sessions)')}:
                # Legacy sessions belonged to the original demo identity. Never
                # reassign them to a newly configured user at startup.
                db.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'U00001'")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self):
        session_id = str(uuid4())
        with self.connection() as db:
            db.execute('INSERT INTO sessions (id,user_id) VALUES (?,?)', (session_id,self.user_id))
        return session_id

    def get(self, session_id):
        with self.connection() as db:
            row = db.execute('SELECT history, memories FROM sessions WHERE id=? AND user_id=?', (session_id,self.user_id)).fetchone()
        if row is None:
            raise KeyError(session_id)
        messages = [ConversationMessage.from_storage(item).to_storage() for item in json.loads(row[0])]
        return {'session_id': session_id, 'messages': messages, 'memories': json.loads(row[1])}

    @contextmanager
    def edit(self, session_id, *, timeout_seconds=None):
        # Database-level claim also protects against concurrent backend workers.
        token = str(uuid4())
        with self.connection() as db:
            updated = db.execute('UPDATE sessions SET busy=?,lease_token=? WHERE id=? AND user_id=? AND busy<?',
                                 (time.time() + max(self.lease_seconds, (timeout_seconds or 0) + 30),
                                  token, session_id, self.user_id,time.time())).rowcount
            if not updated:
                self.get(session_id)
                raise RuntimeError('该会话正在处理请求，请稍后重试。')
        context = self.lease.set((session_id, token))
        try:
            yield self.get(session_id)
        finally:
            with self.connection() as db:
                db.execute('UPDATE sessions SET busy=0,lease_token=NULL WHERE id=? AND lease_token=?', (session_id,token))
            self.lease.reset(context)

    def save(self, session_id, history, memories):
        history = [ConversationMessage.from_storage(item).to_storage() for item in history]
        with self.connection() as db:
            self.get(session_id)
            args = (json.dumps(history, ensure_ascii=False), json.dumps(memories, ensure_ascii=False), session_id,self.user_id)
            lease = self.lease.get()
            if lease and lease[0] == session_id:
                changed = db.execute('UPDATE sessions SET history=?, memories=? WHERE id=? AND user_id=? AND lease_token=? AND busy>?',
                                     (*args,lease[1],time.time())).rowcount
            else:
                changed = db.execute('UPDATE sessions SET history=?, memories=? WHERE id=? AND user_id=? AND busy<?', (*args,time.time())).rowcount
            if not changed:
                raise RuntimeError('会话写入租约已失效或会话不存在，请重试。')

    def delete(self, session_id):
        with self.edit(session_id):
            with self.connection() as db:
                db.execute('DELETE FROM sessions WHERE id=?', (session_id,))
