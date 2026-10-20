"""Run backend tests with temporary SQLite storage and TCP access forbidden.

Usage: .venv/bin/python -B scripts/test_offline.py [tests.test_catalog_evidence]
MCP integration uses the bundled stdio process with explicit temporary DB paths.
No model server, GPU, package install, or project database is needed.
"""
import argparse
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT)]
sys.dont_write_bytecode = True

from app.core.config import Settings, override_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modules', nargs='*', help='Optional tests.* module names; default: entire backend suite')
    args = parser.parse_args()
    if any(not name.startswith('tests.') for name in args.modules):
        parser.error('Only backend tests.* modules are supported')
    with tempfile.TemporaryDirectory(prefix='commerce-offline-tests-') as directory:
        root = Path(directory).resolve()
        old_tempdir = tempfile.tempdir
        connect = sqlite3.connect
        socket_connect, socket_connect_ex = socket.socket.connect, socket.socket.connect_ex

        def isolated_connect(database, *positional, **kwargs):
            raw = os.fsdecode(database)
            if raw != ':memory:':
                path = Path(unquote(urlsplit(raw).path) if raw.startswith('file:') else raw).resolve()
                if not path.is_relative_to(root):
                    raise AssertionError('SQLite access outside temporary test directory is forbidden')
            return connect(database, *positional, **kwargs)

        def no_tcp(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise AssertionError('TCP access is forbidden in offline tests')
            return socket_connect(self, address)

        def no_tcp_ex(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise AssertionError('TCP access is forbidden in offline tests')
            return socket_connect_ex(self, address)

        try:
            tempfile.tempdir = str(root)
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=None, memory_db_path=root/'memory.sqlite3',
                                    commerce_db_path=root/'commerce.sqlite3',
                                    knowledge_db_path=root/'knowledge.sqlite3', chroma_path=root/'chroma')
            with patch.dict(os.environ, {'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(root)}), \
                    override_settings(settings), patch('sqlite3.connect', isolated_connect), \
                    patch.object(socket.socket, 'connect', no_tcp), patch.object(socket.socket, 'connect_ex', no_tcp_ex):
                loader = unittest.defaultTestLoader
                suite = loader.loadTestsFromNames(args.modules) if args.modules else loader.discover(
                    str(ROOT/'backend/tests'), top_level_dir=str(ROOT/'backend'))
                result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
        finally:
            tempfile.tempdir = old_tempdir


if __name__ == '__main__':
    raise SystemExit(main())
