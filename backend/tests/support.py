"""Run every test against explicit defaults, independently of developer .env files."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from contextvars import copy_context
import unittest
from unittest.mock import patch

from app.core.config import Settings, override_settings


class IsolatedSettings:
    def run(self, result=None):
        with TemporaryDirectory(prefix='commerce-test-') as directory:
            with patch.dict(os.environ, {}, clear=True):
                root = Path(directory)
                settings = Settings(_env_file=None, memory_db_path=root/'memory.sqlite3',
                                    structured_routing=False,
                                    natural_response_enabled=False,
                                    commerce_db_path=root/'commerce.sqlite3',
                                    knowledge_db_path=root/'knowledge.sqlite3', chroma_path=root/'chroma')
            with override_settings(settings):
                if isinstance(self, unittest.IsolatedAsyncioTestCase):
                    # unittest captures its async context when constructing the test.
                    self._asyncioTestContext = copy_context()
                return super().run(result)


class ConfiguredTestCase(IsolatedSettings, unittest.TestCase):
    pass


class ConfiguredAsyncTestCase(IsolatedSettings, unittest.IsolatedAsyncioTestCase):
    pass
