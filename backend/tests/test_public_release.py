"""Publication boundaries and launcher configuration, with temporary files only."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_public_files import ROOT, candidate_files, check, main, scan_files
from scripts.export_public import export_public
from scripts.model_settings import load_settings
from tests.support import ConfiguredTestCase
from app.core.config import override_settings
from scripts.generate_mock_data import generate
from scripts import check_shared_model


class RuntimePrivacyTests(ConfiguredTestCase):
    def test_data_generator_does_not_return_local_database_path(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory)/'private-database.sqlite3'
            with override_settings(COMMERCE_DB_PATH=database):
                result = generate(users=1, orders=1)
            self.assertTrue(database.exists())
            self.assertEqual(result['orders_requested'], 1)
            self.assertNotIn(directory, json.dumps(result))
            self.assertNotIn('path', result)

    def test_shared_model_probe_uses_application_settings_without_printing_them(self):
        output = io.StringIO()
        endpoint, key = 'http://configured-model.invalid/v1', 'probe-test-key'
        with override_settings(OPENAI_BASE_URL=endpoint, OPENAI_API_KEY=key, QWEN_MODEL='configured-model'), \
                patch.object(sys, 'argv', ['check_shared_model']), \
                patch('scripts.check_shared_model.urllib.request.build_opener') as opener, \
                contextlib.redirect_stdout(output):
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"choices":[{}]}'
            self.assertEqual(check_shared_model.main(), 0)
        self.assertEqual(opener.return_value.open.call_count, 2)
        for call in opener.return_value.open.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, endpoint+'/chat/completions')
            self.assertEqual(json.loads(request.data)['model'], 'configured-model')
            self.assertEqual(request.get_header('Authorization'), 'Bearer '+key)
        self.assertNotIn(endpoint, output.getvalue())
        self.assertNotIn(key, output.getvalue())


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'source'
        self.root.mkdir()
        (self.root/'.gitignore').write_bytes((ROOT/'.gitignore').read_bytes())

    def write(self, path, body='local-only'):
        target = self.root/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        return target

    def test_ignore_rules_keep_templates_and_hide_local_material(self):
        hidden = ['.env', '.env.local', '.env.model.local', 'frontend/.env.production', '.envrc', 'deployment.env', 'backup.env.old',
                  'secrets.pem', 'server.key',
                  'README.local.md', 'local/notes.md', 'reports/environment.json', 'reports/raw/result.json',
                  'data/snapshot.sqlite3', 'models/weight.safetensors', 'frontend/.next/file.js']
        public = ['README.md', '.env.example', 'frontend/.env.example', 'backend/app.py',
                  'reports/README.md', 'reports/public/summary.json']
        for name in hidden+public: self.write(name, 'OPENAI_API_KEY=local-dev-key' if name.endswith('.env.example') else 'example')
        paths = set(map(str, candidate_files(self.root)))
        self.assertTrue(set(public) <= paths)
        self.assertFalse(set(hidden) & paths)
        self.assertFalse((self.root/'.git').exists())

    def test_already_tracked_private_file_is_rejected(self):
        subprocess.run(['git', 'init', '--template=', '--quiet', str(self.root)], check=True)
        (self.root/'.gitignore').unlink()
        self.write('.env', 'OPENAI_API_KEY=private-value')
        subprocess.run(['git', '-C', str(self.root), 'add', '.env'], check=True)
        (self.root/'.gitignore').write_bytes((ROOT/'.gitignore').read_bytes())
        _, findings = check(self.root)
        self.assertTrue(any(f.path == '.env' and f.category == 'private_file' for f in findings))

    def test_personal_paths_private_networks_and_credentials_are_flagged(self):
        values = ['/'+'home'+'/developer/project', '192.'+'168.42.17', 'sk-'+'A'*32]
        self.write('config.txt', '\n'.join(values))
        findings = scan_files(self.root, [Path('config.txt')])
        self.assertEqual({f.category for f in findings}, {'personal_path', 'private_ipv4', 'credential_pattern'})
        output = io.StringIO()
        with patch.object(sys, 'argv', ['check', '--root', str(self.root)]), contextlib.redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertTrue(all(value not in output.getvalue() for value in values))

    def test_public_template_requires_placeholder_credentials(self):
        self.write('.env.example', 'OPENAI_API_KEY=private-value\n')
        self.assertEqual(scan_files(self.root, [Path('.env.example')])[0].category, 'non_placeholder_credential')
        self.write('.env.example', 'OPENAI_API_KEY=local-dev-key\nJEV_API_KEY=\n')
        self.assertEqual(scan_files(self.root, [Path('.env.example')]), [])

    def test_export_preserves_source_and_copies_only_public_candidates(self):
        self.write('README.md', 'public-readme')
        private = self.write('.env', 'private-value')
        self.write('README.local.md', 'private-notes')
        self.write('frontend/.env.example', 'NEXT_PUBLIC_SHOW_DEV_DETAILS=true')
        output = self.base/'public'
        count = export_public(self.root, output)
        self.assertEqual(count, 3)
        self.assertEqual((output/'README.md').read_text(), 'public-readme')
        self.assertTrue((output/'frontend/.env.example').exists())
        self.assertFalse((output/'.env').exists())
        self.assertFalse((output/'README.local.md').exists())
        self.assertFalse((output/'.git').exists())
        self.assertEqual(private.read_text(), 'private-value')
        self.assertEqual(check(output)[1], [])

    def test_export_refuses_existing_or_internal_destinations(self):
        existing = self.base/'existing'; existing.mkdir()
        (existing/'keep').write_text('unchanged')
        with self.assertRaises(FileExistsError): export_public(self.root, existing)
        self.assertEqual((existing/'keep').read_text(), 'unchanged')
        with self.assertRaises(ValueError): export_public(self.root, self.root/'public')
        self.assertFalse((self.root/'public').exists())

    def test_failed_check_does_not_create_export(self):
        self.write('config.txt', 'sk-'+'A'*32)
        output = self.base/'public'
        with self.assertRaises(ValueError): export_public(self.root, output)
        self.assertFalse(output.exists())

    def test_external_symlink_is_not_read_or_exported(self):
        outside = self.base/'private'; outside.write_text('private-value')
        (self.root/'config.txt').symlink_to(outside)
        findings = scan_files(self.root, [Path('config.txt')])
        self.assertEqual(findings[0].category, 'symlink')
        with self.assertRaises(ValueError): export_public(self.root, self.base/'public')

    def test_cleanup_incomplete_export_without_touching_source(self):
        self.write('README.md', 'public-readme')
        output = self.base/'public'
        with patch('scripts.export_public.shutil.copy2', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): export_public(self.root, output)
        self.assertFalse(output.exists())
        self.assertEqual((self.root/'README.md').read_text(), 'public-readme')


class ModelLauncherSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_literal_values_and_profile_override(self):
        marker = self.root/'must-not-exist'
        base = self.root/'.env'; local = self.root/'.env.model.local'
        base.write_text('MODEL_PATH=base\nVLLM_ENV=runtime\nUNRELATED=ignored\n')
        literal = f'$(touch {marker})'
        local.write_text('MODEL_PATH='+literal+'\n')
        values = load_settings([base, local])
        self.assertEqual(values['MODEL_PATH'], literal)
        self.assertEqual(values['VLLM_ENV'], 'runtime')
        self.assertNotIn('UNRELATED', values)
        self.assertFalse(marker.exists())

    def test_multiline_setting_fails_closed(self):
        path = self.root/'.env'; path.write_text('MODEL_PATH="line1\nline2"\n')
        with self.assertRaises(ValueError): load_settings([path])

    def test_shell_profile_and_dotenv_precedence_without_model_start(self):
        scripts = self.root/'scripts'; scripts.mkdir()
        for name in ('model_env.sh', 'model_settings.py'):
            shutil.copyfile(ROOT/'scripts'/name, scripts/name)
        (self.root/'.env').write_text(f'VLLM_ENV={sys.prefix}\nMODEL_PATH=base\nCUDA_VISIBLE_DEVICES=1\n')
        (self.root/'.env.model.local').write_text('MODEL_PATH=profile\nCUDA_VISIBLE_DEVICES=2\n')
        environment = {'PATH': os.defpath, 'MODEL_CONFIG_PYTHON': sys.executable, 'MODEL_PATH': 'shell'}
        command = ['bash', '-c', 'source "$1"; printf "%s|%s" "$MODEL_PATH" "$CUDA_VISIBLE_DEVICES"',
                   'launcher-test', str(scripts/'model_env.sh')]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'shell|2')
        environment.pop('MODEL_PATH')
        result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'profile|2')

    def test_missing_runtime_is_clear_error_without_model_start(self):
        scripts = self.root/'scripts'; scripts.mkdir()
        for name in ('model_env.sh', 'model_settings.py'):
            shutil.copyfile(ROOT/'scripts'/name, scripts/name)
        result = subprocess.run(['bash', str(scripts/'model_env.sh')],
                                env={'PATH': os.defpath, 'MODEL_CONFIG_PYTHON': sys.executable},
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Set VLLM_ENV', result.stderr)
