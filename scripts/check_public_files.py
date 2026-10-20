"""Read-only checks of upload candidates; prints locations/categories, never matches.

Uses Git ignore semantics, including tracked files when a repository exists.
Without .git, Git metadata is created only in a temporary directory. This is a
limited working-tree check, not a history scan or a guarantee of secret absence.
"""
import argparse
from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {'.git', '.venv', 'venv', 'node_modules', '.next', '__pycache__',
                'data', 'model', 'models', 'artifacts', 'local', 'secrets', '.idea', '.vscode',
                '.pytest_cache', '.mypy_cache', '.ruff_cache'}
PRIVATE_PATH = re.compile(r'/(?:home|Users)/[A-Za-z0-9._-]+/|/data/datashare/personal/[A-Za-z0-9._-]+|'
                          r'[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+')
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|'
                    r'sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|'
                    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
IPV4 = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
PRIVATE_NETWORKS = tuple(ipaddress.ip_network((number, prefix)) for number, prefix in
                         ((10 << 24, 8), ((172 << 24) + (16 << 16), 12), ((192 << 24) + (168 << 16), 16)))
PLACEHOLDERS = {'', 'local-dev-key', 'your-api-key', 'your-key', 'change-me'}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    category: str


def _git(arguments):
    return subprocess.run(['git', '-c', 'core.excludesFile=/dev/null', *arguments],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def candidate_files(root):
    """Public working-tree candidates, not staged blobs; local Git is never changed."""
    root = Path(root).resolve()
    try:
        _git(['-C', str(root), 'rev-parse', '--show-toplevel'])
    except subprocess.CalledProcessError:
        with tempfile.TemporaryDirectory(prefix='public-file-index-') as directory:
            _git(['init', '--bare', '--template=', '--quiet', directory])
            raw = _git(['--git-dir='+directory, '--work-tree='+str(root),
                        'ls-files', '--others', '--exclude-standard', '-z'])
    else:
        raw = _git(['-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'])
    paths = sorted(set(Path(name.decode('utf-8', errors='surrogateescape'))
                       for name in raw.split(b'\0') if name))
    # Deleted tracked paths have no working-tree content to export.
    return [p for p in paths if (root/p).exists() or (root/p).is_symlink()]


def private_file(path):
    if path.is_absolute() or '..' in path.parts or any(part in PRIVATE_DIRS for part in path.parts):
        return True
    if (path.name.startswith('.env') or path.name.endswith('.env') or '.env.' in path.name) and path.name != '.env.example':
        return True
    if path.name == 'README.local.md':
        return True
    if path.parts[0] == 'reports' and path != Path('reports/README.md') and path.parts[1:2] != ('public',):
        return True
    return bool(re.search(r'\.(?:log|sqlite[^/]*|db(?:-wal|-shm)?|safetensors|pt|pth|pem|key|p12|pfx|pyc|tsbuildinfo)$', path.name))


def scan_files(root, paths):
    root, findings = Path(root).resolve(), []
    for path in paths:
        full = root/path
        if private_file(path):
            findings.append(Finding(str(path), 0, 'private_file')); continue
        # Do not follow symlinks, including a linked parent directory.
        if any(parent.is_symlink() for parent in [full, *full.parents] if parent != root.parent):
            findings.append(Finding(str(path), 0, 'symlink')); continue
        if not full.is_file():
            findings.append(Finding(str(path), 0, 'unsupported_file')); continue
        if full.stat().st_size > 2_000_000:
            findings.append(Finding(str(path), 0, 'large_file')); continue
        raw = full.read_bytes()
        try:
            content = raw.decode('utf-8')
            if '\0' in content: raise UnicodeError()
        except UnicodeError:
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.woff', '.woff2'}:
                findings.append(Finding(str(path), 0, 'unexpected_binary'))
            continue  # Image/font metadata still requires manual inspection.
        for number, line in enumerate(content.splitlines(), 1):
            if PRIVATE_PATH.search(line): findings.append(Finding(str(path), number, 'personal_path'))
            if SECRET.search(line): findings.append(Finding(str(path), number, 'credential_pattern'))
            for match in IPV4.findall(line):
                try:
                    address = ipaddress.ip_address(match)
                except ValueError:
                    continue
                if any(address in network for network in PRIVATE_NETWORKS):
                    findings.append(Finding(str(path), number, 'private_ipv4')); break
            if path.name == '.env.example':
                assignment = re.match(r'\s*([A-Z_]*(?:API_KEY|SECRET|PASSWORD|ACCESS_TOKEN))\s*=\s*(.*?)\s*$', line)
                if assignment and assignment[2].strip('\"\'') not in PLACEHOLDERS:
                    findings.append(Finding(str(path), number, 'non_placeholder_credential'))
    return findings


def check(root):
    paths = candidate_files(root)
    return paths, scan_files(root, paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    paths, findings = check(args.root)
    for item in findings:
        print(f'{item.path}:{item.line}: {item.category}')
    print(f'Checked {len(paths)} public candidate files; findings: {len(findings)}. Working tree only; manually review before publishing.')
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
