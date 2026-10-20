"""Online SQLite snapshots with checksums. Cross-database snapshots require paused writes."""
import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def backup(source, destination):
    source, destination = Path(source), Path(destination)
    files = sorted(source.glob('*.sqlite3'))
    if not files:
        raise ValueError('源目录没有 SQLite 数据库')
    destination.mkdir(parents=True, exist_ok=False)
    checksums = {}
    try:
        for file in files:
            target = destination/file.name
            with sqlite3.connect(file.resolve().as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
                if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':raise ValueError('备份完整性校验失败')
                if file.name=='memory.sqlite3':dst.execute('UPDATE sessions SET busy=0')
            checksums[file.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest={'created_at':datetime.now(timezone.utc).isoformat(),'sha256':checksums,'scope':'SQLite only; reindex Chroma from normalized JSONL'}
        (destination/'backup.json').write_text(json.dumps(manifest,indent=2))
        return manifest
    except BaseException:
        shutil.rmtree(destination)
        raise


def restore(source, destination):
    source,destination=Path(source),Path(destination)
    manifest=json.loads((source/'backup.json').read_text())
    for name,digest in manifest['sha256'].items():
        if Path(name).name!=name or not name.endswith('.sqlite3'):raise ValueError('非法备份文件名')
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('备份校验和不匹配')
    destination.mkdir(parents=True,exist_ok=False)
    for name in manifest['sha256']:shutil.copyfile(source/name,destination/name)
    return {'restored_files':list(manifest['sha256'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['backup','restore'])
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps((backup if args.operation=='backup' else restore)(args.source,args.destination),indent=2))
