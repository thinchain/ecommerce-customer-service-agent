"""Download a bounded, provenance-tracked public PQA sample; never fetch the full archive."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

CATEGORIES = ('headsets', 'keyboards', 'earbud_headphones', 'portable_bluetooth_speakers')


def fetch(output, per_category=250, max_bytes=20_000_000):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, used, sources = 0, 0, []
    # An incomplete sample never becomes the final file.
    temporary = output.with_suffix('.partial')
    if output.exists() or temporary.exists():
        raise ValueError('输出文件已存在，请使用新路径或复用已下载的样本')
    try:
        with temporary.open('x', encoding='utf-8') as file, httpx.Client(timeout=30, trust_env=False, follow_redirects=True) as client:
            for category in CATEGORIES:
                url = f'https://amazon-pqa.s3.amazonaws.com/amazon_pqa_{category}.json'
                count, buffered = 0, b''
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(chunk_size=65536):
                        used += len(chunk)
                        if used > max_bytes:
                            raise ValueError('下载超过字节预算，已停止')
                        buffered += chunk
                        while b'\n' in buffered and count < per_category:
                            line, buffered = buffered.split(b'\n', 1)
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            file.write(json.dumps(record, ensure_ascii=False) + '\n')
                            count += 1
                            records += 1
                        if count >= per_category:
                            break
                if count < per_category:
                    raise ValueError(f'{category} 实际数据条数不足或格式已改变')
                sources.append({'url': url, 'records': count})
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest = {'downloaded_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
                'records': records, 'downloaded_bytes': used, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'license_registry': 'https://registry.opendata.aws/amazon-pqa/',
                'upstream_notice': 'https://amazon-pqa.s3.amazonaws.com/readme.txt'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--per-category', type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.per_category <= 2000:
        parser.error('per-category 应为 1..2000')
    print(json.dumps(fetch(args.output, args.per_category)))
