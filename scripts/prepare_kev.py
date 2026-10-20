"""Offline Kev export; preserve downloaded composite Qwen weights and tokenizer."""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def prepare(source: Path, base: Path, output: Path):
    from filelock import FileLock
    from vllm_jev.checkpoint import sha256
    from vllm_jev.decision_export import _copy, _text_backbone, export_kev, verify_decision
    source, base, output = source.resolve(), base.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [source / name for name in ('head.pt', 'adapter_config.json', 'adapter_model.safetensors',
                                      'tokenizer.json', 'tokenizer_config.json')]
    files += [base / 'config.json', base / 'model.safetensors.index.json']
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    for shard in sorted(set(index['weight_map'].values())):
        if Path(shard).name != shard:
            raise ValueError('Base shard filenames must stay within the local model directory')
        files.append(base / shard)
    identity = {str(path): sha256(path) for path in files}
    with FileLock(str(output) + '.lock'):
        if output.exists():
            if json.loads((output / 'local_sources.json').read_text()) != identity:
                raise FileExistsError('Sources changed. Choose a new KEV_CHECKPOINT_PATH; existing export was preserved.')
            verify_decision(output, full=True)
            print('Kev checkpoint verified:', output, flush=True)
            return
        with tempfile.TemporaryDirectory(prefix='.kev-export-', dir=output.parent) as directory:
            root = Path(directory)
            text_base = root / 'base'
            text_base.mkdir()
            print('Preparing local text backbone (original weights remain unchanged)', flush=True)
            _copy(base, text_base)
            _text_backbone(text_base, json.loads((base / 'config.json').read_text()))
            # vllm-jev's text extraction omits this key; Transformers 5.10 requires it.
            text_index_path = text_base / 'model.safetensors.index.json'
            text_index = json.loads(text_index_path.read_text())
            text_index.setdefault('metadata', {})
            text_index_path.write_text(json.dumps(text_index, indent=2) + '\n')
            exported = root / 'checkpoint'
            print('Merging Kev adapter and pointer head on CPU', flush=True)
            export_kev(text_base, source, exported, 'kev-4b', 'local-' + identity[str(source / 'head.pt')])
            verify_decision(exported, full=True)
            (exported / 'local_sources.json').write_text(json.dumps(identity, indent=2) + '\n')
            exported.rename(output)
        print('Kev checkpoint ready:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.base, args.output)
