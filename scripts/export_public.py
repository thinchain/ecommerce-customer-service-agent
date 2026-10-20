"""Copy checked public working-tree files to a NEW directory outside the project.

No Git history, local configuration, data, dependencies or model files are copied.
The source workspace is not modified. Manual content review is still required.
"""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_public_files import check


def export_public(root, destination):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Destination already exists; choose a new directory')
    if destination.resolve().is_relative_to(root):
        raise ValueError('Choose a destination outside the source project')
    paths, findings = check(root)
    if findings:
        locations = ', '.join(f'{f.path}:{f.line}:{f.category}' for f in findings)
        raise ValueError('Public file check failed: '+locations)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for path in paths:
            target = destination/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/path, target)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    count = export_public(ROOT, args.destination)
    print(f'Exported {count} files. Review the clean directory before initializing Git or publishing.')


if __name__ == '__main__':
    main()
