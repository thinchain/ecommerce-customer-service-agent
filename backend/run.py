"""Start the local API: python backend/run.py (or python run.py from backend)."""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.getenv('API_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=os.getenv('API_PORT', '8000'))
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port / API_PORT must be between 1 and 65535')
    uvicorn.run('app.main:app', host=args.host, port=args.port, reload=args.reload,
                reload_dirs=[str(ROOT / 'backend/app')] if args.reload else None)


if __name__ == '__main__':
    main()
