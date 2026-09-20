"""Console entry points for the API."""

import argparse
import os

import uvicorn


def api():
    parser = argparse.ArgumentParser(description="Run the survival-agent HTTP API.")
    parser.add_argument(
        "--host",
        default=os.environ.get("SURVIVAL_HOST", "0.0.0.0"),
        help="listen address (default: SURVIVAL_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SURVIVAL_PORT", "9082")),
        help="listen port (default: SURVIVAL_PORT or 9082)",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="enable uvicorn's per-request access log",
    )
    args = parser.parse_args()
    uvicorn.run(
        "survival_agent.api:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=args.access_log,
    )
