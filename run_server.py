#!/usr/bin/env python
"""Smart NVR Application Server Entrypoint.

Starts the Uvicorn ASGI server hosting the FastAPI backend and web dashboard.
"""

import sys
import uvicorn
from smart_nvr.config import get_settings


def main() -> None:
    """Launch the Smart NVR server."""
    settings = get_settings()
    host = settings.HOST or "0.0.0.0"
    port = settings.PORT or 8000

    print(f"Starting Smart NVR server on http://{host}:{port}")
    print(f"Interactive API documentation available at http://{host}:{port}/docs")
    print(f"Web Dashboard available at http://{host}:{port}/")

    uvicorn.run(
        "smart_nvr.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
