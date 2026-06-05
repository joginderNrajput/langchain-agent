"""Console entrypoint to run the API with uvicorn.

For local/single-process use. In production, run multiple workers behind a
process manager, e.g.:

    gunicorn agentic_research_agent.api.app:app \\
        -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 --timeout 120
"""

from __future__ import annotations

import uvicorn

from agentic_research_agent.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "agentic_research_agent.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
