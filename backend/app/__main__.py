"""Server entry point.

Runs uvicorn on a loop we create ourselves. `uvicorn.run()` calls
`Config.setup_event_loop()`, which reinstalls the platform default policy and
undoes the Windows fix in app/runtime.py — so the loop is configured with
`loop="none"` and driven directly instead.

The cost is no auto-reload: the reloader supervises child processes that build
their own loops before importing anything of ours, so the policy cannot be set
early enough there. Restart after a code change.
"""

import asyncio

from app.runtime import use_compatible_event_loop

use_compatible_event_loop()

import uvicorn  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    config = uvicorn.Config(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        loop="none",
    )
    asyncio.run(uvicorn.Server(config).serve())


if __name__ == "__main__":
    main()
