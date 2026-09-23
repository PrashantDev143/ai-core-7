import asyncio
import sys


def use_compatible_event_loop() -> None:
    """psycopg's async driver refuses Windows' default ProactorEventLoop.

    Must run before any loop is created, which is why the entry points call it
    first and why the server is started via `python -m app` rather than the
    uvicorn CLI — uvicorn builds its loop before it imports the app, so setting
    the policy on import would be too late.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
