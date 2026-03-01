"""Agent Host process entry point.

Loads config, bootstraps components, runs the JSON-RPC read-dispatch-respond
loop until Shutdown or EOF.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog

from agent_host.config import AgentHostConfig
from agent_host.server.handlers import Handlers
from agent_host.server.json_rpc import (
    JsonRpcError,
    JsonRpcResponse,
    parse_request,
    serialize_response,
)
from agent_host.server.method_dispatcher import MethodDispatcher
from agent_host.server.stdio_transport import StdioTransport
from agent_host.session.session_manager import SessionManager
from tool_runtime import ToolRouter

logger = structlog.get_logger()


def _configure_logging(log_level: str) -> None:
    """Configure structlog for stderr output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper()),
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


async def run() -> None:
    """Main async entry point — runs the JSON-RPC server loop."""
    # Load configuration
    config = AgentHostConfig.from_env()
    _configure_logging(config.log_level)

    logger.info("agent_host_starting", model=config.llm_model)

    # Initialize stdin reader
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    # Initialize transport
    transport = StdioTransport(reader=reader, writer=sys.stdout)

    # Initialize tool router
    tool_router = ToolRouter()

    # Initialize session manager (EventEmitter created lazily after session creation)
    session_manager = SessionManager(
        config=config,
        tool_router=tool_router,
        transport=transport,
    )

    # Set up method dispatcher
    dispatcher = MethodDispatcher()
    handlers = Handlers(session_manager)
    handlers.register_all(dispatcher)

    logger.info("agent_host_ready")

    # Read-dispatch-respond loop
    try:
        shutdown_requested = False
        while not shutdown_requested:
            raw = await transport.read_message()
            if raw is None:
                # EOF — clean exit
                logger.info("agent_host_eof")
                break

            if not raw:
                continue

            # Parse request
            try:
                request = parse_request(raw)
            except JsonRpcError as e:
                error_response = JsonRpcResponse(
                    id=None,
                    error=e,
                )
                await transport.write_message(serialize_response(error_response))
                continue

            # Skip notifications (no response expected)
            if request.is_notification:
                await dispatcher.dispatch(request)
                continue

            # Dispatch and respond
            response = await dispatcher.dispatch(request)
            await transport.write_message(serialize_response(response))

            # Check if shutdown was requested
            if request.method == "Shutdown":
                shutdown_requested = True
    finally:
        # Ensure HTTP clients are closed even on crash/EOF without Shutdown
        if not shutdown_requested:
            try:
                await session_manager.shutdown()
            except Exception:
                logger.warning("emergency_shutdown_failed", exc_info=True)

    logger.info("agent_host_exiting")


def main() -> None:
    """Synchronous entry point."""
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
