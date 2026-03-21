"""Agent Host process entry point.

Loads config, bootstraps components, runs the JSON-RPC read-dispatch-respond
loop until Shutdown or EOF.

Supports two transport modes:
- stdio (default): JSON-RPC over stdin/stdout for desktop app
- http: HTTP/SSE server for web/sandbox mode
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

from agent_host.config import AgentHostConfig
from agent_host.logging import configure_logging
from agent_host.server.handlers import Handlers
from agent_host.session.session_manager import SessionManager
from agent_host.transport.json_rpc import (
    JsonRpcError,
    JsonRpcResponse,
    parse_request,
    serialize_response,
)
from agent_host.transport.method_dispatcher import MethodDispatcher
from tool_runtime import ToolRouter

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Cowork Agent Host")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: stdio (default) for desktop, http for web/sandbox",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP server port (only used with --transport http, default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="HTTP server bind address (only used with --transport http, default: 0.0.0.0)",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Workspace directory for file operations (only used with --transport http)",
    )
    return parser.parse_args()


async def run_stdio(config: AgentHostConfig, args: argparse.Namespace) -> None:  # noqa: ARG001
    """Run the agent host with stdio transport (desktop mode)."""
    from agent_host.transport.stdio_transport import StdioTransport

    # Initialize stdin reader
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    # Initialize transport
    transport = StdioTransport(reader=reader, writer=sys.stdout)

    # Initialize tool router
    tool_router = ToolRouter()

    # Initialize session manager
    session_manager = SessionManager(
        config=config,
        tool_router=tool_router,
        transport=transport,
    )

    # Set up method dispatcher
    dispatcher = MethodDispatcher()
    handlers = Handlers(session_manager)
    handlers.register_all(dispatcher)

    logger.info("agent_host_ready", transport="stdio")

    # Read-dispatch-respond loop
    try:
        shutdown_requested = False
        while not shutdown_requested:
            raw = await transport.read_message()
            if raw is None:
                logger.info("agent_host_eof")
                break

            if not raw:
                continue

            try:
                request = parse_request(raw)
            except JsonRpcError as e:
                error_response = JsonRpcResponse(id=None, error=e)
                await transport.write_message(serialize_response(error_response))
                continue

            if request.is_notification:
                await dispatcher.dispatch(request)
                continue

            response = await dispatcher.dispatch(request)
            await transport.write_message(serialize_response(response))

            if request.method == "Shutdown":
                shutdown_requested = True
    finally:
        if not shutdown_requested:
            try:
                await session_manager.shutdown()
            except Exception:
                logger.warning("emergency_shutdown_failed", exc_info=True)

    logger.info("agent_host_exiting")


async def run_http(config: AgentHostConfig, args: argparse.Namespace) -> None:
    """Run the agent host with HTTP transport (web/sandbox mode).

    In sandbox mode (SESSION_ID is set): runs self-registration with Session
    Service, syncs workspace files, then serves HTTP.  On shutdown, syncs
    workspace back before exiting.
    """
    from agent_host.events.event_buffer import EventBuffer
    from agent_host.transport.http_transport import HttpTransport

    # Shared event buffer — owned by EventEmitter, read by HttpTransport for SSE
    event_buffer = EventBuffer()

    workspace_dir = args.workspace_dir

    # Initialize transport (shares the event buffer for SSE streaming)
    transport = HttpTransport(
        host=args.host,
        port=args.port,
        workspace_dir=workspace_dir,
        event_buffer=event_buffer,
    )

    # Initialize tool router
    tool_router = ToolRouter()

    # Initialize session manager (shares event buffer with transport)
    session_manager = SessionManager(
        config=config,
        tool_router=tool_router,
        transport=transport,
        event_buffer=event_buffer,
    )

    # Set up method dispatcher
    dispatcher = MethodDispatcher()
    handlers = Handlers(session_manager)
    handlers.register_all(dispatcher)

    # Register workspace.sync handler (HTTP/sandbox mode only)
    dispatcher.register("workspace.sync", transport.handle_workspace_sync)

    # Wire dispatcher into transport
    transport.set_dispatcher(dispatcher)

    # Sandbox startup: register + sync workspace BEFORE serving HTTP
    #
    # Two modes for obtaining session config:
    #   1. SQS mode (SQS_QUEUE_URL set): poll SQS for session config, then register
    #   2. Env var mode (SESSION_ID set): read from env vars, then register (legacy/debug)
    sqs_mode = bool(config.sqs_queue_url)
    sandbox_mode = sqs_mode or bool(config.session_id)
    registration_result = None
    metrics_publisher = None

    if sqs_mode:
        import aioboto3

        from agent_host.sandbox.metrics import NoOpMetricsPublisher, TaskUtilizationPublisher
        from agent_host.sandbox.sqs_consumer import delete_message, poll_for_session
        from agent_host.sandbox.startup import run_sandbox_startup
        from agent_host.sandbox.workspace_sync import download_workspace
        from agent_host.session.session_client import SessionClient

        import dataclasses

        # Set up boto session for AWS clients
        boto_session = aioboto3.Session()
        boto_kwargs: dict[str, str] = {}
        if config.aws_endpoint_url:
            boto_kwargs["endpoint_url"] = config.aws_endpoint_url

        # Set up CloudWatch metrics (best-effort — no-op if unavailable)
        cw_client = None
        try:
            cw_client = await boto_session.client("cloudwatch", **boto_kwargs).__aenter__()
            metrics_publisher = TaskUtilizationPublisher(
                cw_client,
                service_name=config.sandbox_service_name,
                environment=config.environment,
            )
        except Exception:
            logger.warning("cloudwatch_client_init_failed", exc_info=True)
            if cw_client is not None:
                try:
                    await cw_client.__aexit__(None, None, None)
                except Exception:
                    pass
                cw_client = None
            metrics_publisher = NoOpMetricsPublisher()

        # Report idle while polling
        await metrics_publisher.report_idle()

        # Poll SQS for session config
        sqs_client = await boto_session.client("sqs", **boto_kwargs).__aenter__()
        try:
            sqs_config = await poll_for_session(sqs_client, config.sqs_queue_url)

            # Delete message immediately — before registration
            await delete_message(sqs_client, config.sqs_queue_url, sqs_config.receipt_handle)
        finally:
            await sqs_client.__aexit__(None, None, None)

        # Report busy — now serving a session
        await metrics_publisher.report_busy()

        # Override config with SQS message values
        config = dataclasses.replace(
            config,
            session_id=sqs_config.session_id,
            registration_token=sqs_config.registration_token,
            session_service_url=sqs_config.session_service_url or config.session_service_url,
            workspace_service_url=sqs_config.workspace_service_url or config.workspace_service_url,
        )

        # Register with Session Service
        session_client = SessionClient(config.session_service_url)
        try:
            registration_result = await run_sandbox_startup(
                config,
                session_client,
                port=args.port,
            )
        finally:
            await session_client.close()

    elif sandbox_mode:
        # Legacy env var mode (SESSION_ID set directly)
        from agent_host.sandbox.startup import run_sandbox_startup
        from agent_host.session.session_client import SessionClient

        session_client = SessionClient(config.session_service_url)
        try:
            registration_result = await run_sandbox_startup(
                config,
                session_client,
                port=args.port,
            )
        finally:
            await session_client.close()

    # Post-registration setup (common to both SQS and legacy modes)
    if sandbox_mode and registration_result:
        from agent_host.sandbox.workspace_sync import download_workspace

        # Wire workspace sync context into transport (for workspace.sync RPC)
        ws_url = registration_result.workspace_service_url
        if ws_url and registration_result.workspace_id:
            transport.set_workspace_sync_context(
                workspace_service_url=ws_url,
                workspace_id=registration_result.workspace_id,
            )

        # Sync workspace files from Workspace Service
        if ws_url and registration_result.workspace_id and workspace_dir:
            try:
                await download_workspace(
                    ws_url,
                    registration_result.workspace_id,
                    workspace_dir,
                )
            except Exception:
                logger.warning("workspace_download_failed", exc_info=True)
            finally:
                # Always mark complete — even on failure — so workspace.sync
                # RPCs don't hang forever waiting on the gate.
                transport.mark_startup_sync_complete()
        else:
            transport.mark_startup_sync_complete()

        # Initialize session from registration response (skip CreateSession RPC)
        await session_manager.init_from_registration(
            session_id=registration_result.session_id,
            workspace_id=registration_result.workspace_id,
            policy_bundle_data=registration_result.policy_bundle,
            workspace_dir=workspace_dir,
        )

    # Start HTTP server
    await transport.start()
    transport.set_ready()

    logger.info(
        "agent_host_ready",
        transport="http",
        host=args.host,
        port=args.port,
        sandbox_mode=sandbox_mode,
    )

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def _handle_shutdown() -> None:
        stop_event.set()

    loop = asyncio.get_event_loop()
    try:
        import signal

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _handle_shutdown)
    except NotImplementedError:
        pass  # Windows doesn't support add_signal_handler

    try:
        await stop_event.wait()
    finally:
        logger.info("agent_host_shutting_down", sqs_mode=sqs_mode)

        # Report idle metric before shutdown (so auto-scaling sees reduced utilization)
        if metrics_publisher:
            await metrics_publisher.report_idle()

        # Sandbox shutdown: sync workspace back BEFORE session cancellation
        # (session_manager.shutdown() may cancel the session on the backend,
        # after which workspace uploads could be rejected)
        if sandbox_mode and registration_result and workspace_dir:
            from agent_host.sandbox.workspace_sync import upload_workspace

            try:
                await upload_workspace(
                    registration_result.workspace_service_url,
                    registration_result.workspace_id,
                    workspace_dir,
                )
            except Exception:
                logger.warning("workspace_upload_on_shutdown_failed", exc_info=True)

        try:
            await session_manager.shutdown()
        except Exception:
            logger.warning("emergency_shutdown_failed", exc_info=True)
        await transport.shutdown()

        # Clean up CloudWatch client (SQS mode only)
        if cw_client is not None:
            try:
                await cw_client.__aexit__(None, None, None)
            except Exception:
                pass

    logger.info("agent_host_exiting")


async def run() -> None:
    """Main async entry point — runs the JSON-RPC server loop."""
    args = parse_args()

    # Load configuration
    config = AgentHostConfig.from_env()
    configure_logging(config.log_level, Path(config.log_dir))

    logger.info("agent_host_starting", model=config.llm_model, transport=args.transport)

    if args.transport == "http":
        await run_http(config, args)
    else:
        await run_stdio(config, args)


def main() -> None:
    """Synchronous entry point."""
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
