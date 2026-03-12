"""Sandbox self-registration: read env, resolve container IP, register with Session Service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from agent_sdk.exceptions import SandboxStartupError

if TYPE_CHECKING:
    from agent_host.config import AgentHostConfig
    from agent_host.session.session_client import SessionClient

logger = structlog.get_logger()

# ECS metadata endpoint for container IP discovery
_ECS_METADATA_URI_ENV = "ECS_CONTAINER_METADATA_URI_V4"


@dataclass(frozen=True)
class EcsMetadata:
    """Container IP and task ARN from ECS metadata endpoint."""

    container_ip: str
    task_arn: str


@dataclass(frozen=True)
class RegistrationResult:
    """Result of sandbox self-registration with Session Service."""

    session_id: str
    workspace_id: str
    workspace_service_url: str
    policy_bundle: dict[str, Any]


async def _fetch_ecs_metadata() -> EcsMetadata:
    """Read container IP and task ARN from ECS metadata endpoint.

    The ECS agent injects $ECS_CONTAINER_METADATA_URI_V4 into every task.
    GET {uri} returns container metadata (Networks[0].IPv4Addresses[0]).
    GET {uri}/task returns task metadata (TaskARN).
    """
    metadata_uri = os.environ.get(_ECS_METADATA_URI_ENV)
    if not metadata_uri:
        raise SandboxStartupError(
            f"Environment variable {_ECS_METADATA_URI_ENV} is not set. "
            "Are you running inside an ECS task?"
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        # Fetch container metadata (IP address)
        try:
            resp = await client.get(metadata_uri)
            resp.raise_for_status()
            container_data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxStartupError(f"Failed to read ECS metadata: {exc}") from exc

        # Fetch task metadata (task ARN)
        try:
            resp = await client.get(f"{metadata_uri}/task")
            resp.raise_for_status()
            task_data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxStartupError(f"Failed to read ECS task metadata: {exc}") from exc

    # Extract container IP
    networks = container_data.get("Networks", [])
    if not networks:
        raise SandboxStartupError("No networks in ECS metadata response")
    ipv4_addresses = networks[0].get("IPv4Addresses", [])
    if not ipv4_addresses:
        raise SandboxStartupError("No IPv4 addresses in ECS metadata response")
    container_ip: str = ipv4_addresses[0]

    # Extract task ARN
    task_arn: str = task_data.get("TaskARN", "")
    if not task_arn:
        raise SandboxStartupError("TaskARN not found in ECS task metadata")

    return EcsMetadata(container_ip=container_ip, task_arn=task_arn)


async def run_sandbox_startup(
    config: AgentHostConfig,
    session_client: SessionClient,
    *,
    port: int,
) -> RegistrationResult:
    """Run sandbox self-registration sequence.

    1. Read SESSION_ID, REGISTRATION_TOKEN from config (loaded from env)
    2. Resolve container IP and task ARN (ECS metadata or localhost in local mode)
    3. Call POST /sessions/{sessionId}/register via SessionClient (with retry)
    4. Return session context + policy bundle

    Raises SandboxStartupError on failure.
    """
    session_id = config.session_id
    if not session_id:
        raise SandboxStartupError("SESSION_ID environment variable is required in sandbox mode")

    # Resolve container endpoint
    if config.sandbox_local_mode:
        container_ip = "127.0.0.1"
        task_arn = f"local:{os.getpid()}"
        logger.info("sandbox_startup_local_mode", session_id=session_id)
    else:
        ecs = await _fetch_ecs_metadata()
        container_ip = ecs.container_ip
        task_arn = ecs.task_arn

    sandbox_endpoint = f"http://{container_ip}:{port}"

    logger.info(
        "sandbox_registering",
        session_id=session_id,
        sandbox_endpoint=sandbox_endpoint,
        task_arn=task_arn,
    )

    # Register with Session Service (uses SessionClient with retry + error handling)
    try:
        result = await session_client.register_sandbox(
            session_id,
            sandbox_endpoint=sandbox_endpoint,
            task_arn=task_arn,
            registration_token=config.registration_token or None,
        )
    except Exception as exc:
        raise SandboxStartupError(f"Registration failed: {exc}") from exc

    policy_bundle = result.get("policyBundle")
    if not policy_bundle:
        raise SandboxStartupError("Registration response missing policyBundle")

    workspace_id: str = result.get("workspaceId", "")
    workspace_service_url: str = result.get("workspaceServiceUrl", "")

    logger.info(
        "sandbox_registered",
        session_id=session_id,
        workspace_id=workspace_id,
    )

    return RegistrationResult(
        session_id=session_id,
        workspace_id=workspace_id,
        workspace_service_url=workspace_service_url,
        policy_bundle=policy_bundle,
    )
