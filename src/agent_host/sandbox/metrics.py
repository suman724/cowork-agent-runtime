"""CloudWatch metrics publisher for sandbox worker tasks.

Publishes TaskUtilization metric to CloudWatch for ECS Service auto-scaling.
Value 1.0 when serving a session, 0.0 when idle/polling SQS.

The ECS auto-scaling policy uses target tracking on this metric with
scale_in_enabled=false. See docs/design/sqs-sandbox-dispatch.md.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()

_NAMESPACE = "Cowork/Sandbox"
_METRIC_NAME = "TaskUtilization"


class TaskUtilizationPublisher:
    """Publishes TaskUtilization metric to CloudWatch.

    Gracefully no-ops when CloudWatch is unavailable (local dev without
    LocalStack CloudWatch, or missing credentials).
    """

    def __init__(
        self,
        cloudwatch_client: Any,
        *,
        service_name: str = "sandbox-workers",
        environment: str = "dev",
    ) -> None:
        self._cw = cloudwatch_client
        self._dimensions = [
            {"Name": "ServiceName", "Value": service_name},
            {"Name": "Environment", "Value": environment},
        ]

    async def report_idle(self) -> None:
        """Report that this task is idle (polling SQS, not serving a session)."""
        await self._publish(0.0)

    async def report_busy(self) -> None:
        """Report that this task is serving a session."""
        await self._publish(1.0)

    async def _publish(self, value: float) -> None:
        """Publish metric value to CloudWatch. Best-effort — never raises."""
        try:
            await self._cw.put_metric_data(
                Namespace=_NAMESPACE,
                MetricData=[
                    {
                        "MetricName": _METRIC_NAME,
                        "Value": value,
                        "Unit": "None",
                        "Dimensions": self._dimensions,
                    }
                ],
            )
        except Exception as exc:
            # Best-effort — don't crash the worker if CloudWatch is unavailable.
            # This is expected in local dev without CloudWatch support.
            logger.debug(
                "cloudwatch_publish_failed",
                metric=_METRIC_NAME,
                value=value,
                error=str(exc),
            )


class NoOpMetricsPublisher:
    """No-op metrics publisher for local dev or when CloudWatch is unavailable."""

    async def report_idle(self) -> None:
        pass

    async def report_busy(self) -> None:
        pass
