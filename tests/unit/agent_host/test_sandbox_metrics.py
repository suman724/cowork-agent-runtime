"""Tests for CloudWatch TaskUtilization metrics publisher."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_host.sandbox.metrics import NoOpMetricsPublisher, TaskUtilizationPublisher


@pytest.fixture
def mock_cw() -> AsyncMock:
    cw = AsyncMock()
    cw.put_metric_data = AsyncMock()
    return cw


@pytest.fixture
def publisher(mock_cw: AsyncMock) -> TaskUtilizationPublisher:
    return TaskUtilizationPublisher(
        mock_cw,
        service_name="test-workers",
        environment="test",
    )


@pytest.mark.unit
class TestTaskUtilizationPublisher:
    async def test_report_busy(
        self, publisher: TaskUtilizationPublisher, mock_cw: AsyncMock
    ) -> None:
        await publisher.report_busy()

        mock_cw.put_metric_data.assert_called_once()
        call_kwargs = mock_cw.put_metric_data.call_args.kwargs
        assert call_kwargs["Namespace"] == "Cowork/Sandbox"
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "TaskUtilization"
        assert metric["Value"] == 1.0
        assert metric["Unit"] == "None"

    async def test_report_idle(
        self, publisher: TaskUtilizationPublisher, mock_cw: AsyncMock
    ) -> None:
        await publisher.report_idle()

        metric = mock_cw.put_metric_data.call_args.kwargs["MetricData"][0]
        assert metric["Value"] == 0.0

    async def test_dimensions(
        self, publisher: TaskUtilizationPublisher, mock_cw: AsyncMock
    ) -> None:
        await publisher.report_busy()

        metric = mock_cw.put_metric_data.call_args.kwargs["MetricData"][0]
        dims = {d["Name"]: d["Value"] for d in metric["Dimensions"]}
        assert dims == {"ServiceName": "test-workers", "Environment": "test"}

    async def test_cloudwatch_error_does_not_raise(self, mock_cw: AsyncMock) -> None:
        mock_cw.put_metric_data.side_effect = Exception("CloudWatch unavailable")
        publisher = TaskUtilizationPublisher(mock_cw, environment="test")

        # Should not raise
        await publisher.report_busy()
        await publisher.report_idle()

    async def test_custom_service_name(self, mock_cw: AsyncMock) -> None:
        publisher = TaskUtilizationPublisher(
            mock_cw, service_name="custom-workers", environment="prod"
        )
        await publisher.report_busy()

        metric = mock_cw.put_metric_data.call_args.kwargs["MetricData"][0]
        dims = {d["Name"]: d["Value"] for d in metric["Dimensions"]}
        assert dims["ServiceName"] == "custom-workers"
        assert dims["Environment"] == "prod"


@pytest.mark.unit
class TestNoOpMetricsPublisher:
    async def test_noop_does_not_raise(self) -> None:
        publisher = NoOpMetricsPublisher()
        await publisher.report_idle()
        await publisher.report_busy()
