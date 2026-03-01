"""Tests for JSON-RPC 2.0 parsing and serialization."""

from __future__ import annotations

import json

import pytest

from agent_host.server.json_rpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    JsonRpcError,
    JsonRpcResponse,
    make_error_response,
    make_success_response,
    parse_request,
    serialize_notification,
    serialize_response,
)


class TestParseRequest:
    def test_valid_request(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": "test", "params": {"key": "val"}, "id": 1})
        req = parse_request(raw)
        assert req.method == "test"
        assert req.params == {"key": "val"}
        assert req.id == 1

    def test_request_without_params(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
        req = parse_request(raw)
        assert req.params == {}

    def test_notification_no_id(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": "notify", "params": {}})
        req = parse_request(raw)
        assert req.is_notification
        assert req.id is None

    def test_string_id(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": "test", "id": "abc"})
        req = parse_request(raw)
        assert req.id == "abc"

    def test_invalid_json(self) -> None:
        with pytest.raises(JsonRpcError) as exc_info:
            parse_request("{invalid")
        assert exc_info.value.code == PARSE_ERROR

    def test_not_an_object(self) -> None:
        with pytest.raises(JsonRpcError) as exc_info:
            parse_request('"string"')
        assert exc_info.value.code == INVALID_REQUEST

    def test_missing_method(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "id": 1})
        with pytest.raises(JsonRpcError) as exc_info:
            parse_request(raw)
        assert exc_info.value.code == INVALID_REQUEST

    def test_non_string_method(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": 42, "id": 1})
        with pytest.raises(JsonRpcError) as exc_info:
            parse_request(raw)
        assert exc_info.value.code == INVALID_REQUEST

    def test_non_object_params(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": "test", "params": [1, 2], "id": 1})
        with pytest.raises(JsonRpcError) as exc_info:
            parse_request(raw)
        assert exc_info.value.code == INVALID_PARAMS


class TestSerializeResponse:
    def test_success_response(self) -> None:
        resp = JsonRpcResponse(id=1, result={"data": "ok"})
        raw = serialize_response(resp)
        data = json.loads(raw)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["result"] == {"data": "ok"}
        assert "error" not in data

    def test_error_response(self) -> None:
        resp = JsonRpcResponse(
            id=1,
            error=JsonRpcError(code=-32001, message="Not found"),
        )
        raw = serialize_response(resp)
        data = json.loads(raw)
        assert data["error"]["code"] == -32001
        assert data["error"]["message"] == "Not found"
        assert "result" not in data

    def test_error_with_data(self) -> None:
        resp = JsonRpcResponse(
            id=1,
            error=JsonRpcError(code=-32000, message="Error", data={"detail": "info"}),
        )
        raw = serialize_response(resp)
        data = json.loads(raw)
        assert data["error"]["data"] == {"detail": "info"}

    def test_null_id(self) -> None:
        resp = JsonRpcResponse(id=None, result={"ok": True})
        raw = serialize_response(resp)
        data = json.loads(raw)
        assert data["id"] is None

    def test_empty_result(self) -> None:
        resp = JsonRpcResponse(id=1)
        raw = serialize_response(resp)
        data = json.loads(raw)
        assert data["result"] == {}


class TestSerializeNotification:
    def test_notification(self) -> None:
        raw = serialize_notification("SessionEvent", {"type": "text_chunk"})
        data = json.loads(raw)
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "SessionEvent"
        assert data["params"]["type"] == "text_chunk"
        assert "id" not in data


class TestHelpers:
    def test_make_error_response(self) -> None:
        resp = make_error_response(1, -32601, "Method not found")
        assert resp.id == 1
        assert resp.error is not None
        assert resp.error.code == -32601

    def test_make_success_response(self) -> None:
        resp = make_success_response(1, {"ok": True})
        assert resp.id == 1
        assert resp.result == {"ok": True}
        assert resp.error is None
