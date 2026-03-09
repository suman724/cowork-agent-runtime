"""Tests for MailboxRouter."""

from __future__ import annotations

import asyncio

import pytest

from agent_host.teams.mailbox import MailboxRouter


class TestRegister:
    def test_register_creates_inbox(self) -> None:
        mb = MailboxRouter()
        mb.register("agent-1")
        assert not mb.has_messages("agent-1")

    def test_register_is_idempotent(self) -> None:
        mb = MailboxRouter()
        mb.register("agent-1")
        mb.register("agent-1")  # no error

    def test_unregister_removes_inbox(self) -> None:
        mb = MailboxRouter()
        mb.register("agent-1")
        mb.unregister("agent-1")
        assert not mb.has_messages("agent-1")

    def test_unregister_is_idempotent(self) -> None:
        mb = MailboxRouter()
        mb.unregister("ghost")  # no error


class TestSend:
    async def test_send_delivers_to_recipient(self) -> None:
        mb = MailboxRouter()
        mb.register("alice")
        mb.register("bob")
        await mb.send("alice", "bob", "hello")
        messages = await mb.poll("bob")
        assert len(messages) == 1
        assert messages[0].content == "hello"
        assert messages[0].from_agent == "alice"
        assert messages[0].to_agent == "bob"

    async def test_send_to_unregistered_raises(self) -> None:
        mb = MailboxRouter()
        mb.register("alice")
        with pytest.raises(KeyError, match="Agent not registered"):
            await mb.send("alice", "ghost", "hello")

    async def test_send_does_not_deliver_to_sender(self) -> None:
        mb = MailboxRouter()
        mb.register("alice")
        mb.register("bob")
        await mb.send("alice", "bob", "msg")
        assert not mb.has_messages("alice")


class TestBroadcast:
    async def test_broadcast_reaches_all_except_sender(self) -> None:
        mb = MailboxRouter()
        mb.register("lead")
        mb.register("w1")
        mb.register("w2")
        await mb.broadcast("lead", "team update")
        assert mb.has_messages("w1")
        assert mb.has_messages("w2")
        assert not mb.has_messages("lead")

    async def test_broadcast_message_type(self) -> None:
        mb = MailboxRouter()
        mb.register("lead")
        mb.register("w1")
        await mb.broadcast("lead", "update")
        messages = await mb.poll("w1")
        assert messages[0].message_type == "broadcast"


class TestPoll:
    async def test_poll_empty_returns_empty(self) -> None:
        mb = MailboxRouter()
        mb.register("agent")
        messages = await mb.poll("agent")
        assert messages == []

    async def test_poll_drains_queue(self) -> None:
        mb = MailboxRouter()
        mb.register("alice")
        mb.register("bob")
        await mb.send("alice", "bob", "msg1")
        await mb.send("alice", "bob", "msg2")
        messages = await mb.poll("bob")
        assert len(messages) == 2
        # Queue is drained
        assert await mb.poll("bob") == []

    async def test_poll_unregistered_raises(self) -> None:
        mb = MailboxRouter()
        with pytest.raises(KeyError, match="Agent not registered"):
            await mb.poll("ghost")

    async def test_poll_with_timeout_waits_for_message(self) -> None:
        mb = MailboxRouter()
        mb.register("agent")

        async def send_delayed() -> None:
            await asyncio.sleep(0.05)
            await mb.send("other", "agent", "delayed")

        mb.register("other")
        task = asyncio.create_task(send_delayed())
        messages = await mb.poll("agent", timeout=1.0)
        await task
        assert len(messages) == 1
        assert messages[0].content == "delayed"

    async def test_poll_with_timeout_returns_empty_on_timeout(self) -> None:
        mb = MailboxRouter()
        mb.register("agent")
        messages = await mb.poll("agent", timeout=0.05)
        assert messages == []


class TestHasMessages:
    async def test_has_messages_true_when_queued(self) -> None:
        mb = MailboxRouter()
        mb.register("alice")
        mb.register("bob")
        await mb.send("alice", "bob", "ping")
        assert mb.has_messages("bob") is True

    def test_has_messages_false_when_empty(self) -> None:
        mb = MailboxRouter()
        mb.register("agent")
        assert mb.has_messages("agent") is False

    def test_has_messages_false_for_unregistered(self) -> None:
        mb = MailboxRouter()
        assert mb.has_messages("ghost") is False


class TestConcurrency:
    async def test_concurrent_sends(self) -> None:
        mb = MailboxRouter()
        mb.register("receiver")
        for i in range(5):
            mb.register(f"sender-{i}")

        async def send_one(i: int) -> None:
            await mb.send(f"sender-{i}", "receiver", f"msg-{i}")

        await asyncio.gather(*(send_one(i) for i in range(5)))
        messages = await mb.poll("receiver")
        assert len(messages) == 5
