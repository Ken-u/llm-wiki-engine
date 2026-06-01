"""Unit tests for Agent conversation file storage."""

from app.agents.conversations import (
    append_turn,
    delete_conversation,
    get_conversation,
    list_conversations,
)


def test_append_turn_creates_persistent_agent_conversation(tmp_path):
    conv = append_turn(
        tmp_path,
        agent_id="agent-1",
        user_id=42,
        conversation_id=None,
        user_message="How do I fix boot failure?",
        assistant_answer="Check [[boot-failure]].",
    )

    assert conv["title"] == "How do I fix boot failure?"
    assert conv["message_count"] == 2

    listed = list_conversations(tmp_path, agent_id="agent-1", user_id=42)
    assert listed == [conv]

    loaded = get_conversation(tmp_path, agent_id="agent-1", user_id=42, conversation_id=conv["id"])
    assert loaded["messages"] == [
        {"role": "user", "content": "How do I fix boot failure?"},
        {
            "role": "assistant",
            "content": "Check [[boot-failure]].",
            "rawContent": "Check [[boot-failure]].",
        },
    ]


def test_delete_conversation_removes_metadata_and_messages(tmp_path):
    conv = append_turn(
        tmp_path,
        agent_id="agent-1",
        user_id=42,
        conversation_id=None,
        user_message="First",
        assistant_answer="Answer",
    )

    assert delete_conversation(tmp_path, agent_id="agent-1", user_id=42, conversation_id=conv["id"])
    assert list_conversations(tmp_path, agent_id="agent-1", user_id=42) == []
    assert get_conversation(tmp_path, agent_id="agent-1", user_id=42, conversation_id=conv["id"]) is None


def test_conversations_are_scoped_by_agent_and_user(tmp_path):
    mine = append_turn(
        tmp_path,
        agent_id="agent-1",
        user_id=42,
        conversation_id=None,
        user_message="Mine",
        assistant_answer="A",
    )
    append_turn(
        tmp_path,
        agent_id="agent-1",
        user_id=7,
        conversation_id=None,
        user_message="Other user",
        assistant_answer="B",
    )
    append_turn(
        tmp_path,
        agent_id="agent-2",
        user_id=42,
        conversation_id=None,
        user_message="Other agent",
        assistant_answer="C",
    )

    assert list_conversations(tmp_path, agent_id="agent-1", user_id=42) == [mine]

