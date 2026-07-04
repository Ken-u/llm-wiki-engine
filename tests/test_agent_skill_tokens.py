"""Unit tests for Agent Skill Token helpers and model fields."""

import asyncio

from app.agents.models import Agent
from app.agents import service


def test_generate_skill_token_has_stable_prefix_and_entropy():
    first = service.generate_skill_token()
    second = service.generate_skill_token()

    assert first.startswith("lws_")
    assert second.startswith("lws_")
    assert first != second
    assert len(first) > 32


def test_agent_skill_token_columns_present():
    cols = {c.name for c in Agent.__table__.columns}

    assert {"skill_token_hash", "skill_token_created_at"}.issubset(cols)


def test_regenerate_skill_token_replaces_hash_without_storing_raw():
    class FakeDb:
        def __init__(self):
            self.committed = False

        async def commit(self):
            self.committed = True

    agent = Agent(
        id="agent-1",
        name="Agent",
        description="",
        system_prompt="",
        is_public=True,
        require_api_key=True,
        api_key_hash=None,
        created_by=1,
    )
    db = FakeDb()

    first = asyncio.run(service.regenerate_skill_token(db, agent))  # type: ignore[arg-type]
    first_hash = agent.skill_token_hash
    second = asyncio.run(service.regenerate_skill_token(db, agent))  # type: ignore[arg-type]

    assert first.startswith("lws_")
    assert second.startswith("lws_")
    assert first != second
    assert first_hash != agent.skill_token_hash
    assert agent.skill_token_hash not in {first, second}
    assert agent.skill_token_created_at is not None
    assert db.committed is True
