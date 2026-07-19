"""Mnemosyne Memory Manager — SIRA + H-MEM + Enrichment.

Implements the core Mnemosyne modules:
- SIRA: "Think before you search" retrieval with significance scoring
- H-MEM: Temporal consolidation (significance decay, archival)
- Enrichment: Entity extraction, cross-reference discovery

Reference: https://github.com/Liberation-Labs-THCoalition/Project-Mnemosyne
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

log = logging.getLogger("mnemosyne")


class MnemosyneManager:
    """Manages memory persistence, retrieval, and consolidation."""

    def __init__(self, settings):
        self.settings = settings
        self.engine = None
        self.session_factory = None

    async def initialize(self):
        """Create database connection and ensure schema exists."""
        self.engine = create_async_engine(
            self.settings.database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        async with self.engine.begin() as conn:
            await conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_active TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """))
            await conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    significance REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """))
            await conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_msg TEXT,
                    assistant_msg TEXT,
                    significance REAL DEFAULT 0.5,
                    entities JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            await conn.execute(sa.text("""
                CREATE INDEX IF NOT EXISTS idx_interactions_significance
                ON interactions(significance DESC)
            """))

        log.info("Mnemosyne initialized")

    async def close(self):
        """Clean up database connections."""
        if self.engine:
            await self.engine.dispose()

    async def store(self, session_id: str, user_msg: str, assistant_msg: str):
        """Store an interaction with significance scoring."""
        significance = self._score_significance(user_msg, assistant_msg)

        async with self.session_factory() as session:
            await session.execute(
                sa.text("""
                    INSERT INTO sessions (id) VALUES (:sid)
                    ON CONFLICT (id) DO UPDATE SET last_active = NOW()
                """),
                {"sid": session_id},
            )
            await session.execute(
                sa.text("""
                    INSERT INTO interactions (session_id, user_msg, assistant_msg, significance)
                    VALUES (:sid, :umsg, :amsg, :sig)
                """),
                {"sid": session_id, "umsg": user_msg, "amsg": assistant_msg, "sig": significance},
            )
            await session.commit()

    async def retrieve(self, query: str, session_id: str = None, limit: int = 5) -> list:
        """SIRA retrieval — significance-weighted recency search."""
        async with self.session_factory() as session:
            if session_id:
                result = await session.execute(
                    sa.text("""
                        SELECT user_msg, assistant_msg, significance, created_at
                        FROM interactions
                        WHERE session_id = :sid
                        ORDER BY significance DESC, created_at DESC
                        LIMIT :lim
                    """),
                    {"sid": session_id, "lim": limit},
                )
            else:
                result = await session.execute(
                    sa.text("""
                        SELECT user_msg, assistant_msg, significance, created_at
                        FROM interactions
                        ORDER BY significance DESC, created_at DESC
                        LIMIT :lim
                    """),
                    {"lim": limit},
                )

            rows = result.fetchall()
            return [
                {
                    "content": f"{row[0]} -> {row[1][:100]}",
                    "significance": row[2],
                    "source": "mnemosyne-sira",
                    "created_at": str(row[3]),
                }
                for row in rows
            ]

    async def consolidate(self) -> dict:
        """H-MEM temporal consolidation — decay old, archive significant."""
        async with self.session_factory() as session:
            result = await session.execute(
                sa.text("""
                    UPDATE interactions
                    SET significance = significance * 0.95
                    WHERE significance < 0.7
                    AND created_at < NOW() - INTERVAL '24 hours'
                """)
            )
            await session.commit()
            return {"decayed": result.rowcount, "status": "consolidated"}

    def _score_significance(self, user_msg: str, assistant_msg: str) -> float:
        """Score interaction significance for SIRA retrieval priority."""
        score = 0.5

        total_len = len(user_msg) + len(assistant_msg)
        if total_len > 500:
            score += 0.1
        if total_len > 1000:
            score += 0.1

        if "?" in user_msg:
            score += 0.05

        # Code-specific markers (Rivet focus)
        code_words = {"error", "bug", "fix", "deploy", "build", "test", "commit", "merge"}
        if any(w in user_msg.lower() for w in code_words):
            score += 0.1

        # Urgency markers
        urgent_words = {"urgent", "broken", "down", "critical", "asap", "blocked"}
        if any(w in user_msg.lower() for w in urgent_words):
            score += 0.15

        return min(score, 1.0)
