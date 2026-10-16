"""
Local SQLite-backed persistence for user preferences, feedback, and learning state.
No raw business data is stored here.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PreferenceStore:
    """
    Persistent local store for:
    - User preference key-value pairs
    - Recommendation feedback (explicit ratings)
    - Usage events (lightweight)
    - Bandit arm estimates (recommendation scores)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS preferences (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rec_id TEXT NOT NULL,
        rating TEXT NOT NULL,           -- 'useful', 'not_useful', 'incorrect', 'modified', 'saved'
        context TEXT,                   -- JSON: dataset shape, profile, etc.
        created_at TEXT NOT NULL,
        counted INTEGER DEFAULT 0       -- 1 once included in bandit update
    );

    CREATE TABLE IF NOT EXISTS bandit_arms (
        rec_id TEXT PRIMARY KEY,
        alpha REAL DEFAULT 1.0,         -- successes (Thompson sampling)
        beta  REAL DEFAULT 1.0,         -- failures
        total_shown INTEGER DEFAULT 0,
        total_explicit INTEGER DEFAULT 0,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS usage_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        rec_id TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL
    );
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Preferences ────────────────────────────────────────────────────────────

    def set_preference(self, key: str, value: Any) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?,?,?)",
                (key, json.dumps(value), now),
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]

    def all_preferences(self) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    def delete_preference(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM preferences WHERE key=?", (key,))

    # ── Feedback ───────────────────────────────────────────────────────────────

    REWARD_MAP = {
        "useful": +2.0,
        "saved": +2.0,
        "not_useful": -2.0,
        "incorrect": -2.0,
        "modified": -0.5,
        "completed": +1.0,
    }

    def record_feedback(
        self, rec_id: str, rating: str, context: Optional[Dict] = None
    ) -> None:
        now = datetime.utcnow().isoformat()
        ctx = json.dumps(context or {})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (rec_id, rating, context, created_at) VALUES (?,?,?,?)",
                (rec_id, rating, ctx, now),
            )
        self._update_bandit(rec_id, rating)

    def _update_bandit(self, rec_id: str, rating: str) -> None:
        now = datetime.utcnow().isoformat()
        reward = self.REWARD_MAP.get(rating, 0.0)

        # Thompson Sampling: alpha = successes, beta = failures
        # We normalise reward to 0–1 for Beta distribution
        is_positive = reward > 0
        is_negative = reward < 0

        with self._connect() as conn:
            row = conn.execute(
                "SELECT alpha, beta, total_shown, total_explicit FROM bandit_arms WHERE rec_id=?",
                (rec_id,),
            ).fetchone()

            if row is None:
                alpha, beta_val = 1.0, 1.0
                total_shown, total_explicit = 0, 0
            else:
                alpha = row["alpha"]
                beta_val = row["beta"]
                total_shown = row["total_shown"]
                total_explicit = row["total_explicit"]

            if is_positive:
                alpha += abs(reward)
            elif is_negative:
                beta_val += abs(reward)

            total_explicit += 1

            conn.execute(
                """
                INSERT OR REPLACE INTO bandit_arms
                    (rec_id, alpha, beta, total_shown, total_explicit, updated_at)
                VALUES (?,?,?,?,?,?)
                """,
                (rec_id, alpha, beta_val, total_shown, total_explicit, now),
            )

    def record_shown(self, rec_id: str) -> None:
        """Record that a recommendation was shown (neutral — no reward)."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT total_shown FROM bandit_arms WHERE rec_id=?", (rec_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO bandit_arms (rec_id, alpha, beta, total_shown, total_explicit, updated_at)"
                    " VALUES (?,1.0,1.0,1,0,?)",
                    (rec_id, now),
                )
            else:
                conn.execute(
                    "UPDATE bandit_arms SET total_shown=total_shown+1, updated_at=? WHERE rec_id=?",
                    (now, rec_id),
                )

    def get_bandit_scores(self) -> Dict[str, float]:
        """
        Sample Thompson scores for all known arms.
        Returns {rec_id: float 0-1} — higher = more likely to be useful.
        """
        import numpy as np

        with self._connect() as conn:
            rows = conn.execute("SELECT rec_id, alpha, beta FROM bandit_arms").fetchall()

        scores = {}
        for row in rows:
            try:
                score = float(np.random.beta(row["alpha"], row["beta"]))
            except Exception:
                score = 0.5
            scores[row["rec_id"]] = round(score, 4)

        return scores

    def get_learning_summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total_feedback = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
            pos = conn.execute(
                "SELECT COUNT(*) as c FROM feedback WHERE rating IN ('useful','saved','completed')"
            ).fetchone()["c"]
            neg = conn.execute(
                "SELECT COUNT(*) as c FROM feedback WHERE rating IN ('not_useful','incorrect')"
            ).fetchone()["c"]
            total_shown = conn.execute(
                "SELECT SUM(total_shown) as s FROM bandit_arms"
            ).fetchone()["s"] or 0
            n_arms = conn.execute("SELECT COUNT(*) as c FROM bandit_arms").fetchone()["c"]

        acceptance = pos / total_feedback if total_feedback else 0

        return {
            "total_feedback": total_feedback,
            "positive_feedback": pos,
            "negative_feedback": neg,
            "acceptance_rate": round(acceptance * 100, 1),
            "total_shown": int(total_shown),
            "n_recommendation_types": n_arms,
            "has_enough_data": total_feedback >= 10,
            "status": (
                "Sufficient feedback to begin adapting recommendations."
                if total_feedback >= 10
                else "Insufficient feedback to evaluate personalisation. More interactions needed."
            ),
        }

    def get_feedback_history(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rec_id, rating, created_at FROM feedback ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def reset_all(self) -> None:
        """Delete all learning data. Irreversible."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM preferences;
                DELETE FROM feedback;
                DELETE FROM bandit_arms;
                DELETE FROM usage_events;
            """)
        logger.info("Learning store reset.")

    def export_preferences(self) -> Dict[str, Any]:
        return self.all_preferences()


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: Optional[PreferenceStore] = None


def get_store() -> PreferenceStore:
    global _store
    if _store is None:
        from utils.configuration import get_config
        _store = PreferenceStore(get_config().learning_db_path)
    return _store
