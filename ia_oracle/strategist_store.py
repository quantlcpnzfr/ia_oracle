"""StrategistStore — Persistence and aggregation for the Global Pulse (SITREP).

Collections:
    global_pulse  — time-bucketed synthesized snapshots (BLUF, Key Developments, Regional).
    intel_items   — read-only access for semantic aggregation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from forex_shared.mongo_manager import MongoManager
from forex_shared.worker_api.store import BaseStore
from forex_shared.logging.get_logger import get_logger

log = get_logger(__name__)

class StrategistStore(BaseStore):
    """Async MongoDB persistence for Global Pulse snapshots and aggregation."""

    COLLECTION_PULSE = "global_pulse"
    COLLECTION_INTEL = "intel_items"
    COLLECTION_SYNTHESIS = "global_pulse_synthesis_runs"

    def __init__(self, mongo_manager: Optional[MongoManager] = None) -> None:
        super().__init__(mongo_manager)

    async def ensure_indexes(self) -> None:
        """Idempotent index creation for the Strategist."""
        # Pulse indexes
        await self._mongo.async_ensure_indexes(
            self.COLLECTION_PULSE,
            [
                [("timestamp", -1)],
                [("affected_currencies", 1)],
            ],
        )
        # Ensure we have the indexes on intel_items needed for fast aggregation
        # (Note: IntelMongoStore already handles its own indexes, but we ensure our needs here)
        await self._mongo.async_ensure_indexes(
            self.COLLECTION_INTEL,
            [
                [("event_type", 1), ("extra.oracle_review_candidate", 1), ("published_at", -1)],
            ],
        )
        await self._mongo.async_ensure_indexes(
            self.COLLECTION_SYNTHESIS,
            [
                [("run_id", 1), ("stage", 1), ("created_at", -1)],
                [("created_at", -1)],
            ],
        )
        log.info("StrategistStore: indexes ensured.")

    # ── Aggregation Logic ─────────────────────────────────────────────

    async def fetch_semantic_gems(self, hours: int = 6, limit: int = 100, min_danger: float = 0.5) -> List[Dict[str, Any]]:
        """Fetch recently enriched items ranked by danger and impact.
        
        Refinement #4: Semantic Gem Selection
        Prefer Oracle-reviewed semantic items, but allow high-danger raw/enriched
        intel as a fallback so a fresh pipeline can produce an initial pulse
        before Oracle summaries exist.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        high_danger_floor = max(min_danger, 0.75)
        
        query = {
            "danger_score": {"$gte": min_danger},
            "created_at": {"$gte": cutoff},
            "$or": [
                {"extra.oracle_review_candidate": True},
                {"danger_score": {"$gte": high_danger_floor}},
            ],
        }
        
        # Pipeline Refinement #4: Confidence-Weighted Aggregation
        pipeline = [
            {"$match": query},
            {
                "$lookup": {
                    "from": "oracle_resolved",
                    "localField": "item_id",
                    "foreignField": "trigger_event_id",
                    "as": "oracle_info"
                }
            },
            {
                "$addFields": {
                    "oracle_resolution": {"$arrayElemAt": ["$oracle_info", 0]}
                }
            },
            {
                "$addFields": {
                    "oracle_confidence": {"$ifNull": ["$oracle_resolution.oracle_confidence", 0.0]},
                    "oracle_action": {"$ifNull": ["$oracle_resolution.action", "PENDING"]}
                }
            },
            {
                "$addFields": {
                    "weighted_score": {
                        "$add": [
                            {"$multiply": ["$danger_score", 0.4]},
                            {"$multiply": ["$oracle_confidence", 0.6]}
                        ]
                    }
                }
            },
            {"$sort": {"weighted_score": -1, "created_at": -1}},
            {"$limit": limit}
        ]
        
        # MongoManager is sync-based with async wrappers. 
        # For aggregate, we need to wrap the whole blocking operation.
        def _run_aggregation():
            coll = self._mongo.get_collection(self.COLLECTION_INTEL)
            return list(coll.aggregate(pipeline))

        results = await self._mongo._run_mongo_io(_run_aggregation)
        for item in results:
            extra = item.setdefault("extra", {})
            if not extra.get("analysis_summary"):
                text = " ".join(
                    str(part).strip()
                    for part in (item.get("title"), item.get("body"))
                    if str(part or "").strip()
                )
                if text:
                    extra["analysis_summary"] = text[:500]
        return results

    # ── Pulse Persistence ─────────────────────────────────────────────

    async def save_pulse(self, pulse_doc: Dict[str, Any]) -> str:
        """Store a new Global Pulse snapshot.
        
        Returns the generated document ID.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        if "_id" not in pulse_doc:
            pulse_doc["_id"] = str(uuid.uuid4())
            
        if "timestamp" not in pulse_doc:
            pulse_doc["timestamp"] = now
            
        pulse_doc["created_at"] = now
        
        await self._mongo.async_replace_one(
            self.COLLECTION_PULSE,
            {"_id": pulse_doc["_id"]},
            pulse_doc,
            upsert=True
        )
        
        log.info("StrategistStore: saved new Global Pulse snapshot id=%s", pulse_doc["_id"])
        return pulse_doc["_id"]

    async def start_pulse_run(
        self,
        *,
        run_id: str,
        reason: str,
        min_danger: float,
        item_ids: List[str],
        previous_pulse_id: Optional[str] = None,
    ) -> str:
        """Create the live Global Pulse document at synthesis start."""
        now = datetime.now(timezone.utc).isoformat()
        doc = {
            "synthesis_run_id": run_id,
            "status": "IN_PROGRESS",
            "stage": "started",
            "reason": reason,
            "min_danger": min_danger,
            "item_count": len(item_ids),
            "item_ids": item_ids,
            "previous_pulse_id": previous_pulse_id,
            "updated_at": now,
            "progress": {},
        }

        await self._mongo.async_update_one(
            self.COLLECTION_PULSE,
            {"_id": run_id},
            {
                "$set": doc,
                "$setOnInsert": {
                    "created_at": now,
                    "started_at": now,
                },
                "$inc": {"run_attempts": 1},
            },
            upsert=True,
        )

        log.info("StrategistStore: started Global Pulse run id=%s", run_id)
        return run_id

    async def update_pulse_run(
        self,
        run_id: str,
        *,
        stage: str,
        status: str = "IN_PROGRESS",
        fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the live Global Pulse document as synthesis progresses."""
        now = datetime.now(timezone.utc).isoformat()
        update_fields: Dict[str, Any] = {
            "stage": stage,
            "status": status,
            "updated_at": now,
            f"progress.{stage}.updated_at": now,
            f"progress.{stage}.status": status,
        }
        if status == "COMPLETED":
            update_fields["completed_at"] = now
            update_fields["timestamp"] = now

        for key, value in (fields or {}).items():
            update_fields[key] = value

        await self._mongo.async_update_one(
            self.COLLECTION_PULSE,
            {"_id": run_id},
            {"$set": update_fields},
            upsert=True,
        )

        log.info(
            "StrategistStore: updated Global Pulse run id=%s stage=%s status=%s",
            run_id,
            stage,
            status,
        )

    async def fetch_recoverable_failed_global_pulses(
        self,
        *,
        limit: int = 5,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """Fetch failed pulse runs whose story/domain stages are complete."""
        query = {
            "status": "FAILED",
            "$and": [
                {
                    "$or": [
                        {"stories": {"$exists": True, "$ne": []}},
                        {"domain_pulses": {"$exists": True, "$ne": {}}},
                    ],
                },
                {
                    "$or": [
                        {"global_retry_attempts": {"$exists": False}},
                        {"global_retry_attempts": {"$lt": max_retries}},
                    ],
                },
            ],
        }
        return await self._mongo.async_find_many(
            self.COLLECTION_PULSE,
            query,
            sort=[("updated_at", -1)],
            limit=limit,
        )

    async def mark_global_retry_start(self, run_id: str) -> None:
        """Mark a failed pulse as being retried at the final global stage."""
        now = datetime.now(timezone.utc).isoformat()
        await self._mongo.async_update_one(
            self.COLLECTION_PULSE,
            {"_id": run_id},
            {
                "$set": {
                    "status": "IN_PROGRESS",
                    "stage": "global_retry",
                    "updated_at": now,
                    "progress.global_retry.updated_at": now,
                    "progress.global_retry.status": "IN_PROGRESS",
                },
                "$inc": {"global_retry_attempts": 1},
            },
            upsert=False,
        )

    async def save_synthesis_checkpoint(
        self,
        *,
        run_id: str,
        stage: str,
        payload: Dict[str, Any],
    ) -> str:
        """Persist intermediate Strategist LLM outputs so completed calls are not lost."""
        now = datetime.now(timezone.utc).isoformat()
        checkpoint_id = f"{run_id}:{stage}"
        doc = {
            "_id": checkpoint_id,
            "run_id": run_id,
            "stage": stage,
            "payload": payload,
            "created_at": now,
            "updated_at": now,
        }

        await self._mongo.async_replace_one(
            self.COLLECTION_SYNTHESIS,
            {"_id": checkpoint_id},
            doc,
            upsert=True,
        )

        log.info(
            "StrategistStore: saved synthesis checkpoint run_id=%s stage=%s",
            run_id,
            stage,
        )
        return checkpoint_id

    async def get_latest_pulse(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent Global Pulse snapshot."""
        results = await self._mongo.async_find_many(
            self.COLLECTION_PULSE,
            filter_={"status": "COMPLETED"},
            sort=[("timestamp", -1), ("completed_at", -1), ("updated_at", -1)],
            limit=1
        )
        return results[0] if results else None

    async def fetch_market_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch the latest identified market opportunities (divergences)."""
        return await self._mongo.async_find_many(
            "market_opportunities",
            filter_={},
            sort=[("updated_at", -1), ("created_at", -1)],
            limit=limit
        )

    async def get_pulse_for_currencies(self, currencies: List[str], hours: int = 24) -> Optional[Dict[str, Any]]:
        """Retrieve the latest pulse that affected any of the specified currencies."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        query = {
            "timestamp": {"$gte": cutoff},
            "affected_currencies": {"$in": [c.upper() for c in currencies]}
        }
        
        results = await self._mongo.async_find_many(
            self.COLLECTION_PULSE,
            query,
            sort=[("timestamp", -1)],
            limit=1
        )
        return results[0] if results else None

    # ── BaseStore contract ────────────────────────────────────────────

    async def store_item(self, item: Any) -> str:
        """Alias for save_pulse (BaseStore contract)."""
        return await self.save_pulse(item)

    async def get_item(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """General retrieval (BaseStore contract)."""
        results = await self._mongo.async_find_many(
            self.COLLECTION_PULSE,
            query,
            sort=[("timestamp", -1)],
            limit=1
        )
        return results[0] if results else None
