"""OracleMongoStore — async MongoDB persistence for ia_oracle resolved decisions.

Collections:
    oracle_resolved  — one document per oracle decision (idempotent upsert by trigger_event_id)

Deduplication:
    trigger_event_id is the idempotent key.
    Re-processing the same event (e.g. after a crash) updates the existing document
    rather than creating a duplicate.

Usage::

    store = OracleMongoStore()
    await store.ensure_indexes()              # once at startup
    await store.store_item(response)          # per OracleReviewResponse
    item = await store.get_item({"trigger_event_id": event_id})
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from forex_shared.domain.oracle import OracleReviewResponse
from forex_shared.mongo_manager import MongoManager
from forex_shared.worker_api.store import BaseStore


class OracleMongoStore(BaseStore):
    """Async MongoDB persistence for OracleReviewResponse objects.

    Inherits from ``BaseStore`` (shared_lib) which provides ``self._mongo``
    (MongoManager singleton) and ``self.log`` (Loggable mixin).

    Each resolved oracle decision is stored in ``oracle_resolved`` collection.
    Re-processing the same trigger_event_id updates the document in place.
    """

    COLLECTION = "oracle_resolved"
    TAGS_COLLECTION = "global_tags"

    def __init__(self, mongo_manager: Optional[MongoManager] = None) -> None:
        super().__init__(mongo_manager)

    # ------------------------------------------------------------------
    # Index setup
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Idempotent index creation. Call once at service startup."""
        await self._mongo.async_ensure_indexes(
            self.COLLECTION,
            [
                [("trigger_event_id", 1)],      # primary lookup key
                [("action", 1)],                 # filter by EMIT / HOLD / DISCARD
                [("created_at", -1)],            # recency queries
                [("oracle_confidence", -1)],     # sort by confidence
            ],
        )
        await self._mongo.async_ensure_indexes(
            self.TAGS_COLLECTION,
            [
                [("trigger_event_id", 1), ("asset", 1)], # composite lookup key
                [("trigger_event_id", 1), ("record_type", 1)],
                [("action", 1)],
                [("asset", 1)],
                [("established_at", -1)],
            ],
        )
        self.log.debug("OracleMongoStore: indexes ensured on '%s'", self.COLLECTION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store_item(self, item: OracleReviewResponse) -> str:
        """Persist or update an OracleReviewResponse.

        Returns 'new' on insert, 'updated' on upsert of existing document.
        Idempotent: same trigger_event_id → updates in place.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Check if already exists
        existing = await self._mongo.async_find_many(
            self.COLLECTION,
            {"trigger_event_id": item.trigger_event_id},
            {"_id": 1},
            limit=1,
        )

        if existing:
            await self._mongo.async_update_one(
                self.COLLECTION,
                {"trigger_event_id": item.trigger_event_id},
                {
                    "$set": {
                        **self._build_fields(item),
                        "updated_at": now,
                    },
                    "$inc": {"update_count": 1},
                },
            )
            self.log.debug(
                "OracleMongoStore: updated  id=%s  action=%s",
                item.trigger_event_id, item.action,
            )
            return "updated"

        # Insert new document
        doc = {
            "_id":               str(uuid.uuid4()),
            "trigger_event_id":  item.trigger_event_id,
            "created_at":        now,
            "updated_at":        now,
            "update_count":      0,
            **self._build_fields(item),
        }
        await self._mongo.async_update_one(
            self.COLLECTION,
            {"trigger_event_id": item.trigger_event_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        self.log.info(
            "OracleMongoStore: stored  id=%s  action=%s  confidence=%.2f",
            item.trigger_event_id, item.action, item.oracle_confidence,
        )
        return "new"

    async def get_item(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a single oracle_resolved document matching ``query``."""
        results = await self._mongo.async_find_many(
            self.COLLECTION, query, limit=1
        )
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Global Tags API
    # ------------------------------------------------------------------

    async def store_global_tag(self, tag: Any) -> str:
        """Persist a GlobalTag.
        
        Idempotent by trigger_event_id and asset.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        doc = tag.to_dict() if hasattr(tag, 'to_dict') else tag.__dict__.copy()
        
        # Upsert composite key
        query = {
            "trigger_event_id": tag.trigger_event_id,
            "asset": tag.asset
        }
        
        await self._mongo.async_update_one(
            self.TAGS_COLLECTION,
            query,
            {
                "$set": {**doc, "updated_at": now},
                "$setOnInsert": {"created_at": now}
            },
            upsert=True,
        )
        
        self.log.info(
            "OracleMongoStore: stored global tag for asset=%s, trigger=%s",
            tag.asset, tag.trigger_event_id
        )
        return f"{tag.asset}_{tag.trigger_event_id}"

    async def store_oracle_decision_tag(self, response: OracleReviewResponse) -> str:
        """Persist every Oracle result into global_tags, including HOLD/DISCARD.

        Operational EMIT directives are still stored by ``store_global_tag``.
        This method stores the decision envelope itself so a successful Oracle
        return is never invisible just because no trade tag was emitted.
        """
        now = datetime.now(timezone.utc).isoformat()
        doc_id = f"oracle_decision:{response.trigger_event_id}"
        doc = {
            "record_type": "oracle_decision",
            "event_type": "ORACLE_REVIEW_RESOLVED",
            "trigger_event_id": response.trigger_event_id,
            "asset": "ORACLE_DECISION",
            "action": response.action,
            "active": False,
            "emitted": response.action == "EMIT",
            "oracle_confidence": response.oracle_confidence,
            "reasoning": response.reasoning,
            "tags_to_emit": response.tags_to_emit or [],
            "resolved_at": response.resolved_at,
            "updated_at": now,
        }

        await self._mongo.async_update_one(
            self.TAGS_COLLECTION,
            {"_id": doc_id},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": now},
                "$inc": {"update_count": 1},
            },
            upsert=True,
        )

        self.log.info(
            "OracleMongoStore: stored oracle decision tag id=%s action=%s",
            response.trigger_event_id,
            response.action,
        )
        return doc_id

    async def get_global_tag(self, trigger_event_id: str, asset: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single global_tag document."""
        results = await self._mongo.async_find_many(
            self.TAGS_COLLECTION,
            {"trigger_event_id": trigger_event_id, "asset": asset},
            limit=1
        )
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fields(item: OracleReviewResponse) -> dict:
        """Mutable fields written on both insert and update."""
        fields = {
            "action":           item.action,
            "oracle_confidence": item.oracle_confidence,
            "reasoning":        item.reasoning,
            "tags_to_emit":     item.tags_to_emit or [],
        }
        if item.observability != {}:
            fields["observability"] = item.to_dict()["observability"]
        return fields
