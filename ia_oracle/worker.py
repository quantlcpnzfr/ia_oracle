"""OracleWorker — IA Oracle MQ worker (production).

Pipeline:
    intel.oracle.review  ->  [IA provider]  ->  intel.oracle.resolved  (audit)
                                             ->  intel.global_tags       (if EMIT, one GlobalTag per directive)

The GlobalTag published to intel.global_tags is consumed by:
    - session_manager   → injects intel bias into strategy additional_data
    - executor_trading  → applies confidence penalty / blocks contra-trend orders
    - api_gateway       → WebSocket streaming of active tags

Architecture matches EventDrivenTradingSession:
    - One MQ connection (own instance via MQFactory)
    - Subscribes to input topic  → _handle_review()
    - Publishes to two output topics inside the same handler

Usage (via main.py):
    python ia_oracle/main.py --worker-id oracle_1 --max-sessions 1

Or directly:
    from ia_oracle.worker import OracleWorker
    worker = OracleWorker()
    await worker.start()
    await worker.run_forever()
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from forex_shared.domain.intel import GlobalTag
from forex_shared.domain.oracle import OracleReviewRequest, OracleReviewResponse
from forex_shared.logging.loggable import Loggable
from ia_oracle.store import OracleMongoStore
from ia_oracle.strategist_store import StrategistStore
from forex_shared.providers.mq.mq_factory import MQFactory
from forex_shared.providers.mq.topics import IntelTopics
from forex_shared.providers.cache.redis_provider import RedisProvider
from forex_shared.worker_api.ia_factory import IAProviderFactory
from ia_oracle.providers.ollama_provider import OllamaProvider
from ia_oracle.providers.gemini_provider import GeminiIAProvider
from ia_oracle.providers.openai_provider import OpenAIIAProvider

# ── topic constants (via shared IntelTopics) ──────────────────────────────────
INPUT_TOPIC  = IntelTopics.ORACLE_REVIEW    # intel.oracle.review
OUTPUT_TOPIC = IntelTopics.ORACLE_RESOLVED  # intel.oracle.resolved
TAGS_TOPIC   = IntelTopics.GLOBAL_TAGS      # intel.global_tags

# ── default TTL when directive has no volatility_duration_minutes ─────────────
_DEFAULT_TAG_TTL_MINUTES = 240  # 4 hours

# ── prompt mapping ────────────────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent / "prompts"

_PROMPT_MAP = {
    "trend":       "ia_trend_oracle.md",
    "trend_fast":  "ia_trend_oracle_fast.md",
    "summarizer":  "ia_summarizer.md",
    "fundamental": "ia_fundamental_oracle.md",
    "interest_rate": "ia_interest_rate_oracle.md",
}


def _load_system_prompt(prompt_type: str = "trend") -> str:
    filename = _PROMPT_MAP.get(prompt_type, "ia_trend_oracle.md")
    path = _PROMPTS_DIR / filename
    
    if path.exists():
        return path.read_text(encoding="utf-8")
    
    # Fallback for trend
    if prompt_type == "trend":
        return (
            "You are an expert geopolitical and financial analyst. "
            "Analyse the event provided and return a JSON object with keys: "
            "action (EMIT|DISCARD|HOLD), oracle_confidence (0.0-1.0), "
            "reasoning (string), tags_to_emit (list)."
        )
    # Fallback for interest_rate
    if prompt_type == "interest_rate":
        return (
            "You are a Central Bank Policy Analyst. Analyze the interest rate event "
            "and determine the Guidance (HAWKISH|DOVISH|NEUTRAL). "
            "Return a JSON object with bias, confidence, and global_tag."
        )
    # Generic fallback
    return "You are a helpful AI assistant. Return your answer in JSON format."


def _build_user_prompt(req: OracleReviewRequest) -> str:
    return json.dumps(
        {
            "trigger_event_id":   req.trigger_event_id,
            "title":              req.title,
            "body":               req.body,
            "analysis_summary":   req.analysis_summary,
            "forex_impact":       req.forex_impact,
            "macro_context":      req.macro_context,
            "domain_context":     req.domain_context,
            "regional_context":    req.regional_context,
            "domain":             req.domain,
            "source":             req.source,
            "reason":             req.reason,
            "scores":             req.scores,
            "trade_emit_score":   req.trade_emit_score,
            "candidate_directives": req.candidate_directives,
            "market_context":     req.market_context,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_response(raw: str, event_id: str) -> OracleReviewResponse:
    """Parse LLM JSON response → OracleReviewResponse.

    Supports two schemas:
      - ia_trend_oracle_v1 (nested: oracle_decision, directives, audit)
      - Legacy flat schema (action, oracle_confidence, reasoning)
    """
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)

        # ── ia_trend_oracle_v1 schema ─────────────────────────────────────
        if data.get("oracle_version") == "ia_trend_oracle_v1" or "oracle_decision" in data:
            decision   = data.get("oracle_decision", {})
            directives = data.get("directives", [])
            audit      = data.get("audit", {})

            # action
            if decision.get("ignore"):
                action = "DISCARD"
            elif decision.get("review_only"):
                action = "HOLD"
            elif decision.get("create_tag"):
                action = "EMIT"
            else:
                action = "DISCARD"

            # confidence — average of directive confidences, or oracle_review_score
            if directives:
                confidence = sum(float(d.get("confidence", 0.0)) for d in directives) / len(directives)
            else:
                confidence = float(
                    data.get("input_assessment", {}).get("oracle_review_score", 0.0)
                )

            reasoning = str(
                audit.get("reasoning_summary")
                or decision.get("decision_reason")
                or ""
            )

            tags_to_emit = [
                {
                    "asset":                      d.get("asset", ""),
                    "bias":                       d.get("bias", ""),
                    "confidence":                 float(d.get("confidence", 0.0)),
                    "risk_score":                 float(d.get("risk_score", 0.0)),
                    "volatility_duration_minutes": int(d.get("volatility_duration_minutes", 0)),
                    "reason":                     d.get("reason", ""),
                    "transmission_channel":       d.get("transmission_channel", ""),
                }
                for d in directives
            ]

            return OracleReviewResponse(
                trigger_event_id=event_id,
                action=action,
                oracle_confidence=round(min(max(confidence, 0.0), 1.0), 4),
                reasoning=reasoning,
                tags_to_emit=tags_to_emit,
            )

        # ── Legacy flat schema ────────────────────────────────────────────
        return OracleReviewResponse(
            trigger_event_id=event_id,
            action=data.get("action", "DISCARD").upper(),
            oracle_confidence=float(data.get("oracle_confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            tags_to_emit=data.get("tags_to_emit", []),
        )

    except Exception as exc:
        return OracleReviewResponse(
            trigger_event_id=event_id,
            action="DISCARD",
            oracle_confidence=0.0,
            reasoning=f"[parse error] {exc} — raw: {raw[:200]}",
        )


def _build_global_tags(
    response: OracleReviewResponse,
    *,
    domain: str = "",
    source: str = "ia_oracle",
) -> List[GlobalTag]:
    """Convert OracleReviewResponse directives → list of GlobalTag objects.

    Only called when action == EMIT and tags_to_emit is non-empty.
    TTL defaults to _DEFAULT_TAG_TTL_MINUTES when directive has no duration.
    """
    now = datetime.now(timezone.utc)
    tags: List[GlobalTag] = []

    for directive in response.tags_to_emit:
        asset = directive.get("asset", "").strip()
        if not asset:
            continue

        ttl_minutes = int(directive.get("volatility_duration_minutes", 0)) or _DEFAULT_TAG_TTL_MINUTES
        expires_at  = (now + timedelta(minutes=ttl_minutes)).isoformat()

        tags.append(
            GlobalTag(
                asset=asset,
                bias=directive.get("bias", "neutral"),
                risk_score=float(directive.get("risk_score", response.oracle_confidence)),
                trigger_event_id=response.trigger_event_id,
                established_at=now.isoformat(),
                expires_at=expires_at,
                active=True,
                domain=directive.get("domain") or domain,
                source=directive.get("source") or source,
                observability=deepcopy(response.observability),
            )
        )

    return tags


class OracleWorker(Loggable):
    """MQ-driven IA Oracle worker — production class.

    Subscribes to ``intel.oracle.review``, processes with the configured IA provider, and:

    1. Publishes resolved decision to ``intel.oracle.resolved``  (audit / monitoring)
    2. If action == EMIT: publishes one ``GlobalTag`` per directive to
       ``intel.global_tags``  → consumed by session_manager & executor_trading

    Lifecycle::

        worker = OracleWorker(worker_id="oracle_1", max_sessions=2)
        await worker.start()
        await worker.run_forever()   # blocks; Ctrl-C / SIGTERM triggers stop()
        await worker.stop()
    """

    def __init__(
        self,
        worker_id: str = "oracle_worker_1",
        max_sessions: int = 1,
        store: Optional[OracleMongoStore] = None,
        output_file: Optional[str | Path] = None,
        reset_output_file: bool = False,
    ) -> None:
        self.worker_id    = worker_id
        self.max_sessions = max_sessions
        self._mq          = None
        self._provider    = None
        self._store: Optional[OracleMongoStore] = store  # None = MongoDB disabled
        self._strategist_store: Optional[StrategistStore] = None
        self._semaphore:  Optional[asyncio.Semaphore] = None
        self._stop_event  = asyncio.Event()
        self._output_file = Path(output_file) if output_file else None
        self._reset_output_file = reset_output_file
        self._output_file_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self.log.info("[OracleWorker:%s] Starting...", self.worker_id)
        
        import os
        from pathlib import Path
        audit_dir = os.environ.get("TEST_AUDIT_DIR")
        if audit_dir and not self._output_file:
            self._output_file = Path(audit_dir) / "intel_item_oracle.json"
            
        await self._prepare_output_file()

        # ── Register Local Providers ──────────────────────────────────
        IAProviderFactory.register_provider("OLLAMA", OllamaProvider)
        IAProviderFactory.register_provider("GEMINI", GeminiIAProvider)
        IAProviderFactory.register_provider("OPENAI_NATIVE", OpenAIIAProvider)

        # ── IA provider ───────────────────────────────────────────────
        self._provider = IAProviderFactory.create_from_env(ollama_profile="oracle")
        await self._provider.initialize()

        # ── Strategist store ──────────────────────────────────────────
        try:
            self._strategist_store = StrategistStore()
            await self._strategist_store.ensure_indexes()
        except Exception as e:
            self.log.warning("[OracleWorker] StrategistStore unavailable: %s", e)

        # self._provider is initialized; no default prompt set here anymore
        # as it is now per-request.

        self.log.info(
            "[OracleWorker:%s] Provider ready: %s | model=%s",
            self.worker_id,
            self._provider.provider_type,
            self._provider.model_name,
        )

        # ── MQ — one connection, both subscribe and publish ───────────
        self._mq = MQFactory.create_async_from_env()
        await self._mq.connect()
        self.log.info("[OracleWorker:%s] MQ connected.", self.worker_id)

        await self._mq.subscribe_event(INPUT_TOPIC, self._handle_review, prefetch_count=self.max_sessions)
        self.log.info("[OracleWorker:%s] Subscribed to %s", self.worker_id, INPUT_TOPIC)

        self._semaphore = asyncio.Semaphore(self.max_sessions)
        self.log.info("[OracleWorker:%s] Ready. concurrency=%d", self.worker_id, self.max_sessions)

    async def stop(self) -> None:
        self.log.info("[OracleWorker:%s] Stopping...", self.worker_id)
        self._stop_event.set()
        if self._mq:
            try:
                await self._mq.stop_consuming()
            except Exception:
                pass
            try:
                await self._mq.disconnect()
            except Exception:
                pass
        if self._provider:
            try:
                await self._provider.close()
            except Exception:
                pass
        self.log.info("[OracleWorker:%s] Stopped.", self.worker_id)

    async def run_forever(self) -> None:
        """Block until stop() is called (SIGTERM / KeyboardInterrupt)."""
        self.log.info("[OracleWorker:%s] Running. Waiting for messages...", self.worker_id)
        consume_task = asyncio.create_task(self._mq.start_consuming())
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            consume_task.cancel()
            await asyncio.gather(consume_task, return_exceptions=True)

    # ------------------------------------------------------------------
    # MQ handler — core production logic
    # ------------------------------------------------------------------

    async def _handle_review(self, payload: Dict[str, Any]) -> None:
        """Called by the MQ consumer for each intel.oracle.review message.

        Flow:
            1. Deserialize OracleReviewRequest from payload
            2. Build user prompt
            3. Call the configured IA provider
            4. Parse response → OracleReviewResponse
            5. Publish audit to intel.oracle.resolved
            6. If action == EMIT: publish GlobalTag(s) to intel.global_tags
        """
        req      = OracleReviewRequest.from_dict(payload)
        event_id = req.trigger_event_id

        self.log.info(
            "[OracleWorker:%s] Received review  id=%s  title=%.80s",
            self.worker_id,
            event_id,
            req.title or "(no title)",
        )

        async with self._semaphore:
            try:
                # ── Step 2-4: prompt -> IA provider -> parse ─────────
                
                # Fetch latest Global Pulse to provide macro awareness (MongoDB is the System of Record)
                pulse = None
                
                # 1. MongoDB Priority (The GOLD source)
                if self._strategist_store:
                    try:
                        pulse = await self._strategist_store.get_latest_pulse()
                        if pulse:
                            self.log.debug("[OracleWorker] Fetched macro context from MongoDB (System of Record).")
                    except Exception as e:
                        self.log.warning("[OracleWorker] MongoDB pulse lookup failed: %s", e)

                # 2. Redis Fallback (Superficial/Ephemeral Cache)
                if not pulse:
                    try:
                        redis = await RedisProvider.shared_from_env()
                        pulse = await redis.get_json("global_pulse:latest")
                        if pulse:
                            self.log.debug("[OracleWorker] Fetched macro context from Redis (Fallback).")
                    except Exception as re_e:
                        self.log.debug("[OracleWorker] Redis pulse lookup failed (ignoring): %s", re_e)

                if pulse:
                    # Global context (fallback)
                    req.macro_context = pulse.get("bluf", "Stable global market conditions.")
                    
                    # Domain context (Zoom-in)
                    domain_pulses = pulse.get("domain_pulses", {})
                    # Try exact match or case-insensitive match
                    target_domain = (req.domain or "").lower()
                    domain_sitrep = domain_pulses.get(target_domain)
                    if not domain_sitrep:
                        # Fallback: check if target_domain is a substring of any key (e.g. 'economic' in 'economic_us')
                        for k, v in domain_pulses.items():
                            if target_domain in k:
                                domain_sitrep = v
                                break
                    
                    req.domain_context = domain_sitrep if domain_sitrep else req.macro_context
                    
                    # Regional context
                    regional = pulse.get("regional_highlights", {})
                    # If the event has a country, check that region
                    # For now, we join all highlights as a summary of hotspots
                    hotspots = []
                    for reg, highlights in regional.items():
                        if highlights:
                            hotspots.append(f"{reg.upper()}: {', '.join(highlights)}")
                    
                    req.regional_context = " | ".join(hotspots) if hotspots else "No major regional hotspots identified."
                
                # --- Grounding with Market Opportunities (Technical Context) ---
                if self._strategist_store:
                    try:
                        opps = await self._strategist_store.fetch_market_opportunities(limit=5)
                        if opps:
                            lines = []
                            for o in opps:
                                lines.append(
                                    f"{o['symbol']}: Div={o['divergence']:+.2f} "
                                    f"(Base:{o['base_strength']:.2f} Quote:{o['quote_strength']:.2f})"
                                )
                            req.market_context = " | ".join(lines)
                            self.log.debug("[OracleWorker] Injected Market Opportunities context.")
                    except Exception as me:
                        self.log.warning("[OracleWorker] Market context lookup failed: %s", me)

                prompt_type = req.prompt_type or "trend"
                if (
                    prompt_type == "trend"
                    and getattr(self._provider, "provider_type", "").upper() == "OLLAMA"
                ):
                    prompt_type = "trend_fast"
                system_prompt = _load_system_prompt(prompt_type)
                user_prompt = _build_user_prompt(req)
                
                self.log.info(
                    "[OracleWorker:%s] Calling %s for id=%s  model=%s  prompt=%s",
                    self.worker_id,
                    self._provider.provider_type,
                    event_id,
                    self._provider.model_name,
                    prompt_type,
                )

                raw      = await self._provider.generate(user_prompt, system_prompt=system_prompt)
                response = _parse_response(raw, event_id)
                # Provenance comes from the received request, never the LLM.
                response.observability = deepcopy(req.observability)

                self.log.info(
                    "[OracleWorker:%s] Resolved  id=%s  action=%s  confidence=%.2f  reasoning=%.100s",
                    self.worker_id,
                    event_id,
                    response.action,
                    response.oracle_confidence,
                    response.reasoning,
                )

                # ── Step 5: publish audit ─────────────────────────────
                await self._publish_resolved(response)

                # ── Step 6: emit GlobalTag(s) if EMIT ────────────────
                if response.action == "EMIT":
                    await self._emit_global_tags(
                        response,
                        domain=req.domain,
                        source=req.source or "ia_oracle",
                    )

                # ── Step 7: persist to MongoDB (optional) ─────────────
                await self._persist(response)

            except Exception as exc:
                self.log.error(
                    "[OracleWorker:%s] Error processing id=%s: %s",
                    self.worker_id,
                    event_id,
                    exc,
                    exc_info=True,
                )

    async def _publish_resolved(self, response: OracleReviewResponse) -> None:
        """Publish resolved decision to intel.oracle.resolved (audit topic)."""
        payload = response.to_dict()
        ok = await self._mq.publish_event(OUTPUT_TOPIC, payload)
        if ok:
            self.log.info(
                "[OracleWorker:%s] Published resolved  id=%s  topic=%s",
                self.worker_id,
                response.trigger_event_id,
                OUTPUT_TOPIC,
            )
            await self._write_output_result(payload)
        else:
            self.log.warning(
                "[OracleWorker:%s] Failed to publish resolved id=%s",
                self.worker_id,
                response.trigger_event_id,
            )

    async def _prepare_output_file(self) -> None:
        """Initialize the optional validation output file as a JSON array."""
        if self._output_file is None:
            return

        path = self._output_file

        def _prepare() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._reset_output_file or not path.exists() or path.stat().st_size == 0:
                path.write_text("[]\n", encoding="utf-8")

        try:
            await asyncio.to_thread(_prepare)
            self.log.info(
                "[OracleWorker:%s] Output validation file enabled: %s",
                self.worker_id,
                path,
            )
        except Exception as exc:
            self.log.warning(
                "[OracleWorker:%s] Failed to prepare output file %s: %s",
                self.worker_id,
                path,
                exc,
            )

    async def _write_output_result(self, payload: Dict[str, Any]) -> None:
        """Append one resolved response to the validation output JSON file."""
        if self._output_file is None:
            return

        async with self._output_file_lock:
            path = self._output_file

            def _append() -> int:
                path.parent.mkdir(parents=True, exist_ok=True)
                existing: list[dict[str, Any]] = []
                if path.exists() and path.stat().st_size > 0:
                    try:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(loaded, list):
                            existing = loaded
                    except json.JSONDecodeError:
                        backup = path.with_suffix(path.suffix + ".invalid")
                        path.replace(backup)

                existing.append(payload)
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                tmp.replace(path)
                return len(existing)

            try:
                count = await asyncio.to_thread(_append)
                self.log.info(
                    "[OracleWorker:%s] Output file updated  count=%d  path=%s",
                    self.worker_id,
                    count,
                    path,
                )
            except Exception as exc:
                self.log.warning(
                    "[OracleWorker:%s] Failed to update output file %s: %s",
                    self.worker_id,
                    path,
                    exc,
                )

    async def _emit_global_tags(
        self,
        response: OracleReviewResponse,
        *,
        domain: str = "",
        source: str = "ia_oracle",
    ) -> None:
        """Build GlobalTag(s) from EMIT directives and publish to intel.global_tags.

        Each directive in tags_to_emit becomes one GlobalTag.
        Directives without an 'asset' field are silently skipped.
        TTL = directive.volatility_duration_minutes (default: 240 min / 4h).
        """
        tags = _build_global_tags(response, domain=domain, source=source)
        if not tags:
            self.log.debug(
                "[OracleWorker:%s] EMIT with no emittable directives for id=%s",
                self.worker_id,
                response.trigger_event_id,
            )
            return

        for tag in tags:
            payload = tag.to_mq_payload(event_type="GLOBAL_TAG_UPDATED")
            ok = await self._mq.publish_event(TAGS_TOPIC, payload)
            if ok:
                self.log.info(
                    "[OracleWorker:%s] GlobalTag emitted  asset=%s  bias=%s  "
                    "risk=%.2f  ttl_until=%s  topic=%s",
                    self.worker_id,
                    tag.asset,
                    tag.bias,
                    tag.risk_score,
                    tag.expires_at,
                    TAGS_TOPIC,
                )
                
                # TEST: Persistir e recuperar GlobalTag no MongoDB
                if getattr(self, "_store", None):
                    try:
                        await self._store.store_global_tag(tag)
                        saved_tag = await self._store.get_global_tag(tag.trigger_event_id, tag.asset)
                        self.log.info(
                            "[OracleWorker:%s] MongoDB GlobalTag saved! Retrieved (stored in mongodb): %s",
                            self.worker_id,
                            saved_tag,
                        )
                    except Exception as exc:
                        self.log.warning(
                            "[OracleWorker:%s] Failed to persist GlobalTag to MongoDB: %s",
                            self.worker_id, exc
                        )
            else:
                self.log.warning(
                    "[OracleWorker:%s] Failed to emit GlobalTag for asset=%s",
                    self.worker_id,
                    tag.asset,
                )

    async def _persist(self, response: OracleReviewResponse) -> None:
        """Persist resolved response to MongoDB (graceful — skipped if store is None).

        MongoDB failure logs a warning but never interrupts the MQ pipeline.
        Store is set at startup by main.py; None means MongoDB is disabled/unavailable.
        """
        if self._store is None:
            return
        try:
            outcome = await self._store.store_item(response)
            decision_tag_id = await self._store.store_oracle_decision_tag(response)
            
            # TEST: Lendo de volta do MongoDB para validar a gravação
            saved_item = await self._store.get_item({"trigger_event_id": response.trigger_event_id})
            
            self.log.info(
                "[OracleWorker:%s] MongoDB %s id=%s decision_tag=%s. Retrieved (stored in mongodb): %s",
                self.worker_id,
                outcome,
                response.trigger_event_id,
                decision_tag_id,
                saved_item.get("action") if saved_item else "NOT FOUND",
            )
        except Exception as exc:
            self.log.warning(
                "[OracleWorker:%s] MongoDB persistence failed for id=%s: %s",
                self.worker_id,
                response.trigger_event_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Compat shim for main.py (which calls create_session on broker mode)
    # ------------------------------------------------------------------

    async def create_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """No-op — OracleWorker has a single built-in session."""
        self.log.debug(
            "[OracleWorker:%s] create_session called (no-op): %s",
            self.worker_id,
            payload,
        )
        return {"status": "ok", "session_id": payload.get("session_id", "default")}
