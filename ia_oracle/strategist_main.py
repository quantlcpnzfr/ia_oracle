"""Entrypoint for the Strategist daemon within the ia_oracle service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from forex_shared.env_config_manager import EnvConfigManager
from forex_shared.logging.get_logger import get_logger, setup_logging


async def run(args: argparse.Namespace) -> int:
    from ia_oracle.strategist_store import StrategistStore
    from ia_oracle.strategist_worker import StrategistWorker

    logger = get_logger(__name__)
    worker = StrategistWorker(
        store=StrategistStore(),
        worker_id=args.worker_id,
        model_override=args.model or None,
        emit_downstream_actions=args.emit_downstream_actions,
    )
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_requested.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop_requested.set))

    try:
        await worker.start()
        logger.info("STRATEGIST_READY worker_id=%s", args.worker_id)
        result = await worker.perform_full_synthesis(reason="STARTUP_PULSE")
        logger.info("STRATEGIST_STARTUP_PULSE_RESULT status=%s", result)
        if args.once:
            return 0 if result == "COMPLETED" else 2
        await stop_requested.wait()
        return 0
    finally:
        await worker.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ia_oracle Strategist daemon.")
    parser.add_argument("--worker-id", default="strategist_1")
    parser.add_argument("--model", default="")
    parser.add_argument("--once", action="store_true", help="Run one synthesis and exit.")
    parser.add_argument(
        "--emit-downstream-actions",
        action="store_true",
        help="Enable legacy GlobalTag and trading.signal publication after synthesis.",
    )
    args = parser.parse_args()
    setup_logging(level=logging.INFO)
    EnvConfigManager.startup()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
