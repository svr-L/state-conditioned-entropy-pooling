from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def design_hash(design: dict) -> str:
    payload = json.dumps(design, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def open_lockbox(design: dict, results_summary: dict, stamp_path, *, force: bool = False) -> dict:
    """Record one opening of a frozen research design.

    A later opening under a different design hash is refused unless force=True,
    which is itself recorded. This mirrors the audit-control logic added in v24.
    """
    stamp_path = Path(stamp_path)
    h = design_hash(design)
    stamp = {
        "design_hash": h,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": design,
        "results_summary": results_summary,
        "forced": bool(force),
    }
    if stamp_path.exists():
        prior = json.loads(stamp_path.read_text(encoding="utf-8"))
        if prior.get("design_hash") != h and not force:
            raise RuntimeError(
                "Lockbox already opened under a different design hash. "
                "Opening a new design would invalidate the original out-of-sample claim."
            )
        history = prior.get("history", []) + [{k: v for k, v in prior.items() if k != "history"}]
    else:
        history = []
    stamp["history"] = history
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    return stamp
