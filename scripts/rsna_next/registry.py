"""Atomic run registry and paid-budget reservations (no cloud provisioning)."""
import argparse
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

from .contracts import assert_budget, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["status", "record", "reserve", "settle"])
    p.add_argument("--registry", type=Path, default=Path("experiments/next_registry.json"))
    p.add_argument("--run-id")
    p.add_argument("--state", choices=["planned", "running", "failed", "validated", "rejected", "blocked"])
    p.add_argument("--artifact", type=Path)
    p.add_argument("--usd", type=float, default=0)
    p.add_argument("--note", default="")
    args = p.parse_args()
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    with args.registry.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(args.registry.read_text()) if args.registry.exists() else {
            "budget_cap_usd": 500, "paid_jobs": [], "runs": {}, "fallback_submission": "56077025"}
        if args.action != "status" and not args.run_id:
            raise ValueError("run-id required")
        if args.action == "reserve":
            if any(r["run_id"] == args.run_id for r in data["paid_jobs"]):
                raise ValueError("Budget reservation already exists")
            assert_budget(data["paid_jobs"], args.usd, data["budget_cap_usd"])
            data["paid_jobs"].append({"run_id": args.run_id, "reserved_usd": args.usd, "actual_usd": 0})
        elif args.action == "settle":
            matches = [r for r in data["paid_jobs"] if r["run_id"] == args.run_id]
            if len(matches) != 1 or args.usd < 0:
                raise ValueError("A matching reservation and nonnegative actual cost are required")
            entry = matches[0]
            # Record actual costs even when an overrun has already occurred.
            entry.update(reserved_usd=0, actual_usd=args.usd)
        elif args.action == "record":
            if not args.state:
                raise ValueError("State required")
            data["runs"][args.run_id] = {"state": args.state, "note": args.note,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "artifact": str(args.artifact) if args.artifact else None,
                "artifact_sha256": sha256(args.artifact) if args.artifact else None}
        if args.action != "status":
            write_json(args.registry, data)
        remaining = data["budget_cap_usd"] - sum(r["actual_usd"] + r["reserved_usd"] for r in data["paid_jobs"])
        print(json.dumps({"remaining_usd": remaining, "runs": data["runs"]}, indent=2))


if __name__ == "__main__":
    main()
