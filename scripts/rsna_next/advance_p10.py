"""One-shot coordinator: verify completed prerequisites, then launch the fixed wave.

Only private free Kaggle jobs are launched. No cloud provisioning or submissions.
Reruns verify registry/build hashes and do not re-push a recorded accepted run.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .contracts import sha256, write_json


def command(args):
    result = subprocess.run(["rtk", "proxy", *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-4000:])
    return result.stdout


def record(run_id, state, artifact, note):
    command([sys.executable, "-m", "scripts.rsna_next.registry", "record", "--run-id", run_id,
             "--state", state, "--artifact", str(artifact), "--note", note])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poll-seconds", type=int, default=45)
    p.add_argument("--timeout-hours", type=float, default=3)
    args = p.parse_args()
    started = time.monotonic()
    jobs = {"verify": ("rsna-knee-p10-duplicate-replay", Path("outputs/kaggle/rsna-p10-duplicate-replay-v1")),
            "imagenet": ("rsna-knee-next-student-imagenet-report-0-smoke", Path("outputs/kaggle/rsna-p12-imagenet-smoke-v1"))}
    completed, previous = set(), {}
    while len(completed) != len(jobs):
        for name, (slug, output) in jobs.items():
            if name in completed:
                continue
            status = command(["kaggle", "kernels", "status", "vallepuvamsikrishna/" + slug])
            if status != previous.get(name):
                print(status.strip(), flush=True)
                previous[name] = status
            if "COMPLETE" in status:
                command(["kaggle", "kernels", "output", "vallepuvamsikrishna/" + slug, "-p", str(output)])
                completed.add(name)
                print(f"Downloaded {name}", flush=True)
            elif any(s in status for s in ("ERROR", "CANCEL", "FAILED")):
                command(["kaggle", "kernels", "output", "vallepuvamsikrishna/" + slug, "-p", str(output)])
                raise RuntimeError(f"Prerequisite {name} failed; inspect {output}")
        if len(completed) < len(jobs):
            if time.monotonic() - started > args.timeout_hours * 3600:
                raise TimeoutError("Prerequisite wait exceeded its limit; no full jobs launched")
            time.sleep(max(10, args.poll_seconds))

    verify_root = jobs["verify"][1]
    verification = verify_root / "duplicate_verification.json"
    replay_path = verify_root / "replay.json"
    replay = json.loads(replay_path.read_text())
    smoke_root = Path("outputs/kaggle/rsna-p10-teacher-smoke-v1/next_run")
    if (replay["status"] != "replay_validated" or replay["rows"] != 12 or
            replay["run_manifest_sha256"] != sha256(smoke_root / "manifest.json") or
            replay["checkpoint_sha256"] != sha256(smoke_root / "best.pt") or
            replay["predictions_sha256"] != sha256(smoke_root / "predictions.csv")):
        raise ValueError("Original smoke replay provenance failed")
    command([sys.executable, "-m", "scripts.rsna_next.repair_split", "--pixel-matches",
             "outputs/kaggle/rsna-p10-pixel-audit-v1/result/exact_selected_pixel_matches.csv",
             "--verification", str(verification), "--output", "outputs/experiments/P10-contract-v2"])
    split_root = Path("outputs/experiments/P10-contract-v2")
    split_manifest = json.loads((split_root / "manifest.json").read_text())
    write_json("experiments/results/P10-contract-v2.json", {k: v for k, v in split_manifest.items() if k != "image_pairs"})
    write_json("experiments/results/P10-smoke-replay.json", replay)
    raw = json.loads(verification.read_text())
    write_json("experiments/results/P10-duplicate-verification.json", {
        "status": raw["status"], "full_decoded_study_equal": raw["full_decoded_study_equal"],
        "studies": len(raw["studies"]), "decoded_slices": [r["decoded_slices"] for r in raw["studies"]],
        "reconstructed_cache_matches": [r["reconstructed_cache_match"] for r in raw["studies"]],
        "source_sha256": sha256(verification), "note": "Decoded pixel equality, not patient identity"})
    record("P10-DUPLICATE-REPLAY", "validated", "experiments/results/P10-smoke-replay.json", "Raw duplicate reconstructed and all 12 smoke predictions replayed")
    record("P10-SPLITS-V2", "validated", "experiments/results/P10-contract-v2.json", "Verified report/image split; 3909 pseudo train; shared validation and expert folds preserved")
    record("P10-PIXEL-AUDIT", "validated", "experiments/results/P10-duplicate-verification.json", "Raw-image overlap verified; training twin excluded in v2; full-training split gate passed")

    image_root = jobs["imagenet"][1] / "next_run"
    command([sys.executable, "-m", "scripts.rsna_next.validate", "--roots", str(image_root), "--splits",
             str(split_root / "splits.csv"), "--train", "data/raw/train.csv", "--smoke", "--output",
             "outputs/experiments/P12-imagenet-smoke-validation"])
    ir = json.loads((image_root / "replay.json").read_text())
    if ir["status"] != "replay_validated" or ir["run_manifest_sha256"] != sha256(image_root / "manifest.json"):
        raise ValueError("ImageNet smoke replay failed")
    record("P12-IMAGENET-SMOKE", "validated", image_root / "replay.json", "Corrected-split ImageNet smoke passed artifact checks and full image replay")

    wave = [("P10-TEACHERS-012", "teacher", [0, 1, 2]),
            ("P10-TEACHERS-34", "teacher", [3, 4]),
            ("P12-FASTMRI-BASELINE-F0", "student", [0])]
    launched = []
    for run_id, mode, folds in wave:
        registry = json.loads(Path("experiments/next_registry.json").read_text())
        if run_id in registry["runs"] and registry["runs"][run_id]["state"] in {"running", "validated"}:
            entry = registry["runs"][run_id]
            if sha256(entry["artifact"]) != entry["artifact_sha256"]:
                raise ValueError(f"Recorded launch artifact changed: {run_id}")
            launched.append({"run_id": run_id, "status": "already_recorded"})
            continue
        bundle = Path("kaggle") / run_id.lower()
        build = [sys.executable, "scripts/build_next_kernels.py", "training", "--recipe", "fastmri-report",
                 "--mode", mode, "--seed", "20260917", "--splits", str(split_root / "splits.csv"),
                 "--smoke-evidence", "outputs/experiments/P10-teacher-smoke-validation/validation.json",
                 "--replay-evidence", str(replay_path), "--output", str(bundle)]
        build += ["--folds", *map(str, folds)] if mode == "teacher" else ["--fold", "0"]
        command(build)
        result = command(["kaggle", "kernels", "push", "-p", str(bundle)])
        if "successfully pushed" not in result:
            raise RuntimeError(f"Uncertain push outcome for {run_id}: inspect before retry")
        metadata = json.loads((bundle / "kernel-metadata.json").read_text())
        record(run_id, "running", bundle / "build.json", f"Private Kaggle accepted: {metadata['id']}; folds={folds}; not yet validated")
        launched.append({"run_id": run_id, "slug": metadata["id"], "folds": folds, "status": "accepted"})
        write_json("experiments/results/P10-training-wave.json", {"status": "launching", "jobs": launched, "paid_usd": 0})
        print(result.strip(), flush=True)
    write_json("experiments/results/P10-training-wave.json", {"status": "launched", "jobs": launched, "paid_usd": 0})
    print("All three full-training jobs accepted; five teacher folds and baseline fold 0.", flush=True)


if __name__ == "__main__":
    main()
