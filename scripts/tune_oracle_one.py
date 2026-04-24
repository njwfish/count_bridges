"""Single-eval worker: runs one (seed, nfe, kind, sigma, lam) oracle run.

Designed to be spawned many in parallel. Outputs a per-eval JSON so results
can be merged after.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tune_oracle_bridge import build_dataset, run_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--nfe", type=int, required=True)
    p.add_argument("--kind", choices=["fixed", "adaptive"], required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    assert args.kind == "fixed" or args.lam is not None, "adaptive needs --lam"

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "outputs" / "tune_oracle_grid"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = dict(seed=args.seed, nfe=args.nfe, kind=args.kind,
                sigma=args.sigma, lam=args.lam, n_samples=args.n_samples)
    tag = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    lam_str = "na" if args.lam is None else f"{args.lam:g}"
    fname = f"{args.kind}_s{args.seed}_nfe{args.nfe}_sig{args.sigma:g}_lam{lam_str}_{tag}.json"
    out_path = out_dir / fname

    if out_path.exists():
        print(f"SKIP (exists): {fname}")
        return

    t0 = time.time()
    dataset = build_dataset(seed=args.seed)
    e, dt = run_eval(dataset, args.kind, args.sigma, args.lam,
                     n_steps=args.nfe, n_samples=args.n_samples, device="cuda",
                     seed=0)
    total = time.time() - t0
    result = dict(**spec, energy_distance=e, seconds=dt, total_seconds=total,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    out_path.write_text(json.dumps(result, indent=2))
    lam_display = 'na' if args.lam is None else f'{args.lam:g}'
    print(f"DONE: {args.kind} sig={args.sigma:g} lam={lam_display} "
          f"seed={args.seed} nfe={args.nfe}  E={e:.4f}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
