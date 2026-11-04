#!/usr/bin/env python3
"""Fast structural and import smoke test for a simple_baselines checkout."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "Bio",
    "torch",
    "transformers",
    "tokenizers",
    "loralib",
)

EXPECTED_DATASETS = (
    "aamyl",
    "casp",
    "gcn4",
    "gfp",
    "his2",
    "his5",
    "lov",
    "nmt",
    "pard3",
    "pte",
    "trpb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail when the separately distributed data payload is absent.",
    )
    parser.add_argument(
        "--model-check",
        action="store_true",
        help="Load the bundled ESM2 8M model and run a small CPU forward pass.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_structure(strict_data: bool) -> None:
    for relative in ("code", "notebooks", "models", "scripts"):
        require((ROOT / relative).is_dir(), f"missing directory: {relative}")

    scripts = sorted(CODE.glob("*.py"))
    notebooks = sorted((ROOT / "notebooks").glob("lean_*.ipynb"))
    require(len(scripts) == 35, f"expected 35 Python modules, found {len(scripts)}")
    require(len(notebooks) == 7, f"expected 7 lean notebooks, found {len(notebooks)}")

    missing_data = []
    for dataset in EXPECTED_DATASETS:
        filename = "gfp_dataset_10mut.csv" if dataset == "gfp" else f"{dataset}.csv"
        path = ROOT / "data" / dataset / filename
        if not path.is_file():
            missing_data.append(str(path.relative_to(ROOT)))

    if missing_data and strict_data:
        raise RuntimeError("missing dataset files: " + ", ".join(missing_data))
    if missing_data:
        print("[warning] external data payload is incomplete: " + ", ".join(missing_data))
    else:
        print(f"[ok] data payload: {len(EXPECTED_DATASETS)} representative datasets")


def check_imports() -> None:
    for name in REQUIRED_PACKAGES:
        importlib.import_module(name)
    print(f"[ok] dependencies: {len(REQUIRED_PACKAGES)} imports")

    import utils
    import utils_for_analysis

    require(Path(utils.root_dataset_path) == ROOT / "data" / "merged_datasets", "utils root path is not portable")
    require(utils.DATASET_PATHS["gfp"] == str(ROOT / "data" / "gfp" / "gfp_dataset_10mut.csv"), "GFP path is not portable")
    require(utils_for_analysis.DATASET_PATHS is utils.DATASET_PATHS, "analysis shim does not expose utils")
    print("[ok] project imports and repository-relative paths")


def check_sources() -> None:
    for path in sorted(CODE.glob("*.py")):
        compile(path.read_bytes(), str(path), "exec")
    print("[ok] Python syntax: all code modules")


def check_model() -> None:
    import torch
    from plm_base import load_model, plm_init

    plm_init(str(ROOT))
    wrapper = load_model("esm2_t6_8M_UR50D")
    tokens = torch.tensor([wrapper.get_encode()("ACDE")], dtype=torch.long)
    with torch.no_grad():
        logits, hidden = wrapper.get_forward()(tokens)
    require(logits.shape[:2] == tokens.shape, "unexpected ESM2 logits shape")
    require(hidden.shape[:2] == tokens.shape, "unexpected ESM2 hidden-state shape")
    print("[ok] local ESM2 8M checkpoint and CPU forward pass")


def main() -> None:
    args = parse_args()
    check_structure(args.strict_data)
    check_imports()
    check_sources()
    if args.model_check:
        check_model()
    print("[ok] simple_baselines verification complete")


if __name__ == "__main__":
    main()
