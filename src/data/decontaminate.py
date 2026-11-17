"""Drop training pairs whose query or positive appears verbatim in an evaluation task.

    python -m src.data.decontaminate --src webfaq_full --out-name webfaq_clean

Collects every string of at least 20 characters (normalized: lowercase, punctuation to spaces) from
the MMTEB subset in src/eval/mmteb.py, then rewrites <PAIRS_DIR>/<src>/*.parquet without the pairs
that match. Exact-match only: it catches copied test text, not paraphrases.
"""

from __future__ import annotations

import argparse
import hashlib
import re

from .paths import use_local_hf_cache

use_local_hf_cache()
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ..eval import mmteb  # noqa: E402
from .build_pairs import PAIRS_DIR  # noqa: E402

MIN_CHARS = 20


def key(text: str) -> int:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return int.from_bytes(hashlib.blake2b(norm.encode(), digest_size=8).digest(), "big")


def collect(obj, out: set[int]) -> None:
    if isinstance(obj, str):
        if len(obj) >= MIN_CHARS:
            out.add(key(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect(v, out)
    elif hasattr(obj, "column_names"):  # a datasets.Dataset
        for name in obj.column_names:
            col = obj[name]
            if len(col) and isinstance(col[0], str):
                out.update(key(x) for x in col if len(x) >= MIN_CHARS)
            elif len(col) and isinstance(col[0], list):
                for x in col:
                    collect(x, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect(v, out)


def eval_keys() -> set[int]:
    names = [n for group in mmteb.SUBSET.values() for n in group]
    keys: set[int] = set()
    for task in mmteb.get_subset(names):
        task.load_data()
        collect(task.dataset, keys)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="webfaq_full")
    parser.add_argument("--out-name", default="webfaq_clean")
    args = parser.parse_args()

    keys = eval_keys()
    print(f"{len(keys):,} evaluation strings", flush=True)
    out_dir = PAIRS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    total = dropped = 0
    for path in sorted((PAIRS_DIR / args.src).glob("*.parquet")):
        table = pq.read_table(path)
        bad = [
            key(q) in keys or key(p) in keys
            for q, p in zip(table["query"].to_pylist(), table["positive"].to_pylist(), strict=True)
        ]
        clean = table.filter(pc.invert(pa.array(bad)))
        pq.write_table(clean, out_dir / path.name)
        total += len(table)
        dropped += len(table) - len(clean)
        print(f"{path.stem}: {len(table):,} -> {len(clean):,}", flush=True)
    print(f"total {total:,}, dropped {dropped:,}")


if __name__ == "__main__":
    main()
