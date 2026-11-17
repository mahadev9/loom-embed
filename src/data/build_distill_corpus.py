"""Sample the text corpus that the teacher embeds for distillation (no labels needed).

    python -m src.data.build_distill_corpus --per-language 250000 --out-name distill_3M

Per language, three kinds of text, written to <RAW_DIR>/distill/<out-name>/<lang>.parquet with
columns text, lang, kind, pair_id:
  web     passages decoded from the tokenized FineWeb2 training documents: a random document, cut to a
          log-uniform length of 24-256 tokens from its start (so the lengths are mixed)
  q, a    the question and answer of a random WebFAQ pair (from pairs/<pairs-name>, after its first
          `--skip` validation rows); both texts share a pair_id so the pair can be checked later
          (a question the teacher scores as unrelated to its answer is probably spam)
Texts are deduplicated within a language. The mix is --web-fraction web passages, the rest pairs.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from .paths import RAW_DIR, TOKENIZED_DIR, use_local_hf_cache

use_local_hf_cache()
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from .build_pairs import PAIRS_DIR, WEBFAQ  # noqa: E402

CLS_ID, SEP_ID = 2, 3
WINDOW = 262_144  # tokens read per random window
PER_WINDOW = 16  # documents taken from each window


def sample_web(code: str, n: int, tok: Tokenizer, rng: random.Random) -> list[str]:
    """Random 1 MB windows of the token file, `PER_WINDOW` documents from each (random reads on the
    HDD cost about 13 ms each, so one document per read would take half an hour per language)."""
    data = np.memmap(TOKENIZED_DIR / code / "train.bin", dtype="<u4", mode="r")
    out, seen = [], set()
    while len(out) < n:
        off = rng.randrange(0, len(data) - WINDOW)
        w = np.asarray(data[off : off + WINDOW])
        starts = np.flatnonzero(w == CLS_ID)
        ends = np.flatnonzero(w == SEP_ID)
        if len(starts) < 2:
            continue
        for i in rng.sample(range(len(starts) - 1), min(PER_WINDOW, len(starts) - 1)):
            s0 = int(starts[i]) + 1
            e0 = ends[np.searchsorted(ends, s0)] if np.searchsorted(ends, s0) < len(ends) else s0 + 256
            length = int(np.exp(rng.uniform(np.log(24), np.log(256))))
            text = tok.decode(w[s0 : min(e0, s0 + length)].tolist(), skip_special_tokens=True).strip()
            if len(text) < 30 or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-language", type=int, default=250_000, help="texts per language")
    parser.add_argument("--web-fraction", type=float, default=0.6)
    parser.add_argument("--pairs-name", default="webfaq_clean")
    parser.add_argument("--skip", type=int, default=1000, help="validation rows at file start")
    parser.add_argument("--langs", nargs="+", default=list(WEBFAQ))
    parser.add_argument("--out-name", default="distill_3M")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
    out_dir = RAW_DIR / "distill" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in args.langs:
        out = out_dir / f"{code}.parquet"
        if out.exists():
            print(f"{code}: exists, skipping")
            continue
        rng = random.Random(f"{args.seed}-{code}")
        n_web = int(args.per_language * args.web_fraction)
        n_pairs = (args.per_language - n_web) // 2
        web = sample_web(code, n_web, tok, rng)
        table = pq.read_table(PAIRS_DIR / args.pairs_name / f"{code}.parquet", columns=["query", "positive"])
        idx = rng.sample(range(args.skip, len(table)), min(n_pairs, len(table) - args.skip))
        big = pa.large_string()  # plain string columns overflow on take() for the large languages
        q = table["query"].cast(big).take(idx).to_pylist()
        p = table["positive"].cast(big).take(idx).to_pylist()
        texts = web + q + p
        kinds = ["web"] * len(web) + ["q"] * len(q) + ["a"] * len(p)
        pair_ids = [-1] * len(web) + list(range(len(q))) * 2
        keep, seen = [], set()
        for i, t in enumerate(texts):
            if t not in seen:
                seen.add(t)
                keep.append(i)
        pq.write_table(
            pa.table(
                {
                    "text": [texts[i] for i in keep],
                    "lang": [code] * len(keep),
                    "kind": [kinds[i] for i in keep],
                    "pair_id": [pair_ids[i] for i in keep],
                }
            ),
            out,
        )
        print(f"{code}: {len(keep):,} texts ({len(web):,} web, {len(q):,} pairs) -> {out}", flush=True)


if __name__ == "__main__":
    main()
