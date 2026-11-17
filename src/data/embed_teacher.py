"""Embed the distillation corpus with a teacher and cache the vectors.

    python -m src.data.embed_teacher --corpus distill_3M --teacher Qwen/Qwen3-Embedding-0.6B

Reads <RAW_DIR>/distill/<corpus>/*.parquet (src.data.build_distill_corpus) and writes, per
language file, <RAW_DIR>/distill/<corpus>/<tag>/<lang>.npy: float16, shape (rows, --dim), each row
the teacher's vector cut to its first --dim dimensions and re-normalized. Matryoshka prefixes of the
student then match same-size prefixes of this vector (re-normalized). Texts are embedded as
documents, without the teacher's query instruction. Resumable: a finished language is skipped, and
an interrupted one restarts from its last 50k-row part file.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .paths import RAW_DIR, use_local_hf_cache

use_local_hf_cache()
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

PART = 50_000


def save_atomic(path, array) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.rename(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="distill_3M")
    parser.add_argument("--teacher", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--tag", default="q06b", help="output folder name for this teacher")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--max-len", type=int, default=256, help="teacher tokens per text")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--langs", nargs="*")
    args = parser.parse_args()

    root = RAW_DIR / "distill" / args.corpus
    out_dir = root / args.tag
    out_dir.mkdir(exist_ok=True)
    model = SentenceTransformer(
        args.teacher, model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda"
    )
    model.max_seq_length = args.max_len
    writer = ThreadPoolExecutor(max_workers=1)
    files = sorted(root.glob("*.parquet"))
    for path in files:
        code = path.stem
        if args.langs and code not in args.langs:
            continue
        final = out_dir / f"{code}.npy"
        if final.exists():
            print(f"{code}: done, skipping")
            continue
        texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
        parts, futures = [], []
        start = time.time()
        for i in range(0, len(texts), PART):
            part = out_dir / f"{code}.part{i // PART:04d}.npy"
            parts.append(part)
            if part.exists():
                continue
            emb = model.encode(
                texts[i : i + PART],
                batch_size=args.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )[:, : args.dim].float()
            emb = torch.nn.functional.normalize(emb, dim=-1).half().cpu().numpy()
            futures.append(writer.submit(save_atomic, part, emb))  # the HDD can be slow; keep embedding
            done = min(i + PART, len(texts))
            print(f"  {code}: {done:,}/{len(texts):,} ({done / (time.time() - start):.0f}/s incl. resume)", flush=True)
        for f in futures:
            f.result()
        futures.clear()
        np.save(final, np.concatenate([np.load(p) for p in parts]))
        for p in parts:
            p.unlink()
        print(f"{code}: {len(texts):,} texts -> {final}", flush=True)


if __name__ == "__main__":
    main()
