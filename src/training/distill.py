"""Distil cached teacher embeddings into the embedding model.

    python -m src.training.distill --config configs/train/distill_50m.yaml --run-name distill-a
    python -m src.training.distill --config configs/train/distill_50m.yaml --run-name smoke \\
        --set schedule.total_texts=20000 data.corpus=smoke

Starts from a pretrained encoder (`init_from`) and regresses the student's embedding onto the
teacher's vector for the same text: 1 - cosine, averaged over the Matryoshka dims, where the
teacher target for dim d is its first d dimensions re-normalized (the cache holds the first 768).
Batches mix all languages. Optionally adds an MSE between the student's and the teacher's
similarity matrices inside each chunk (`loss.sim_weight`). Resumes from <RUNS_DIR>/<run>/latest.pt;
batches depend only on (seed, step). latest.pt has the same format as the contrastive trainer's,
so src/eval/mmteb.py loads it directly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from src.data.paths import RAW_DIR, RUNS_DIR, TOKENIZED_DIR
from src.model import EmbeddingModel, load_model_config

from .contrastive import DEVICE, chunks, encode, load_encoder, lr_at, tokenize
from .data import Prefetcher
from .mlm import load_config


class DistillData:
    """All corpus texts with their teacher vectors, and a held-out validation slice."""

    def __init__(self, corpus: str, teacher: str, val_texts: int, seed: int) -> None:
        root = RAW_DIR / "distill" / corpus
        texts, langs, vecs = [], [], []
        for path in sorted(root.glob("*.parquet")):
            t = pq.read_table(path, columns=["text"])["text"].to_pylist()
            v = np.load(root / teacher / f"{path.stem}.npy", mmap_mode="r")
            if len(v) != len(t):
                raise SystemExit(f"{path.stem}: {len(t)} texts but {len(v)} teacher vectors")
            texts += t
            langs += [path.stem] * len(t)
            vecs.append(v)
        self.texts, self.langs = texts, langs
        self.vecs = np.concatenate(vecs)  # fp16 (N, dim); about 1.5 KB per text
        self.seed = seed
        perm = np.random.default_rng([seed, 11]).permutation(len(texts))
        self.val_idx, self.train_idx = perm[:val_texts], perm[val_texts:]

    def order(self, epoch: int) -> np.ndarray:
        return self.train_idx[np.random.default_rng([self.seed, epoch, 7]).permutation(len(self.train_idx))]

    def batch(self, step: int, size: int) -> tuple[list[str], np.ndarray]:
        per_epoch = len(self.train_idx) // size  # whole batches only; the remainder is skipped
        epoch, within = divmod(step, per_epoch)
        idx = np.sort(self.order(epoch)[within * size : (within + 1) * size])
        return [self.texts[i] for i in idx], self.vecs[idx]


def distill_loss(student, teacher, dims, sim_weight):
    """student (n, D) raw, teacher (n, T) unit vectors; returns (loss, per-dim cosine at the first dim)."""
    total, cos0 = 0.0, None
    for d in dims:
        s = F.normalize(student[:, :d], dim=-1)
        t = F.normalize(teacher[:, :d], dim=-1)
        cos = (s * t).sum(dim=-1)
        total = total + (1 - cos).mean()
        if cos0 is None:
            cos0 = cos.detach().mean()
    total = total / len(dims)
    if sim_weight > 0:
        s = F.normalize(student, dim=-1)
        t = F.normalize(teacher[:, : student.size(1)], dim=-1)
        total = total + sim_weight * F.mse_loss(s @ s.T, t @ t.T)
    return total, cos0


def train_step(model, tok, texts, vecs, cfg):
    """One optimizer step's forward/backward over the batch in chunks. Returns (loss, cosine, tokens)."""
    n, chunk = len(texts), cfg["batch"]["chunk_texts"]
    dims, sw = model.cfg.matryoshka_dims, cfg["loss"]["sim_weight"]
    seqs = tokenize(tok, texts, cfg["data"]["max_len"])
    target = torch.from_numpy(vecs).to(DEVICE).float()
    loss_sum, cos_sum = 0.0, 0.0
    for i in range(0, n, chunk):
        emb = encode(model, seqs[i : i + chunk], cfg["data"]["width"])
        loss, cos = distill_loss(emb.float(), target[i : i + chunk], dims, sw)
        (loss * len(emb) / n).backward()
        loss_sum += loss.item() * len(emb) / n
        cos_sum += cos.item() * len(emb) / n
    return loss_sum, cos_sum, sum(map(len, seqs))


@torch.no_grad()
def validate(model, data: DistillData, tok, cfg) -> dict[str, float]:
    """Mean cosine to the teacher on held-out texts, at 384 and 64 dims, overall and per language."""
    model.eval()
    idx = data.val_idx
    seqs = tokenize(tok, [data.texts[i] for i in idx], cfg["data"]["max_len"])
    embs = torch.cat([encode(model, s, cfg["data"]["width"]).float() for s in chunks(seqs, 128)])
    target = torch.from_numpy(data.vecs[idx]).to(DEVICE).float()
    out = {}
    for d in (embs.size(1), 64):
        s, t = F.normalize(embs[:, :d], dim=-1), F.normalize(target[:, :d], dim=-1)
        cos = (s * t).sum(dim=-1).cpu().numpy()
        out[f"cos{d}"] = float(cos.mean())
        if d == embs.size(1):
            for lang in sorted({data.langs[i] for i in idx}):
                mask = np.array([data.langs[i] == lang for i in idx])
                out[f"{lang}"] = float(cos[mask].mean())
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.manual_seed(cfg["seed"])

    model_cfg = load_model_config(cfg["model"])
    model = EmbeddingModel(model_cfg).to(DEVICE)
    if cfg.get("init_from"):
        load_encoder(model, cfg["init_from"])
    model.train()
    o = cfg["optim"]
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": o["weight_decay"]}, {"params": other, "weight_decay": 0.0}],
        lr=o["lr"],
        betas=tuple(o["betas"]),
    )

    tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    d = cfg["data"]
    data = DistillData(d["corpus"], d["teacher"], d["val_texts"], cfg["seed"])
    batch_size = cfg["batch"]["texts"]
    total_steps = math.ceil(cfg["schedule"]["total_texts"] / batch_size)
    warmup = max(1, int(cfg["schedule"]["warmup_fraction"] * total_steps))

    start_step = 0
    latest = run_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}")
    print(
        f"{model_cfg.summary()}\nsteps {total_steps} x {batch_size} texts, corpus "
        f"{len(data.texts):,} texts ({len(data.train_idx):,} train), teacher dim {data.vecs.shape[1]}, "
        f"matryoshka {model_cfg.matryoshka_dims}",
        flush=True,
    )

    def batches():
        for step in range(start_step, total_steps):
            yield (step, *data.batch(step, batch_size))

    log_path = run_dir / "log.jsonl"

    def log(rec: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    eval_every, save_every = cfg["eval"]["every_steps"], cfg["checkpoint"]["every_steps"]
    window_start, window_loss, window_cos, window_tokens, n = time.time(), 0.0, 0.0, 0, 0
    for step, texts, vecs in Prefetcher(batches()):
        lr = lr_at(step, total_steps, o["lr"], warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss, cos, tokens = train_step(model, tok, texts, vecs, cfg)
        torch.nn.utils.clip_grad_norm_(model.parameters(), o["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        window_loss, window_cos, window_tokens, n = (
            window_loss + loss, window_cos + cos, window_tokens + tokens, n + 1
        )  # fmt: skip
        done = step + 1
        if done % cfg["log_every_steps"] == 0 or done == total_steps:
            rec = {
                "type": "train", "step": done, "texts": done * batch_size, "loss": window_loss / n,
                "cos": window_cos / n, "lr": lr, "tok_per_s": window_tokens / (time.time() - window_start),
            }  # fmt: skip
            log(rec)
            print(
                f"step {done}/{total_steps} loss {rec['loss']:.4f} cos384 {rec['cos']:.4f} "
                f"lr {lr:.2e} {rec['tok_per_s'] / 1e3:.0f}k tok/s",
                flush=True,
            )
            window_start, window_loss, window_cos, window_tokens, n = time.time(), 0.0, 0.0, 0, 0
        if done % eval_every == 0 or done == total_steps:
            res = validate(model, data, tok, cfg)
            log({"type": "eval", "step": done, "cos": res})
            print(f"  validation cosine to teacher: " + " ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
        if done % save_every == 0 or done == total_steps:
            tmp = Path(str(latest) + ".tmp")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": done}, tmp)
            tmp.rename(latest)


if __name__ == "__main__":
    main()
