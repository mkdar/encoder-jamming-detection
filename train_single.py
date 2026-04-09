from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from jammer_sim import CLASS_NAMES, OFDMJammingGenerator
from models import MLPBaseline, TransformerEncoderClassifier


torch.set_num_threads(1)


def batch_to_tensors(batch):
    return torch.from_numpy(batch.x), torch.from_numpy(batch.y)


def evaluate(model, batch, batch_size=1024):
    model.eval()
    x, y = batch_to_tensors(batch)
    preds = []
    losses = []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            xb = x[i : i + batch_size]
            yb = y[i : i + batch_size]
            logits = model(xb)
            losses.append(F.cross_entropy(logits, yb).item())
            preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        "loss": float(np.mean(losses)),
        "acc": float((pred == y_np).mean()),
        "macro_f1": float(f1_score(y_np, pred, average="macro")),
        "pred": pred,
        "true": y_np,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "mlp"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    generator = OFDMJammingGenerator()
    input_dim = generator.feature_dim()
    seq_len = generator.seq_len
    num_classes = len(CLASS_NAMES)

    if args.model == "transformer":
        model = TransformerEncoderClassifier(input_dim=input_dim, num_classes=num_classes)
        lr = 3e-4
    else:
        model = MLPBaseline(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes)
        lr = 5e-4

    val_batch = generator.make_fixed_set(args.val_samples, seed=10_000 + args.seed, balanced=True)
    test_batch = generator.make_fixed_set(args.test_samples, seed=20_000 + args.seed, balanced=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    rows = []
    steps_per_epoch = math.ceil(args.train_samples / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + 1000 * epoch)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for step in range(steps_per_epoch):
            cur_bs = min(args.batch_size, args.train_samples - step * args.batch_size)
            labels = np.arange(cur_bs, dtype=np.int64) % num_classes
            rng.shuffle(labels)
            batch = generator.generate(cur_bs, rng, force_labels=labels)
            xb, yb = batch_to_tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * cur_bs
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += cur_bs
        scheduler.step()
        val_metrics = evaluate(model, val_batch)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_seen,
            "train_acc": total_correct / total_seen,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        rows.append(row)
        print(f"[{args.model}][seed={args.seed}] epoch {epoch}/{args.epochs} train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / f"{args.model}_seed{args.seed}_history.csv", index=False)
    test_metrics = evaluate(model, test_batch)
    np.save(args.output_dir / f"{args.model}_seed{args.seed}_pred.npy", test_metrics["pred"])
    np.save(args.output_dir / f"{args.model}_seed{args.seed}_true.npy", test_metrics["true"])
    np.save(args.output_dir / f"{args.model}_seed{args.seed}_test_snr.npy", test_batch.snr_db)

    snr_rows = []
    for snr_db in [0, 4, 8, 12, 16, 20, 24]:
        snr_batch = generator.make_fixed_set(5000, seed=30_000 + args.seed + int(snr_db), snr_db=snr_db, balanced=True)
        snr_metrics = evaluate(model, snr_batch)
        snr_rows.append({"snr_db": snr_db, "acc": snr_metrics["acc"], "macro_f1": snr_metrics["macro_f1"]})
    pd.DataFrame(snr_rows).to_csv(args.output_dir / f"{args.model}_seed{args.seed}_snr.csv", index=False)

    metrics = {
        "model": args.model,
        "seed": args.seed,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "confusion_matrix": confusion_matrix(test_metrics["true"], test_metrics["pred"]).tolist(),
    }
    with open(args.output_dir / f"{args.model}_seed{args.seed}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
