"""
finetune_embedder.py — Fine-tunes all-MiniLM-L6-v2 on TAMU ECE domain data.

Uses MultipleNegativesRankingLoss: treats all other (query, passage) pairs in
the batch as negatives — no manual negative mining needed.

Run on HPRC:
    python finetune_embedder.py --data ../crawler/training_pairs.jsonl

Run locally (MPS/CPU):
    python finetune_embedder.py --data ../crawler/training_pairs.jsonl --epochs 1 --batch 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_training_pairs(path: str) -> list[InputExample]:
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            query    = row.get("query", "").strip()
            positive = row.get("positive", "").strip()
            if query and positive:
                examples.append(InputExample(texts=[query, positive]))
    log.info("Loaded %d training examples", len(examples))
    return examples


def main(args: argparse.Namespace) -> None:
    examples = load_training_pairs(args.data)

    if not examples:
        log.error("No training examples found. Run generate_training_data.py first.")
        return

    log.info("Loading base model: %s", args.base_model)
    model = SentenceTransformer(args.base_model)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch)
    loss = MultipleNegativesRankingLoss(model)

    output_path = args.output
    Path(output_path).mkdir(parents=True, exist_ok=True)

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    log.info(
        "Training: %d examples, %d epochs, batch=%d, warmup=%d steps",
        len(examples), args.epochs, args.batch, warmup_steps,
    )

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=os.path.join(output_path, "checkpoints"),
        checkpoint_save_steps=500,
    )

    log.info("Fine-tuned model saved to: %s", output_path)
    log.info(
        "Next step: set EMBEDDING_MODEL=%s in your .env and re-run python ingest.py",
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,  help="Path to training_pairs.jsonl")
    parser.add_argument("--base-model", default="all-MiniLM-L6-v2", help="Base embedding model")
    parser.add_argument("--output",     default="./tamu-ece-embedder", help="Output model directory")
    parser.add_argument("--epochs",     type=int,   default=3,  help="Training epochs")
    parser.add_argument("--batch",      type=int,   default=64, help="Batch size (use 128+ on A100)")
    args = parser.parse_args()
    main(args)
