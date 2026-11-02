# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: AutoTrainer Engine (High-Level Zero-Boilerplate Model Training)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad AutoTrainer:
Simplifies deep learning training into an intuitive, zero-boilerplate interface.
Handles dataset batching, optimizer stepping, learning rate scheduling, mixed-precision,
progress telemetry, and model checkpointing automatically.
"""

import time
import os
import math
from typing import Optional, Union, List, Dict, Any, Callable

from .tensor import Tensor, randn, zeros
from . import nn
from . import optim
from . import functional as F
from .utils.data import Dataset, TensorDataset, DataLoader

class AutoTrainer:
    """
    High-level, user-friendly Trainer that automates the entire training loop.
    Users can train models in 3 lines of code without writing boilerplate loops.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        epochs: int = 5,
        batch_size: int = 16,
        mixed_precision: bool = False,
        checkpoint_dir: str = "./checkpoints",
        on_epoch_end: Optional[Callable[[int, float, float], None]] = None,
    ):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = checkpoint_dir
        self.on_epoch_end = on_epoch_end

        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer

        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        self.scaler = optim.GradScaler() if mixed_precision else None
        self.history: List[Dict[str, Any]] = []

    def fit(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """
        Executes the end-to-end training loop with live progress telemetry.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        total_batches = len(loader)

        print("\n" + "=" * 80)
        print("  VELOGRAD AUTOTRAINER: Commencing High-Performance Training Session")
        print("=" * 80)
        print(f"  Total Epochs:      {self.epochs}")
        print(f"  Batch Size:        {self.batch_size}")
        print(f"  Dataset Samples:   {len(dataset)}")
        print(f"  Total Batches:     {total_batches}")
        print(f"  Mixed Precision:   {'BF16 / Scaled' if self.mixed_precision else 'FP32'}")
        print(f"  Optimizer:         {self.optimizer.__class__.__name__} (lr={self.optimizer.lr})")
        print("-" * 80)

        start_total_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.perf_counter()

            for b_idx, batch in enumerate(loader):
                inputs_rows = [s[0] for s in batch]
                targets_rows = [s[1] for s in batch]

                b_sz = len(inputs_rows)
                in_dim = len(inputs_rows[0])
                tgt_dim = len(targets_rows[0])

                inp_tensor = Tensor((b_sz, in_dim), data=[float(x) for row in inputs_rows for x in row])
                tgt_tensor = Tensor((b_sz, tgt_dim), data=[float(x) for row in targets_rows for x in row])

                self.optimizer.zero_grad()
                outputs = self.model(inp_tensor)
                loss = self.loss_fn(outputs, tgt_tensor)

                if self.scaler is not None:
                    scaled = self.scaler.scale_loss(loss)
                    scaled.backward()
                    self.scaler.step(self.optimizer)
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_val = float(loss.contiguous()._data[0])
                epoch_loss += loss_val

                # Progress bar display
                progress = (b_idx + 1) / total_batches
                bar_len = 24
                filled = int(bar_len * progress)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\r  Epoch {epoch:02d}/{self.epochs:02d} [{bar}] Batch {b_idx + 1}/{total_batches} | Step Loss: {loss_val:.4f}", end="", flush=True)

            avg_loss = epoch_loss / max(1, total_batches)
            epoch_time = (time.perf_counter() - epoch_start) * 1000.0

            log_entry = {
                "epoch": epoch,
                "loss": avg_loss,
                "epoch_time_ms": epoch_time,
            }
            self.history.append(log_entry)

            print(f"\r  Epoch {epoch:02d}/{self.epochs:02d} | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f} ms")

            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_loss, epoch_time)

        total_elapsed = time.perf_counter() - start_total_time
        print("-" * 80)
        print(f"  Training Session Completed Successfully in {total_elapsed:.2f} seconds!")
        print(f"  Initial Loss: {self.history[0]['loss']:.6f} -> Final Loss: {self.history[-1]['loss']:.6f}")
        print("=" * 80 + "\n")

        # Save checkpoint
        self.save_checkpoint(os.path.join(self.checkpoint_dir, "model_final.json"))
        return self.history

    def save_checkpoint(self, path: str):
        """Saves model parameters to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model.state_dict(), f)
        print(f"  [Checkpoint] Model weights successfully saved to: {path}")
