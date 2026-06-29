import logging
from pathlib import Path
from typing import Any, override

import torch
from lightning import Callback, Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import nn

from src.modules.nets.vision_transformer import EncoderBlock, MLPBlock, VisionTransformer

log = logging.getLogger(__name__)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def prune_mlp_block(mlp: MLPBlock, amount: float, min_hidden: int = 64) -> tuple[MLPBlock, int, int]:
    """Physically prune the MLP expansion dimension by removing whole hidden units."""
    fc1 = mlp[0]
    fc2 = mlp[3]
    tail_dropout = mlp[4]

    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError("Expected MLPBlock with Linear layers at indices 0 and 3")

    in_dim = fc1.in_features
    old_hidden = fc1.out_features
    new_hidden = max(min_hidden, int(old_hidden * (1.0 - amount)))
    if new_hidden >= old_hidden:
        return mlp, old_hidden, old_hidden

    dropout_p = tail_dropout.p if isinstance(tail_dropout, nn.Dropout) else 0.0
    device = fc1.weight.device
    dtype = fc1.weight.dtype
    norms = fc1.weight.detach().norm(dim=1)
    keep = torch.topk(norms, k=new_hidden, largest=True).indices.sort().values

    new_mlp = MLPBlock(in_dim, new_hidden, dropout_p).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_mlp[0].weight.copy_(fc1.weight[keep])
        new_mlp[0].bias.copy_(fc1.bias[keep])
        new_mlp[3].weight.copy_(fc2.weight[:, keep])
        new_mlp[3].bias.copy_(fc2.bias)

    return new_mlp, old_hidden, new_hidden


def prune_vit_mlp_blocks(net: VisionTransformer, amount: float, min_hidden: int = 64) -> dict[str, int]:
    """Replace each encoder MLP with a physically narrower block."""
    old_hidden: int | None = None
    new_hidden: int | None = None
    blocks_pruned = 0

    for module in net.encoder.layers:
        if not isinstance(module, EncoderBlock):
            continue
        new_mlp, block_old, block_new = prune_mlp_block(module.mlp, amount, min_hidden)
        module.mlp = new_mlp
        old_hidden = block_old
        new_hidden = block_new
        blocks_pruned += 1

    if old_hidden is None or new_hidden is None:
        raise ValueError("No encoder MLP blocks found to prune")

    return {
        "blocks_pruned": blocks_pruned,
        "old_mlp_hidden": old_hidden,
        "new_mlp_hidden": new_hidden,
        "prune_amount": amount,
    }


def rebind_optimizers(trainer: Trainer, pl_module: LightningModule) -> None:
    """Rebuild the optimizer after parameter shapes change, keeping the current LR."""
    old_optimizer = trainer.optimizers[0]
    current_lr = old_optimizer.param_groups[0]["lr"]
    weight_decay = old_optimizer.param_groups[0].get("weight_decay", 0.0)

    new_optimizer = pl_module.hparams.optimizer(params=pl_module.parameters())
    for group in new_optimizer.param_groups:
        group["lr"] = current_lr
        group["weight_decay"] = weight_decay

    trainer.optimizers = [new_optimizer]
    trainer.strategy.optimizers = [new_optimizer]

    for config in trainer.lr_scheduler_configs:
        scheduler = config.scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
            for sub_scheduler in scheduler._schedulers:
                sub_scheduler.optimizer = new_optimizer
        else:
            scheduler.optimizer = new_optimizer


def build_prune_amount_schedule(
    prune_at_steps: list[int], mlp_prune_amount: float | list[float], mlp_prune_amount_increment: float | None = None
) -> dict[int, float]:
    """Map each prune step to the fraction of MLP hidden units removed at that step."""
    if isinstance(mlp_prune_amount, list):
        if len(mlp_prune_amount) != len(prune_at_steps):
            raise MisconfigurationException(
                "When `mlp_prune_amount` is a list it must have the same length as "
                f"`prune_at_steps` ({len(prune_at_steps)}), got {len(mlp_prune_amount)}."
            )
        return dict(zip(prune_at_steps, mlp_prune_amount, strict=True))

    if mlp_prune_amount_increment is not None:
        return {
            step: mlp_prune_amount + index * mlp_prune_amount_increment for index, step in enumerate(prune_at_steps)
        }

    return dict.fromkeys(prune_at_steps, mlp_prune_amount)


class ViTProgressiveMLPPruning(Callback):
    """Physically prune ViT MLP hidden units at scheduled training steps.

    Supports multiple prune events with per-step amounts. Each amount is applied to
    the **current** MLP width (successive prunes compound).

    Amount schedule options:
    - ``mlp_prune_amount: 0.12`` — same fraction at every step
    - ``mlp_prune_amount: [0.08, 0.10, 0.12]`` — explicit per-step fractions
    - ``mlp_prune_amount: 0.08`` + ``mlp_prune_amount_increment: 0.04``
      → 0.08 at first step, 0.12 at second, 0.16 at third, ...

    When ``defer_after_checkpoint`` is True (default), pruning runs one step after
    each target so a step checkpoint at N can be saved first (e.g. checkpoint at 10k,
    prune at 10,001).

    Set ``save_step_checkpoints_every_n`` to save ``step-<N>.ckpt`` at every multiple
    of N training steps, including steps with no pruning (e.g. 30k when pruning at
    10k / 20k / 40k).
    """

    def __init__(
        self,
        prune_at_steps: int | list[int] = 40_000,
        mlp_prune_amount: float | list[float] = 0.12,
        mlp_prune_amount_increment: float | None = None,
        min_hidden: int = 64,
        reinit_optimizer: bool = True,
        defer_after_checkpoint: bool = True,
        save_pre_prune_checkpoint: bool = False,
        save_step_checkpoints_every_n: int | None = 10_000,
    ) -> None:
        if isinstance(prune_at_steps, int):
            prune_at_steps = [prune_at_steps]
        self._prune_at_steps = sorted(prune_at_steps)
        self._amount_by_step = build_prune_amount_schedule(
            self._prune_at_steps, mlp_prune_amount, mlp_prune_amount_increment
        )
        self._min_hidden = min_hidden
        self._reinit_optimizer = reinit_optimizer
        self._defer_after_checkpoint = defer_after_checkpoint
        self._save_pre_prune_checkpoint = save_pre_prune_checkpoint
        self._save_step_checkpoints_every_n = save_step_checkpoints_every_n
        self._applied_steps: set[int] = set()

    def _trigger_step(self, target_step: int) -> int:
        return target_step + int(self._defer_after_checkpoint)

    def _pending_target(self, global_step: int) -> int | None:
        for target in self._prune_at_steps:
            if target in self._applied_steps:
                continue
            if global_step == self._trigger_step(target):
                return target
        return None

    @override
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step
        if self._save_step_checkpoints_every_n and step > 0 and step % self._save_step_checkpoints_every_n == 0:
            self._save_step_checkpoint(trainer, step)

        target = self._pending_target(step)
        if target is None:
            return

        self._applied_steps.add(target)
        self._apply_pruning(trainer, pl_module, target)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"applied_steps": sorted(self._applied_steps)}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._applied_steps = set(state_dict.get("applied_steps", []))

    def _checkpoint_dir(self, trainer: Trainer) -> Path:
        if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.dirpath:
            return Path(trainer.checkpoint_callback.dirpath)
        return Path(trainer.default_root_dir) / "checkpoints"

    @rank_zero_only
    def _save_step_checkpoint(self, trainer: Trainer, step: int) -> None:
        ckpt_path = self._checkpoint_dir(trainer) / f"step-{step}.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
        log.info("Saved step checkpoint to %s", ckpt_path)

    @rank_zero_only
    def _apply_pruning(self, trainer: Trainer, pl_module: LightningModule, target_step: int) -> None:
        net = pl_module.net
        if not isinstance(net, VisionTransformer):
            raise TypeError("ViTProgressiveMLPPruning expects pl_module.net to be a VisionTransformer")

        amount = self._amount_by_step[target_step]

        if self._save_pre_prune_checkpoint:
            ckpt_path = self._checkpoint_dir(trainer) / f"pre_prune_step_{target_step}.ckpt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(ckpt_path)
            log.info("Saved pre-prune checkpoint to %s", ckpt_path)

        params_before = count_trainable_params(net)
        stats = prune_vit_mlp_blocks(net, amount, self._min_hidden)
        params_after = count_trainable_params(net)

        if self._reinit_optimizer:
            rebind_optimizers(trainer, pl_module)

        reduction = 1.0 - (params_after / params_before)
        log.info(
            "Applied physical MLP pruning at global_step %s (target %s, amount %.2f): "
            "hidden %s -> %s across %s blocks (%.2fM -> %.2fM params, %.1f%% reduction this step)",
            trainer.global_step,
            target_step,
            amount,
            stats["old_mlp_hidden"],
            stats["new_mlp_hidden"],
            stats["blocks_pruned"],
            params_before / 1e6,
            params_after / 1e6,
            reduction * 100,
        )

        pl_module.log("pruning/step", float(target_step), on_step=True, prog_bar=True)
        pl_module.log("pruning/amount", amount, on_step=True)
        pl_module.log("pruning/mlp_hidden", float(stats["new_mlp_hidden"]), on_step=True)
        pl_module.log("pruning/param_reduction", reduction, on_step=True)
