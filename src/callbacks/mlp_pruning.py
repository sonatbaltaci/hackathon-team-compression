import logging
from pathlib import Path
from typing import Any, List, Union

import torch
import torch.nn as nn
from lightning import Callback, Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import override

from src.modules.nets.vision_transformer import EncoderBlock, MLPBlock, VisionTransformer

log = logging.getLogger(__name__)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def prune_mlp_block(
    mlp: MLPBlock,
    amount: float,
    min_hidden: int = 64,
) -> tuple[MLPBlock, int, int]:
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
    norms = fc1.weight.detach().norm(dim=1)
    keep = torch.topk(norms, k=new_hidden, largest=True).indices.sort().values

    new_mlp = MLPBlock(in_dim, new_hidden, dropout_p)
    with torch.no_grad():
        new_mlp[0].weight.copy_(fc1.weight[keep])
        new_mlp[0].bias.copy_(fc1.bias[keep])
        new_mlp[3].weight.copy_(fc2.weight[:, keep])
        new_mlp[3].bias.copy_(fc2.bias)

    return new_mlp, old_hidden, new_hidden


def prune_vit_mlp_blocks(
    net: VisionTransformer,
    amount: float,
    min_hidden: int = 64,
) -> dict[str, int]:
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


class ViTProgressiveMLPPruning(Callback):
    """Physically prune ViT MLP hidden units at scheduled training steps.

    When ``defer_after_checkpoint`` is True (default), pruning runs one step after
    each target so ``ModelCheckpoint(every_n_train_steps=N)`` can save the full model
    at step N first (e.g. checkpoint at 10k, prune at 10,001).
    """

    def __init__(
        self,
        prune_at_steps: Union[int, List[int]] = 40_000,
        mlp_prune_amount: float = 0.12,
        min_hidden: int = 64,
        reinit_optimizer: bool = True,
        defer_after_checkpoint: bool = True,
        save_pre_prune_checkpoint: bool = True,
    ) -> None:
        if isinstance(prune_at_steps, int):
            prune_at_steps = [prune_at_steps]
        self._prune_at_steps = sorted(prune_at_steps)
        self._mlp_prune_amount = mlp_prune_amount
        self._min_hidden = min_hidden
        self._reinit_optimizer = reinit_optimizer
        self._defer_after_checkpoint = defer_after_checkpoint
        self._save_pre_prune_checkpoint = save_pre_prune_checkpoint
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
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        target = self._pending_target(trainer.global_step)
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
    def _apply_pruning(
        self, trainer: Trainer, pl_module: LightningModule, target_step: int
    ) -> None:
        net = pl_module.net
        if not isinstance(net, VisionTransformer):
            raise TypeError("ViTProgressiveMLPPruning expects pl_module.net to be a VisionTransformer")

        if self._save_pre_prune_checkpoint:
            ckpt_path = self._checkpoint_dir(trainer) / f"pre_prune_step_{target_step}.ckpt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(ckpt_path)
            log.info("Saved pre-prune checkpoint to %s", ckpt_path)

        params_before = count_trainable_params(net)
        stats = prune_vit_mlp_blocks(net, self._mlp_prune_amount, self._min_hidden)
        params_after = count_trainable_params(net)

        if self._reinit_optimizer:
            rebind_optimizers(trainer, pl_module)

        reduction = 1.0 - (params_after / params_before)
        log.info(
            "Applied physical MLP pruning at global_step %s (target %s): hidden %s -> %s "
            "across %s blocks (%.2fM -> %.2fM params, %.1f%% reduction)",
            trainer.global_step,
            target_step,
            stats["old_mlp_hidden"],
            stats["new_mlp_hidden"],
            stats["blocks_pruned"],
            params_before / 1e6,
            params_after / 1e6,
            reduction * 100,
        )

        pl_module.log("pruning/step", float(target_step), on_step=True, prog_bar=True)
        pl_module.log("pruning/mlp_hidden", float(stats["new_mlp_hidden"]), on_step=True)
        pl_module.log("pruning/param_reduction", reduction, on_step=True)
