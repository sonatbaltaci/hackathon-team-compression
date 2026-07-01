"""Token dropping logic.

At configured layers it drops a fraction of patch tokens, either randomly or
at fixed evenly-spaced positions.
Optionally, it aggregates these into a register token.
"""

from typing import Literal

import torch


def token_select(
    class_token: torch.Tensor,  # noqa: ARG001
    patch_tokens: torch.Tensor,
    register_tokens: torch.Tensor | None,
    current_layer: int,
    total_layers: int,  # noqa: ARG001
    drop_ratio: float = 0.5,
    drop_type: Literal["random", "fixed"] = "random",
    activate_after_n_layers: int = 0,
    every_n_layers: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Select patch tokens to keep and patch tokens to merge into registers.

    Args:
        class_token (torch.Tensor): Class token tensor. Shape: (B, 1, C).
        patch_tokens (torch.Tensor): Patch token tensor. Shape: (B, N, C).
        register_tokens (torch.Tensor | None): Register token tensor. If None, no
            register aggregation indices are returned. If not None, only one register
            is allowed. Shape: (B, 1, C).
        current_layer (int): Zero-based index of the current encoder layer.
        total_layers (int): Total number of encoder layers.
        drop_ratio (float, optional): Fraction of patch tokens to drop, in [0, 1]. Default: 0.5.
        drop_type (Literal["random", "fixed"], optional): Whether to drop tokens randomly or at
            fixed evenly-spaced positions. Default: "random".
        activate_after_n_layers (int, optional): Number of initial layers to skip before
            dropping begins. Default: 0.
        every_n_layers (int, optional): Drop tokens once every N layers after activation.
            Default: 1.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor] | None]: Kept patch-token indices and dropped patch-token indices
            assigned to each register. If there is no register, the second item is None.
    """
    num_patches = patch_tokens.shape[1]

    if every_n_layers < 1:
        msg = f"every_n_layers must be >= 1, got {every_n_layers}."
        raise ValueError(msg)

    should_drop = (
        current_layer >= activate_after_n_layers and (current_layer - activate_after_n_layers) % every_n_layers == 0
    )
    if not should_drop:
        all_idcs = torch.arange(num_patches, device=patch_tokens.device)
        if register_tokens is None:
            return [all_idcs], None
        return [all_idcs], [torch.empty(0, device=patch_tokens.device, dtype=torch.long)]

    if num_patches == 0:
        empty = torch.empty(0, device=patch_tokens.device, dtype=torch.long)
        if register_tokens is None:
            return [empty], None
        return [empty], [empty]

    num_kept_patches = num_patches - int(num_patches * drop_ratio)

    if drop_type == "random":
        shuffled_idcs = torch.randperm(num_patches, device=patch_tokens.device)
        kept_idcs = shuffled_idcs[:num_kept_patches].sort().values
        dropped_idcs = shuffled_idcs[num_kept_patches:].sort().values
    else:
        kept_idcs = torch.round(torch.linspace(0, num_patches - 1, num_kept_patches, device=patch_tokens.device)).long()
        mask = torch.ones(num_patches, dtype=torch.bool, device=patch_tokens.device)
        mask[kept_idcs] = False
        dropped_idcs = torch.arange(num_patches, device=patch_tokens.device)[mask]

    if register_tokens is None:
        return [kept_idcs], None

    torch._assert(
        register_tokens.shape[1] == 1,
        f"Random token dropping supports exactly one register, got {register_tokens.shape[1]}.",
    )
    # WARNING: this is cheating, we "should" return list(kept_idcs.unsqueeze(1)) techically, but since we would
    # concat on token_aggregate anyway, we can "cheat" a bit an instead of returning one tensor per output patch
    # we just return a single tensor which is a bit more efficient.
    return [kept_idcs], [dropped_idcs]


def token_aggregate(
    patch_selection_idcs: list[torch.Tensor],
    register_selection_idcs: list[torch.Tensor] | None,
    patch_tokens: torch.Tensor,
    register_tokens: torch.Tensor | None,
    current_layer: int,  # noqa: ARG001
    total_layers: int,  # noqa: ARG001
    register_aggregation_fn: str = "sum",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop unselected patch tokens and aggregate dropped tokens into registers.

    Args:
        patch_selection_idcs (list[torch.Tensor]): Indices of patch tokens to keep.
        register_selection_idcs (list[torch.Tensor] | None): Dropped patch-token indices assigned to each register.
            If None, no register aggregation is performed.
        patch_tokens (torch.Tensor): Patch token tensor. Shape: (B, N, C).
        register_tokens (torch.Tensor | None): Register token tensor. If None, no register output is returned. If not
            None, one output register is returned per input register. Shape: (B, R, C).
        current_layer (int): Zero-based index of the current encoder layer.
        total_layers (int): Total number of encoder layers.
        register_aggregation_fn (str, optional): Register aggregation function. Default: "sum".

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Kept patch tokens and the updated register token. The register output
            is None when `register_tokens` is None.
    """
    if len(patch_selection_idcs) == 0:
        kept_idcs = torch.empty(0, device=patch_tokens.device, dtype=torch.long)
    else:
        kept_idcs = patch_selection_idcs[0]
    merged_patch_tokens = patch_tokens.index_select(dim=1, index=kept_idcs)

    if register_tokens is None:
        torch._assert(register_selection_idcs is None, "Register selection indices require register tokens.")
        return merged_patch_tokens, None

    torch._assert(
        register_selection_idcs is not None and (len(register_selection_idcs) == register_tokens.shape[1] == 1),
        "Expected one set of dropped patch indices per register.",
    )

    register_selection_idcs = register_selection_idcs[0]
    if register_selection_idcs.numel() == 0:
        return merged_patch_tokens, register_tokens

    dropped_patch_tokens = patch_tokens.index_select(dim=1, index=register_selection_idcs)
    register_aggregation_fn = getattr(torch, register_aggregation_fn)
    merged_register_tokens = register_aggregation_fn(
        torch.cat([register_tokens, dropped_patch_tokens], dim=1), dim=1, keepdim=True
    )
    return merged_patch_tokens, merged_register_tokens
