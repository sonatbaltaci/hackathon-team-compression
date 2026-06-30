import copy

import torch
import torch.nn as nn


class Router:
    """Random token dropping router."""

    def __init__(self, percent_to_keep=0.5):
        self.percent_to_keep = percent_to_keep

    def get_tokens_to_keep(self, x):
        b, t, _ = x[:, 1:, :].shape
        num_keep = int(t * self.percent_to_keep)
        rand_noise = torch.rand(b, t, device=x.device)
        sorted_rand_noise = torch.argsort(rand_noise, dim=1)
        indices_to_keep = sorted_rand_noise[:, :num_keep]
        indices_to_keep += 1
        return indices_to_keep

    def drop_tokens(self, x, indices_to_keep):
        b, t, d = x.shape
        masked_x = torch.gather(
            x, 1, indices_to_keep.unsqueeze(-1).expand(-1, -1, d)
        )
        return masked_x

    def recover_tokens(self, x, indices_to_keep, original_x):
        recovered_x = torch.scatter(
            original_x,
            1,
            indices_to_keep.unsqueeze(-1).expand(-1, -1, original_x.size(-1)),
            x,
        )
        return recovered_x


class MoDRouter(nn.Module):
    """Mixture of Depth router with learned token scoring."""

    def __init__(self, hidden_dim, percent_to_keep=0.5):
        super().__init__()
        self.percent_to_keep = percent_to_keep
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        b, t, _ = x[:, 1:, :].shape
        num_keep = int(t * self.percent_to_keep)
        scores = self.linear(x[:, 1:, :]).squeeze(-1)
        topk_vals, topk_idx = scores.topk(num_keep, dim=1)
        indices = topk_idx + 1
        indices, sort_order = indices.sort(dim=1)
        weights = torch.sigmoid(topk_vals.gather(1, sort_order))
        return indices, weights

    def drop_tokens(self, x, indices_to_keep):
        d = x.size(-1)
        cls_token = x[:, :1, :]
        masked_x = torch.gather(
            x, 1, indices_to_keep.unsqueeze(-1).expand(-1, -1, d)
        )
        return torch.cat([cls_token, masked_x], dim=1)

    def recover_tokens(self, x, indices_to_keep, original_x):
        cls_token = x[:, :1, :]
        kept_tokens = x[:, 1:, :]
        recovered = original_x.clone()
        recovered[:, :1, :] = cls_token
        recovered.scatter_(
            1,
            indices_to_keep.unsqueeze(-1).expand(-1, -1, original_x.size(-1)),
            kept_tokens,
        )
        return recovered

    @staticmethod
    def create_per_layer_routers(router, router_layers):
        return nn.ModuleDict(
            {str(i): copy.deepcopy(router) for i in router_layers}
        )
