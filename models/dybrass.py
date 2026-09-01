from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + self.eps)
        return normalized * self.weight


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, output_dims),
        )
        self.residual_layer = nn.Linear(input_dims, output_dims)
        self.ln_layer = nn.LayerNorm(output_dims)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layers(inputs)
        return self.ln_layer(output + self.residual_layer(inputs))


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        centers = torch.zeros(cluster_number, embedding_dim)
        nn.init.xavier_uniform_(centers)
        orthogonal_centers = torch.zeros_like(centers)
        for row in range(cluster_number):
            block_start = (row // embedding_dim) * embedding_dim
            projection = torch.zeros(embedding_dim)
            for previous in range(block_start, row):
                basis = centers[previous]
                projection = projection + (
                    (centers[row] @ basis) / (basis @ basis) * basis
                )
            centers[row] = centers[row] - projection
            orthogonal_centers[row] = centers[row] / centers[row].norm(p=2)
        centers = orthogonal_centers
        self.cluster_centers = nn.Parameter(centers, requires_grad=False)

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = F.normalize(states, p=2, dim=-1, eps=1e-6)
        batch_size, n_channels, _ = states.shape
        flattened_states = states.reshape(batch_size * n_channels, -1)

        similarities = flattened_states @ self.cluster_centers.T
        norms = self.cluster_centers.norm(p=2, dim=-1).clamp_min(1e-6)
        probabilities = F.softmax(similarities.square() / norms, dim=-1)
        probabilities = probabilities.reshape(batch_size, n_channels, -1)

        global_embeddings = torch.bmm(probabilities.transpose(1, 2), states)
        global_states = probabilities @ global_embeddings
        return global_states, global_embeddings, probabilities


class InteractingMambaBlock(nn.Module):
    def __init__(self, config: DictConfig, n_channels: int, layer_index: int) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        cluster_number = (
            config.n_out
            if layer_index == config.n_layers - 1
            else config.n_clusters
        )
        self.dt_rank = math.ceil(config.d_model / 16)

        self.in_proj = nn.Linear(
            config.d_model, config.d_model * 2, bias=config.bias
        )
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.d_conv,
            groups=config.d_model,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )
        self.x_proj = nn.Linear(
            config.d_model, 2 * config.d_state * config.d_model, bias=False
        )
        self.x_delta = nn.Linear(config.d_model, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, 1, bias=False)
        self.dt_bias = nn.Parameter(torch.rand(n_channels))

        dt_init_std = self.dt_rank**-0.5 * config.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.empty(n_channels).uniform_(config.dt_min, config.dt_max)
        with torch.no_grad():
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        transition = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        transition = transition.repeat(n_channels, 1)
        self.A_log = nn.Parameter(torch.log(transition))
        self.D = nn.Parameter(torch.ones(n_channels))

        self.assignment = ClusterAssignment(
            cluster_number=cluster_number,
            embedding_dim=config.d_state,
        )

    def forward(
        self, inputs: torch.Tensor, return_intermediates: bool = False
    ) -> dict[str, torch.Tensor]:
        projected = self.in_proj(inputs)
        x, z = projected.chunk(2, dim=-1)

        original_shape = x.shape
        x = rearrange(x, "b l r d -> (b r) d l")
        x = self.conv1d(x)[:, :, : original_shape[1]]
        x = rearrange(
            x,
            "(b r) d l -> b l r d",
            b=original_shape[0],
            r=original_shape[2],
        )
        x = F.silu(x)

        output = self._selective_scan(x, return_intermediates)
        output["outputs"] = output["y"] * F.silu(z)
        return output

    def _selective_scan(
        self, x: torch.Tensor, return_intermediates: bool
    ) -> dict[str, torch.Tensor]:
        transition = -torch.exp(self.A_log.float())
        skip = self.D.float()

        delta = self.x_delta(x)
        bc = self.x_proj(x)
        b_matrix, c_matrix = torch.chunk(bc, 2, dim=-1)
        delta = (self.dt_proj.weight @ delta.transpose(-2, -1)).squeeze(-2)
        delta = F.softplus(delta + self.dt_bias)

        delta_a = torch.exp(delta.unsqueeze(-1) * transition)
        delta_b = delta.unsqueeze(-1) * b_matrix
        bx = (
            rearrange(
                delta_b,
                "b l r (d n) -> b l r n d",
                d=self.d_model,
                n=self.d_state,
            )
            @ x.unsqueeze(-1)
        ).squeeze(-1)

        batch_size, length, n_channels, _ = x.shape
        hidden = torch.zeros(
            batch_size, n_channels, self.d_state, device=x.device, dtype=x.dtype
        )
        global_state = torch.zeros_like(hidden)
        hidden_sequence = []
        global_embeddings = []
        assignments = []

        for step in range(length):
            if step:
                global_state, global_embedding, assignment = self.assignment(hidden)
                global_embeddings.append(global_embedding)
                if return_intermediates:
                    assignments.append(assignment)
            hidden = (
                delta_a[:, step] * hidden
                + bx[:, step]
                + delta[:, step].unsqueeze(-1) * global_state
            )
            hidden_sequence.append(hidden)

        hidden_states = torch.stack(hidden_sequence, dim=1)
        global_embeddings_tensor = torch.stack(global_embeddings, dim=1)
        y = (
            hidden_states.unsqueeze(-2)
            @ rearrange(
                c_matrix,
                "b l r (d n) -> b l r n d",
                d=self.d_model,
                n=self.d_state,
            )
        ).squeeze(-2)
        y = y + skip.unsqueeze(-1) * x

        output = {
            "y": y,
            "global_embs": global_embeddings_tensor,
        }
        if return_intermediates:
            output.update(
                {
                    "assignments": torch.stack(assignments, dim=1),
                    "delta": delta,
                    "hidden_states": hidden_states,
                }
            )
        return output


class ResidualBlock(nn.Module):
    def __init__(self, config: DictConfig, n_channels: int, layer_index: int) -> None:
        super().__init__()
        self.mixer = InteractingMambaBlock(config, n_channels, layer_index)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(
        self, inputs: torch.Tensor, return_intermediates: bool = False
    ) -> dict[str, torch.Tensor]:
        output = self.mixer(self.norm(inputs), return_intermediates)
        output["embeds"] = output["outputs"] + inputs
        return output


class Mamba(nn.Module):
    def __init__(self, config: DictConfig, n_channels: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(config, n_channels, layer_index)
                for layer_index in range(config.n_layers)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, return_intermediates: bool = False
    ) -> dict[str, Any]:
        layer_intermediates = []
        output: dict[str, Any] = {}
        for layer in self.layers:
            output = layer(inputs, return_intermediates)
            inputs = output["embeds"]
            if return_intermediates:
                layer_intermediates.append(
                    {
                        key: output[key]
                        for key in ("assignments", "delta", "hidden_states")
                    }
                )
        if return_intermediates:
            output["layer_intermediates"] = layer_intermediates
        return output


class MaskedBatchNorm2d(nn.BatchNorm2d):
    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None or not self.training:
            return super().forward(inputs)

        weights = mask[:, None, None, :].to(inputs.dtype)
        count = weights.sum() * inputs.size(2)
        if count.item() == 0:
            raise ValueError("A masked batch contains no valid time points.")
        mean = (inputs * weights).sum(dim=(0, 2, 3)) / count
        variance = (
            (inputs - mean[None, :, None, None]).square() * weights
        ).sum(dim=(0, 2, 3)) / count

        with torch.no_grad():
            momentum = self.momentum if self.momentum is not None else 0.1
            unbiased = variance * count / (count - 1) if count.item() > 1 else variance
            self.running_mean.lerp_(mean, momentum)
            self.running_var.lerp_(unbiased, momentum)
            self.num_batches_tracked.add_(1)

        normalized = (inputs - mean[None, :, None, None]) / torch.sqrt(
            variance[None, :, None, None] + self.eps
        )
        if self.affine:
            normalized = (
                normalized * self.weight[None, :, None, None]
                + self.bias[None, :, None, None]
            )
        return normalized


class StateRouter(nn.Module):
    def __init__(self, d_state: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_state)
        self.scorer = nn.Sequential(
            nn.Linear(d_state, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, global_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(self.input_norm(global_embeddings)).squeeze(-1)
        return scores, F.softmax(scores, dim=-1)


class StateDynamicClassifier(nn.Module):
    def __init__(
        self,
        n_states: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=(n_states, temporal_kernel_size),
            padding=(0, temporal_kernel_size // 2),
        )
        self.bn1 = MaskedBatchNorm2d(hidden_dim)
        self.act = nn.SiLU()
        self.step_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, scores: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(scores.dtype)
            scores = scores * mask.unsqueeze(-1)

        x = scores.unsqueeze(1).permute(0, 1, 3, 2)
        x = self.act(self.bn1(self.conv1(x), mask))
        x, mask = self._masked_step_pool(x, mask)

        if mask is None:
            x = self.global_pool(x)
        else:
            weights = mask[:, None, None, :]
            x = (x * weights).sum(dim=(2, 3), keepdim=True)
            x = x / weights.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        return self.fc(x)

    def _masked_step_pool(
        self, inputs: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is None:
            return self.step_pool(inputs), None
        minimum = torch.finfo(inputs.dtype).min
        inputs = inputs.masked_fill(mask[:, None, None, :] == 0, minimum)
        inputs = self.step_pool(inputs)
        mask = -F.max_pool2d(
            -mask[:, None, None, :], kernel_size=(1, 2), stride=(1, 2)
        )
        mask = mask.squeeze(1).squeeze(1)
        inputs = inputs.masked_fill(mask[:, None, None, :] == 0, 0.0)
        return inputs, mask


class DyBraSS(nn.Module):
    def __init__(self, dataset: DictConfig, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.n_channels = dataset.n_channels
        self.embedding = ResidualMLPBlock(
            input_dims=dataset.n_channels,
            hidden_dims=config.input_hid,
            output_dims=config.d_model,
        )
        self.mamba = Mamba(config, dataset.n_channels)
        self.norm_f = RMSNorm(config.d_model, config.rms_norm_eps)
        self.unembedding = ResidualMLPBlock(
            input_dims=config.d_model,
            hidden_dims=config.input_hid,
            output_dims=dataset.n_channels,
        )
        self.state_router = StateRouter(
            d_state=config.d_state,
            hidden_dim=config.rt_hidden_dim,
        )
        self.lm_head = StateDynamicClassifier(
            n_states=config.n_out,
            hidden_dim=config.conv_hid,
            output_dim=dataset.num_classes,
            dropout=config.dropout_prob,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, time, ROI, ROI].")
        if inputs.size(-1) != self.n_channels or inputs.size(-2) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels}x{self.n_channels} dFC matrices."
            )
        if inputs.size(1) < 4:
            raise ValueError("Each sequence must contain at least four dFC windows.")

        full_mask = mask.to(inputs.dtype) if mask is not None else None
        model_inputs = inputs[:, :-1]
        input_mask = full_mask[:, 1:] if full_mask is not None else None

        embedded = self.embedding(model_inputs)
        mamba_output = self.mamba(embedded, return_intermediates)
        decoded = self.unembedding(self.norm_f(mamba_output["embeds"]))

        state_scores, state_probabilities = self.state_router(
            mamba_output["global_embs"]
        )
        state_mask = (
            input_mask[:, 1 : state_scores.size(1) + 1]
            if input_mask is not None
            else None
        )
        logits = self.lm_head(state_scores, state_mask)
        output: dict[str, Any] = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1)[:, 1],
            "state_scores": state_scores,
            "state_probabilities": state_probabilities,
        }

        if labels is not None:
            output.update(
                self._losses(
                    logits=logits,
                    labels=labels,
                    state_probabilities=state_probabilities,
                    state_mask=state_mask,
                    reconstruction=decoded,
                    target=inputs[:, 1:],
                    reconstruction_mask=(
                        full_mask[:, 1:] if full_mask is not None else None
                    ),
                )
            )
        if return_intermediates:
            output["layer_intermediates"] = mamba_output["layer_intermediates"]
        return output

    def _losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        state_probabilities: torch.Tensor,
        state_mask: torch.Tensor | None,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        reconstruction_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if labels.ndim == 1:
            labels = F.one_hot(labels.long(), num_classes=logits.size(-1))
        if labels.shape != logits.shape:
            raise ValueError("labels must have shape [batch] or [batch, classes].")
        labels = labels.to(logits.dtype)
        classification = F.cross_entropy(logits, labels)

        log_probabilities = (state_probabilities + 1e-10).log()
        entropy_per_step = -(state_probabilities * log_probabilities).sum(-1)
        if state_mask is None:
            entropy = entropy_per_step.mean()
            mean_usage = state_probabilities.reshape(
                -1, state_probabilities.size(-1)
            ).mean(0)
        else:
            entropy = (entropy_per_step * state_mask).sum()
            entropy = entropy / state_mask.sum().clamp(min=1)
            mean_usage = (
                state_probabilities * state_mask.unsqueeze(-1)
            ).sum(dim=(0, 1)) / state_mask.sum().clamp(min=1)

        uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        balance = F.kl_div((mean_usage + 1e-10).log(), uniform, reduction="batchmean")

        if reconstruction_mask is None:
            reconstruction_mae = F.l1_loss(reconstruction, target)
        else:
            weights = reconstruction_mask[:, :, None, None]
            count = (
                reconstruction_mask.sum() * target.size(-1) * target.size(-2)
            ).clamp(min=1)
            reconstruction_mae = ((reconstruction - target).abs() * weights).sum()
            reconstruction_mae = reconstruction_mae / count

        loss = (
            classification
            + self.config.lambda_ent * entropy
            + self.config.lambda_bal * balance
            + self.config.rec_gamma * reconstruction_mae
        )
        return {
            "loss": loss,
            "classification_loss": classification,
            "entropy_loss": entropy,
            "balance_loss": balance,
            "reconstruction_mae": reconstruction_mae,
        }
