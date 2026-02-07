# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.until_config import PretrainedConfig

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)
                    )
                )
            if len(unexpected_keys) > 0:
                logger.info(
                    "Weights from pretrained model not used in {}: {}".format(
                        model.__class__.__name__,
                        "\n   " + "\n   ".join(unexpected_keys),
                    )
                )
            if len(error_msgs) > 0:
                logger.error(
                    "Weights from pretrained model cause errors in {}: {}".format(
                        model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)
                    )
                )

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [
                    (k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)
                ]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model


##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(
        self,
    ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class MILNCELoss(nn.Module):
    """Multiple Instance Learning Noise Contrastive Estimation (MIL-NCE) Loss.

    Computes contrastive loss with support for multiple positive pairs per sample.
    Positive-pair masks are built dynamically from the input similarity matrix,
    so the loss works correctly even when the actual batch size differs from the
    configured default (e.g., the last mini-batch in an epoch).

    Args:
        n_pair: Number of positive pairs per sample.
    """

    def __init__(self, n_pair=1):
        super(MILNCELoss, self).__init__()
        self.n_pair = n_pair

    def _build_positive_mask(self, batch_size, device):
        """Build block-diagonal mask identifying positive pairs.

        Returns a float tensor of shape (batch_size * n_pair, batch_size * n_pair)
        where 1.0 marks a positive pair.
        """
        eye = torch.eye(batch_size, device=device)
        # Kronecker product: expand each scalar in the identity to an (n_pair x n_pair) block
        mask = eye.repeat_interleave(self.n_pair, dim=0).repeat_interleave(
            self.n_pair, dim=1
        )
        return mask

    def forward(self, sim_matrix):
        """Compute MIL-NCE loss.

        Args:
            sim_matrix: Similarity scores, shape (batch_size * n_pair, batch_size * n_pair).

        Returns:
            Scalar loss value.
        """
        batch_size = sim_matrix.size(0) // self.n_pair
        device = sim_matrix.device

        # Positive-pair mask (block-diagonal)
        positive_mask = self._build_positive_mask(batch_size, device)

        # Mask out positive pairs from the text-to-video direction
        text_to_video_scores = sim_matrix + positive_mask * -1e12
        video_to_text_scores = sim_matrix.transpose(1, 0)

        combined_scores = torch.cat(
            [video_to_text_scores, text_to_video_scores], dim=-1
        )
        log_probs = F.log_softmax(combined_scores, dim=-1)

        # Keep only positive-pair log-probs, mask out negatives
        combined_mask = torch.cat(
            [positive_mask, torch.zeros_like(positive_mask)], dim=-1
        )
        masked_log_probs = log_probs + (1.0 - combined_mask) * -1e12

        # Aggregate positive log-probs via logsumexp
        aggregated_log_probs = -torch.logsumexp(masked_log_probs, dim=-1)

        # Select one representative index per sample (middle of each positive group)
        selector = torch.zeros_like(aggregated_log_probs, dtype=torch.bool)
        representative_indices = torch.arange(
            batch_size, device=device
        ) * self.n_pair + (self.n_pair // 2)
        selector[representative_indices] = True

        loss = aggregated_log_probs.masked_select(selector).mean()
        return loss


class MaxMarginRankingLoss(nn.Module):
    """Max-Margin Ranking Loss with optional hard-negative weighting.

    Computes a bidirectional ranking loss with a configurable margin.
    When negative_weighting is enabled with n_pair > 1, hard negatives
    receive higher loss weight.  Weighting masks are built dynamically
    from the input size to handle variable batch sizes correctly.

    Args:
        margin: Margin for the ranking loss.
        negative_weighting: Whether to apply non-uniform negative weighting.
        n_pair: Number of positive pairs per sample.
        hard_negative_rate: Proportion of hard negatives (vs. easy negatives).
    """

    def __init__(
        self, margin=1.0, negative_weighting=False, n_pair=1, hard_negative_rate=0.5
    ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.negative_weighting = negative_weighting
        self.easy_negative_rate = 1.0 - hard_negative_rate

    def _build_weighting_mask(self, batch_size, device):
        """Build weighting mask distinguishing hard vs. easy negatives.

        Returns a float tensor of shape (batch_size * n_pair, batch_size * n_pair)
        with higher weights for hard negatives.
        """
        alpha = self.easy_negative_rate / (
            (batch_size - 1) * (1.0 - self.easy_negative_rate)
        )
        eye = torch.eye(batch_size, device=device)
        base = (1.0 - alpha) * eye + alpha
        # Kronecker-expand to (batch_size * n_pair, batch_size * n_pair)
        mask = base.repeat_interleave(self.n_pair, dim=0).repeat_interleave(
            self.n_pair, dim=1
        )
        return mask * (batch_size * (1.0 - self.easy_negative_rate))

    def forward(self, x):
        """Compute max-margin ranking loss.

        Args:
            x: Similarity matrix, shape (batch_size * n_pair, batch_size * n_pair).

        Returns:
            Scalar loss value.
        """
        batch_size = x.size(0) // self.n_pair
        diagonal = torch.diag(x)

        # Bidirectional margin violations
        margin_violations = F.relu(self.margin + x - diagonal.view(-1, 1)) + F.relu(
            self.margin + x - diagonal.view(1, -1)
        )

        if self.negative_weighting and self.n_pair > 1 and batch_size > 1:
            weighting_mask = self._build_weighting_mask(batch_size, x.device)
            margin_violations = margin_violations * weighting_mask

        return margin_violations.mean()
