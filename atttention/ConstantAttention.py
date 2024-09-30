import torch
from torch import nn
import math
import torch.nn.functional as F


class ConstantAttention(nn.Module):
    #### WARNING, SIGMA_A IS NOT IMPLEMENTED HERE ####
    def __init__(self, seq_length, d, distribution_A, sigma_A):
        super(ConstantAttention, self).__init__()
        self.unnormalized_weights = nn.Linear(seq_length, seq_length, bias=False)
        self.distribution_A = distribution_A

        if distribution_A == "normal":
            nn.init.normal_(self.unnormalized_weights.weight.data)

        elif distribution_A == "identity":
            nn.init.eye_(self.unnormalized_weights.weight.data)

        elif distribution_A == "chafai" or distribution_A == "chafai_perp":
            nn.init.normal_(
                self.unnormalized_weights.weight.data,
                mean=-torch.log(torch.tensor(2, dtype=torch.int8)) / 2,
                std=torch.sqrt(torch.log(torch.tensor(2, dtype=torch.int8))),
            )
        elif distribution_A == "uniform":
            nn.init.ones_(self.unnormalized_weights.weight.data)

        else:
            raise ValueError(
                "Please input a suitable distribution for the attention matrices."
            )

    def forward(self, x_Q, x_K, x_V):
        attn_weights = self.get_attention_weights(0, 0)
        T = attn_weights.shape[-1]
        upscaled_attention_weights = math.sqrt(T) * attn_weights
        # s = torch.linalg.svdvals(upscaled_attention_weights)
        # print('singular values of upscaled A ', s)

        if (
            self.distribution_A == "chafai_perp"
        ):  ## EACH attention matrix A gets the troublesome direction removed
            upscaled_attention_weights = upscaled_attention_weights - 1 / math.sqrt(
                T
            ) * torch.ones_like(upscaled_attention_weights)

        output = torch.matmul(upscaled_attention_weights, 1 / math.sqrt(T) * x_V)
        return output

    def get_attention_weights(self, x_Q, x_K):
        if (
            self.distribution_A == "chafai"
            or self.distribution_A == "chafai_perp"
            or self.distribution_A == "uniform"
        ):
            return F.softmax(self.unnormalized_weights.weight, dim=-1)
        # elif self.distribution_A == "chafai_perp":
        #  return F.softmax(self.unnormalized_weights.weight, dim=-1) - 1/self.unnormalized_weights.weight.shape[-1] * torch.ones_like(self.unnormalized_weights.weight)
        else:
            return self.unnormalized_weights.weight
