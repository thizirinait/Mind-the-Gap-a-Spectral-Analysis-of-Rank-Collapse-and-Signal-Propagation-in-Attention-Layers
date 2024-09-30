import torch
from torch import nn
import math
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product attention (self-attention)
    If distribution_A = chafai_perp, then the Scaled Dot Product attention is computed
    And the troublesome direction is removed
    """

    def __init__(
        self,
        d,
        d_k,
        distribution_A=None,
        distribution_Q="normal",
        distribution_K="normal",
    ):
        # Initialize the attention module
        super(ScaledDotProductAttention, self).__init__()
        self.query_embed = nn.Linear(d, d, bias=False)  # no bias
        self.key_embed = nn.Linear(d, d, bias=False)  # no bias
        self.d_k = d_k
        self.distribution_A = distribution_A
        self.attention_weights = None

        if distribution_Q == "normal":
            nn.init.normal_(self.query_embed.weight.data)
        elif distribution_Q == "xavier_normal":
            nn.init.xavier_normal_(self.query_embed.weight.data)
        else:
            raise ValueError(
                f"Distribution for queries {distribution_Q} not recognised."
            )

        if distribution_K == "normal":
            nn.init.normal_(self.key_embed.weight.data)
        elif distribution_K == "xavier_normal":
            nn.init.xavier_normal_(self.key_embed.weight.data)
        else:
            raise ValueError(
                f"Distribution for queries {distribution_K} not recognised."
            )
        ## otherwise by default, it is uniform with scaled variance like 1/sqrt(d)

    def get_attention_weights(self, query_in, key_in):
        # attention depending on the inputs A(X) = softmax(Q(X) K(X).T / sqrt(d_k))
        # Compute attention weights

        attention_weights = torch.matmul(
            query_in, key_in.transpose(-2, -1)
        )  # (n_query,n_key)
        attention_weights = attention_weights / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_weights, dim=-1)  # (n_query,n_key)

        return attention_weights

    def forward(self, query_in, key_in, value_in):
        # Compute attention
        attention_weights = self.get_attention_weights(query_in, key_in)
        T = attention_weights.shape[-1]
        upscaled_attention_weights = math.sqrt(T) * attention_weights

        if (
            self.distribution_A == "chafai_perp"
        ):  ## EACH attention matrix A gets the troublesome direction removed
            upscaled_attention_weights = upscaled_attention_weights - 1 / math.sqrt(
                T
            ) * torch.ones_like(
                upscaled_attention_weights
            )  # removing the troublesome direction with - 1/T * torch.ones(T,T)

        # Multiply by values to obtain the final output
        output = torch.matmul(upscaled_attention_weights, 1 / math.sqrt(T) * value_in)

        return output
