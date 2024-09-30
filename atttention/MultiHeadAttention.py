import torch
from torch import nn
import math
from .ScaledDotProductAttention import ScaledDotProductAttention
from .ConstantAttention import ConstantAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        seq_length,
        d,
        num_heads,
        attention_type,
        distribution_A,
        distribution_V,
        sigma_A=1,
        sigma_V=1,
        distribution_H="identity",
        distribution_Q="normal",
        distribution_K="normal",
    ):
        super(MultiHeadAttention, self).__init__()

        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d % num_heads == 0, "d must be divisible by num_heads"

        # Initialize dimensions
        self.d = d  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d // num_heads  # Dimension of each head's key, query, and value
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_h = nn.Linear(d, d, bias=False)
        self.attention_type = attention_type

        ## Initialize distribution of W_V
        if distribution_V == "normal":
            nn.init.normal_(self.W_V.weight.data, std=sigma_V)

        elif distribution_V == "xavier_normal":
            nn.init.xavier_normal_(self.W_V.weight.data, std=sigma_V)

        elif distribution_V == "orthogonal":
            nn.init.orthogonal_(
                self.W_V.weight.data, gain=math.sqrt(seq_length) * sigma_V
            )
        elif distribution_V == "identity":
            nn.init.eye_(self.W_V.weight.data)
            self.W_V.weight.data = (
                math.sqrt(seq_length) * sigma_V * self.W_V.weight.data
            )
        else:
            raise Exception(
                f"Distribution for distribution_V type {distribution_V} not recognized."
            )

        ## Initialize distribution of W_H
        if distribution_H == "identity":
            nn.init.eye_(self.W_h.weight.data)
        elif distribution_H == "normal":
            nn.init.xavier_(self.W_h.weight.data)
        elif distribution_H == "orthogonal":
            nn.init.orthogonal_(self.W_h.weight.data)
        else:
            raise Exception(
                f"Distribution for distribution_H type {distribution_H} not recognized."
            )

        if self.attention_type == "default":
            self.attention = ScaledDotProductAttention(
                self.d, self.d_k, distribution_A, distribution_Q, distribution_K
            )
        elif self.attention_type == "constant":
            self.attention = ConstantAttention(
                seq_length, self.d, distribution_A, sigma_A
            )
        else:
            raise Exception(f"Attention type {attention_type} not recognized.")

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        # (B, T, d) --> (B, h, T, d_k)
        batch_size, seq_length, d = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d)

    def forward(self, x_Q, x_K, x_V):
        # Apply linear transformations and split heads
        V = self.split_heads(self.W_V(x_V))

        if self.attention_type == "default":
            # print('shape x_K ', x_K.shape)
            # print('W_K ', self.attention.key_embed)
            # print('shape K before split heads ', self.attention.key_embed(x_K).shape)
            K = self.split_heads(self.attention.key_embed(x_K))
            Q = self.split_heads(self.attention.query_embed(x_Q))

        elif self.attention_type == "constant":
            K, Q = x_K, x_Q

        # print('Q after split', Q.shape)
        # print('K after split', K.shape)
        # print('V after split', V.shape)

        attn_output = self.attention(Q, K, V)
        # print("mean/std attn output ", attn_output.mean(), attn_output.std())

        # Combine heads and apply output transformation
        output = self.W_h(self.combine_heads(attn_output))
        # print("mean/std attn output after multiplying by W_h ", attn_output.mean(), attn_output.std())

        return output
