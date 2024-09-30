import torch
from torch import nn
import math
from atttention.MultiHeadAttention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    A simple transformer block.
    """

    def __init__(
        self,
        seq_length,
        gamma,
        heads,
        attention_type="constant",
        distribution_A="chafai",
        distribution_V="normal",
        sigma_A=1,
        sigma_V=1,
        norm_attention=False,
        mlp_flag=False,
        norm_mlp=False,
        ff_hidden_mult=4,
        skip_connections=False,
        activation="relu",
        distribution_Q="normal",
        distribution_K="normal",
        distribution_H="identity",
    ):
        super().__init__()
        self.emb = int(1 / gamma * seq_length)
        self.heads = heads
        self.skip_connections = skip_connections
        if skip_connections == "upscaled":
            self.scale_residual = 1 / math.sqrt(self.emb)
        else:
            self.scale_residual = 1
        # self.scale_residual = 1 / math.sqrt(self.emb)

        self.multi_head_attention = MultiHeadAttention(
            seq_length=seq_length,
            d=self.emb,
            num_heads=self.heads,
            attention_type=attention_type,
            distribution_A=distribution_A,
            distribution_V=distribution_V,
            sigma_A=sigma_A,
            sigma_V=sigma_V,
            distribution_Q=distribution_Q,
            distribution_K=distribution_K,
            distribution_H=distribution_H,
        )

        norms = {True: nn.LayerNorm(self.emb), False: nn.Identity()}
        self.norm = norms[norm_attention]
        self.norm_mlp = norms[norm_mlp]

        self.mlp_flag = mlp_flag
        self.mlp = nn.Identity()

        activations_funcs = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
        }
        self.activation = activations_funcs[activation]

        if self.mlp_flag is True:
            self.mlp = nn.Sequential(
                nn.Linear(self.emb, ff_hidden_mult * self.emb),
                self.activation,
                nn.Linear(ff_hidden_mult * self.emb, self.emb),
            )

    def forward(self, x):

        attended = self.multi_head_attention(x, x, x)

        # print('attended ', attended)

        # Add skip connection, followed by LayerNorm
        if self.skip_connections is True or self.skip_connections == "upscaled":
            x = self.scale_residual * attended + x

        else:
            x = attended
        # print('before layernorm mean/std ', x.mean(), x.std())
        y = self.norm(x)
        # print('after layernorm mean/std ', y.mean(), y.std())
        # print('')

        # Fully connected network
        fedforward = self.mlp(y)
        # print('after mlp', fedforward)

        # Add skip connection, followed by LayerNorm
        if (
            self.skip_connections is True
            or self.skip_connections == "upscaled"
            and self.mlp_flag is True
        ):
            x = self.scale_residual * fedforward + y

        else:
            x = fedforward
        # print('after skip 2', x)
        x = self.norm_mlp(x)
        # print('output ', x)

        return x


class TransformerModel(nn.Module):
    """
    Simplified Transformer (encoder-only) Network
    """

    def __init__(
        self,
        n_layers,
        seq_length,
        gamma,
        heads,
        attention_type="constant",
        distribution_A="chafai",
        distribution_V="normal",
        sigma_A=1,
        sigma_V=1,
        mlp=False,
        LN=False,
        activation="relu",
        skip_connections=False,
        distribution_Q="normal",
        distribution_K="normal",
        distribution_H="identity",
        seed=None,
    ):
        super().__init__()

        if mlp is False:
            norm_mlp = False
            ff_hidden_mult = 1
            activation = "relu"

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_length,
                    gamma,
                    heads,
                    attention_type=attention_type,
                    distribution_A=distribution_A,
                    distribution_V=distribution_V,
                    sigma_A=sigma_A,
                    sigma_V=sigma_V,
                    norm_attention=LN,
                    mlp_flag=mlp,
                    norm_mlp=norm_mlp,
                    ff_hidden_mult=ff_hidden_mult,
                    skip_connections=skip_connections,
                    activation=activation,
                    distribution_Q=distribution_Q,
                    distribution_K=distribution_K,
                    distribution_H=distribution_H,
                )
                for _ in range(n_layers)
            ]
        )

    def forward_with_embeds(self, x):  # x is (B, seq_length, d)
        tokens_embeddings = []
        for i, block in enumerate(self.blocks):
            # print(f'layer {i}')
            x = block(x)
            tokens_embeddings.append(x)
        return x, tokens_embeddings

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            # print(f'layer {i}')
            x = block(x)
            # print(f'layer {i} block {x}')
        return x
