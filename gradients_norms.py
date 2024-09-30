import torch
from torch import nn
import numpy as np
import math
import argparse
from model import TransformerModel
from utils import generate_isometric_input
from configs import configs
import pickle
import logging
from pathlib import Path
import pandas as pd
import os


def produce_gradients_norms(
    conf_name,
    model_params,
    max_depth,
    intermediary_layer,
    list_n=[50],
    n_simulations=1,
    list_gammas=[1],
    device="cpu",
):
    """
    This function computes the frobenius norm of d(X_L)/d(W_\ell)
    for \ell fixed to a value called "intermediary_layer"
    and L going from "intermediary layer" to max_depth (the depth of the network)
    """

    directory_path = (
        Path(Path().absolute()) / "results" / "gradient_norm_layer1" / str(conf_name)
    )
    directory_path.mkdir(exist_ok=True, parents=True)
    with (directory_path / "config.pkl").open("wb") as f:
        pickle.dump(model_params, f)

    for k, gamma in enumerate(list_gammas):

        for sim in range(n_simulations):

            for i, T in enumerate(list_n):

                d_in = int(1 / gamma * T)
                X_0 = generate_isometric_input(seq_length=T, gamma=gamma).to(device)
                model = TransformerModel(
                    max_depth + 1,
                    T,
                    gamma,
                    heads=model_params.get("heads"),
                    attention_type=model_params.get("attention_type"),
                    distribution_A=model_params.get("distribution_A"),
                    distribution_V=model_params.get("distribution_V"),
                    LN=model_params.get("layernorm"),
                    skip_connections=model_params.get("skip_connections"),
                    sigma_A=model_params.get("sigma_A"),
                    sigma_V=model_params.get("sigma_V"),
                    mlp=model_params.get("mlp"),
                    distribution_Q=model_params.get("distribution_Q"),
                    distribution_K=model_params.get("distribution_K"),
                ).to(device)
                input = X_0.reshape(1, X_0.shape[0], X_0.shape[1])

                for L in range(intermediary_layer, max_depth + 1):

                    ## jacobian of the output wrt intermediary feature map : first part of the gradient
                    ## Important note: this computation heavily depends on whether A is a function of X or not
                    ## Therefore, we use autograd functions
                    def intermediary_map(x):
                        for l in range(intermediary_layer, L):
                            x = model.blocks[l](x)
                        return x

                    J = torch.autograd.functional.jacobian(intermediary_map, input)

                    ## derivative of intermediary inputs wrt weights : second part of the gradient
                    ## Important note: this computation always holds whether A is a function of X or not
                    prod = torch.eye(T, device=device)
                    for l in range(intermediary_layer, 0, -1):
                        prod = prod @ model.blocks[
                            l
                        ].multi_head_attention.attention.get_attention_weights(
                            input, input
                        ).detach().view(
                            T, T
                        )
                    prod = prod @ X_0
                    for l in range(1, intermediary_layer):
                        prod = (
                            prod
                            @ model.blocks[
                                l
                            ].multi_head_attention.W_V.weight.data.detach()
                        )
                    norm = torch.linalg.norm(
                        J.view(T * d_in, T * d_in)
                        @ torch.kron(prod, torch.eye(d_in, device=device))
                    )

                    df = pd.DataFrame(
                        {
                            "gamma": [gamma],
                            "sim": [sim],
                            "T": [T],
                            "intermediary_layer": [intermediary_layer],
                            "L": [L],
                            "gradient_norm": [
                                norm.detach().cpu().numpy()
                            ],  # norm.item() instead?
                        }
                    )
                    output_path = directory_path / "".join([str(conf_name), ".csv"])

                    df.to_csv(
                        output_path,
                        mode="a",
                        header=not os.path.exists(output_path),
                        index=False,
                    )

    logging.info("done!")

    # return array_norm_gradient


def produce_gradients_norms_bis(
    conf_name,
    model_params,
    max_depth,
    intermediary_layer,
    list_n=[50],
    n_simulations=1,
    list_gammas=[1],
    device="cpu",
):
    """
    This function computes the frobenius norm of d(X_L)/d(W_\ell)
    for \ell fixed to a value called "intermediary_layer"
    and L going from "intermediary layer" to max_depth (the depth of the network)
    """

    directory_path = (
        Path(Path().absolute())
        / "results"
        / "gradient_norm_layer3_rectified_nokron"
        / str(conf_name)
    )
    directory_path.mkdir(exist_ok=True, parents=True)
    with (directory_path / "config.pkl").open("wb") as f:
        pickle.dump(model_params, f)

    for k, gamma in enumerate(list_gammas):

        for sim in range(n_simulations):

            for i, T in enumerate(list_n):

                d_in = int(1 / gamma * T)
                X_0 = generate_isometric_input(seq_length=T, gamma=gamma).to(device)
                model = TransformerModel(
                    max_depth,
                    T,
                    gamma,
                    heads=model_params.get("heads"),
                    attention_type=model_params.get("attention_type"),
                    distribution_A=model_params.get("distribution_A"),
                    distribution_V=model_params.get("distribution_V"),
                    LN=model_params.get("layernorm"),
                    skip_connections=model_params.get("skip_connections"),
                    sigma_A=model_params.get("sigma_A"),
                    sigma_V=model_params.get("sigma_V"),
                    mlp=model_params.get("mlp"),
                    distribution_Q=model_params.get("distribution_Q"),
                    distribution_K=model_params.get("distribution_K"),
                ).to(device)
                input = X_0.reshape(1, X_0.shape[0], X_0.shape[1])

                output, embeddings = model.forward_with_embeds(
                    input
                )  ## embeddings[0] = X_1, embeddings[1] = X_2
                embeddings_completed = [
                    input
                ] + embeddings  ## embeddings_completed[0] = X_0, embeddings_completed[1] = X_1

                X_previous_l = embeddings_completed[intermediary_layer - 1]
                if model_params.get("attention_type") == "default":
                    queries_l = model.blocks[
                        intermediary_layer - 1
                    ].multi_head_attention.attention.query_embed(
                        X_previous_l
                    )  ## Q_{l}(X_{l-1}) = W_Q^{l}(X_{l-1})
                    keys_l = model.blocks[
                        intermediary_layer - 1
                    ].multi_head_attention.attention.key_embed(X_previous_l)

                else:
                    queries_l = 0
                    keys_l = 0
                ## A(X_{l-1})
                A = (
                    model.blocks[intermediary_layer - 1]
                    .multi_head_attention.attention.get_attention_weights(
                        queries_l, keys_l
                    )
                    .detach()
                    .view(T, T)
                )  ## blocks[0].get_attention() = A_1, blocks[1].get_attention()= A_2
                ## We need the l-th block's attention, evaluated at X_{l-1}
                if "perp" in model_params.get("distribution_A"):
                    A = A - 1 / T * torch.ones_like(A)

                AX = A @ X_previous_l
                # first_derivative = torch.kron(
                #    A @ X_previous_l, torch.eye(d_in, device=device)
                # )

                for L in range(intermediary_layer, max_depth + 1):

                    ## jacobian of the output wrt intermediary feature map : first part of the gradient
                    ## Important note: this computation heavily depends on whether A is a function of X or not
                    ## Therefore, we use autograd functions
                    def intermediary_map(x):
                        for l in range(intermediary_layer, L):
                            x = model.blocks[l](x)
                        return x

                    J = torch.autograd.functional.jacobian(intermediary_map, input)

                    ## derivative of intermediary inputs wrt weights : second part of the gradient
                    ## Important note: this computation always holds whether A is a function of X or not

                    # this way, we avoid computing the kron product
                    norm_J = torch.linalg.norm(J)
                    norm_JAX = torch.linalg.norm(J @ AX)
                    norm = norm_JAX * norm_J

                    # norm = torch.linalg.norm(
                    #    J.view(T * d_in, T * d_in).T @ first_derivative
                    # )

                    df = pd.DataFrame(
                        {
                            "gamma": [gamma],
                            "sim": [sim],
                            "T": [T],
                            "intermediary_layer": [intermediary_layer],
                            "L": [L],
                            "gradient_norm": [
                                norm.detach().cpu().numpy()
                            ],  # norm.item() instead?
                        }
                    )
                    output_path = directory_path / "".join([str(conf_name), ".csv"])

                    df.to_csv(
                        output_path,
                        mode="a",
                        header=not os.path.exists(output_path),
                        index=False,
                    )

    logging.info("done!")

    # return array_norm_gradient


if __name__ == "__main__":

    options = argparse.ArgumentParser(
        description="Producing the gradients norms of the outputs at all layers from intermediary_layer to max_depth with respect to the values at layer intermediary_layer.  All settings are described in configs.py file."
    )
    options.add_argument(
        "--n_simulations",
        type=int,
        default=1,
        help="Number of simulations to iterate over.",
    )
    options.add_argument(
        "--intermediary_layer",
        type=int,
        default=2,
        help="Layer from which to take the gradient from.",
    )
    options.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximal depth in the network.",
    )
    options.add_argument(
        "--list_n",
        nargs="+",
        type=int,
        default=50,
        help="List of sequence lengths to iterate over.",
    )

    options.add_argument(
        "--list_gammas",
        nargs="+",
        type=float,
        default=1,
        help="List of gammas to iterate over.",
    )
    options = options.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    for _, (config, params) in enumerate(configs.items()):
        produce_gradients_norms_bis(
            config,
            params,
            options.max_depth,
            options.intermediary_layer,
            list_n=options.list_n,
            n_simulations=options.n_simulations,
            list_gammas=options.list_gammas,
            device=device,
        )
