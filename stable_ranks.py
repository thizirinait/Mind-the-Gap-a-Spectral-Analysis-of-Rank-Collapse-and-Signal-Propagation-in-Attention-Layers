import torch
from torch import nn
import numpy as np
import argparse
from model import TransformerModel
from utils import generate_isometric_input
from configs import configs
import pandas as pd
import os
import pickle
import logging
from pathlib import Path


def produce_stable_ranks(
    conf_name, model_params, n_layers, list_n=[50], n_simulations=1, list_gammas=[1]
):
    # os.makedirs(os.getcwd() + "/stable_rank/" + str(conf_name))

    ## let us first save the configuration that lead to these results
    # with open(
    #    os.getcwd() + "/stable_rank/" + str(conf_name) + "/config.pkl", "wb"
    # ) as f:
    #    pickle.dump(model_params, f)

    directory_path = (
        Path(Path().absolute()) / "results" / "stable_rank_XAVIER_KEYS" / str(conf_name)
    )
    directory_path.mkdir(exist_ok=True, parents=True)
    with (directory_path / "config.pkl").open("wb") as f:
        pickle.dump(model_params, f)

    for k, gamma in enumerate(list_gammas):

        for sim in range(n_simulations):

            for j, T in enumerate(list_n):

                X_0 = generate_isometric_input(seq_length=T, gamma=gamma)

                train_loader = torch.utils.data.DataLoader(
                    X_0.reshape(1, X_0.shape[0], X_0.shape[1]), batch_size=1
                )

                model = TransformerModel(
                    n_layers,
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
                )

                for i, batch in enumerate(train_loader):
                    output, tokens_embs = model.forward_with_embeds(batch)

                for l, X in enumerate(tokens_embs):

                    try:
                        sigma = X[0] @ X[0].T
                        s = torch.linalg.svdvals(sigma)
                        stable_rank = ((s**2).sum() / (s**2).max()).detach().numpy()
                    except:
                        stable_rank = float("nan")

                    df = pd.DataFrame(
                        {
                            "gamma": [gamma],
                            "sim": [sim],
                            "T": [T],
                            "l": [l],
                            "stable_rank": [stable_rank],
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
    # return array_stable_rank


if __name__ == "__main__":

    options = argparse.ArgumentParser(
        description="Producing stable ranks for all settings as described in configs.py file."
    )
    options.add_argument(
        "--n_simulations",
        type=int,
        default=1,
        help="Number of simulations to iterate over.",
    )
    options.add_argument(
        "--n_layers",
        type=int,
        default=5,
        help="Maximal depth to iterate over.",
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

    # stable_ranks = {}

    for _, (config, params) in enumerate(configs.items()):

        produce_stable_ranks(
            config,
            params,
            options.n_layers,
            list_n=options.list_n,
            n_simulations=options.n_simulations,
            list_gammas=options.list_gammas,
        )
