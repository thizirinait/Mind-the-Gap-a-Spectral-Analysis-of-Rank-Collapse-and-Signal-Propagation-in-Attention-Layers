import torch
from torch import nn
import numpy as np
import argparse
import shutil
import copy
from omegaconf import OmegaConf
from model import TransformerModel
from utils import generate_isometric_dataset, params2experiment_name
from configs import configs, train_config
from pathlib import Path
import pickle
import os
import pandas as pd
import math
import json
from matplotlib import pyplot as plt


def produce_losses(
    conf_name,
    model_params,
    train_config,
    list_lr,
    seq_length,
    gamma,
    n_layers,
    n_simulations,
    f_to_learn,
    device,
):
    ### Assuming I want to learn the constant matrix-valued function equal to identity i.e. torch.ones(seq_length, d)

    # add to the dictionnary theses specific parameters
    model_params["gamma"] = gamma
    model_params["seq_length"] = seq_length
    model_params["n_layers"] = n_layers
    train_config["device"] = str(device)

    params = {}
    params.update(model_params)
    params.update(train_config)
    params.update(
        {
            "list_lr": list_lr,
            "n_simulations": n_simulations,
            "f_to_learn": f_to_learn.__name__,
        }
    )

    # Create experiment name
    conf = OmegaConf.create(params)
    experiment_name = OmegaConf.to_container(copy.deepcopy(conf))
    experiment_name.pop("attention_type")
    experiment_name.pop("distribution_A")
    experiment_name.pop("skip_connections")
    experiment_name.pop("layernorm")
    experiment_name = params2experiment_name(experiment_name)
    print(f"Experiment name {experiment_name}")

    directory_path = (
        Path(Path().absolute()) / "results" / "loss" / experiment_name / str(conf_name)
    )
    ## Deletes the directory if it already exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    directory_path.mkdir(exist_ok=True, parents=True)
    with (directory_path / "config_model.pkl").open("wb") as f:
        pickle.dump(model_params, f)
    with (directory_path / "config_training.pkl").open("wb") as f:
        pickle.dump(train_config, f)

    source_data = generate_isometric_dataset(
        train_config["n_training_data"], seq_length, gamma
    )
    n_test = int(train_config["ratio_test_train"] * train_config["n_training_data"])
    random_test_data = generate_isometric_dataset(n_test, seq_length, gamma)

    criterion = nn.MSELoss()

    for k, lr in enumerate(list_lr):

        for sim in range(n_simulations):
            # print(f'************************************** SIMULATION {sim} **************************************')
            train_loader = torch.utils.data.DataLoader(
                source_data, batch_size=train_config["batch_size"]
            )
            test_loader = torch.utils.data.DataLoader(
                random_test_data, batch_size=train_config["batch_size"]
            )

            model = TransformerModel(
                n_layers,
                seq_length,
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

            if train_config["optimizer"] == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif train_config["optimizer"] == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else:
                raise Exception(
                    f'Optimizer {train_config["optimizer"]} not recognized.'
                )

            train_losses, test_losses = [], []

            for epoch in range(0, train_config["n_epochs"]):

                # print(f'************************************** EPOCH {epoch} **************************************')
                stop_because_of_nans = False

                ## Training step
                model.train()
                for i, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(batch.to(device))
                    target = f_to_learn(batch).to(device)
                    loss = criterion(output, target)
                    train_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    ## we stop training when the training loss is nan
                    if math.isnan(train_losses[-1]) is True:
                        stop_because_of_nans = True
                        break

                ## Validation step
                # model.eval()
                # for i, batch in enumerate(test_loader):
                #    output = model(batch.to(device))
                #    target = f_to_learn(batch).to(device)
                #    test_loss = criterion(output, target)

                # train_loss, test_loss = loss.item(), test_loss.item()
                # train_losses.append(train_loss)
                # test_losses.append(test_loss)

            df = pd.DataFrame(
                {
                    "seq_lenght": [seq_length],
                    "gamma": [gamma],
                    "sim": [sim],
                    "n_layers": [n_layers],
                    "n_epochs": [train_config["n_epochs"]],
                    "lr": [lr],
                    "train_loss": [train_losses],
                    # "test_loss": [test_losses],
                }
            )
            output_path = directory_path / "".join([str(conf_name), ".csv"])

            df.to_csv(
                output_path,
                mode="a",
                header=not os.path.exists(output_path),
                index=False,
            )

            if stop_because_of_nans:  # to break the outer loop
                break

            # plt.figure()
            # plt.scatter(
            #    batch.flatten().cpu().numpy(), output.flatten().detach().cpu().numpy()
            # )
            # plt.savefig(directory_path / "".join([str(conf_name), ".png"]))
            # plt.close()

            # if epoch%(train_config['n_epochs']//5)==0:
            #  print(f"Epoch: {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
    # return train_losses, test_losses


def identity(x):
    return x


def square(x):
    return x**2


def polynomial_order4(x):
    return x**4 + x**3 - x**2 + x + 1


def shifted_sigmoid(x):
    return torch.sigmoid(x + 1 / 2)


def expanded_shifted_sigmoid(x):
    return torch.sigmoid(20 * (x + 1 / 2))


def heaviside(x):
    return torch.heaviside(x, torch.zeros_like(x))


if __name__ == "__main__":

    options = argparse.ArgumentParser(
        description="Producing the training/testing losses of Transformers whose settings are described in configs.py file."
    )

    options.add_argument(
        "--n_simulations",
        type=int,
        default=1,
        help="Number of simulations to iterate over.",
    )
    options.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Number of layers for all the Transformer models.",
    )

    options.add_argument(
        "--list_lr",
        nargs="+",
        type=float,
        default=3e-3,
        help="List of learning rates to iterate over.",
    )

    options.add_argument(
        "--function_to_learn",
        type=str,
        default="expanded shifted sigmoid",
        help="Name of the entry-wise mapping function to learn.",
    )

    options = options.parse_args()
    if options.function_to_learn == "identity":
        f = identity
    elif options.function_to_learn == "square":
        f = square
    elif options.function_to_learn == "shifted sigmoid":
        f = shifted_sigmoid
    elif options.function_to_learn == "expanded shifted sigmoid":
        f = expanded_shifted_sigmoid
    elif options.function_to_learn == "polynomial_order4":
        f = polynomial_order4
    elif options.function_to_learn == "heaviside":
        f = heaviside
    else:
        raise ValueError(
            f"Mapping to learn {options.function_to_learn} not recognized."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    for _, (config, params) in enumerate(configs.items()):
        produce_losses(
            config,
            params,
            train_config,
            options.list_lr,
            train_config["seq_length"],
            train_config["gamma"],
            options.depth,
            options.n_simulations,
            f_to_learn=f,
            device=device,
        )
