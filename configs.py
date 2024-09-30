configs = {}

configs["A"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A^{\perp}$"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A + LN"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A^{\perp}$ + LN"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A + skip"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A^{\perp}$ + skip"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A + upscaled skip"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A^{\perp}$ + upscaled skip"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A + skip + LN"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A^{\perp}$ + skip + LN"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A + upscaled skip + LN"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}


configs["$A^{\perp}$ + upscaled skip + LN"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["Identity A"] = {
    "heads": 1,
    "attention_type": "constant",
    "distribution_A": "identity",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A(X)"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A(X)^{\perp}$"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A(X) + LN"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A(X)^{\perp}$ + LN"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": False,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A(X) + skip"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A(X)^{\perp}$ + skip"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A(X) + upscaled skip"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}


configs["$A(X)^{\perp}$ + upscaled skip"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": False,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A(X) + skip + LN"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["$A(X)^{\perp}$ + skip + LN"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": True,
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

configs["A(X) + upscaled skip + LN"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}


configs["$A(X)^{\perp}$ + upscaled skip + LN"] = {
    "heads": 1,
    "attention_type": "default",
    "distribution_A": "chafai_perp",
    "distribution_V": "normal",
    "distribution_H": "identity",
    "sigma_A": 1,
    "sigma_V": 1,
    "mlp": False,
    "layernorm": True,
    "skip_connections": "upscaled",
    "distribution_Q": "xavier_normal",
    "distribution_K": "xavier_normal",
}

train_config = {
    "optimizer": "adam",
    "n_training_data": 50,
    "n_epochs": 50,
    "batch_size": 64,
    "ratio_test_train": 1 / 10,
    "seq_length": 500,
    "gamma": 1,
}
