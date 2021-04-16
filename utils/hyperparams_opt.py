from typing import Any, Dict

import numpy as np
import optuna
# from sb3_contrib import TQC
from stable_baselines3 import DDPG, DQN, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

from utils import linear_schedule


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256, 512, 1024, 2048])

    # gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])

    # learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 0.01)

    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])

    # ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    # ent_coef = 0

    # clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])

    # n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10, 20])
    
    # gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 1.0])
    
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])    
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small":  [dict(pi=[ 64,  64], vf=[ 64,  64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
        "large":  [dict(pi=[256, 256], vf=[256, 256])]
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        # "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

def default_ppo_params() -> Dict[str, Any]:
    return {
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "lr": 0.0003,
            #"lr_schedule": "constant",
            # "ent_coef": ent_coef,
            "clip_range": 0.2,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "net_arch": "small",
            "activation_fn": "tanh"
            }
                                    
def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    # lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small":  [dict(pi=[ 64,  64], vf=[ 64,  64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
        "large":  [dict(pi=[256, 256], vf=[256, 256])]
    }[net_arch]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }

def default_a2c_params() -> Dict[str, Any]:
    return {
            "gamma": 0.99,
            "normalize_advantage": False,
            "max_grad_norm": 0.5,
            "use_rms_prop": True,
            "gae_lambda": 1.0,
            "n_steps": 5,
            # "lr_schedule": 0.95,
            "lr": 0.0007,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "ortho_init": False,
            "net_arch": "small",
            "activation_fn": "tanh",
            }



def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    
    
    # learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 0.01)
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #    learning_rate = linear_schedule(learning_rate)
    
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 100, 256, 512, 1024, 2048])
    
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    
    # learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 10000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [100, 1000, 10000, 20000])
    
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    # train_freq = trial.suggest_categorical("train_freq", [8, 16, 32, 64, 128, 256, 512])
    train_freq = trial.suggest_categorical('train_freq', [1, 2, 10, 180, 360])
    
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05])
    
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    
    # You can comment that out when not using gSDE
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    
    # NOTE: Add "verybig" to net_arch when tuning HER
    # net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = trial.suggest_categorical("net_arch", ["small", "large", "big"])
    
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        # "medium": [256, 256],
        "large":  [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "verybig": [256, 256, 256],
    }[net_arch]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(
                              # log_std_init=log_std_init, 
                              net_arch=net_arch),
    }


def default_sac_params() -> Dict[str, Any]:
    return {
            "gamma": 0.99,
            "lr": 0.0003,
            # "lr_schedule": "constant",
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_starts": 100,
            "train_freq": 1,
            # "gradient_steps": 1,
            # "ent_coef": 'auto',
            "tau": 0.005,
            # "target_entropy": 'auto',
            "net_arch": "large",
            }


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    
    # learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 0.01)
    
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 100, 256, 512, 1024, 2048])
    
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])

    episodic = trial.suggest_categorical("episodic", [True, False])

    if episodic:
        train_freq, gradient_steps = (1, "episode"), -1
    else:
        train_freq = trial.suggest_categorical("train_freq", [1, 16, 128, 256, 1000, 2000])
        gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    # NOTE: Add "verybig" to net_arch when tuning HER
    # net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = trial.suggest_categorical("net_arch", ["small", "large", "big"])

    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    
    # Parâmetros adicionados
    learning_starts = trial.suggest_categorical("learning_starts", [100, 1000, 10000, 20000])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05])

    net_arch = {
        "small": [64, 64],
        # "medium":  [256, 256],
        "large":  [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "verybig": [256, 256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
        "learning_starts": learning_starts,
        "tau": tau,
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    return hyperparams


def default_td3_params() -> Dict[str, Any]:
    return {
            "gamma": 0.99,
            "lr": 0.001,
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_starts": 100,
            "tau": 0.005,
            "episodic": True,
            "noise_type": None,
            "net_arch": "big",  
            }


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02])

    episodic = trial.suggest_categorical("episodic", [True, False])

    if episodic:
        train_freq, gradient_steps = (1, "episode"), -1
    else:
        train_freq = trial.suggest_categorical("train_freq", [1, 16, 128, 256, 1000, 2000])
        gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    return hyperparams


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    return hyperparams


def sample_her_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for HER hyperparams.

    :param trial:
    :return:
    """
    model_class_str = {
        SAC: "sac",
        DDPG: "ddpg",
        DQN: "dqn",
        TD3: "td3",
        # TQC: "tqc",
    }[trial.model_class]

    hyperparams = HYPERPARAMS_SAMPLER[model_class_str](trial)

    hyperparams["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    hyperparams["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["online_sampling"] = trial.suggest_categorical("online_sampling", [True, False])

    return hyperparams


# def sample_tqc_params(trial: optuna.Trial) -> Dict[str, Any]:
#     """
#     Sampler for TQC hyperparams.
# 
#     :param trial:
#     :return:
#     """
#     # TQC is SAC + Distributional RL
#     hyperparams = sample_sac_params(trial)
# 
#     n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
#     top_quantiles_to_drop_per_net = trial.suggest_int("top_quantiles_to_drop_per_net", 0, n_quantiles - 1)
# 
#     hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})
#     hyperparams["top_quantiles_to_drop_per_net"] = top_quantiles_to_drop_per_net
# 
#     return hyperparams


HYPERPARAMS_SAMPLER = {
    "a2c": sample_a2c_params,
    "ddpg": sample_ddpg_params,
    "dqn": sample_dqn_params,
    "her": sample_her_params,
    "sac": sample_sac_params,
    # "tqc": sample_tqc_params,
    "ppo": sample_ppo_params,
    "td3": sample_td3_params,
}

def default_params(algo: str) -> Dict[str, Any]:
    if algo == "ppo":
        return default_ppo_params()
    elif algo == "sac":
        return default_sac_params()
    elif algo == "td3":
        return default_td3_params()
    elif algo == "a2c":
        return default_a2c_params()
    else:
        raise NotImplementedError("Missing default parameters for", algo, "algorithm!")

