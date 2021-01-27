"""
Plot training reward
"""
import argparse
import importlib
import os

from matplotlib import pyplot as plt
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, plot_results

# For tensorflow imported with tensorboard
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str)
parser.add_argument("--env", type=str)
parser.add_argument("--exp-folder", type=str)
parser.add_argument("--axis", choices=["steps", "episodes", "time"], type=str)
parser.add_argument(
    "--gym-packages",
    type=str,
    nargs="+",
    default=[],
    help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
)
args = parser.parse_args()

# Going through custom gym packages to let them register in the global registory
for env_module in args.gym_packages:
    importlib.import_module(env_module)

algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)
x_axis = {"steps": X_TIMESTEPS, "episodes": X_EPISODES, "time": X_WALLTIME}[args.axis]

dirs = [
    os.path.join(log_path, folder)
    for folder in os.listdir(log_path)
    if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
]

try:
    plot_results(dirs, 2e6, x_axis, env)
except Exception as e:
    print(e)

plt.show()
