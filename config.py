import os
from argparse import ArgumentParser

import yaml

from utilities.helpers import is_rank_zero
from utilities.io import expand_path, load_yaml

# Here 'storage' a symlink to the real storage disk
CHECKPOINT_DIRECTORY = expand_path(os.getenv("CHECKPOINT_DIRECTORY", "~/storage/checkpoints"))
DATASET_DIRECTORY = expand_path(os.getenv("DATASET_DIRECTORY", "~/storage/datasets"))
CACHE_DIRECTORY = expand_path(os.getenv("CACHE_DIRECTORY", "~/storage/cache"))


def read_cli_configuration():
    cli = ArgumentParser(add_help=True)

    # Initial args to parse
    cli.add_argument("--config", type=str, required=True, help="Path to a configuration .yml")
    cli.add_argument("--seed", type=int, default=3741)
    cli.add_argument("--version", type=str, default=None)
    cli.add_argument("--gpu", type=int, default=0)

    # Determine if "--config" was provided
    args = cli.parse_args()

    # Loading args from config .yml
    config = load_yaml(expand_path(args.config))
    if config is None:
        raise FileNotFoundError("Unable to find configuration .yml file.")

    # Store random seed in config
    config["seed"] = args.seed

    # Store version in config
    config["version"] = args.version

    # Get gpu to use
    config["gpu"] = args.gpu

    # By default the session name is the file name.
    if "name" not in config:
        config["name"] = os.path.splitext(os.path.basename(args.config))[0]

    # Store configuration
    os.environ["SESSION_CONFIG"] = yaml.safe_dump(config)

    return config


# GATHER CONFIGURATION

CONFIG_DATA = os.getenv("SESSION_CONFIG", None)
if CONFIG_DATA is not None:
    # We are multi-gpu, load the config from env instead.
    CONFIG_DATA = yaml.safe_load(CONFIG_DATA)
else:
    # Get configuration from cli args.
    CONFIG_DATA = read_cli_configuration()

# VALIDATE CONFIGURATION

# Ensure configuration has required keys
for required_key in ["model", "dataset", "augmentation", "seed", "epochs", "name"]:
    if required_key not in CONFIG_DATA:
        raise RuntimeError(f"Configuration file did not specify '{required_key}'.")

# By default, the session version is the random seed.
if "version" not in CONFIG_DATA or CONFIG_DATA["version"] is None:
    CONFIG_DATA["version"] = str(CONFIG_DATA["seed"])

# ASSIGN CONFIGURATION GLOBALS

# Load module configuration
MODEL_ARGS: dict = CONFIG_DATA.get("model_args", {}) or {}
MODEL_NAME: str = CONFIG_DATA["model"]

# Load data module configuration
DATASET_ARGS: dict = CONFIG_DATA.get("dataset_args", {}) or {}
DATASET_NAME: str = CONFIG_DATA["dataset"]

# Load augmentation configuration
AUGMENTATION_ARGS: dict = CONFIG_DATA.get("augmentation_args", {}) or {}
AUGMENTATION_NAME: str = CONFIG_DATA["augmentation"]

# Load optimizer configuration
OPTIMIZER_ARGS: dict = CONFIG_DATA.get("optimizer_args", {}) or {}

# Load LR scheduler configuration
SCHEDULER_ARGS: dict = CONFIG_DATA.get("scheduler_args", {}) or {}

# Training configuration
RANDOM_SEED: int = int(CONFIG_DATA["seed"])
EPOCHS: int = int(CONFIG_DATA["epochs"])

# Superpixel configuration
SP_COUNT: int = int(CONFIG_DATA['seg_count'])

# Session configuration
SESSION_VERSION: str = CONFIG_DATA["version"]
SESSION_NAME: str = CONFIG_DATA["name"]

GPU: int = CONFIG_DATA['gpu']


def report_configuration():
    print()
    print("- CHECKPOINT_DIRECTORY", CHECKPOINT_DIRECTORY)
    print("- DATASET_DIRECTORY", DATASET_DIRECTORY)
    print("- CACHE_DIRECTORY", CACHE_DIRECTORY)
    print()
    print("- MODEL_NAME", MODEL_NAME)
    print("- MODEL_ARGS", MODEL_ARGS)
    print()
    print("- DATASET_NAME", DATASET_NAME)
    print("- DATASET_ARGS", DATASET_ARGS)
    print()
    print("- AUGMENTATION_NAME", AUGMENTATION_NAME)
    print("- AUGMENTATION_ARGS", AUGMENTATION_ARGS)
    print()
    print("- OPTIMIZER_ARGS", OPTIMIZER_ARGS)
    print("- SCHEDULER_ARGS", SCHEDULER_ARGS)
    print()
    print("- RANDOM_SEED", RANDOM_SEED)
    print("- EPOCHS", EPOCHS)
    print()
    print("- SESSION_NAME", SESSION_NAME)
    print("- SESSION_VERSION", SESSION_VERSION)
    print()


if is_rank_zero():
    report_configuration()

# print(f"LOADED CONFIGURATION ON ({os.getenv('GLOBAL_RANK', 0)}:{os.getenv('LOCAL_RANK', 0)})")
