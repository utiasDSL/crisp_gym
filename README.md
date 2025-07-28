![crisp_gym](media/crisp_gym_logo.webp)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![MIT Badge](https://img.shields.io/badge/MIT-License-blue?style=flat)
<a href="https://github.com/utiasDSL/crisp_gym/actions/workflows/ruff_ci.yml"><img src="https://github.com/utiasDSL/crisp_gym/actions/workflows/ruff_ci.yml/badge.svg"/></a>
<a href="https://github.com/utiasDSL/crisp_gym/actions/workflows/pixi_ci.yml"><img src="https://github.com/utiasDSL/crisp_gym/actions/workflows/pixi_ci.yml/badge.svg"/></a>

This repository contains Gymnasium environments to train and deploy high-level learning-based policies using [CRISP_PY](https://github.com/utiasDSL/crisp_py) and the [CRISP controllers](https://github.com/utiasDSL/crisp_controllers).

The installation steps are equal to [CRISP_PY](https://github.com/utiasDSL/crisp_py/tree/feat-ruff-check?tab=readme-ov-file#git-installation-with-pixi) installation.

## Installation

The installation steps are similar to [crisp_py](https://github.com/utiasDSL/crisp_py/tree/feat-ruff-check?tab=readme-ov-file#git-installation-with-pixi) installation.
In particular, if you work on a multi-machine setup, you should check how to setup CycloneDDS / Zenoh.
> [!WARNING]
> It is important that you set the path for your configuration files.
> Write a script in `scripts/set_env.sh` and export the environment variable:
>```bash
> export CRISP_CONFIG_PATH=/path/to/config
> # e.g. export CRISP_CONFIG_PATH=$HOME/repos/crisp_py/config
>```


## Recording Datasets

You can record datasets in `LeRobotDataset` format by running `python scripts/record_lerobot_dataset.py`.
Check the available options with `-h` to modify the metadata of the dataset created.
You can interactively save episodes using the keyboard to save or delete failed episodes.
For now the data-collection works with:
- [x] Kinesthetic teaching,
- [x]  In a dual arm setup with follower and leader,

It is important to note, that to record a dataset in `lerobot` format, you need to install lerobot.
Lerobot has a lot of dependencies and makes it really hard to install it in your system without breaking some previous config.
[This fork of lerobot](https://github.com/danielsanjosepro/lerobot) has only the dependencies needed to create datasets (which is still quite a lot).

After recording a dataset, you can visualize it using the tools provided by lerobot as well as train a policy with it.
You will then need to install (separately) a full version of lerobot and not only use the fork mentioned above.
