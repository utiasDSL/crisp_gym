#!/bin/bash

export GIT_LFS_SKIP_SMUDGE=1

# Check if .pixi.sh exists
if [ ! -f "scripts/set_env.sh" ]; then
  echo "Error: scripts/set_env.sh file not found. You need to define this scripts te set personal environment variables."
  echo "Make sure to export the CRISP_CONFIG_PATH variable to point to your configuration folder as well as the ROS_DOMAIN_ID variable (set to 100 if working with demos)."
  echo "For multi-machine setups check https://utiasdsl.github.io/crisp_controllers/misc/multi_machine_setup/"
  exit 1
fi
