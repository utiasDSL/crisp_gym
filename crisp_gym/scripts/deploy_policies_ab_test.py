"""Script to deploy two policies for A/B testing in Crisp Gym environment."""

import random

from crisp_gym.scripts.deploy_policy import DeploymentArgs, deploy_policy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy policies for A/B testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deployment.")
    parser.add_argument(
        "--policy_repo_id_a", type=str, required=True, help="Name of the first policy to deploy."
    )
    parser.add_argument(
        "--policy_repo_id_b", type=str, required=True, help="Name of the second policy to deploy."
    )
    parser.add_argument(
        "--deployment_args",
        type=str,
        required=True,
        help="Path to YAML file with deployment arguments.",
    )
    args = parser.parse_args()

    deployment_args = DeploymentArgs.from_yaml(args.deployment_args)

    random.seed(args.seed)
    policies_repo_ids = [args.policy_repo_id_a, args.policy_repo_id_b]
    random.shuffle(policies_repo_ids)

    for i, policy_repo_id in enumerate(policies_repo_ids):
        print(f"Policy {i + 1}: {policy_repo_id}")
        deployment_args.policy_repo_id_or_path = policy_repo_id
        deploy_policy(deployment_args)
