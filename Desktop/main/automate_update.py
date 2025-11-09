#!/usr/bin/env python3
"""
automate_update.py - Automate directory updates for a specified target run

This script automates the process of updating a specified directory (e.g., a git repository)
for a given target run, such as updating model repositories or datasets.

Usage:
    python automate_update.py --target <target_name> --action <action>

Arguments:
    --target: The target to update (e.g., 'ml_deploy', 'torchserve', etc.)
    --action: The action to perform (e.g., 'pull', 'status', 'log')

Example:
    python automate_update.py --target ml_deploy --action pull
"""

import argparse
import subprocess
import os
from pathlib import Path
from datetime import datetime

# Define your model endpoints (same as in unified_api.py)
REPOS = {
    "ml_deploy": "ML-Projects-with-Deployment",
    "torchserve": "torchserve-dashboard",
    "langchain": "langchain",
    "haystack": "haystack",
    "madewithml": "MadeWithML",
    "ml_template": "ml-serving-template",
    "aiforbeginners": "AI-For-Beginners",
    "deepspeed": "DeepSpeed",
    "sd_webui": "stable-diffusion-webui"
}

def log_print(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def update_repo(target: str, action: str):
    """
    Update the specified target repository with the given action.
    """
    repo_path = REPOS.get(target)
    if not repo_path:
        log_print(f"❌ Invalid target name: {target}")
        log_print(f"Available targets: {', '.join(REPOS.keys())}")
        return {"error": "Invalid target name"}

    # Assume repos are in the same parent directory as this script
    repo_full_path = Path(__file__).parent / repo_path

    if not repo_full_path.exists():
        log_print(f"⚠️ Repository directory does not exist: {repo_full_path}")
        return {"error": "Repository directory not found"}

    try:
        if action == "pull":
            result = subprocess.run(
                ["git", "pull"],
                cwd=repo_full_path,
                capture_output=True,
                text=True
            )
        elif action == "status":
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_full_path,
                capture_output=True,
                text=True
            )
        elif action == "log":
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=repo_full_path,
                capture_output=True,
                text=True
            )
        else:
            log_print(f"❌ Invalid action: {action}")
            log_print("Available actions: pull, status, log")
            return {"error": "Invalid action"}

        log_print(f"✅ Action '{action}' completed for target '{target}'")
        return {
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "action": action,
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        log_print(f"❌ Failed to perform action '{action}' on target '{target}': {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Automate directory updates for a specified target run")
    parser.add_argument("--target", required=True, help="The target to update (e.g., 'ml_deploy')")
    parser.add_argument("--action", required=True, choices=["pull", "status", "log"], help="The action to perform")
    args = parser.parse_args()

    log_print(f"🚀 Starting automation for target '{args.target}' with action '{args.action}'")
    result = update_repo(args.target, args.action)
    log_print(f"📊 Result: {result}")

if __name__ == "__main__":
    main()
