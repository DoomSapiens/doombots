import argparse
import os
import time

from stable_baselines3 import PPO

from agents.base_agent import BaseAgent
from utils.game_utils import setup_game


def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO agent for ViZDoom.")
    parser.add_argument('--load-model', type=str, required=True, help='Path to the pre-trained model .zip file to load for testing.')
    parser.add_argument('--scenario', type=str, default='basic', help='Name of the scenario to run (e.g., "basic", "deadly_corridor").')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run for evaluation.')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering of the game window for faster evaluation.')
    args = parser.parse_args()

    if not os.path.exists(args.load_model):
        raise FileNotFoundError(f"Model file not found at: {args.load_model}")

    is_rendered = not args.no_render

    game = setup_game(scenario=args.scenario, render=is_rendered)
    env = BaseAgent(game)

    print(f"--- Loading model from: {args.load_model} ---")
    model = PPO.load(args.load_model, device='cuda:0', env=env)

    print(f"--- Running evaluation for {args.episodes} episodes on scenario: {args.scenario} ---")
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions for evaluation
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Total Reward for episode {episode + 1}: {total_reward}")
        # A short pause between episodes can be helpful for visualization
        if is_rendered:
            time.sleep(1)

    env.close()
    print("--- Evaluation finished ---")


if __name__ == "__main__":
    main()