import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agents.base_agent import BaseAgent
from utils.game_utils import setup_game


class TrainAndLoggingCallback(BaseCallback):
    """
    A custom callback that saves the model at a specified frequency.
    """
    def __init__(self, save_interval, save_path, load_steps=0):
        super(TrainAndLoggingCallback, self).__init__(1)
        self.save_interval = save_interval
        self.save_path = save_path
        self.load_steps = load_steps

    def _on_step(self):
        if self.n_calls % self.save_interval == 0:
            model_path = f'{self.save_path}_{self.n_calls + self.load_steps}'
            self.model.save(model_path)
            print(f"Saving model to {model_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train or test a PPO agent for ViZDoom.")
    parser.add_argument('--scenario', type=str, default='basic', help='Name of the scenario to run (e.g., "basic", "deadly_corridor").')
    parser.add_argument('--load-model', type=str, default=None, help='Name of a pre-trained model to load for continued training.')
    parser.add_argument('--load-steps', type=int, default=0, help='Total number of timesteps of the pre-trained model to load.')
    parser.add_argument('--total-timesteps', type=int, default=100000, help='Total number of timesteps for the training session.')
    parser.add_argument('--save-interval', type=int, default=10000, help='Interval (in steps) to save a model.')
    args = parser.parse_args()

    # --- Setup Directories ---
    os.makedirs('../models', exist_ok=True)
    model_path = f'../models/{args.load_model if args.load_model else args.scenario}'
    log_path = f'../logs/{args.load_model if args.load_model else args.scenario}'

    game = setup_game(scenario=args.scenario, render=False)
    env = BaseAgent(game)
    callback = TrainAndLoggingCallback(save_interval=args.save_interval, save_path=model_path, load_steps=args.load_steps)

    if args.load_model:
        print(f"--- Loading model from: {args.load_model} ---")
        model_name = model_path if args.load_steps == 0 else f"{model_path}_{args.load_steps}"
        model = PPO.load(model_name, env=env, device='cuda:0')
        # Ensure the model is using the new environment instance
        model.set_env(env)
    else:
        print("--- Starting new training session ---")
        model = PPO('CnnPolicy', env, verbose=1, learning_rate=0.0001, n_steps=2048, tensorboard_log=log_path)

    # --- Start Training ---
    model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)
    print("--- Training finished ---")


if __name__ == "__main__":
    main()
