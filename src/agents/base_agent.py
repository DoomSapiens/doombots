import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import vizdoom as vzd

from utils.game_utils import grayscale, get_action_space, get_state_info


class BaseAgent(Env):

    def __init__(self, game: vzd.DoomGame):
        super().__init__()
        self.game = game

        # Create the action space and observation space
        self.observation_space = self.get_observation_space()

        action_space = get_action_space(self.game)
        self.action_space = Discrete(len(action_space))

    def step(self, action):
        actions = get_action_space(self.game)
        reward = self.game.make_action(actions[action], 1)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = grayscale(state)
        else:
            state = np.zeros(self.observation_space.shape)

        info = get_state_info(self.game)
        done = self.game.is_episode_finished()

        return state, reward, done, info

    def reset(self, **kvargs):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return grayscale(state)

    def close(self):
        self.game.close()

    def get_next_action(self, action):
        """
        Returns the next action to be taken.
        """
        actions = get_action_space(self.game)
        return actions[action]

    def get_observation_space(self):
        """
        Returns the observation space of the game.
        """
        return Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
