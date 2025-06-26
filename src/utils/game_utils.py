import os

import cv2
import numpy as np
import vizdoom as vzd


def setup_vizdoom(scenario, render):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, scenario + '.cfg'))
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, scenario + '.wad'))
    game.set_window_visible(render)
    game.set_render_hud(render)
    game.init()
    return game


def get_action_space(game: vzd.DoomGame):
    """
    Returns the action space of the game.
    """
    return np.identity(len(game.get_available_buttons()))


def grayscale(observation):
    gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
    state = np.reshape(resize, (100, 160, 1))
    return state