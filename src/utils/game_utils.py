import cv2
import json5
import numpy as np
import vizdoom as vzd

CONFIG_PATH = '../scenarios/{}.json5'
SCENARIO_PATH = '../scenarios/{}.wad'


def setup_game(scenario, render):
    """
    Sets up the ViZDoom game environment based on the provided scenario configuration.

    :param scenario: Path to the scenario configuration file (JSON5 format).
    :param render: Boolean indicating whether to render the game window.
    :return: An initialized ViZDoom game instance.
    """

    # 1. Load the configuration file for the specified scenario.
    try:
        with open(CONFIG_PATH.format(scenario), 'r') as f:
            config = json5.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found for '{scenario}'")
        raise
    except Exception as e:
        print(f"An error occurred while parsing configuration file for '{scenario}': {e}")
        raise

    # Normalize keys to snake_case (e.g., "episode-timeout" -> "episode_timeout")
    config = {key.lower().replace('-', '_'): value for key, value in config.items()}

    # Helper function to convert string representations to ViZDoom's internal enum types.
    def get_vzd_enum(enum_name, value_str):
        try:
            # Access the enum class (e.g., vzd.ScreenResolution)
            enum_class = getattr(vzd, enum_name)
            # Access the specific enum member (e.g., vzd.ScreenResolution.RES_320X240)
            return getattr(enum_class, value_str)
        except AttributeError:
            print(f"Warning: Could not find enum value '{value_str}' for enum '{enum_name}'.")
            return None

    # 2. Create the game instance and configure it based on the provided settings.
    game = vzd.DoomGame()

    # --- General Settings ---
    game.set_doom_scenario_path(SCENARIO_PATH.format(scenario))
    game.set_window_visible(render)

    if 'doom_skill' in config:
        game.set_doom_skill(config['doom_skill'])

    # --- Rewards ---
    if 'living_reward' in config:
        game.set_living_reward(config['living_reward'])
    if 'death_penalty' in config:
        game.set_death_penalty(config['death_penalty'])

    # --- Rendering Options ---
    # game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    # if 'screen_resolution' in config:
    #     if res := get_vzd_enum('ScreenResolution', config['screen_resolution']):
    #         game.set_screen_resolution(res)
    # if 'screen_format' in config:
    #     if fmt := get_vzd_enum('ScreenFormat', config['screen_format']):
    #         game.set_screen_format(fmt)

    if 'render_hud' in config:
        game.set_render_hud(config['render_hud'])
    if 'render_crosshair' in config:
        game.set_render_crosshair(config['render_crosshair'])
    if 'render_weapon' in config:
        game.set_render_weapon(config['render_weapon'])
    if 'render_decals' in config:
        game.set_render_decals(config['render_decals'])
    if 'render_particles' in config:
        game.set_render_particles(config['render_particles'])

    # --- Episode Control ---
    if 'episode_start_time' in config:
        game.set_episode_start_time(config['episode_start_time'])
    if 'episode_timeout' in config:
        game.set_episode_timeout(config['episode_timeout'])

    # --- Player Controls ---
    if 'available_buttons' in config and isinstance(config['available_buttons'], list):
        buttons = [btn for b in config['available_buttons'] if (btn := get_vzd_enum('Button', b))]
        game.set_available_buttons(buttons)

    # --- Game State Information ---
    if 'available_game_variables' in config and isinstance(config['available_game_variables'], list):
        variables = [var for v in config['available_game_variables'] if (var := get_vzd_enum('GameVariable', v))]
        game.set_available_game_variables(variables)

    # --- Game Mode ---
    if 'mode' in config:
        if mode := get_vzd_enum('Mode', config['mode']):
            game.set_mode(mode)

    # 3. Initialize the game with the configured settings.
    game.init()
    return game


def grayscale(observation):
    """
    Converts the observation to grayscale and resizes it to a fixed size.

    :param observation: The observation from the game, expected to be in RGB format.
    :return: A grayscale image of shape (100, 160, 1).
    """
    # 1) Input is RGB (320x240), convert to grayscale and resize
    # gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    # resize = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_CUBIC)
    # state = np.reshape(resize, (120, 160, 1))
    # return state

    # 2) Input is already grayscale (160x120), resize to (120, 160, 1)
    # return np.resize(observation, (120, 160, 1))

    # 3) Input is already grayscale (160x120), resize to (100, 160, 1)
    resize = cv2.resize(observation, (160, 100), interpolation=cv2.INTER_CUBIC)
    state = np.reshape(resize, (100, 160, 1))
    return state


def get_action_space(game: vzd.DoomGame):
    """
    Returns the action space of the game.

    :param game: The initialized ViZDoom game instance.
    :return: A numpy array representing the action space, where each action corresponds to a button.
    """
    return np.identity(len(game.get_available_buttons()))


def get_state_info(game: vzd.DoomGame):
    """
    Returns the state information of the game. This includes available game variables and their values.

    :param game: The initialized ViZDoom game instance.
    :return: A dictionary with game variable names and their values.
    """
    info = game.get_available_game_variables()
    info = {f"{info[i].name}": info[i].value for i in range(len(info))}
    return info