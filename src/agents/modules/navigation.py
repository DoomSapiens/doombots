from src.agent import BaseAgent


class NavigationModule(BaseAgent):
    """
    Navigation module for the agent, inheriting from BaseAgent.
    This module is responsible for navigating through the environment.
    """

    def __init__(self, scenario='basic', render=False):
        super().__init__(scenario=scenario, render=render)


    def navigate(self, action):
        """
        Perform navigation based on the given action.
        :param action: The action to perform.
        :return: The result of the step in the environment.
        """
        return self.step(action)