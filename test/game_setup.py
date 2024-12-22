# Import necessary libraries
import random  # For generating random numbers
import os  # For interacting with the operating system
from itertools import cycle  # For creating an iterator that cycles through an iterable
from typing import override

# Import TYPE_CHECKING to allow for type hints without circular imports.
from typing import TYPE_CHECKING, Union, List, Set, Dict

# Import modules from the text_adventure_games package
from text_adventure_games import games, things
from text_adventure_games.agent.persona import (
    Persona,
)  # Import Persona class for character personas
from text_adventure_games.agent.agent_cognition.reflect import Reflect
from text_adventure_games.assets.prompts import (
    world_info_prompt,
)  # Import world info prompt
from text_adventure_games.things.characters import (
    GenerativeAgent,
    DiscoveryAgent,
)  # Import character classes
from text_adventure_games.utils.consts import (
    get_assets_path,
)  # Function to get the path for assets
from text_adventure_games.utils.build_agent import (
    build_agent,
)  # Function to build agents
from text_adventure_games.utils.general import (
    get_logger_extras,
)  # Function to get logger extras
from text_adventure_games.actions import talk

# Mapping of groups to boolean values indicating certain properties
GROUP_MAPPING = {
    "A": (False, False),  # Group A has both properties set to False
    "B": (True, False),  # Group B has the first property set to True
    "C": (False, True),  # Group C has the second property set to True
    "D": (True, True),  # Group D has both properties set to True
}


class ClassicGame(games.SurvivorGame):
    """
    Represents a classic game setup for the SurvivorGame.

    This class initializes a game with specified parameters, allowing for customization of player characters, actions,
    and game settings.

    Args:
        start_at (things.Location): The starting location of the game.
        player (things.Character): The main character controlled by the player.
        characters (list, optional): A list of additional characters in the game. Defaults to None.
        custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
    """

    def __init__(
        self,
        start_at: things.Location,
        player: things.Character,
        characters=None,
        custom_actions=None,
        max_ticks=5,
        num_finalists=2,
        experiment_name="exp1",
        experiment_id=1,
    ):
        """
        Initializes the game with specified parameters.

        This constructor sets up the game environment, including the starting location, player character, and various
        game settings. It allows for customization of characters, actions, and experiment details.

        Args:
            start_at (things.Location): The starting location of the game.
            player (things.Character): The main character controlled by the player.
            characters (list, optional): A list of additional characters in the game. Defaults to None.
            custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
            max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
            num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
            experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
            experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        """

        # Call the initializer of the parent class to set up the game with the provided parameters.
        super().__init__(
            start_at,  # The starting location of the game.
            player,  # The main character controlled by the player.
            characters,  # Additional characters in the game (if any).
            custom_actions,  # Custom actions available in the game (if any).
            max_ticks=max_ticks,  # Maximum number of game ticks allowed.
            num_finalists=num_finalists,  # Number of finalists at the end of the game.
            experiment_name=experiment_name,  # Name of the experiment.
            experiment_id=experiment_id,
        )  # Identifier for the experiment.


class ExplorationGame(games.SurvivorGame):
    """
    Represents an exploration game that extends the SurvivorGame.

    This class initializes the exploration game with specific parameters and overrides the winning condition to include
    a unique end state check. It allows players to navigate through the game while aiming to achieve specific
    objectives.

    Args:
        start_at (things.Location): The starting location of the game.
        player (things.Character): The main character controlled by the player.
        characters (list, optional): A list of additional characters in the game. Defaults to None.
        custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
    """

    def is_won(self):
        """
        Determines if the exploration game has been won.

        This method checks if any character has found the idol, which signifies the end of the game. If a character is
        immune and has found the idol, the game is declared won.

        Returns:
            bool: True if the game has been won, otherwise False.
        """

    def __init__(
        self,
        start_at: things.Location,
        player: things.Character,
        characters=None,
        custom_actions=None,
        max_ticks=5,
        num_finalists=2,
        experiment_name="exp1",
        experiment_id=1,
    ):
        """
        Initializes the exploration game with specified parameters.

        This constructor sets up the game environment, including the starting location, player character, and various
        game settings. It allows for customization of characters, actions, and experiment details, while also defining
        the end state check for the game.

        Args:
            start_at (things.Location): The starting location of the game.
            player (things.Character): The main character controlled by the player.
            characters (list, optional): A list of additional characters in the game. Defaults to None.
            custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
            max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
            num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
            experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
            experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        """

        super().__init__(
            start_at,
            player,
            characters,
            custom_actions,
            max_ticks=max_ticks,
            num_finalists=num_finalists,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            end_state_check="on_action",
        )

    @override
    def is_won(self):
        """
        Determines if the exploration game has been won.

        This method checks each character in the game to see if any have the "immune" property, indicating they have
        found the idol. If such a character is found, the game is declared won, and a message is printed to indicate the
        winner.

        Override the default behavior of SurvivorGame is_won to specify the
        end state for the exploration game.

        Returns:
            bool: True if the game has been won, otherwise False.
        """

        # Iterate through all characters in the game to check for the winning condition.
        for character in list(self.characters.values()):
            # Check if the character has the "immune" property, indicating they found the idol.
            if character.get_property("immune"):
                # Print a message announcing the character who found the idol and end the game.
                print("{name} found the idol! Game is over".format(name=character.name))
                return True  # Return True to indicate the game has been won.
        return False  # Return False if no character has won the game.


class DiscoveryGame(games.SurvivorGame):
    """
    Represents a discovery game that extends the SurvivorGame.

    This class initializes the discovery game with specific parameters, including the number of rounds and idols. It
    provides mechanisms to determine the game's end state based on the discovery of idols or the reaching of a maximum
    number of rounds.

    Args:
        start_at (things.Location): The starting location of the game.
        player (things.Character): The main character controlled by the player.
        characters (list, optional): A list of additional characters in the game. Defaults to None.
        custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        max_rounds (int, optional): The maximum number of rounds in the game. Defaults to 10.
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
    """

    def __init__(
        self,
        start_at: things.Location,
        player: things.Character,
        characters=None,
        custom_actions=None,
        max_ticks=5,
        num_finalists=2,
        max_rounds=10,
        experiment_name="exp1",
        experiment_id=1,
    ):
        """
        Initializes the discovery game with specified parameters.

        This constructor sets up the game environment, including the starting location, player character, and various
        game settings. It allows for customization of characters, actions, and experiment details, while also
        initializing the count of remaining idols and defining the maximum number of rounds.

        Args:
            start_at (things.Location): The starting location of the game.
            player (things.Character): The main character controlled by the player.
            characters (list, optional): A list of additional characters in the game. Defaults to None.
            custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
            max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
            num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
            max_rounds (int, optional): The maximum number of rounds in the game. Defaults to 10.
            experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
            experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        """

        # Call the initializer of the parent class to set up the discovery game with the provided parameters.
        super().__init__(
            start_at,
            player,
            characters,
            custom_actions,
            max_ticks=max_ticks,
            num_finalists=num_finalists,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            end_state_check="on_action",
        )

        # Initialize the count of remaining idols in the game.
        self.remaining_idols = self._get_idols_count()

        # Set the maximum number of rounds for the game.
        self.max_rounds = max_rounds

    @override
    def is_won(self):
        """
        Determines if the discovery game has been won.

        This method checks the current state of the game to see if all idols have been found or if the maximum number of
        rounds has been reached. If either condition is met, the game is declared won, player scores are logged, and a
        message is printed to indicate the game's conclusion.

        Override the default behavior of SurvivorGame is_won to specify the end state for the discovery game.

        Returns:
            bool: True if the game has been won, otherwise False.
        """

        # Check the number of remaining idols in the game.
        remaining_idols = self._get_idols_count()

        # If all idols have been found, declare the game over and log player scores.
        if remaining_idols == 0:
            print(
                "All idols have been found! The game is over."
            )  # Notify that the game has ended.
            self._log_player_scores()  # Log the final scores of the players.
            return True  # Return True to indicate the game has been won.

        # If the maximum number of rounds has been reached, declare the game over and log player scores.
        if self.round > self.max_rounds - 1:
            print(
                "The time-limit of the game has been reached."
            )  # Notify that the time limit has been reached.
            self._log_player_scores()  # Log the final scores of the players.
            return True  # Return True to indicate the game has been won.

        # Update the remaining idols count for the next evaluation.
        self.remaining_idols = remaining_idols

        # Return False to indicate that the game is still ongoing.
        return False

    def _get_idols_count(self):
        """
        Counts the number of remaining idols in the game.

        This method iterates through all game locations to determine how many idols are still available. It checks each
        location for the presence of an idol and returns the total count of remaining idols.

        Returns:
            int: The number of remaining idols in the game.
        """

        # Initialize a counter for the remaining idols.
        remaining_idols = 0

        # Iterate through all locations in the game to check for idols.
        for location in list(self.locations.values()):
            # Check if the current location has an idol.
            if location.get_property("has_idol"):
                remaining_idols += 1  # Increment the counter if an idol is found.

        # Return the total count of remaining idols.
        return remaining_idols

    def _log_player_scores(self):
        """
        Logs the final scores of all players in the game.

        This method iterates through each character in the game and formats their final score for logging. It utilizes a
        logging mechanism to record each player's score along with additional contextual information.

        Returns:
            None
        """

        # Iterate through all characters in the game to log their final scores.
        for agent in list(self.characters.values()):
            # Create a message string that includes the agent's name and their final score formatted to two decimal places.
            message = f"{agent.name}'s final score: {agent.score:.2f}"

            # Retrieve additional logging information for the agent.
            extras = get_logger_extras(self, agent)

            # Add the type of log entry to the extras dictionary.
            extras["type"] = "Scores"

            # Log the message at the debug level, including the message and additional context.
            self.logger.debug(msg=message, extra=extras)

    def _get_player_alliance(self, ids_only=False, names_only=False, as_str=False):
        """
        Retrieves the player's alliance information.

        This method provides access to the player's teammates, allowing for different formats of the returned data.
        Depending on the specified flags, it can return only the IDs, only the names, or the full teammate objects.

        Args:
            ids_only (bool, optional): If True, return only the IDs of the teammates. Defaults to False.
            names_only (bool, optional): If True, return only the names of the teammates. Defaults to False.
            as_str (bool, optional): If True, return names as a single string. Defaults to False.

        Returns:
            list: A list of IDs, names, or teammate objects based on the specified flags.
        """

        # If the ids_only flag is set to True, return a list of IDs for each teammate.
        if ids_only:
            return [ally.id for ally in self.player.get_teammates()]

        # If the names_only flag is set to True, return a list of names for each teammate.
        if names_only:
            return list(self.player.get_teammates(names_only=names_only, as_str=as_str))

        # If neither flag is set, return the full list of teammate objects.
        return self.player.get_teammates()

    @override
    def update_world_info(self):
        """
        Updates the world information for the discovery game.

        This method gathers various parameters related to the current state of the game, including the value of
        remaining idols, contestant locations, partner counts, and remaining rounds. It formats this information into a
        structured string for use in the game's world information display.

        Returns:
            None
        """

        params = {
            "idol_value": 100 - self.total_ticks,
            "contestant_names_locs": ", ".join(
                [
                    f"{c.name} who is at {c.location.name}"
                    for c in self.characters.values()
                    if (c.id != self.player.id)
                    and (c.id not in self._get_player_alliance(ids_only=True))
                ]
            ),
            "partner_count": len(self._get_player_alliance()),
            "teammates": self._get_player_alliance(names_only=True, as_str=True),
            "game_locations": ", ".join(list(self.locations.keys())),
            "remaining_idols": self.remaining_idols,
            "rounds_remaining": 11 - self.round,
            "turns_left_this_round": self.max_ticks_per_round - (self.tick - 1),
            "n": self.round,
        }
        self.world_info = world_info_prompt.discovery_world_info.format(**params)

    def get_basic_game_goal(self):
        """
        Retrieves the basic game goal for the discovery game.

        This method constructs a string that outlines the primary objective of the game, incorporating information about
        the player's teammates. It formats this information for display in the game's world information.

        Returns:
            str: A formatted string representing the basic game goal.
        """

        params = {"teammates": self._get_player_alliance(names_only=True, as_str=True)}
        return world_info_prompt.discovery_basic_goal.format(**params)


class ConferenceGame(games.SurvivorGame):
    """
    Represents a classic game setup for the SurvivorGame.

    This class initializes a game with specified parameters, allowing for customization of player characters, actions,
    and game settings.

    Args:
        start_at (things.Location): The starting location of the game.
        player (things.Character): The main character controlled by the player.
        characters (list, optional): A list of additional characters in the game. Defaults to None.
        custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
    """

    def __init__(
        self,
        start_at: things.Location,
        player: things.Character,
        characters=None,
        custom_actions=None,
        max_ticks=5,
        experiment_name="exp1",
        experiment_id=1,
    ):
        """
        Initializes the game with specified parameters.

        This constructor sets up the game environment, including the starting location, player character, and various
        game settings. It allows for customization of characters, actions, and experiment details.

        Args:
            start_at (things.Location): The starting location of the game.
            player (things.Character): The main character controlled by the player.
            characters (list, optional): A list of additional characters in the game. Defaults to None.
            custom_actions (list, optional): A list of custom actions available in the game. Defaults to None.
            max_ticks (int, optional): The maximum number of game ticks. Defaults to 5.
            num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
            experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
            experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        """

        # Call the initializer of the parent class to set up the game with the provided parameters.
        super().__init__(
            start_at,  # The starting location of the game.
            player,  # The main character controlled by the player.
            characters,  # Additional characters in the game (if any).
            custom_actions,  # Custom actions available in the game (if any).
            max_ticks=max_ticks,  # Maximum number of game ticks allowed.
            experiment_name=experiment_name,  # Name of the experiment.
            experiment_id=experiment_id,
        )  # Identifier for the experiment.

        self.meeting_name = "Apple Inc. Q2 2025 Board Meeting"
        self.topic = (
            "Strategic Planning for the Next Quarter and Review of Q1 2025 Results"
        )

    @override
    def update_world_info(self, character=None):
        """
        Updates the world information for the conference game.

        This method gathers various parameters related to the current state of the game, including the meeting name,
        topic, and participants. It formats this information into a structured string for use in the game's world
        information display.

        Returns:
            None
        """

        characters_list = [
            c.name for c in self.characters.values() if c.id != character.id
        ]
        if len(characters_list) == 2:
            formatted_characters = " and ".join(characters_list)
        elif len(characters_list) > 2:
            formatted_characters = (
                ", ".join(characters_list[:-1]) + ", and " + characters_list[-1]
            )
        else:
            formatted_characters = ""

        params = {
            "meeting_name": self.meeting_name,
            "topic": self.topic,
            "characters": formatted_characters,
        }
        self.world_info = world_info_prompt.conference_world_info.format(**params)

    def update_impressions(self) -> None:
        """
        Update the impressions of the characters.

        This method iterates through all characters in the game and updates their impressions based on the current
        game state.

        Returns:
            None
        """
        for character in self.characters.values():
            # print("+" * 100)
            # print("UPDATING IMPRESSIONS FOR:", character.name)

            self.update_world_info(character=character)

            character.update_character_impressions(self)

            # for obs in character.memory.observations:
            #     if obs.node_type.value == 6:
            #         print(
            #             "-",
            #             obs.node_id,
            #             obs.node_round,
            #             obs.node_type,
            #             obs.node_description,
            #         )
            # print("DONE UPDATING IMPRESSIONS FOR:", character.name)
            # print("CURRENT IMPRESSIONS:")
            # print(character.impressions.impressions.items())
            # print("+" * 100)

    def update_goals(self) -> None:
        """
        Update the goals of the characters.

        This method iterates through all characters in the game and generates new goals for each character based
        on the current game state.

        Returns:
            None
        """
        for character in self.characters.values():
            # print("+" * 100)
            # print("GENERATING GOALS FOR:", character.name)

            self.update_world_info(character=character)

            character.generate_goals(self)

            # for obs in character.memory.observations:
            #     if obs.node_type.value == 5:
            #         print(
            #             "-",
            #             obs.node_id,
            #             obs.node_round,
            #             obs.node_type,
            #             obs.node_description,
            #         )

            # print("DONE GENERATING GOALS FOR:", character.name)
            # print("CURRENT GOALS:")
            # print(character.goals.goals.items())
            # print("+" * 100)

    def evaluate_goals(self) -> None:
        """
        Evaluate the goals of the characters.

        This method iterates through all characters in the game and evaluates their goals based on the current game state.

        Returns:
            None
        """
        for character in self.characters.values():
            # print("+" * 100)
            # print("EVALUATING GOALS FOR:", character.name)

            self.update_world_info(character=character)

            character.goals.evaluate_goals(self)

            # for obs in character.memory.observations:
            #     if obs.node_type.value == 5:
            #         print(
            #             "-",
            #             obs.node_id,
            #             obs.node_round,
            #             obs.node_type,
            #             obs.node_description,
            #         )

            # print("DONE EVALUATING GOALS FOR:", character.name)
            # print("+" * 100)

    def update_reflections(self) -> None:
        """
        Update the reflections of the characters.

        This method iterates through all characters in the game and forces each character to reflect on the
        current game state.

        Returns:
            None
        """
        for character in self.characters.values():
            # print("+" * 100)
            # print("REFLECTING FOR:", character.name)

            self.update_world_info(character=character)

            Reflect.reflect(self, character)

            # for obs in character.memory.observations:
            #     if obs.node_type.value == 3:
            #         print(
            #             "-",
            #             obs.node_id,
            #             obs.node_round,
            #             obs.node_type,
            #             obs.node_description,
            #         )

            # print("DONE REFLECTING FOR:", character.name)
            # print("+" * 100)

    def update_perceptions(self) -> None:
        """
        Update the perceptions of the characters.

        This method iterates through all characters in the game and allows each character to perceive their
        surroundings based on the current game state.

        Returns:
            None
        """
        for character in self.characters.values():

            # print("+" * 100)
            # print("PERCEPTING FOR:", character.name)

            self.update_world_info(character=character)

            character.perceive(self)

            # for obs in character.memory.observations:
            #     if obs.node_type.value == 4:
            #         print(
            #             "-",
            #             obs.node_id,
            #             obs.node_round,
            #             obs.node_type,
            #             obs.node_description,
            #         )

            # print("DONE PERCEPTING FOR:", character.name)
            # print("+" * 100)

    def update_cognitive_functions(
        self,
        update_round: bool = True,
        update_impressions: bool = True,
        evaluate_goals: bool = True,
        update_reflections: bool = True,
        update_perceptions: bool = True,
        update_goals: bool = True,
    ) -> None:
        """
        Update the cognitive functions of the characters.

        This method calls the update methods for impressions, goals, reflections, and perceptions in sequence,
        based on the parameters provided. Each component can be selectively updated.

        Args:
            update_round (bool): If True, update the round and ticks of the game. Defaults to True.
            update_impressions (bool): If True, update the impressions of the characters. Defaults to True.
            evaluate_goals (bool): If True, evaluate the goals of the characters. Defaults to True.
            update_reflections (bool): If True, update the reflections of the characters. Defaults to True.
            update_perceptions (bool): If True, update the perceptions of the characters. Defaults to True.
            update_goals (bool): If True, update the goals of the characters. Defaults to True.

        Returns:
            None
        """

        # print("~ INITIALIZING IMPRESSIONS UPDATE ~")
        if update_impressions:
            self.update_impressions()

        if update_reflections:
            self.update_reflections()

        if evaluate_goals:
            self.evaluate_goals()

        if update_round:
            self.round += 1
            self.tick = 0

        if update_goals:
            self.update_goals()

    @override
    def game_loop(self):
        # Set goals for all characters at the beginning of the round

        from text_adventure_games.agent.memory_stream import MemoryType
        
        # for character in self.characters.values():
        #     self.update_world_info(character=character)

        self.update_cognitive_functions(
            update_round=False,
            update_impressions=False,
            evaluate_goals=False,
            update_reflections=False,
            update_perceptions=False,
            update_goals=True,
        )

        # self.update_cognitive_functions(
        #     update_round=False,
        #     update_impressions=True,
        #     evaluate_goals=False,
        #     update_reflections=False,
        #     update_perceptions=False,
        #     update_goals=False,
        # )

        # print("PLAYER MEMORY:")
        # for observation in self.player.memory.observations:
        #     print(
        #         observation.node_id,
        #         observation.node_type,
        #         observation.node_description,
        #         observation.node_keywords,
        #         observation.node_importance,
        #     )

        # print("ATTEMPT 1", self.player.memory.get_observations_by_type(MemoryType.GOAL))
        # print("ATTEMPT 2", self.player.memory.get_observations_by_type(5))
        # print("ATTEMPT 3", self.player.memory.get_observations_by_type(MemoryType.GOAL.value))

        # Reset the dialogue state for all characters
        self.reset_character_dialogue()

        talk_action = talk.Talk(
            self,
            "Discuss the strategic plan for the next quarter and review of Q1 2025 results.",
            self.player,
            talking_to=set(self.characters.values()).difference({self.player}),
            dialogue_duration=10,
            max_iterations=None,
        )
        talk_action()
        print("~ DONE TALKING ~")

        # Save the game results so far for later analysis
        self.save_end_game_data()


def build_exploration(
    experiment_name: str = "exp1",
    experiment_id: int = 1,
    max_ticks: int = 6,
    num_finalists: int = 2,
    architecture: str = "A",
    personas_path: str = ".",
    random_placement: bool = False,
) -> games.Game:
    """
    Builds and initializes an exploration game.

    This function sets up the game environment by creating locations, adding items, and populating characters based on
    provided parameters. It returns an instance of the ExplorationGame configured with the specified settings.

    Args:
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 6.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        architecture (str, optional): The architecture type for the characters. Defaults to "A".
        personas_path (str, optional): The path to the character personas. Defaults to ".".
        random_placement (bool, optional): If True, characters are placed randomly. Defaults to False.

    Returns:
        games.Game: An instance of the ExplorationGame configured with the specified parameters.
    """

    # Build the game locations using the predefined function.
    locations = build_game_locations()

    # Create and add a machete item to the ocean location.
    machete4 = things.Item(
        "machete2",  # Unique identifier for the item.
        "a sharp machete",  # Short description of the item.
        "A SHARP MACHETE USED FOR CUTTING VINES.",  # Detailed description of the item.
    )
    locations.get("ocean").add_item(machete4)  # Add the machete to the ocean location.

    # Create and add another machete item to the jungle path location.
    machete5 = things.Item(
        "machete3",  # Unique identifier for the item.
        "a sharp machete",  # Short description of the item.
        "A SHARP MACHETE USED FOR CUTTING VINES.",  # Detailed description of the item.
    )
    locations.get("jungle_path").add_item(
        machete5
    )  # Add the machete to the jungle path location.

    # Create and add a clue item to the cliffs location.
    clue = things.Item(
        "idol clue",  # Unique identifier for the item.
        "a clue to the idol",  # Short description of the item.
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND IN THE JUNGLE WITH A MACHETE",  # Detailed description of the item.
    )
    locations.get("cliffs").add_item(clue)  # Add the clue to the cliffs location.

    # Initialize an empty list to hold character instances.
    characters = []

    # Set the starting location for the game.
    start_at = locations.get("camp")

    # Collect character data from the specified personas path.
    character_jsons = collect_game_characters(personas_path)

    # Create character instances based on the collected persona data.
    for f in character_jsons:
        persona = Persona.import_persona(f)  # Import the persona from the file.
        character = GenerativeAgent(
            persona, architecture
        )  # Create a character instance.
        location = locations.get("camp")  # Get the starting location.
        location.add_character(character)  # Add the character to the starting location.
        characters.append(character)  # Append the character to the list of characters.

        # Print a message indicating the character's starting location and group.
        print(
            f"Character {character.name} starts at {location.name} and belongs to Group {architecture}"
        )

    # Remove the first character from the list to designate them as the player.
    player = characters.pop(0)

    # Return an instance of the ExplorationGame with the specified parameters.
    return ExplorationGame(
        start_at,  # The starting location of the game.
        player,  # The player character.
        characters,  # The list of other characters in the game.
        custom_actions=None,  # Custom actions for the game (if any).
        max_ticks=max_ticks,  # Maximum number of game ticks.
        num_finalists=num_finalists,  # Number of finalists at the end of the game.
        experiment_name=experiment_name,  # Name of the experiment.
        experiment_id=experiment_id,  # Identifier for the experiment.
    )


def build_discovery(
    experiment_name: str = "exp1",
    experiment_id: int = 1,
    num_characters: int = 6,
    max_ticks: int = 6,
    num_finalists: int = 2,
    max_rounds: int = 10,
    personas_path: str = ".",
    random_placement: bool = False,
) -> games.Game:
    """
    Builds and initializes a discovery game.

    This function sets up the game environment by creating locations, assigning characters, and configuring game
    parameters based on the provided settings. It returns an instance of the DiscoveryGame configured with the
    specified parameters.

    Args:
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        num_characters (int, optional): The number of characters in the game. Defaults to 6.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 6.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        max_rounds (int, optional): The maximum number of rounds in the game. Defaults to 10.
        personas_path (str, optional): The path to the character personas. Defaults to ".".
        random_placement (bool, optional): If True, characters are placed randomly. Defaults to False.

    Returns:
        games.Game: An instance of the DiscoveryGame configured with the specified parameters.
    """

    # Build the initial game locations using the predefined functions.
    # Create basic game locations.
    locations = build_game_locations()
    # Enhance locations for the discovery game.
    locations = build_discovery_locations(locations)

    # Initialize an empty list to hold character instances.
    characters = []
    # Set the starting location for the game.
    start_at = locations.get("camp")

    # Create a cycler for character groups based on the number of characters.
    group_cycler = (
        cycle(["A", "E", "D"]) if num_characters == 6 else cycle(["A", "B", "C", "D"])
    )

    # Collect character data from the specified personas path, partitioning by role.
    character_jsons = collect_game_characters(
        personas_path, partition=["Detective", "Explorer"]
    )
    # Get detective persona files.
    detective_files = character_jsons.get("Detective", [])
    # Get explorer persona files.
    explorer_files = character_jsons.get("Explorer", [])

    # Ensure both lists have the same length by truncating the longer list.
    # Randomize the ordering of the persona files to ensure variability across experiments.
    # Find the minimum length and truncate the lists.
    min_length = min(len(detective_files), len(explorer_files))
    detective_files = detective_files[:min_length]
    explorer_files = explorer_files[:min_length]
    # Shuffle detective and explorer files.
    random.shuffle(detective_files)
    random.shuffle(explorer_files)

    # Create teams by pairing detective and explorer persona files.
    for teams in zip(detective_files, explorer_files):
        architecture = next(group_cycler)  # Get the next group architecture.
        team = []  # Initialize a new team list.
        for f in teams:
            # Import the persona from the file and create a character instance.
            persona = Persona.import_persona(f)
            character = DiscoveryAgent(persona, group=architecture)
            team.append(character)
            location = locations.get("camp")
            location.add_character(character)
            characters.append(character)
            print(
                f"Character {character.name} starts at {location.name} and belongs to Group {architecture}"
            )

        # Set teammates for each character in the team.
        for character in team:
            character.set_teammates(members=team)

    # Remove the first character from the list to designate them as the player.
    player = characters.pop(0)

    # Return an instance of the DiscoveryGame with the specified parameters.
    return DiscoveryGame(
        start_at,  # The starting location of the game.
        player,  # The player character.
        characters,  # The list of other characters in the game.
        custom_actions=None,  # Custom actions for the game (if any).
        max_ticks=max_ticks,  # Maximum number of game ticks.
        num_finalists=num_finalists,  # Number of finalists at the end of the game.
        max_rounds=max_rounds,  # Maximum number of rounds in the game.
        experiment_name=experiment_name,  # Name of the experiment.
        experiment_id=experiment_id,  # Identifier for the experiment.
    )


def build_classic(
    experiment_name: str = "exp1",
    experiment_id: int = 1,
    num_characters: int = 4,
    max_ticks: int = 6,
    num_finalists: int = 2,
    personas_path: str = ".",
    random_placement: bool = False,
) -> games.Game:
    """
    Builds and initializes a classic game.

    This function sets up the game environment by creating locations, assigning characters, and configuring game
    parameters based on the provided settings. It returns an instance of the ClassicGame configured with the specified
    parameters.

    Args:
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        num_characters (int, optional): The number of characters in the game. Defaults to 4.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 6.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        personas_path (str, optional): The path to the character personas. Defaults to ".".
        random_placement (bool, optional): If True, characters are placed randomly. Defaults to False.

    Returns:
        games.Game: An instance of the ClassicGame configured with the specified parameters.
    """

    # Build the valid starting locations for the game.
    locs = build_game_locations()  # Create game locations.
    # Initialize an empty list to hold character instances.
    characters = []

    # Determine character placement based on the random_placement flag.
    if random_placement:
        # Create a cycler for random location assignments.
        location_cycler = cycle(list(locs.values()))
        # Assign random locations to each character.
        location_assignments = [next(location_cycler) for _ in range(num_characters)]
    else:
        # Assign all characters to the camp location if random placement is not enabled.
        location_assignments = [locs.get("camp")] * num_characters

    # Create a cycler for character groups based on the group mapping.
    group_cycler = cycle(GROUP_MAPPING.keys())
    # Assign groups to characters.
    group_assignments = [next(group_cycler) for _ in range(num_characters)]
    # Shuffle the group assignments for randomness.
    random.shuffle(group_assignments)
    # Shuffle the location assignments for randomness.
    random.shuffle(location_assignments)
    # Set the starting location for the game.
    start_at = location_assignments[0]

    # Collect character data from the specified personas path.
    character_jsons = collect_game_characters(personas_path)

    # Ensure the character_jsons list has enough entries by adding None if necessary.
    if len(character_jsons) < num_characters:
        diff = num_characters - len(character_jsons)  # Calculate the difference.
        character_jsons.extend([None] * diff)  # Extend the list with None values.

    # Create character instances based on the collected persona data.
    for i, filename in enumerate(character_jsons):
        if not filename:
            # Create a default persona if no filename is provided.
            persona = build_agent(
                "An quirky contestant that is must see TV on a reality show.",
                facts_new=True,
            )
        else:
            # Import the persona from the file.
            persona = Persona.import_persona(filename)

        # Create a character instance with the persona and assigned group.
        character = GenerativeAgent(persona, group_assignments[i])
        location = location_assignments[i]
        location.add_character(character)
        characters.append(character)

        # Print the character's starting information.
        print(
            f"Character {character.name} starts at {location.name} and belongs to Group {group_assignments[i]}"
        )

    # Remove the first character from the list to designate them as the player.
    player = characters.pop(0)

    # Return an instance of the ClassicGame with the specified parameters.
    return ClassicGame(
        start_at,  # The starting location of the game.
        player,  # The player character.
        characters,  # The list of other characters in the game.
        custom_actions=None,  # Custom actions for the game (if any).
        max_ticks=max_ticks,  # Maximum number of game ticks.
        num_finalists=num_finalists,  # Number of finalists at the end of the game.
        experiment_name=experiment_name,  # Name of the experiment.
        experiment_id=experiment_id,  # Identifier for the experiment.
    )


# TODO: This is a placeholder for the conference game. It still needs to be properly implemented.
def build_conference(
    experiment_name: str = "exp1",
    experiment_id: int = 1,
    num_characters: int = 2,  # TODO: Add the actual number of characters
    max_ticks: int = 1,  # TODO: Add the actual number of ticks
    personas_path: str = ".",
    leader: str = None,
) -> games.Game:
    """
    Builds and initializes a classic game.

    This function sets up the game environment by creating locations, assigning characters, and configuring game
    parameters based on the provided settings. It returns an instance of the ClassicGame configured with the specified
    parameters.

    Args:
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        num_characters (int, optional): The number of characters in the game. Defaults to 10.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 1.
        personas_path (str, optional): The path to the character personas. Defaults to ".".
        leader (str, optional): The name of the leader character. Defaults to None.

    Returns:
        games.Game: An instance of the ClassicGame configured with the specified parameters.
    """

    # Build the valid starting locations for the game.
    locs = build_conference_locations()  # Create game locations.
    
    # Initialize an empty list to hold character instances.
    characters = []

    # Assign all characters to the conference location.
    location_assignments = [locs.get("conference")] * num_characters

    # Assign all characters to group D.
    group_assignments = ["D" for _ in range(num_characters)]
    # Set the starting location for the game.
    start_at = location_assignments[0]

    # Collect character data from the specified personas path.
    character_jsons = collect_game_characters(personas_path)

    # Ensure the character_jsons list has enough entries by adding None if necessary.
    if len(character_jsons) < num_characters:
        diff = num_characters - len(character_jsons)  # Calculate the difference.
        character_jsons.extend([None] * diff)  # Extend the list with None values.

    # Create character instances based on the collected persona data.
    for i, filename in enumerate(character_jsons):
        # Skip if the index is greater than the number of characters.
        if i >= num_characters:
            break

        if not filename:
            # Create a default persona if no filename is provided.
            persona = build_agent(
                "An quirky contestant that is must see TV on a reality show.",
                facts_new=True,
            )
        else:
            # Get the character info path from the filename.
            character_info_path = os.path.join(os.path.dirname(os.path.dirname(filename)), "General")
            # Import the persona from the file.
            persona = Persona.import_persona(filename=filename, character_info_path=character_info_path)

        # Create a character instance with the persona and assigned group.
        character = GenerativeAgent(persona, group_assignments[i])

        print(
            "Adding",
            character.name,
            f"(group {group_assignments[i]})"
        )

        location = location_assignments[i]

        location.add_character(character)

        characters.append(character)

    # Create a dictionary of characters for quick lookup.
    character_dict = {character.name: character for character in characters}
    
    # Set the player character.
    if leader is None:
        player = characters.pop(0)
    elif leader in character_dict:
        player = character_dict[leader]
    else:
        raise ValueError(f"Leader {leader} not found in characters")

    # Return an instance of the ConferenceGame with the specified parameters.
    return ConferenceGame(
        start_at,  # The starting location of the game.
        player,  # The player character.
        characters,  # The list of other characters in the game.
        custom_actions=None,  # Custom actions for the game (if any).
        max_ticks=max_ticks,  # Maximum number of game ticks.
        experiment_name=experiment_name,  # Name of the experiment.
        experiment_id=experiment_id,  # Identifier for the experiment.
    )


def collect_game_characters(personas_path, partition: List[str] = None):
    """
    Collects game character files from the specified directory.

    This function retrieves character files in JSON format from a given directory, optionally filtering them based on
    specified partitions. It returns a list or dictionary of character file paths, depending on whether partitions are
    provided.

    Args:
        personas_path (str): The path to the directory containing character persona files.
        partition (List[str], optional): A list of keywords to filter character files. Defaults to None.

    Returns:
        Union[List[str], Dict[str, List[str]]]: A list of character file paths or a dictionary of lists categorized by
        partition.
    """

    # Check if the provided personas_path is a valid directory.
    if not os.path.isdir(personas_path):
        # If not, retrieve the path to the package assets.
        package_assets = get_assets_path()
        # Construct the full path to the personas directory within the package assets.
        personas_path = os.path.join(package_assets, personas_path)

    # Initialize a dictionary to hold character files categorized by partition, or an empty list if no partition is
    # provided.
    character_files = {key: [] for key in partition} if partition else []

    # Check if the personas_path exists before attempting to list its contents.
    if os.path.exists(personas_path):
        # Iterate through all files in the personas directory.
        for filename in os.listdir(personas_path):
            # Check if the file has a .json extension.
            if filename.endswith(".json"):
                # Construct the full path to the character file.
                character_path = os.path.join(personas_path, filename)
                # If partitions are specified, categorize the character files accordingly.
                if partition:
                    for key in partition:
                        # Check if the partition key is in the filename.
                        if key in filename:
                            character_files[key].append(
                                character_path
                            )  # Add the file path to the corresponding partition.
                            break  # Exit the loop once the file is added to a partition.
                else:
                    # If no partitions are specified, simply add the character file path to the list.
                    character_files.append(character_path)

    # Return the collected character files, either as a list or a dictionary of lists.
    return character_files


def build_game_locations():
    """
    Builds and initializes the game locations.

    This function creates various locations within the game, sets their properties, and establishes connections between
    them. It also adds items to specific locations and returns a dictionary of the initialized locations for use in the
    game.

    Returns:
        dict: A dictionary containing the initialized game locations, including camp, cliffs, beach, ocean, jungle path,
        and well.
    """

    # Create various locations for the game with descriptions.
    camp = things.Location(
        "Camp", "the tribe's base camp."
    )  # The main base for the tribe.
    cliffs = things.Location(
        "Cliffs",
        """the front of some steep cliffs.
            Climb them carefully so you don't fall.""",  # A steep cliff area.
    )
    beach = things.Location(
        "Beach",
        "the beach, toes in the sand. In front of you is the vast ocean.",  # A sandy beach area.
    )
    ocean = things.Location(
        "Ocean",
        "the edge of the ocean with waves washing up around your knees.",  # The ocean area.
    )
    jungle_path = things.Location(
        "Jungle Path",
        "a jungle path towards the well.",  # A path leading through the jungle.
    )
    well = things.Location(
        "Well",
        "the water well where you can get water for your tribe.",  # A well for obtaining water.
    )
    jungle = things.Location(
        "Jungle",
        "the deep jungle. There could be treasures hiding nearby.",  # A dense jungle area.
    )

    # Set properties for the jungle location.
    jungle.set_property("has_idol", True)
    jungle.set_property("tool_required", True)
    jungle.set_property("idol_found", False)
    jungle_fail_message = "but the vines get in the way and it becomes impossible without the right tool (a machete!)."
    jungle.set_property("search_fail", jungle_fail_message)
    jungle.set_property(
        "found_message",
        "This idol has already been found by another team! Hurry to find one of the remaining idols!",
    )

    # Establish connections between locations.
    camp.add_connection("out", beach)
    beach.add_connection("north", jungle_path)
    beach.add_connection("south", ocean)
    beach.add_connection("west", cliffs)
    beach.add_connection("in", camp)
    jungle_path.add_connection("south", beach)
    jungle_path.add_connection("east", well)
    jungle_path.add_connection("north", jungle)
    well.add_connection("west", jungle_path)
    jungle.add_connection("south", jungle_path)
    ocean.add_connection("north", beach)
    cliffs.add_connection("east", beach)

    # Create and add gettable items to specific locations.
    fishing_pole = things.Item(
        "pole",
        "a fishing pole",
        "A SIMPLE FISHING POLE.",
    )
    ocean.add_item(fishing_pole)
    ocean.set_property("has_fish", True)

    # Create and add machete items to various locations.
    machete1 = things.Item(
        "machete1",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    camp.add_item(machete1)

    machete2 = things.Item(
        "machete2",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    well.add_item(machete2)

    machete3 = things.Item(
        "machete3",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    beach.add_item(machete3)

    # Return a dictionary of all initialized locations, ensuring the jungle is not a starting point.
    return {
        "camp": camp,
        "cliffs": cliffs,
        "beach": beach,
        "ocean": ocean,
        "jungle_path": jungle_path,
        "well": well,
    }


def build_discovery_locations(base_locations):
    """
    Builds additional discovery locations and their properties.

    This function enhances the base locations by adding new locations, establishing connections between them, and
    setting properties for items and clues. It returns the updated dictionary of locations, including newly created
    locations such as the waterfall, rocky shore, and lazy river.

    Args:
        base_locations (dict): A dictionary of existing base locations to enhance.

    Returns:
        dict: The updated dictionary of locations, including additional locations and their properties.
    """

    # Establish additional locations and their connections.
    base_locations.get("camp").add_connection(
        "north", base_locations.get("well")
    )  # Connect camp to the well.

    # Create a new location for the waterfall with a description.
    waterfall = things.Location(
        "Waterfall",
        "A stunning waterfall creates a veil of mist.",
    )
    # Set properties for the waterfall location.
    waterfall.set_property("has_idol", True)
    waterfall.set_property("tool_required", True)
    waterfall.set_property("idol_found", False)
    waterfall.set_property(
        "found_message",
        "This idol has already been found by another team! Hurry to find one of the remaining idols!",
    )
    waterfall_fail_message = "but the rocks are too slippery and it becomes impossible without the right tool (a sturdy stick!)."
    waterfall.set_property("search_fail", waterfall_fail_message)
    waterfall.add_connection("west", base_locations.get("well"))
    base_locations.get("well").add_connection("east", waterfall)

    # Create a new location for the rocky shore with a description.
    rocky_shores = things.Location(
        "rocky shore", "Slippery tidepools with rocks beaten by waves."
    )
    # Set properties for the rocky shores location.
    rocky_shores.set_property("has_idol", True)
    rocky_shores.set_property("tool_required", False)
    rocky_shores.set_property("idol_found", False)
    rocky_shores.set_property(
        "found_message",
        "This idol has already been found by another team! Hurry to find one of the remaining idols!",
    )
    rocky_shores.set_property(
        "search_fail",
        "but the tide is too high and dangerous to wade across the rocks. It will subside next round and you should try again then! ",
    )
    rocky_shores.add_connection("north", base_locations.get("camp"))
    base_locations.get("camp").add_connection("south", rocky_shores)

    # Create a new location for the lazy river with a description.
    lazy_river = things.Location("lazy river", "the banks of a murky, winding river")
    lazy_river.add_connection("south", base_locations.get("well"))
    base_locations.get("well").add_connection("north", lazy_river)

    # Create and add gettable items to specific locations.
    stick1 = things.Item(
        "stick", "a long stick", "A sturdy stick to keep balanced on slippery rocks."
    )
    base_locations.get("well").add_item(stick1)

    stick2 = things.Item(
        "stick", "a long stick", "A sturdy stick to keep balanced on slippery rocks."
    )
    base_locations.get("beach").add_item(stick2)

    stick3 = things.Item(
        "stick", "a long stick", "A sturdy stick to keep balanced on slippery rocks."
    )
    base_locations.get("ocean").add_item(stick3)

    stick4 = things.Item(
        "stick", "a long stick", "A sturdy stick to keep balanced on slippery rocks."
    )
    base_locations.get("jungle_path").add_item(stick4)

    # Create and add machete items to various locations.
    machete4 = things.Item(
        "machete2",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    base_locations.get("ocean").add_item(machete4)

    machete5 = things.Item(
        "machete3",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    base_locations.get("jungle_path").add_item(machete5)

    # Create clues that provide hints about idol locations.
    clue1 = things.Item(
        "idol clue",
        "a clue to the idol",
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND IN THE JUNGLE WITH A MACHETE",
    )
    base_locations.get("cliffs").add_item(clue1)  # Add the clue to the cliffs location.
    clue1_message = "".join(
        [
            "'An idol can be found by searching the jungle with a machete.' ",
            "'You can fail this action but keep trying as long as you have a machete and are in the jungle!' ",
            "'If you pick up and hold this clue while searching, you'll have a better chance of discovering the idol!'",
        ]
    )
    clue1.set_property("clue_content", clue1_message)  # Set the clue content for clue1.

    clue2 = things.Item(
        "idol clue",
        "a clue to the idol",
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND IN THE WATERFALL WITH A STICK",
    )
    base_locations.get("well").add_item(clue2)  # Add the clue to the well location.
    clue2_message = "".join(
        [
            "'An idol can be found by searching the waterfall with a sturdy stick.' ",
            "'You can fail this action but keep trying as long as you have a stick and are at the waterfall!' ",
            "'If you pick up and hold this clue while searching, you'll have a better chance of discovering the idol!'",
        ]
    )
    clue2.set_property("clue_content", clue2_message)  # Set the clue content for clue2.

    clue3 = things.Item(
        "idol clue",
        "a clue to the idol",
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND IN THE JUNGLE WITH A MACHETE",
    )
    base_locations.get("ocean").add_item(clue3)  # Add the clue to the ocean location.
    clue3_message = "".join(
        [
            "'An idol can be found by searching the jungle with a machete.' ",
            "'You can fail this action but keep trying as long as you have a machete and are in the jungle!' ",
            "'If you pick up and hold this clue while searching, you'll have a better chance of discovering the idol!'",
        ]
    )
    clue3.set_property("clue_content", clue3_message)  # Set the clue content for clue3.

    clue4 = things.Item(
        "idol clue",
        "a clue to the idol",
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND ON THE ROCKY SHORES DURING CERTAIN ROUNDS",
    )
    lazy_river.add_item(clue4)  # Add the clue to the lazy river location.
    clue4_message = "".join(
        [
            "'An idol can be found by searching rocky shores, but be careful of the tide!' ",
            "'The tide behaves in a cyclic manner, so you must plan your search for the correct timing. ' ",
            "'If you pick up and hold this clue while searching, you'll have a better chance of discovering the idol!'",
        ]
    )
    clue4.set_property("clue_content", clue4_message)  # Set the clue content for clue4.

    # Add the new locations to the base_locations dictionary.
    base_locations["waterfall"] = waterfall  # Add waterfall to the locations.
    base_locations["rocky_shore"] = rocky_shores  # Add rocky shore to the locations.
    base_locations["lazy_river"] = lazy_river  # Add lazy river to the locations.

    # Return the updated dictionary of locations.
    return base_locations


def build_mini_discovery(
    experiment_name: str = "exp1",
    experiment_id: int = 1,
    max_ticks: int = 6,
    num_finalists: int = 2,
    personas_path: str = ".",
    random_placement: bool = False,
) -> games.Game:
    """
    Builds and initializes a mini discovery game.

    This function sets up a simplified game environment by creating a location, adding items and clues, and configuring
    game parameters based on the provided settings. It returns an instance of the DiscoveryGame configured with the
    specified parameters.

    Args:
        experiment_name (str, optional): The name of the experiment. Defaults to "exp1".
        experiment_id (int, optional): The identifier for the experiment. Defaults to 1.
        max_ticks (int, optional): The maximum number of game ticks. Defaults to 6.
        num_finalists (int, optional): The number of finalists at the end of the game. Defaults to 2.
        personas_path (str, optional): The path to the character personas. Defaults to ".".
        random_placement (bool, optional): If True, characters are placed randomly. Defaults to False.

    Returns:
        games.Game: An instance of the DiscoveryGame configured with the specified parameters.
    """

    # Create a new location for the cliffs with a description.
    cliffs = things.Location(
        "Cliffs",
        """the front of some steep cliffs.
            Climb them carefully so you don't fall.""",
    )

    # Create a clue item that provides information about the idol's location.
    clue = things.Item(
        "idol clue",
        "a clue to the idol",
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND IN THE JUNGLE WITH A MACHETE",
    )
    cliffs.add_item(clue)

    # Construct the message content for the clue.
    clue1_message = "".join(
        [
            "'An idol can be found by searching the jungle with a machete.' ",
            "'You can fail this action but keep trying as long as you have a machete and are in the jungle!' ",
            "'If you pick up and hold this clue while searching, you'll have a better chance of discovering the idol!'",
        ]
    )
    clue.set_property("clue_content", clue1_message)

    # Set properties for the cliffs location.
    cliffs.set_property("has_idol", True)
    cliffs.set_property("tool_required", True)
    cliffs_message = "but the rocks are too slippery and it becomes impossible without the right tool (a sturdy stick!)."
    cliffs.set_property(
        "search_fail", cliffs_message
    )  # Message for failed search attempts.

    # Create a stick item that can be used for balance.
    stick4 = things.Item(
        "stick",
        "a long stick",
        "A sturdy stick to keep balanced on slippery rocks.",
    )
    cliffs.add_item(stick4)

    # Initialize an empty list to hold character instances.
    characters = []
    # Set the starting location for the game.
    start_at = cliffs  # The game starts at the cliffs location.

    # Iterate through the files in the exploration personas directory to create characters.
    for i, filename in enumerate(os.listdir("exploration_personas")):
        if i > 1:  # Limit the number of characters to 2.
            break
        if filename.endswith(".json"):  # Check if the file is a JSON file.
            persona = Persona.import_persona(
                f"exploration_personas/{filename}"
            )  # Import the persona from the file.
            character = DiscoveryAgent(
                persona, "B"
            )  # Create a character instance with the imported persona.
            location = cliffs  # Set the character's location to the cliffs.
            location.add_character(
                character
            )  # Add the character to the cliffs location.
            characters.append(
                character
            )  # Append the character to the list of characters.
            # Print the character's starting information.
            print(
                f"Character {character.name} starts at {location.name} and belongs to Group B"
            )

    # Remove the first character from the list to designate them as the player.
    player = characters.pop(0)  # The player is the first character in the list.

    return DiscoveryGame(
        start_at,
        player,
        characters,
        custom_actions=None,
        max_ticks=max_ticks,
        num_finalists=num_finalists,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
    )


def build_conference_locations():
    """
    Builds and initializes the game locations.

    This function creates various locations within the game, sets their properties, and establishes connections between
    them. It also adds items to specific locations and returns a dictionary of the initialized locations for use in the
    game.

    Returns:
        dict: A dictionary containing the initialized game locations, including camp, cliffs, beach, ocean, jungle path,
        and well.
    """

    # Create various locations for the game with descriptions.
    conference = things.Location(
        "Conference", "the conference where the meetings take place."
    )  # The main base for the tribe.

    # Set properties for the jungle location.
    # There are no properties for the conference

    # Add a connections
    # There are no connections for the conference

    # Return a dictionary of all initialized locations, ensuring the jungle is not a starting point.
    return {
        "conference": conference,
    }
