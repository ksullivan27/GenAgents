"""
Author: 

File: agent_cognition/perceive.py
Description: defines how agents perceive their environment
"""

from typing import TYPE_CHECKING, Dict

# Local imports for memory types and utility functions.
from text_adventure_games.agent.memory_stream import MemoryType
from text_adventure_games.utils.general import (
    parse_location_description,  # Function to parse descriptions of locations.
    find_difference_in_dict_lists,  # Function to find differences between lists of dictionaries.
)
from text_adventure_games.gpt.gpt_helpers import (
    gpt_get_action_importance,
)  # Function to assess the importance of actions based on GPT.

# Conditional import statements for type checking to avoid circular dependencies.
if TYPE_CHECKING:
    from text_adventure_games.games import Game  # Import Game class for type hinting.
    from text_adventure_games.things import (
        Character,  # Import Character class for type hinting.
    )


def collect_perceptions(game: "Game"):
    """
    Collects the latest information about the character's current location.

    Args:
        game (Game): The current game instance from which to collect location data.

    Returns:
        The description of the current location as provided by the game.
    """

    # Call the game's describe method to get the latest location information.
    return game.describe()


def perceive_location(game: "Game", character: "Character"):
    """
    Updates the character's perception of their current location by collecting and analyzing observations.
    This function identifies any changes in the environment since the last observation and updates the character's
    memory accordingly to store these observations as new memories (of type MemoryType.ACTION).

    Args:
        game (Game): The current game instance that contains the game state and mechanics.
        character (Character): The character whose perceptions are being updated.

    Returns:
        None
    """

    # Collect the current perceptions of the character's location from the game (includes exits, items, other
    # characters, and agent's inventory).
    location_description = collect_perceptions(game)

    # Parse the collected location description into structured observations.
    location_observations = parse_location_description(location_description)

    # Check for differences between the last observations and the current observations.
    diffs_perceived = find_difference_in_dict_lists(
        character.last_location_observations, location_observations
    )

    # Update the character's last location observations with the current observations.
    character.last_location_observations = location_observations.copy()

    # Add any new observations that have been perceived since the last update.
    add_new_observations(game, character, new_percepts=diffs_perceived)


def add_new_observations(game: "Game", character: "Character", new_percepts: Dict):
    """
    Processes and adds new observations made by a character in the game.
    This function iterates through the new percepts, logs what the character sees, and updates the character's memory
    with relevant details.

    Args:
        game (Game): The current game instance that contains the game state and mechanics.
        character (Character): The character making the observations.
        new_percepts (Dict): A dictionary containing new percepts observed by the character.

    Returns:
        None
    """

    # Iterate through the new percepts to create observations based on the differences.
    for observations in new_percepts.values():
        # Print what the character sees for debugging or logging purposes.
        print(f"{character.name} sees: {observations}")

        # Define the command that represents the action of looking around.
        command = "Look around at the surroundings"

        # Process each statement in the observations.
        for statement in observations:
            # Summarize and score the action based on the statement and the character's context.
            action_statement, action_importance, action_keywords = (
                game.parser.summarize_and_score_action(
                    statement, character, command=command
                )
            )

            # Add the summarized action to the character's memory with relevant details.
            character.memory.add_memory(
                round=game.round,  # Current round of the game.
                tick=game.tick,  # Current tick of the game.
                description=action_statement,  # Description of the action.
                keywords=action_keywords,  # Keywords associated with the action.
                location=character.location.name,  # Location of the character.
                success_status=True,  # Status indicating the action was successful.
                memory_importance=action_importance,  # Importance level of the memory.
                memory_type=MemoryType.PERCEPT.value,  # Type of memory being added.
                actor_id=character.id,  # ID of the character making the observation.
            )
