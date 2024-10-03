import logging
import os
import random
from SurvivorWorld.text_adventure_games import games, things, actions, blocks
from SurvivorWorld.text_adventure_games.agent.persona import Persona
from SurvivorWorld.text_adventure_games.things.characters import GenerativeAgent
from SurvivorWorld.text_adventure_games.utils.build_agent import build_agent
from SurvivorWorld.text_adventure_games.utils.custom_logging import logging_setup
from SurvivorWorld.text_adventure_games.utils.custom_logging import logger

GROUP_MAPPING = {"A": (False, False),
                 "B": (True, False),
                 "C": (False, True),
                 "D": (True, True)}


class ExperimentGame(games.SurvivorGame):
    """
    Represents an experimental version of the Survivor game, extending the functionality of the base SurvivorGame class.
    This class initializes the game with specific parameters for conducting experiments, including the starting
    location, player character, and various game settings.

    Args:
        start_at (things.Location): The starting location of the player in the game.
        player (things.Character): The player character controlled by the user.
        characters (list, optional): A list of additional characters (NPCs) to include in the game.
        custom_actions (list, optional): A list of custom actions to be added to the game's parser.
        max_ticks (int, optional): The maximum number of ticks per round, defaulting to 5.
        num_finalists (int, optional): The number of finalists in the game, defaulting to 2.
        experiment_name (str, optional): The name of the experiment, defaulting to "exp1".
        experiment_id (int, optional): The ID of the experiment, defaulting to 1.

    Returns:
        None
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
            experiment_id=1
    ):
        """
        Initializes an instance of the ExperimentGame class, which is a specialized version of the SurvivorGame. This
        constructor sets up the game with specific parameters for conducting experiments, including the starting
        location, player character, and various game settings.

        Args:
            start_at (things.Location): The starting location of the player in the game.
            player (things.Character): The player character controlled by the user.
            characters (list, optional): A list of additional characters (NPCs) to include in the game.
            custom_actions (list, optional): A list of custom actions to be added to the game's parser.
            max_ticks (int, optional): The maximum number of ticks per round, defaulting to 5.
            num_finalists (int, optional): The number of finalists in the game, defaulting to 2.
            experiment_name (str, optional): The name of the experiment, defaulting to "exp1".
            experiment_id (int, optional): The ID of the experiment, defaulting to 1.

        Returns:
            None
        """

        super().__init__(start_at,
                         player, 
                         characters, 
                         custom_actions, 
                         max_ticks=max_ticks, 
                         num_finalists=num_finalists,
                         experiment_name=experiment_name,
                         experiment_id=experiment_id)
            
def build_experiment(experiment_name, experiment_id, max_ticks=6, num_finalists=2, architecture="A") -> games.Game:
    """
    Builds and initializes an experimental game environment with specified parameters. This function sets up locations,
    items, and characters, and returns an instance of the ExperimentGame configured for the given experiment.

    Args:
        experiment_name (str): The name of the experiment.
        experiment_id (int): The ID of the experiment.
        max_ticks (int, optional): The maximum number of ticks per round, defaulting to 6.
        num_finalists (int, optional): The number of finalists in the game, defaulting to 2.
        architecture (str, optional): The architecture type for the characters, defaulting to "A".

    Returns:
        games.Game: An instance of the ExperimentGame configured with the specified parameters.
    """

    # Define Locations
    camp = things.Location(
        "Camp",
        "the tribe's base camp."
    )
    cliffs = things.Location(
        "Cliffs",
        """the front of some steep cliffs.
            Climb them carefully so you don't fall.""",
    )
    beach = things.Location(
        "Beach",
        "the beach, toes in the sand. In front of you is the vast ocean."
    )
    ocean = things.Location(
        "Ocean",
        "the edge of the ocean with waves washing up around your knees.",
    )
    jungle_path = things.Location(
        "Jungle Path",
        "a jungle path towards the well.",
    )
    well = things.Location(
        "Well",
        "the water well where you can get water for your tribe.",
    )
    jungle = things.Location(
        "Jungle",
        "the deep jungle. There could be treasures hiding nearby.",
    )
    # Set a property for the jungle location indicating it has an idol
    jungle.set_property("has_idol", True)

    # Define connections between locations
    camp.add_connection("out", beach)  # From camp to beach
    beach.add_connection("north", jungle_path)  # From beach to jungle path
    beach.add_connection("south", ocean)  # From beach to ocean
    beach.add_connection("west", cliffs)  # From beach to cliffs
    beach.add_connection("in", camp)  # From beach back to camp
    jungle_path.add_connection("south", beach)  # From jungle path to beach
    jungle_path.add_connection("east", well)  # From jungle path to well
    jungle_path.add_connection("north", jungle)  # From jungle path to jungle
    well.add_connection("west", jungle_path)  # From well to jungle path
    jungle.add_connection("south", jungle_path)  # From jungle to jungle path
    ocean.add_connection("north", beach)  # From ocean to beach
    cliffs.add_connection("east", beach)  # From cliffs to beach

    # Define Gettable Items
    fishing_pole = things.Item(
        "pole",
        "a fishing pole",
        "A SIMPLE FISHING POLE.",
    )
    # Add fishing pole to the ocean location
    ocean.add_item(fishing_pole)
    # Set a property for the ocean indicating it has fish
    ocean.set_property("has_fish", True)

    # Define machete items and add them to respective locations
    machete1 = things.Item(
        "machete1",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    camp.add_item(machete1)  # Add machete1 to camp

    machete2 = things.Item(
        "machete2",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    well.add_item(machete2)  # Add machete2 to well

    machete3 = things.Item(
        "machete3",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    beach.add_item(machete3)  # Add machete3 to beach

    machete4 = things.Item(
        "machete2",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    ocean.add_item(machete4)  # Add machete4 to ocean

    machete5 = things.Item(
        "machete3",
        "a sharp machete",
        "A SHARP MACHETE USED FOR CUTTING VINES.",
    )
    jungle_path.add_item(machete5)  # Add machete5 to jungle path

    # Define exploration clue item
    clue = things.Item(
        "idol clue",
        "a clue to the idol",
        "A CLUE THAT SAYS THE IDOL CAN BE FOUND IN THE JUNGLE WITH A MACHETE",
    )
    # Add clue to the cliffs location
    cliffs.add_item(clue)

    # Initialize characters list
    characters = []
    # Set starting location for exploration
    start_at = camp


    # Iterate over files in the "exploration_personas" directory
    for i, filename in enumerate(os.listdir("exploration_personas")):
        # Check if the file has a .json extension
        if filename.endswith(".json"):
            # Import the persona from the JSON file
            persona = Persona.import_persona("exploration_personas/" + filename)
            # Create a GenerativeAgent instance using the imported persona
            character = GenerativeAgent(persona, architecture)
            # Set the character's location to the camp
            location = camp
            # Add the character to the specified location
            location.add_character(character)
            # Append the character to the characters list
            characters.append(character)
            # Print the character's name, location, and group information
            print(f"Character {character.name} starts at {location.name} and belongs to Group {architecture}")

    # Remove the first character from the list and assign it to the player variable
    player = characters.pop(0)

    # Initialize the game with the specified parameters
    game = ExperimentGame(start_at, 
                        player, 
                        characters, 
                        custom_actions=None,
                        max_ticks=max_ticks,
                        num_finalists=num_finalists,
                        experiment_name=experiment_name,
                        experiment_id=experiment_id)

    # Return the initialized game instance
    return game

