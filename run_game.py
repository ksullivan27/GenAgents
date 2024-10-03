import traceback  # Importing traceback for error handling and stack trace generation
from typing import TYPE_CHECKING  # Importing TYPE_CHECKING for conditional type checking
import argparse  # Importing argparse for command-line argument parsing
import sys  # Importing sys for system-specific parameters and functions

# Conditional import for type checking; Game class is imported only if type checking is enabled
if TYPE_CHECKING:
    from text_adventure_games.games import Game  # Importing Game class from text_adventure_games module

# Importing GptParser3 for parsing game input
from text_adventure_games.parsing import GptParser3  

# Importing functions to build different types of games from the game_setup module
from test.game_setup import build_exploration, build_classic, build_discovery  

def main():
    """Run the main game execution flow.

    This function orchestrates the game setup and execution process. It parses command-line arguments, sets up the game
    based on those arguments, and then runs the game.

    Args:
        None

    Returns:
        None
    """

    args = parse_args()  # Parse command-line arguments to retrieve user input and options
    experiment_game = setup(args)  # Set up the game environment based on the parsed arguments
    run(experiment_game)  # Execute the game with the configured environment


def parse_args():
    """Parse command-line arguments for the experiment configuration.

    This function sets up the argument parser to handle user input for running an experiment. It defines required and
    optional parameters, ensuring that the necessary information is provided to execute the game correctly.

    Args:
        None

    Returns:
        Namespace: An object containing the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")  # Create an argument parser with a description of its purpose

    # Required arguments
    parser.add_argument("experiment_method", type=str, choices=['classic', 'exploration', 'discovery'], help="Method of the experiment. Supported: 'classic', 'exploration', 'discovery'.")  # Define the experiment method as a required argument with specific choices
    parser.add_argument("experiment_name", type=str, help="Name of the experiment.")  # Define the experiment name as a required argument
    parser.add_argument("experiment_id", type=int, help="ID of the experiment.")  # Define the experiment ID as a required integer argument
    parser.add_argument("personas_path", type=str, help="The full path to persona files you want to use or their folder name within the assets folder.")  # Define the path to persona files as a required argument

    # Optional arguments with default values
    parser.add_argument("--num_characters", type=int, default=4, help="The number of agents to create in the game (default: 4)")  # Define the number of characters as an optional argument with a default value
    parser.add_argument("--max_ticks", type=int, default=6, help="Maximum number of ticks per round (default: 6).")  # Define the maximum ticks per round as an optional argument with a default value
    parser.add_argument("--max_rounds", type=int, default=10, help="The maximum number of rounds to play. Currently only valid for DiscoveryGame. (default: 10).")  # Define the maximum rounds as an optional argument with a default value
    parser.add_argument("--num_finalists", type=int, default=2, help="Number of finalists (default: 2).")  # Define the number of finalists as an optional argument with a default value
    parser.add_argument("--architecture", type=str, default="A", help="Type of architecture (default: 'A').")  # Define the architecture type as an optional argument with a default value
    parser.add_argument("--random_placement", type=bool, default=False, help="Should characters be placed randomly across the map? (default: False)")  # Define random placement as an optional argument with a default value

    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])  # Parse the arguments and return them; show help if no arguments are provided


def setup(args) -> "Game":
    """Set up the game based on the provided arguments.

    This function initializes a game instance according to the specified experiment method and parameters. It configures
    the game settings and prepares the game for execution, returning the created game object.

    Args:
        args (Namespace): The parsed command-line arguments containing configuration for the game setup.

    Returns:
        Game: An instance of the game configured with the specified parameters.

    Raises:
        ValueError: If the experiment method is not recognized or if required parameters are missing.
    """

    print("Setting up the game")  # Output a message indicating that the game setup process has started
    game_created = False  # Initialize a flag to track whether the game has been successfully created

    # Prepare a dictionary of game arguments based on the provided command-line arguments
    game_args = {
        "experiment_name": args.experiment_name,  # Name of the experiment
        "experiment_id": args.experiment_id,      # ID of the experiment
        "max_ticks": args.max_ticks,              # Maximum number of ticks per round
        "num_finalists": args.num_finalists,      # Number of finalists in the game
        "personas_path": args.personas_path,      # Path to persona files
        "random_placement": args.random_placement   # Flag for random placement of characters
    }

    # Check the experiment method and create the corresponding game instance
    if args.experiment_method == "classic":
        game_args["num_characters"] = args.num_characters  # Add number of characters for classic method
        game = build_classic(**game_args)  # Build the classic game using the provided arguments
        game_created = True  # Set the flag to indicate the game has been created

    if args.experiment_method == "exploration":
        game_args["architecture"] = args.architecture  # Add architecture type for exploration method
        game = build_exploration(**game_args)  # Build the exploration game using the provided arguments
        game_created = True  # Set the flag to indicate the game has been created

    if args.experiment_method == "discovery":
        game_args["num_characters"] = args.num_characters  # Add number of characters for discovery method
        game_args["max_rounds"] = args.max_rounds  # Add maximum rounds for discovery method
        game = build_discovery(**game_args)  # Build the discovery game using the provided arguments
        game_created = True  # Set the flag to indicate the game has been created

    # If the game was successfully created, configure it further
    if game_created:
        game.give_hints = True  # Enable hints for the game
        parser = GptParser3(game, verbose=False)  # Initialize the parser for the game
        game.set_parser(parser)  # Set the parser for the game
        parser.refresh_command_list()  # Refresh the command list in the parser
        return game  # Return the created game instance


def run(game):
    """Execute the game loop and handle any exceptions that occur.

    This function runs the main game loop and captures any exceptions that may arise during execution. Regardless of
    whether the game completes successfully or encounters an error, it ensures that the simulation data is saved for
    later analysis.

    Args:
        game (Game): The game instance to be executed.

    Returns:
        None

    Raises:
        Exception: Any exception raised during the game loop execution will be caught and printed.
    """

    try:
        game.game_loop()  # Attempt to execute the main game loop
    except Exception as e:
        print(e)  # Print the exception message if an error occurs
        print(traceback.format_exc())  # Print the full traceback for debugging purposes
    finally:
        # This block will execute after the game loop finishes or if an error occurs
        # Save the simulation data to allow for analysis, even if it's incomplete
        game.save_simulation_data()  # Save the current state of the simulation


if __name__ == "__main__":
    print("Entering main")
    main()
