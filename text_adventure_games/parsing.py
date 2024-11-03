"""The Parser

The parser is the module that handles the natural language understanding in the game. The players enter commands in
text, and the parser interprets them and performs the actions that the player intends. This is the module with the most
potential for improvement using modern natural language processing. The implementation that I have given below only uses
simple keyword matching.
"""

circular_import_prints = False

if circular_import_prints:
    print("Importing Parser")

# Import necessary modules and types for the text adventure game parsing functionality
from typing import TYPE_CHECKING  # For conditional type checking
from collections import defaultdict  # For creating default dictionaries
import inspect  # For introspection of live objects
import textwrap  # For wrapping text
import re  # For regular expressions
import json  # For JSON handling
import tiktoken  # For tokenization
import spacy  # For natural language processing
import nltk
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from inflect import engine
from jellyfish import (
    jaro_winkler_similarity,
    levenshtein_distance,
)  # For string similarity metrics

# Download the 'wordnet' dataset.
nltk.download("wordnet")

# Importing game-related classes and functions
if circular_import_prints:
    print(f"\t{__name__} calling imports for Character")
from .things import Character  # Character class from the things module

if TYPE_CHECKING:
    if circular_import_prints:
        print(f"\t{__name__} calling type checking imports for Item and Location")
    from .things import Item, Location  # Conditional imports for type checking

    if circular_import_prints:
        print(f"\t{__name__} calling type checking imports for Thing")
    from text_adventure_games.things.base import Thing  # Base class for game objects

if circular_import_prints:
    print(f"\t{__name__} calling imports for Actions")
from . import actions  # Importing actions module

if circular_import_prints:
    print(f"\t{__name__} calling imports for Consts")
from .utils.consts import get_models_config

if circular_import_prints:
    print(f"\t{__name__} calling imports for Normalize Name")
from .utils.general import (
    normalize_name
)  # Utility function to normalize names

if circular_import_prints:
    print(f"\t{__name__} calling imports for ActionSequence")
from text_adventure_games.actions.base import ActionSequence  # Action sequence handling

# Importing GPT-related helper functions
if circular_import_prints:
    print(f"\t{__name__} calling imports for GptCallHandler")
from .gpt.gpt_helpers import (
    GptCallHandler,  # Handler for GPT calls
    limit_context_length,  # Function to limit context length
    gpt_get_action_importance,  # Function to assess action importance
    gpt_get_summary_description_of_action,  # Function to get action summary
    gpt_pick_an_option,  # Function to pick an option
    get_prompt_token_count,  # Function to count tokens in a prompt
    get_token_remainder,
)  # Function to get remaining tokens

if circular_import_prints:
    print(f"\t{__name__} calling imports for MemoryType")
from .agent.memory_stream import MemoryType  # Memory type for agent's memory stream


class Parser:
    """
    Parser class for managing and interpreting player commands in a text adventure game.

    This class facilitates the parsing of user input, maintaining command history, determining the intent behind
    commands, and how that intent is reflected in the simulated world. It provides methods for adding actions and
    blocks, processing commands, and interacting with game characters and items.

    Attributes:
        command_history (list): A list of commands issued by the player along with their responses.
        character_histories (defaultdict): A dictionary storing histories of characters.
        actions (dict): A dictionary of actions available in the game.
        blocks (dict): A dictionary of blocks available in the game.
        game (Game): A reference to the game instance.
        perspective (str): The perspective from which the game is narrated.
        echo_commands (bool): A flag indicating whether to print user commands.

    Args:
        game (Game): The game instance to be associated with the parser.
        echo_commands (bool, optional): Whether to print user commands. Defaults to False.
    """

    def __init__(self, game, echo_commands=False):
        """
        Initializes the Parser with the game instance and optional command echoing.

        This constructor sets up the command history, character histories, and the default actions and blocks for the
        game. It also establishes a reference to the game and determines whether to print user commands.

        Args:
            game (Game): The game instance to be associated with the parser.
            echo_commands (bool, optional): Whether to print user commands. Defaults to False.
        """

        if circular_import_prints:
            print(f"-\tInitializing Parser")

        # Initialize a list to store the commands issued by the player
        # along with the corresponding responses provided by the game.
        self.command_history = []

        # Initialize a defaultdict to keep track of the histories of characters.
        self.character_histories = defaultdict(list)

        # Retrieve and set the default actions available in the game.
        self.actions = game.default_actions()

        # Retrieve and set the default blocks available in the game.
        self.blocks = game.default_blocks()

        # Store a reference to the game instance for later use.
        self.game = game

        # Assign the parser instance to the game, allowing the game to access the parser.
        self.game.parser = self

        # Set the narrative perspective for the game, defaulting to third-person.
        self.perspective = "3rd"

        # Determine whether to print the user's commands based on the provided flag.
        self.echo_commands = echo_commands

    def ok(self, command: str, description: str, thing: "Thing"):
        """
        Prints a description of a successful command execution and updates the command history.

        This method is used to provide feedback to the player when a command is successfully executed. It formats the
        description for better readability and appends it to the command history for future reference.

        Args:
            command (str): The command that was executed.
            description (str): A description of the outcome of the command.
            thing (Thing): The object or entity related to the command.
        """

        print(Parser.wrap_text(description))
        self.add_description_to_history(description)

    def fail(self, command: str, description: str, thing: "Thing"):
        """
        Prints a description of a failed command execution to the console.

        This method is used to inform the player when a command cannot be executed successfully. It formats the failure
        description for better readability before displaying it.

        Args:
            command (str): The command that was attempted.
            description (str): A description of the failure reason.
            thing (Thing): The object or entity related to the command.
        """

        print(f"\n{Parser.wrap_text(description)}")

    @staticmethod
    def wrap_text(text: str, width: int = 80) -> str:
        """
        Wraps the given text to a specified width for better readability.

        This static method takes a string of text and formats it so that each line does not exceed the specified width.
        It is useful for ensuring that text output remains visually appealing and easy to read.

        Args:
            text (str): The text to be wrapped.
            width (int, optional): The maximum width of each line. Defaults to 80.

        Returns:
            str: The wrapped text with line breaks inserted as necessary.
        """

        # Split the input text into individual lines based on newline characters.
        lines = text.split("\n")

        # Wrap each line to the specified width using textwrap.fill,
        # creating a list of wrapped lines.
        wrapped_lines = [textwrap.fill(line, width) for line in lines]

        # Join the wrapped lines back into a single string with newline characters
        # separating each line and return the result.
        return "\n".join(wrapped_lines)

    def add_command_to_history(self, command: str):
        """
        Adds a user command to the command history for tracking purposes.

        This method creates a message dictionary containing the command issued by the user and appends it to the command
        history list. This allows the game to maintain a record of all commands entered by the player.

        Args:
            command (str): The command string entered by the user.
        """

        # Create a message dictionary to represent the user's command,
        # including the role as "user" and the command content.
        message = {"role": "user", "content": command}

        # Append the message dictionary to the command history list,
        # allowing the game to keep track of all user commands.
        self.command_history.append(message)

    def add_description_to_history(self, description: str):
        """
        Appends a descriptive message to the command history as an assistant response.

        This method is used to record an evocative description of game actions or outcomes, allowing the player to
        reference past events. It creates a message dictionary with the role set to "assistant" and adds it to the
        command history for tracking.

        Args:
            description (str): A description of the actions, outcomes, or setting to be recorded.
        """

        # Create a message dictionary to represent the assistant's response,
        # including the role as "assistant" and the descriptive content.
        message = {"role": "assistant", "content": description}

        # Append the message dictionary to the command history list,
        # allowing the game to keep track of all assistant responses.
        self.command_history.append(message)

    def add_action(self, action: actions.Action):
        """
        Adds an action to the parser's list of available actions.

        This method allows the parser to register a new action, making it accessible for command parsing and execution.
        The action is stored in a dictionary using its name as the key, enabling easy retrieval during gameplay.

        Args:
            action (actions.Action): The action instance to be added to the parser's actions.
        """

        self.actions[action.action_name()] = action

    def add_block(self, block):
        """
        Adds a block to the parser's list of available blocks.

        This method allows the parser to register a new block, which can be used for managing game states or scenarios.
        The block is stored in a dictionary using its class name as the key, facilitating easy access during gameplay.

        Args:
            block: The block instance to be added to the parser's blocks.
        """

        # Store the block in the blocks dictionary using its class name as the key.
        # This assignment will overwrite any existing entry for the same block class name,
        # which could result in losing previously added blocks of that type.
        self.blocks[block.__class__.__name__] = block

        # TODO: Address the overwriting issue above.
        # The following commented-out code suggests an alternative approach to avoid overwriting.
        # It retrieves the existing list of blocks for the given class name from the blocks dictionary.
        # If no blocks exist for that class name, it initializes an empty list.
        # extended_block = self.blocks.get(block.__class__.__name__, [])

        # It then extends the retrieved list with the new block instance.
        # This allows multiple instances of the same block class to be stored together.
        # extended_block.extend(block)

        # Finally, it updates the blocks dictionary with the extended list of blocks for the class name.
        # This ensures that the dictionary now contains all instances of the specified block class.
        # self.blocks[block.__class__.__name__] = extended_block

    def init_actions(self):
        """
        Initializes the parser's actions by registering all action classes from the actions module.

        This method scans the actions module for all classes that are subclasses of the Action class, excluding the base
        Action class itself. It then adds each of these action classes to the parser's list of available actions, making
        them accessible for command parsing and execution.
        """

        # Initialize an empty dictionary to store the actions that will be registered.
        self.actions = {}

        # Iterate over all members of the actions module to find action classes.
        for member in dir(actions):
            # Retrieve the attribute corresponding to the member name from the actions module.
            attr = getattr(actions, member)

            # Check if the attribute is a class, is a subclass of actions.Action,
            # and is not the base Action class itself.
            if (
                inspect.isclass(attr)  # Verify that the attribute is a class.
                and issubclass(
                    attr, actions.Action
                )  # Ensure it is a subclass of Action.
                and attr != actions.Action  # Exclude the base Action class.
            ):
                # Add the identified action class to the parser's actions.
                self.add_action(attr)

    def determine_intent(self, command: str, character: Character):
        """
        Determines the intent of a player's command based on the input string.

        This method analyzes the command provided by the player and identifies the intended action or request. It checks
        for various keywords and patterns to classify the command into specific intents such as movement, examination,
        or inventory management. Here we have implemented it with a simple keyword match. Later we will use AI to do
        more flexible matching.

        Args:
            command (str): The command string input by the player.
            character (Character): The character associated with the command.

        Returns:
            str: The identified intent of the command, or None if no intent is recognized.
        """

        # Check which character is acting; defaults to the player.
        # The following line is commented out because the current character is already passed in.
        # character = self.get_character(command, character)  # Not needed if passing in the current character

        # Convert the command to lowercase for uniformity in processing.
        command = command.lower()

        # Check if the command contains a comma, indicating a sequence of commands.
        if "," in command:
            # Allow the player to input a comma-separated sequence of commands.
            return "sequence"

        # Check if the command indicates a direction using the character's location.
        elif self.get_direction(command, character.location):
            # Return the intent as "direction" if a valid direction is found.
            return "direction"

        # Check for the "look" command, which prompts a re-description of the surroundings.
        elif command in {"look", "l"}:
            return "describe"

        # Check for examination commands, either starting with "examine" or "x ".
        elif "examine " in command or command.startswith("x "):
            return "examine"

        # Check for commands related to taking or getting items.
        elif "take " in command or "get " in command:
            return "take"

        # Check for the command to light something.
        elif "light" in command:
            return "light"

        # Check for the command to drop an item.
        elif "drop " in command:
            return "drop"

        # Check for various forms of the "eat" command.
        elif (
            "eat " in command
            or "eats " in command
            or "ate " in command
            or "eating " in command
        ):
            return "eat"

        # Check for the command to drink something.
        elif "drink" in command:
            return "drink"

        # Check for the command to give an item.
        elif "give" in command:
            return "give"

        # Check for commands related to attacking or hitting.
        elif "attack" in command or "hit " in command or "hits " in command:
            return "attack"

        # Check for the command to view the inventory.
        elif "inventory" in command or command == "i":
            return "inventory"

        # Check for the command to quit the game.
        elif "quit" in command:
            return "quit"

        # If no specific intent is recognized, check against custom actions.
        else:
            for _, action in self.actions.items():
                special_command = action.action_name()
                # Return the action name if it matches the command.
                if special_command in command:
                    return action.action_name()

        # Return None if no intent could be determined from the command.
        return None

    def parse_action(self, command: str, character: Character) -> actions.Action:
        """
        Parses a player's command and returns the corresponding action object.

        This method processes the input command to determine the intended action based on the player's input and the
        current character. It identifies the action type and returns an appropriate action object, or triggers a failure
        response if no valid action is found.

        Args:
            command (str): The command string input by the player.
            character (Character): The character associated with the command.

        Returns:
            actions.Action: The action object corresponding to the parsed command, or None if no action is identified.
        """

        # Convert the command to lowercase and remove any leading or trailing whitespace.
        command = command.lower().strip()

        # If the command is empty after stripping, return None to indicate no action.
        if not command:
            return None

        # Determine the intent of the command using the determine_intent method.
        intent = self.determine_intent(command, character)

        # Check if the identified intent corresponds to a registered action.
        if intent in self.actions:
            # Retrieve the action associated with the intent and instantiate it with the game, command, and character.
            action = self.actions[intent]
            return action(self.game, command, character)

        # Check if the intent indicates a directional movement.
        elif intent == "direction":
            # Return a Go action for moving in the specified direction.
            return actions.Go(self.game, command, character)

        # Check if the intent indicates a command to take an item.
        elif intent == "take":
            # Return a Get action for taking the specified item.
            return actions.Get(self.game, command, character)

        # If no valid action is found, trigger a failure response.
        self.fail(command, f"No action found for {command}", character)

        # Return None if no action could be determined.
        return None

    def parse_command(self, command: str, character: Character):
        """
        Processes a player's command and executes the corresponding action.

        This method checks if the command has been repeated. If it hasn't, this adds it to the command history and
        attempts to parse the command into an actionable format. If the command is valid, it executes the action;
        otherwise, it handles failures by providing feedback to the player.

        Args:
            command (str): The command string input by the player.
            character (Character): The character associated with the command.

        Returns:
            bool: True if the action was successfully executed, False otherwise.
        """

        # The following line is commented out; it would print the command for debugging purposes.
        # print("\n>", command, "\n", flush=True)

        # Check if the command has been repeated in the command history.
        # If it has, print a warning message and return False to indicate no action will be taken.
        if self.command_repeated(command):
            print(
                f"Command {command} was repeated. Possibly mis-labeled as an ActionSequence."
            )
            return False

        # Add the current command to the command history for tracking purposes.
        Parser.add_command_to_history(self, command)

        # Attempt to parse the command into an actionable format.
        action = self.parse_action(command, character)

        # If no valid action is returned, trigger a failure response indicating no match was found.
        if not action:
            self.fail(command, "No action could be matched from the command", character)
            return False

        # If the action is an instance of ActionSequence, trigger a failure response indicating that the command parsed
        # to multiple actions.
        elif isinstance(action, ActionSequence):
            self.fail(
                command,
                "Command parsed to multiple actions. Try simpler command that attempts only 1 action.",
                character,
            )
            return False

        # If a valid action is found, execute it and return the result.
        else:
            return action()

    def command_repeated(self, command: str) -> bool:
        """
        Checks if the given command is a repeat of the last command issued by the player.

        This method compares the current command with the most recent command in the command history to determine if
        they are the same. It returns a boolean indicating whether the command has been repeated.

        Args:
            command (str): The command string to check against the command history.

        Returns:
            bool: True if the command is a repeat of the last command, False otherwise.
        """

        # Check if the command history is empty.
        # If it is, return False since there are no previous commands to compare against.
        if len(self.command_history) == 0:
            return False

        # Compare the current command with the content of the last command in the history.
        # Return True if they are the same, indicating the command has been repeated; otherwise, return False.
        return command == self.command_history[-1]["content"]

    @staticmethod
    def split_command(command: str, keyword: str) -> tuple[str, str]:
        """
        Splits a command string into two parts based on a specified keyword.

        This static method searches for the keyword within the command and divides the command into two segments: the
        portion before the keyword and the portion after it. If the keyword is not found, it returns the entire command
        as the first element and an empty string as the second.

        Args:
            command (str): The command string to be split.
            keyword (str): The keyword to split the command string around.

        Returns:
            tuple[str, str]: A tuple containing the part of the command before the keyword and the part after.
        """

        # Convert both the command and the keyword to lowercase for case-insensitive comparison.
        command = command.lower()
        keyword = keyword.lower()

        # Find the position of the keyword within the command string.
        keyword_pos = command.find(keyword)

        # If the keyword is not found (position is -1), return the entire command and an empty string.
        if keyword_pos == -1:
            return (command, "")

        # Split the command into two parts:
        # the segment before the keyword and the segment after the keyword.
        before_keyword = command[:keyword_pos]
        after_keyword = command[keyword_pos + len(keyword) :]

        # Return a tuple containing the two parts of the split command.
        return (before_keyword, after_keyword)

    def get_character(self, command: str, character: Character) -> Character:
        # ST 3/10 - add character arg for sake of override in GptParser3
        """
        Retrieves the character associated with a given command.

        This method searches for a character name within the command string and returns the corresponding character
        object if a match is found. If no character name is identified in the command, it defaults to returning the
        player character.

        Args:
            command (str): The command string input by the player.
            character (Character): The character associated with the command (used for potential overrides).

        Returns:
            Character: The character object that matches the command, or the player character if no match is found.
        """

        # Convert the command string to lowercase for case-insensitive matching.
        command = command.lower()

        # Use a generator expression to find the first character whose name is found in the command.
        # The next() function retrieves the first matching character from the game characters.
        # If no match is found, it defaults to returning the player character.
        return next(
            (
                self.game.characters[name]  # Access the character object by name.
                for name in self.game.characters.keys()  # Iterate over all character names.
                if name.lower()
                in command  # Check if the lowercase name is in the command.
            ),
            self.game.player,  # Default to the player character if no match is found.
        )

    def check_if_character_exists(self, name):
        """
        Checks if a character with the given name exists in the game.

        This method first performs a quick check for an exact match of the character name in the game's character list.
        If no exact match is found, it normalizes the name and conducts more complex similarity checks to identify
        potential matches based on partial names and string similarity metrics.

        Args:
            name (str): The name of the character to check for existence.

        Returns:
            tuple: A tuple containing a boolean indicating whether the character exists and the character name if found;
            otherwise, returns (False, None).
        """

        # Perform a quick O(1) check for an exact match of the character name in the game's character list.
        if name in self.game.characters:
            return True, name

        # If no exact match is found, normalize the input name for further checks.
        norm_name = normalize_name(name)

        # If normalization fails (returns None), indicate that the character does not exist.
        if not norm_name:
            return False, None

        # Get the length of the normalized name for further processing.
        nchar = len(norm_name)

        # If the normalized name is too short (2 characters or less), return False.
        if nchar <= 2:
            return False, None

        # Set a threshold for Levenshtein distance based on the length of the normalized name.
        lev_threshold = 1 if nchar < 5 else 2 if nchar < 12 else 3

        # Calculate a threshold for Jaro similarity based on the length of the normalized name.
        try:
            jaro_threshold = max(0.75, ((nchar - 2) / nchar))
        except ZeroDivisionError:
            # Handle potential division by zero by setting a default threshold.
            jaro_threshold = 0.8

        # Iterate through all character names in the game to find potential matches.
        for char_name in self.game.characters:
            # Normalize the current character name for comparison.
            norm_char_name = normalize_name(char_name)

            # Check if the normalized input name is a partial match with the current character name.
            if self.is_partial_name(norm_name, norm_char_name):
                return True, char_name

            # Calculate the length ratio to ensure it falls within a reasonable range before further comparison.
            length_ratio = len(norm_char_name) / (
                len(norm_name) + 0.01
            )  # Avoid division by zero
            if not (0.5 < length_ratio < 2):
                continue  # Skip to the next character if the length ratio is not suitable.

            # Perform similarity checks using Jaro-Winkler and Levenshtein distance metrics.
            if (
                jaro_winkler_similarity(norm_char_name, norm_name) > jaro_threshold
                and levenshtein_distance(norm_char_name, norm_name) < lev_threshold
            ):
                return True, char_name  # Return True if a match is found.

        # If no matches are found after checking all characters, return False.
        return False, None

    def is_partial_name(self, candidate_name, comparison_name):
        """
        Checks if the candidate name partially matches the comparison name based on specific criteria.

        This method evaluates the first and last parts of both names to determine if they share a common prefix or
        suffix. It returns True if either the first or last parts match, indicating a partial name match; otherwise, it
        returns False.

        Args:
            candidate_name (str): The name to be checked for a partial match.
            comparison_name (str): The name against which the candidate name is compared.

        Returns:
            bool: True if there is a partial match based on the first or last parts of the names, False otherwise.
        """

        # Split the candidate name and comparison name into lists of their constituent parts.
        cand_parts = candidate_name.split()
        comp_parts = comparison_name.split()

        # Check if the first part of the candidate name matches the first part of the comparison name.
        if cand_parts[0] == comp_parts[0]:
            return True  # Return True if the first parts match.

        # Check if the last part of the candidate name matches the last part of the comparison name.
        return (
            cand_parts[-1] == comp_parts[-1]
        )  # Return True if the last parts match, otherwise return False.

    def get_character_location(self, character: Character) -> "Location":
        """
        Retrieves the location of the specified character.

        This method returns the current location of the given character object, allowing other parts of the program to
        access the character's position within the game world.

        Args:
            character (Character): The character whose location is to be retrieved.

        Returns:
            Location: The location object representing the character's current position.
        """

        return character.location

    def match_item(self, command: str, item_dict: dict[str, "Item"]) -> "Item":
        """
        Finds and returns the first item from the item dictionary that matches a name in the command string.

        This method searches through the provided dictionary of items and checks if any item name is present in the
        command. If a match is found, the corresponding item is returned; otherwise, None is returned.

        Args:
            command (str): The command string input by the player.
            item_dict (dict[str, Item]): A dictionary mapping item names to their corresponding Item objects.

        Returns:
            Item: The first matching item if found, or None if no match exists.
        """

        # Use the next() function to find the first item in the item_dict whose name is present in the command string.
        return next(
            (
                item  # The item to return if a match is found.
                for item_name, item in item_dict.items()  # Iterate over the items in the item_dict.
                if item_name
                in command  # Check if the item name is included in the command string.
            ),
            None,  # Return None if no matching item is found.
        )

    def get_items_in_scope(self, character=None) -> dict[str, "Item"]:
        """
        Retrieves a dictionary of items that are currently in the character's location and inventory.

        This method checks the specified character's location for items and combines them with the items in the
        character's inventory. If no character is provided, it defaults to using the player character, ensuring that all
        relevant items are returned in a single dictionary.

        Args:
            character (Character, optional): The character whose items are to be retrieved. Defaults to the player
            character if not provided.

        Returns:
            dict[str, Item]: A dictionary mapping item names to their corresponding Item objects that are in the
            character's scope.
        """

        # If no character is provided, default to using the player character.
        if character is None:
            character = self.game.player

        # Create a dictionary to store items that are currently in the character's location.
        # The dictionary comprehension iterates over the items in the character's location.
        items_in_scope = {
            item_name: character.location.items[
                item_name
            ]  # Map item names to their corresponding Item objects.
            for item_name in character.location.items
        }

        # Iterate over the items in the character's inventory and add them to the items_in_scope dictionary.
        for item_name in character.inventory:
            items_in_scope[item_name] = character.inventory[
                item_name
            ]  # Include inventory items in the scope.

        # Return the dictionary containing all items in the character's location and inventory.
        return items_in_scope

    def get_direction(self, command: str, location: "Location" = None) -> str:
        """
        Determines the intended direction of movement based on the player's command.

        This method analyzes the command string to identify directional keywords and returns the corresponding
        direction. It supports various input formats, including shorthand notations and phrases indicating movement, and
        can also check for valid exits in a specified location.

        Args:
            command (str): The command string input by the player.
            location (Location, optional): The current location of the character, used to validate exits. Defaults to
            None.

        Returns:
            str: The identified direction as a string, or None if no valid direction is found.
        """

        # Convert the command string to lowercase for case-insensitive comparison.
        command = command.lower()

        # Check if the command is a shorthand for "north" or contains the word "north".
        if command == "n" or "north" in command:
            return "north"  # Return "north" if matched.

        # Check if the command is a shorthand for "south" or contains the word "south".
        if command == "s" or "south" in command:
            return "south"  # Return "south" if matched.

        # Check if the command is a shorthand for "east" or contains the word "east".
        if command == "e" or "east" in command:
            return "east"  # Return "east" if matched.

        # Check if the command is a shorthand for "west" or contains the word "west".
        if command == "w" or "west" in command:
            return "west"  # Return "west" if matched.

        # Check if the command indicates movement upwards.
        if command.endswith("go up"):
            return "up"  # Return "up" if matched.

        # Check if the command indicates movement downwards.
        if command.endswith("go down"):
            return "down"  # Return "down" if matched.

        # Check if the command indicates exiting the current location.
        if command.endswith("go out"):
            return "out"  # Return "out" if matched.

        # Check if the command indicates entering a location.
        if command.endswith("go in"):
            return "in"  # Return "in" if matched.

        # If no direction is found and a location is provided, check for valid exits.
        if location:
            for exit in location.connections.keys():
                # Check if any exit name is included in the command.
                if exit.lower() in command:
                    return exit  # Return the matching exit if found.

        # If no valid direction or exit is identified, return None.
        return None


class GptParser(Parser):
    """
    GptParser class for managing interactions with the GPT model in a text adventure game.

    This class extends the Parser class to integrate GPT-based functionalities, including generating descriptions,
    handling commands, and managing character interactions. It utilizes natural language processing and tokenization to
    enhance the gameplay experience by providing context-aware responses and summaries.

    Attributes:
        verbose (bool): A flag indicating whether to print verbose output for debugging.
        tokenizer: The tokenizer used for encoding input text.
        nlp: The natural language processing model for analyzing text.
        gpt_handler: The handler for interacting with the GPT model.
        max_input_tokens (int): The maximum number of tokens allowed for input to the GPT model.
        narrator_turn_limit (int): The limit on the number of turns to consider for context in descriptions.

    Args:
        game (Game): The game instance to be associated with the parser.
        echo_commands (bool, optional): Whether to print user commands. Defaults to True.
        verbose (bool, optional): Whether to enable verbose output. Defaults to False.
    """

    def __init__(self, game, echo_commands=True, verbose=False):
        """
        Initializes the GptParser with the game instance and configuration options.

        This constructor sets up the necessary components for the GptParser, including the tokenizer, natural language
        processing model, and GPT handler. It also allows for configuration of command echoing and verbosity for
        debugging purposes.

        Args:
            game (Game): The game instance to be associated with the parser.
            echo_commands (bool, optional): Whether to print user commands. Defaults to True.
            verbose (bool, optional): Whether to enable verbose output for debugging. Defaults to False.
        """

        # Call the constructor of the parent class (Parser) to initialize the game and command echoing settings.
        super().__init__(game, echo_commands=echo_commands)

        # Set the verbosity level for debugging output.
        self.verbose = verbose

        # Initialize the tokenizer using the specified encoding for processing text.
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Load the English language model for natural language processing using spaCy.
        self.nlp = spacy.load("en_core_web_sm")

        # Set up the GPT handler for interacting with the GPT model.
        self.gpt_handler = self._set_up_gpt()

        # Retrieve the maximum number of input tokens allowed for the GPT model's context.
        self.max_input_tokens = self.gpt_handler.model_context_limit

        # Define the limit on the number of turns to consider for context in narrative descriptions.
        self.narrator_turn_limit = 5

    def _set_up_gpt(self):
        """
        Sets up the GPT handler with the specified model parameters.

        This method initializes the GptCallHandler with configuration settings such as the API key, model type, and
        various parameters that control the behavior of the GPT model. It returns an instance of the GptCallHandler,
        which is used for making requests to the GPT model.

        Returns:
            GptCallHandler: An instance of the GptCallHandler configured with the specified model parameters.
        """

        # Define a dictionary containing the parameters needed to configure the GPT model.
        model_params = {
            # "max_tokens": 400,  # Set the maximum number of tokens for the model's output.
            "temperature": 1,  # Control the randomness of the output; higher values produce more varied responses.
            "top_p": 1,  # Set the cumulative probability for token selection; 1 means all tokens are considered.
            "max_retries": 5,  # Define the maximum number of retries for API calls in case of failure.
        }

        # Create and return an instance of GptCallHandler, passing the model parameters as keyword arguments.
        return GptCallHandler(model_config_type="parser", **model_params)

    def get_handler(self):
        """
        Retrieves the GPT handler, initializing it if it has not been set up yet.

        This method checks if the GPT handler is already created and returns it if available. If the handler does not
        exist, it calls the setup method to create a new instance and then returns it.

        Returns:
            GptCallHandler: The GPT handler instance used for interacting with the GPT model.
        """

        # Check if the GPT handler has not been initialized yet.
        if not self.gpt_handler:
            # If the handler is not set up, call the _set_up_gpt method to initialize it.
            self.gpt_handler = self._set_up_gpt()

        # Return the GPT handler instance, whether it was just created or already exists.
        return self.gpt_handler

    def gpt_describe(
        self, system_instructions, command_history, extra_description=None
    ):
        """
        Generates a description using the GPT model based on system instructions and command history.

        This method constructs a message context for the GPT model, including system instructions and relevant command
        history, to produce a narrative description. It handles additional context through an optional description and
        manages token limits to ensure the input fits within the model's constraints.

        Args:
            system_instructions (str): Instructions that guide the behavior of the GPT model.
            command_history (list): A list of previous commands and descriptions to provide context.
            extra_description (str, optional): Additional context to enhance the description generation. Defaults to
            None.

        Returns:
            str: The generated description from the GPT model, or an error message if the generation fails.
        """
        # TODO: should the context for each description be more limited to focus on recent actions?

        # Attempt to generate a description using the GPT model.
        try:
            # Create a list of messages starting with the system instructions.
            messages = [{"role": "system", "content": system_instructions}]

            # Calculate the number of tokens used in the system instructions.
            system_count = get_prompt_token_count(
                content=system_instructions,
                role="system",
                pad_reply=False,
                tokenizer=self.tokenizer,
            )

            # Determine the number of available tokens for the context based on input limits.
            available_tokens = get_token_remainder(
                self.max_input_tokens, system_count, self.gpt_handler.max_output_tokens
            )

            # Limit the context to the last few turns of command history for relevance.
            narrator_context = command_history[-self.narrator_turn_limit :]

            # If an extra description is provided, format it for inclusion in the context.
            if extra_description:
                user_description = "".join(
                    [
                        "Use the following description as additional context to fulfill your system function. ",
                        "And if the description describes a failed action, provide a helpful embellishment to a user ",
                        "that helps them to learn why their action led to an error. ",
                        f"Description: {extra_description}.",
                    ]
                )
                # Append the formatted user description to the narrator context.
                narrator_context.append({"role": "user", "content": user_description})

            # Limit the context length to fit within the available token count.
            context = limit_context_length(narrator_context, available_tokens)

            # Add the limited context to the messages list.
            messages.extend(context)

            # If verbose mode is enabled, print the messages for debugging.
            if self.verbose:
                print(json.dumps(messages, indent=2))

            # Generate a response from the GPT handler using the constructed messages.
            return self.gpt_handler.generate(messages=messages)

        # Handle any exceptions that occur during the process.
        except Exception as e:
            # Return an error message if something goes wrong with the GPT generation.
            return f"Something went wrong with GPT: {e}"

    def create_action_statement(
        self, command: str, description: str, character: Character
    ):
        """
        Creates a formatted action statement for a character's command and its outcome.

        This method constructs a string that summarizes the action taken by a character, including the actor's name,
        location, command, and the resulting description. It then generates a concise summary of this action statement
        using the GPT model.

        Args:
            command (str): The command string representing the action taken by the character.
            description (str): A description of the outcome of the action.
            character (Character): The character who performed the action.

        Returns:
            str: A summarized description of the action statement generated by the GPT model.
        """

        # Construct a formatted string that summarizes the action taken by the character.
        # This includes the actor's name, their location, the action command, and the outcome description.
        outcome = (
            f"""ACTOR: {character.name}; LOCATION: {character.location.name}, ACTION: {command}; """
            f"""OUTCOME: {description}"""
        )

        # Call the function to generate a summary of the action statement using the GPT model.
        # The outcome string is passed along with the GPT handler and a limit on the number of tokens for the response.
        return gpt_get_summary_description_of_action(
            outcome, call_handler=self.gpt_handler, max_tokens=256
        )

    def extract_keywords(self, text, actions=False):
        """
        Extracts keywords from the provided text, identifying characters, objects, and optionally actions.

        This method processes the input text using natural language processing to identify and categorize keywords, such
        as characters, objects, and actions, while filtering out common stopwords. It returns a dictionary containing sets of
        identified characters, objects, actions, and miscellaneous dependencies based on the text analysis.

        Args:
            text (str): The input text from which to extract keywords.
            actions (bool, optional): Whether to extract actions from the text. Defaults to False.

        Returns:
            dict: A dictionary with keys for "characters", "objects", "actions", "misc_deps", and "other_named_entities",
                  each containing a list of identified keywords.
        """

        # Check if the input text is empty; if so, return None.
        if not text:
            return None

        # Define a set of custom stopwords to filter out common, non-informative words, including basic verbs.
        custom_stopwords = {
            "a",
            "an",
            "and",
            "he",
            "it",
            "i",
            "you",
            "she",
            "they",
            "we",
            "us",
            "'s",
            "this",
            "that",
            "these",
            "those",
            "them",
            "their",
            "my",
            "your",
            "our",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "say",
            "says",
            "said",
            "go",
            "goes",
            "went",
            "make",
            "makes",
            "made",
            "know",
            "knows",
            "knew",
            "think",
            "thinks",
            "thought",
            "take",
            "takes",
            "took",
            "see",
            "sees",
            "saw",
            "come",
            "comes",
            "came",
            "want",
            "wants",
            "wanted",
            "like",
            "likes",
            "liked",
        }

        # Import necessary packages for lemmatization and singularization.
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet
        from inflect import engine

        # Initialize the lemmatizer and inflect engine.
        lemmatizer = WordNetLemmatizer()
        inflect_engine = engine()

        # Function to standardize words by converting to lowercase, lemmatizing, and singularizing.
        def standardize_word(word):
            word = word.lower()
            # Try lemmatizing with different parts of speech
            for pos in [wordnet.VERB, wordnet.NOUN, wordnet.ADJ, wordnet.ADV]:
                lemmatized_word = lemmatizer.lemmatize(word, pos)
                if lemmatized_word != word:  # If lemmatization changed the word
                    word = lemmatized_word
                    if pos == wordnet.NOUN:
                        word = inflect_engine.singular_noun(word) or word
                        break
                return word

        # Process the input text using the natural language processing model.
        doc = self.nlp(text)

        # Initialize a defaultdict to store identified keywords categorized by type.
        keys = defaultdict(set)

        # Iterate over each word in the processed document.
        for w in doc:
            # Skip the word if it is in the custom stopwords set.
            if w.text.lower() in custom_stopwords:
                continue

            # Check if the word is a proper noun (PROPN).
            if w.pos_ in ["PROPN"]:
                # If the proper noun has compound words, handle the entire compound noun.
                compound_noun = " ".join(
                    [
                        child.text
                        for child in w.subtree
                        if child.text.lower() not in custom_stopwords
                    ]
                ).lower()
                exists, name = self.check_if_character_exists(compound_noun)
                if exists:
                    # If the character exists, add it to the characters set.
                    keys["characters"].add(name.lower())
                else:
                    # If not, add the compound noun to miscellaneous dependencies.
                    keys["misc_deps"].add(standardize_word(compound_noun))

                # Process each word in the compound noun separately.
                for part in compound_noun.split():
                    exists, name = self.check_if_character_exists(part)
                    if exists:
                        # If the character exists, add it to the characters set.
                        keys["characters"].add(name.lower())
                    else:
                        # If not, add the word to miscellaneous dependencies.
                        keys["misc_deps"].add(standardize_word(part))
                continue

            # Check if the word is a subject in the dependency parse.
            if "subj" in w.dep_:
                exists, name = self.check_if_character_exists(w.text.lower())
                if exists:
                    # If the character exists, add it to the characters set.
                    keys["characters"].add(name.lower())
                else:
                    # If not, add the word to miscellaneous dependencies.
                    keys["misc_deps"].add(standardize_word(w.text))

            # Check if the word is an object in the dependency parse.
            if "obj" in w.dep_:
                exists, name = self.check_if_character_exists(w.text.lower())
                if exists:
                    # If the character exists, add it to the characters set.
                    keys["characters"].add(name.lower())
                else:
                    # If not, add the word to the objects set.
                    keys["objects"].add(standardize_word(w.text))

            if actions:
                # Check if the word is a verb (action).
                if w.pos_ == "VERB":
                    keys["actions"].add(standardize_word(w.text))

        # Iterate over named entities in the document.
        for ent in doc.ents:
            # Check if the entity is a person, organization, or geopolitical entity.
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                exists, name = self.check_if_character_exists(ent.text.lower())
                if exists:
                    # If the character exists, add it to the characters set.
                    keys["characters"].add(name.lower())
                else:
                    cleaned_entity = " ".join(
                        [
                            standardize_word(w.text)
                            for word in ent.text.split()
                            if word.lower() not in custom_stopwords
                        ]
                    ).lower()
                    keys["other_named_entities"].add(cleaned_entity)

        # Remove duplicates between 'misc_deps' and 'other_named_entities'
        keys["other_named_entities"] = keys["other_named_entities"] - keys["misc_deps"]

        # Remove duplicates between 'characters' and 'other_named_entities'
        keys["other_named_entities"] = keys["other_named_entities"] - keys["characters"]

        # Remove duplicates between 'objects' and 'other_named_entities'
        keys["other_named_entities"] = keys["other_named_entities"] - keys["objects"]

        # Convert the sets in the keys dictionary to lists for easier handling.
        keys = {k: list(v) for k, v in keys.items()}

        # Return the dictionary containing categorized keywords.
        return keys

    def summarize_and_score_action(
        self, description, thing, command="look", needs_summary=True, needs_score=True
    ):
        """
        Generates a summary and importance score for a given action description.

        This method creates an action statement based on the provided description and command, and optionally calculates
        the importance of the action using the GPT model. It also extracts relevant keywords from the action statement,
        returning all three components for further processing.

        Args:
            description (str): A description of the action taken.
            thing (Thing): The object or character associated with the action.
            command (str, optional): The command associated with the action. Defaults to "look".
            needs_summary (bool, optional): Indicates whether to generate a summary of the action. Defaults to True.
            needs_score (bool, optional): Indicates whether to calculate the importance score of the action. Defaults to
            True.

        Returns:
            tuple: A tuple containing the action statement, the importance score of the action, and a list of extracted
            keywords.
        """

        # Check if a summary of the action is needed.
        if needs_summary:
            # Generate an action statement using the command, description, and associated object or character.
            action_statement = self.create_action_statement(command, description, thing)
        else:
            # If no summary is needed, use the provided description directly as the action statement.
            action_statement = description

        # Check if the importance score of the action should be calculated.
        if needs_score:
            # Call the function to get the importance of the action using the GPT model.
            importance_of_action = gpt_get_action_importance(
                action_statement,
                call_handler=self.gpt_handler,
                max_tokens=10,
                top_p=0.25,
            )
        else:
            # If no score is needed, set the importance of the action to 0.
            importance_of_action = 0

        # Extract keywords from the action statement for further analysis.
        keywords = self.extract_keywords(action_statement)

        # Return the action statement, its importance score, and the extracted keywords.
        return action_statement, importance_of_action, keywords

    def add_command_to_history(
        self, command, summary, keywords, character, importance, success, type
    ):
        """
        Logs a command issued by a character and updates their memory with relevant details.

        This method records a command into the command history, while also adding its summary, keywords, importance, and
        other contextual information into both the character's memory and the memories of other characters in view,
        ensuring that all relevant entities are aware of the action taken.

        Args:
            command (str): The command string input by the character.
            summary (str): A summary of the action and its outcome.
            keywords (dict): Keywords extracted from the summary.
            character (Character): The character associated with the command.
            importance (int): The importance level of the action.
            success (bool): Indicates whether the action was successful.
            type (str): The type of memory being recorded.

        Returns:
            None
        """

        # Log the command as a user-supplied input in the command history.
        # This uses the parent class's method to ensure proper logging format.
        super().add_command_to_history(command)

        # Add a memory entry for the character that includes details about the action taken.
        character.memory.add_memory(
            round=self.game.round,  # Record the current game round.
            tick=self.game.tick,  # Record the current game tick.
            description=summary.lower(),  # Store the action summary in lowercase.
            keywords=keywords,  # Include any relevant keywords extracted from the summary.
            location=character.location.name,  # Record the character's current location.
            success_status=success,  # Indicate whether the action was successful.
            memory_importance=importance,  # Set the importance level of the memory.
            memory_type=type,  # Specify the type of memory being recorded.
            actor_id=character.id,  # Store the ID of the character who performed the action.
        )

        # Iterate over other characters in the current character's view.
        for char in character.chars_in_view:
            # Add a similar memory entry for each character in view, ensuring they are aware of the action.
            char.memory.add_memory(
                round=self.game.round,  # Record the current game round for the viewing character.
                tick=self.game.tick,  # Record the current game tick for the viewing character.
                description=summary.lower(),  # Store the action summary in lowercase for consistency.
                keywords=keywords,  # Include the same keywords for context.
                location=character.location.name,  # Record the location of the action.
                success_status=success,  # Indicate the success status of the action.
                memory_importance=importance,  # Set the importance level of the memory for the viewing character.
                memory_type=type,  # Specify the type of memory being recorded for the viewing character.
                actor_id=character.id,  # Store the ID of the character who performed the action.
            )

    def ok(self, command: str, description: str, thing: "Thing") -> None:
        """
        Logs a successful command and generates a summary of the action taken by a character.

        This method processes the command and its outcome, updating the command history and character memories
        accordingly. It also prepares system instructions for the narrator to create descriptive responses based on the
        action performed.

        Example:
            command: "Get the pole"
            description: "The troll got the pole"
            character: troll

        Args:
            command (str): The command string given by the character.
            description (str): A description of the outcome of the command.
            thing (Thing): The object or character associated with the command.

        Returns:
            None
        """

        # FIRST: Summarize the action taken and store it as a memory for the relevant characters.
        if isinstance(thing, Character):
            # Generate a summary, importance score, and keywords for the action.
            summary_of_action, importance_of_action, action_keywords = (
                self.summarize_and_score_action(description, thing, command=command)
            )

            # Format the command to include the character's name for clarity in the command history.
            command = f"{thing.name}'s action: {command}"

            # Log the command and its details into the command history.
            self.add_command_to_history(
                command,
                summary_of_action,
                action_keywords,
                thing,
                importance_of_action,
                success=True,
                type=MemoryType.ACTION.value,
            )

        # # SECOND: Prepare system instructions for generating a narrative description of the action.
        # system_instructions = "".join(
        #     [
        #         "You are the narrator for a text adventure game. You create short, ",
        #         "evocative descriptions of the game. The player can be described in ",
        #         f"the {self.perspective} person, and you should use present tense. ",
        #         "If the command is 'look' then describe the game location and its characters and items. ",
        #         "Focus on describing the most recent events.",
        #     ]
        # )

        # # TODO: I'm commenting this out for now to avoid paying this each time.
        # # It also doesn't seem to add anything aside from a printing/logging a longer statement.
        # response = self.gpt_describe(system_instructions, self.command_history)
        # self.add_description_to_history(response)
        # print(self.wrap_text(response) + '\n')

    def fail(self, command: str, description: str, thing: "Thing"):
        """
        Handles the failure of a command by generating a descriptive error message and logging it.

        This method constructs a narrative that explains why a command failed, utilizing the GPT model to provide
        context and feedback. It updates the command history and character memories with the failure details, ensuring
        that both the character and the game state are aware of the unsuccessful action. Commands that do not pass all
        preconditions lead to a failure. They are logged by this method. Failed commands are still added to the global
        command history and to the memories of characters in view.

        Args:
            command (str): The command string that was attempted by the character.
            description (str): A description of the failure outcome.
            thing (Thing): The object or character associated with the failed command.

        Returns:
            None
        """

        # SECOND: Generate a descriptive message explaining the failure of the command for console output.
        system_instructions = "".join(
            [
                "You are the narrator for a text adventure game. ",  # Introduce the narrator's role.
                f"{thing.name} entered a command that failed in the game. ",  # Specify the character and the failed command.
                f"Try to help {thing.name} understand why the command failed. ",  # Prompt for guidance on the failure.
                f"You will see the last few commands given. {thing.name} attempted the last one and failed. ",  # Reference the recent command history.
                "Summarize why they failed using only the information provided. ",  # Instruct to summarize based on available data.
                "Do not make up rules of the game that you don't know explicitly.",  # Caution against inventing game rules.
            ]
        )

        # Call the gpt_describe method to generate a response based on the system instructions and command history.
        response = self.gpt_describe(
            system_instructions, self.command_history, extra_description=description
        )

        # FIRST: If the thing is a character, log the failure and update their memory with the action details.
        if isinstance(thing, Character):
            print(
                f"{thing.name} action failed. Adding failure memory to history."
            )  # Inform about the failure being logged.

            # Calculate the importance of the action based on the GPT response.
            importance_of_action = gpt_get_action_importance(
                response, call_handler=self.gpt_handler, max_tokens=10, top_p=0.25
            )

            # Extract keywords from the GPT response for further context.
            keywords = self.extract_keywords(response)

            # Format the command to include the character's name for clarity in the command history.
            command = f"{thing.name}'s action: {command}"

            # Log the failed command and its details into the command history.
            self.add_command_to_history(
                command,
                response,
                keywords,
                thing,
                importance_of_action,
                success=False,
                type=MemoryType.ACTION.value,
            )

        # If verbose mode is enabled, print the GPT's error description for debugging purposes.
        if self.verbose:
            print("GPT's Error Description:")

        # Add the generated description of the failure to the command history.
        self.add_description_to_history(response)

        # Print the wrapped response for console output, ensuring it is formatted for readability.
        print(f"\n{self.wrap_text(response)}")


class GptParser2(GptParser):
    """
    GptParser2 class for managing GPT-based command parsing in a text adventure game.

    This class extends the GptParser to include functionality for refreshing the command list and determining user
    intent using GPT. It maps action descriptions and aliases to their corresponding action names, enhancing the
    parser's ability to interpret player commands.

    Args:
        game (Game): The game instance to be associated with the parser.
        echo_commands (bool, optional): Whether to print user commands. Defaults to True.
        verbose (bool, optional): Whether to enable verbose output for debugging. Defaults to False.
    """

    def __init__(self, game, echo_commands=True, verbose=False):
        """
        Initializes the GptParser2 with the game instance and configuration options.

        This constructor sets up the necessary components for the GptParser2, including calling the parent class's
        initializer and refreshing the command list to ensure it is up to date. It allows for configuration of command
        echoing and verbosity for debugging purposes.

        Args:
            game (Game): The game instance to be associated with the parser.
            echo_commands (bool, optional): Whether to print user commands. Defaults to True.
            verbose (bool, optional): Whether to enable verbose output for debugging. Defaults to False.
        """

        # Call the constructor of the parent class (GptParser) to initialize the game instance,
        # command echoing settings, and verbosity level.
        super().__init__(game, echo_commands, verbose)

        # Refresh the command list to ensure it is updated with the latest action descriptions and aliases.
        self.refresh_command_list()

    def refresh_command_list(self):
        """
        Refreshes the command list by mapping action descriptions and aliases to their corresponding action names.

        This method constructs a dictionary that associates each action's description, including any aliases, with the
        action name. It updates the command descriptions attribute, ensuring that the parser has the latest information
        for interpreting player commands.

        Returns:
            self: The instance of the GptParser2 for method chaining.
        """

        # Initialize an empty dictionary to store command descriptions,
        # which will map action descriptions and their aliases to action names.
        command_descriptions = {}

        # Iterate over all actions in the actions dictionary.
        for _, action in self.actions.items():
            # Retrieve the description of the current action.
            description = action.ACTION_DESCRIPTION

            # If the action has aliases, append them to the description for clarity.
            if action.ACTION_ALIASES:
                description += " (can also be invoked with '{aliases}')".format(
                    aliases="', '".join(
                        action.ACTION_ALIASES
                    )  # Join aliases into a single string.
                )

            # Use the walrus operator to assign the action name and check if it exists.
            if action_name := action.ACTION_NAME:
                # Map the constructed description to the action name in the command descriptions dictionary.
                command_descriptions[description] = action_name

        # Update the command_descriptions attribute of the instance with the newly created dictionary.
        self.command_descriptions = command_descriptions

        # Return the instance of the class for method chaining.
        return self

    def determine_intent(self, command, character: Character):
        """
        Determines the intent of a user's command by comparing it to predefined commands.

        This method constructs a set of instructions for the GPT model to identify which command the user's input most
        closely matches. It utilizes the GPT handler to process the command descriptions and return the best matching
        intent.

        Credit: Dr. Chris Callison-Burch (University of Pennsylvania)
        Instead of the keyword based intent determination, we'll use GPT.

        Args:
            command (str): The command string input by the player.
            character (Character): The character associated with the command.

        Returns:
            str: The identified intent of the command as determined by the GPT model.
        """

        # Construct a string of instructions for the GPT model, explaining its role as a parser for the game.
        instructions = "".join(
            [
                "You are the parser for a text adventure game. For a user input, say which ",
                "of the commands it most closely matches. The commands are:",
            ]
        )

        # Call the gpt_pick_an_option function to determine the best matching command for the user's input.
        # Pass the constructed instructions, the command descriptions, the user's command, and the GPT handler.
        return gpt_pick_an_option(
            instructions,
            self.command_descriptions,  # The dictionary of command descriptions to match against.
            command,  # The user's input command to analyze.
            self.gpt_handler,  # The handler for interacting with the GPT model.
            max_tokens=10,  # Limit the number of tokens in the response.
        )


class GptParser3(GptParser2):
    """
    GptParser3 class for enhanced command parsing using GPT in a text adventure game.

    This class extends the GptParser2 to provide functionality for matching characters, items, and directions based on
    user commands. It utilizes the GPT model to interpret player inputs and generate appropriate responses, improving
    the interaction experience in the game.

    Args:
        game (Game): The game instance to be associated with the parser.
        echo_commands (bool, optional): Whether to print user commands. Defaults to True.
        verbose (bool, optional): Whether to enable verbose output for debugging. Defaults to False.
    """

    def __init__(self, game, echo_commands=True, verbose=False):
        """
        Initializes the GptParser3 with the game instance and configuration options.

        This constructor sets up the necessary components for the GptParser3 by calling the parent class's initializer.
        It allows for configuration of command echoing and verbosity for debugging purposes.

        Args:
            game (Game): The game instance to be associated with the parser.
            echo_commands (bool, optional): Whether to print user commands. Defaults to True.
            verbose (bool, optional): Whether to enable verbose output for debugging. Defaults to False.
        """

        super().__init__(game, echo_commands, verbose)

    def get_character(
        self,
        command: str,
        character: Character = None,
        hint: str = None,
        split_words=None,
        position=None,
    ) -> Character:
        """
        Attempts to match a character's name from the command input.

        This method generates a list of character descriptions and uses the GPT model to determine which character, if
        any, corresponds to the command provided by the player. It can also provide hints to narrow down the search for
        a specific character.

        Args:
            command (str): The command string input by the player.
            character (Character, optional): The character associated with the command. Defaults to None.
            hint (str, optional): A hint about the role of the character being searched for. Defaults to None.
            split_words (optional): Not used in this implementation. Defaults to None.
            position (optional): Not used in this implementation. Defaults to None.

        Returns:
            Character: The matched character object based on the command input, or None if no match is found.
        """

        # If verbose mode is enabled, print a message indicating that character matching is in progress.
        if self.verbose:
            print("Matching a character with GPT.")

        # Initialize an empty dictionary to store character descriptions.
        character_descriptions = {}

        # Iterate over all characters in the game to create descriptions.
        for name, character in self.game.characters.items():
            # Check if the character has a location.
            if character.location:
                # Create a description string that includes the character's name, description, and current location.
                d = "{name} - {description} (currently located in {location})"
                description = d.format(
                    name=name,
                    description=character.description,
                    location=character.location.name,
                )
            else:
                # If the character has no location, create a simpler description.
                description = "{name} - {description}".format(
                    name=name, description=character.description
                )

            # The following commented-out code would provide a special description for the player character.
            # if character == self.game.player:
            #     description = "The player: {description}".format(
            #         description=character.description
            #     )

            # Store the description in the character_descriptions dictionary, mapping it to the character object.
            character_descriptions[description] = character

        # Construct instructions for the GPT model, explaining its role in matching characters based on user input.
        # TODO: should the passed in character be the default rather than the game player?
        instructions = "".join(
            [
                "You are the parser for a text adventure game. For an input command try to ",
                "match the character in the command (if no character is mentioned in the ",
                "command, then default to '{player}').".format(
                    player=self.game.player.name
                ),
            ]
        )

        # If a hint is provided, append it to the instructions for additional context.
        if hint:
            instructions += f"\nHint: the character you are looking for is the {hint}. "

        # Add a prompt indicating that the possible characters will follow.
        instructions += "\n\nThe possible characters are:"

        # Call the gpt_pick_an_option function to determine the best matching character based on the instructions and
        # command.
        return gpt_pick_an_option(
            instructions,
            character_descriptions,  # Pass the dictionary of character descriptions for matching.
            command,  # The command input by the player to analyze.
            call_handler=self.gpt_handler,  # The handler for interacting with the GPT model.
            max_tokens=10,  # Limit the number of tokens in the response.
        )

    def match_item(
        self, command: str, item_dict: dict[str, "Item"], hint: str = None
    ) -> "Item":
        """
        Attempts to match an item from the command input against a provided dictionary of items.

        This method generates instructions for the GPT model to identify which item, if any, corresponds to the command
        given by the player. It constructs a list of item descriptions and utilizes the GPT model to determine the best
        match based on the input command.

        Args:
            command (str): The command string input by the player.
            item_dict (dict[str, Item]): A dictionary mapping item names to their corresponding Item objects.
            hint (str, optional): A hint about the type of item being searched for. Defaults to None.

        Returns:
            Item: The matched item object based on the command input, or None if no match is found.
        """

        # If verbose mode is enabled, print a message indicating that item matching is in progress.
        if self.verbose:
            print("Matching an item with GPT.")

        # Construct the initial instructions for the GPT model, explaining its role in matching items.
        instructions = (
            """You are the parser for a text adventure game. For an input command try to """
            """match the item in the command."""
        )

        # If a hint is provided, append it to the instructions for additional context.
        if hint:
            instructions += f"\nHint: {hint}."

        # Add a prompt indicating that the possible items will follow.
        instructions += "\n\nThe possible items are:"

        # Initialize an empty dictionary to store item descriptions.
        item_descriptions = {}

        # Iterate over all items in the provided item dictionary.
        for name, item in item_dict.items():
            # Check if the item has a location.
            if item.location:
                # Create a description string that includes the item's name, description, and current location.
                description = (
                    "{name} - {description} (currently located in {location})".format(
                        name=name,
                        description=item.description,
                        location=item.location.name,
                    )
                )
            else:
                # If the item has no location, create a simpler description.
                description = "{name} - {description}".format(
                    name=name, description=item.description
                )

            # Store the description in the item_descriptions dictionary, mapping it to the item object.
            item_descriptions[description] = item

        # Call the gpt_pick_an_option function to determine the best matching item based on the instructions and command.
        return gpt_pick_an_option(
            instructions,
            item_descriptions,  # Pass the dictionary of item descriptions for matching.
            command,  # The command input by the player to analyze.
            call_handler=self.gpt_handler,  # The handler for interacting with the GPT model.
            max_tokens=10,  # Limit the number of tokens in the response.
        )

    def get_direction(self, command: str, location: "Location" = None) -> str:
        """
        Determines the intended direction of movement based on the player's command.

        This method analyzes the command string to identify directional keywords and returns the corresponding
        direction. It constructs a list of possible directions, including those derived from the current location, and
        utilizes the GPT model to determine the best match based on the input command.

        Args:
            command (str): The command string input by the player.
            location (Location, optional): The current location of the character, used to validate available directions.
            Defaults to None.

        Returns:
            str: The identified direction as a string, or None if no valid direction is found.
        """

        # If verbose mode is enabled, print a message indicating that direction matching is in progress.
        if self.verbose:
            print("Matching a direction with GPT.")

        # Construct the initial instructions for the GPT model, explaining its role in matching directions.
        instructions = "".join(
            [
                "You are the parser for a text adventure game. For an input command try to ",
                "match the direction in the command. Give the closest matching one, or say ",
                "None if none match. The possible directions are:",
            ]
        )

        # Initialize an empty dictionary to store possible directions.
        directions = {}

        # If a location is provided, iterate over its connections to gather possible directions.
        if location:
            for direction, to_loc in location.connections.items():
                # Create a description string that includes the direction and the location it leads to.
                loc_description = "{name} - {description}".format(
                    name=to_loc.name, description=to_loc.description
                )
                location_name_direction = "{direction} toward {loc}".format(
                    direction=direction, loc=loc_description
                )
                # Store the formatted description in the directions dictionary, mapping it to the direction.
                directions[location_name_direction] = direction

        # Define a dictionary of common shorthand directions and their full forms.
        other_directions = {
            "'n' can mean north": "north",
            "'s' can mean south": "south",
            "'e' can mean east": "east",
            "'w' can mean west": "west",
            "'out' can mean 'go out'": "out",
            "'in' can mean 'go in'": "in",
            "'up' can mean 'go up'": "up",
            "'down' can mean 'go down'": "down",
        }

        # Update the directions dictionary with the common shorthand directions.
        directions |= other_directions

        # Call the gpt_pick_an_option function to determine the best matching direction based on the instructions and
        # command.
        return gpt_pick_an_option(
            instructions,
            directions,  # Pass the dictionary of possible directions for matching.
            command,  # The command input by the player to analyze.
            call_handler=self.gpt_handler,  # The handler for interacting with the GPT model.
            max_tokens=10,  # Limit the number of tokens in the response.
        )


# class GptParser3(GptParser2):
#     def __init__(self, game, echo_commands=True, verbose=False, model='gpt-4o-mini'):
#         super().__init__(game, echo_commands, verbose)
#         self.model = model

#     def extract_digit(self, text):
#         return re.findall(r"[-]?\d+", text)[0]

#     def get_characters_and_find_current(self, character=None):
#         current_idx = -999
#         chars = {}
#         for i, char in enumerate(list(self.game.characters)):
#             chars[i] = char
#             if character and char == character.name:
#                 current_idx = i
#         return chars, current_idx

#     def get_character(
#         self, command: str, character: Character = None, hint: str = None, split_words=None, position=None
#     ) -> Character:
#         """
#         This method tries to match a character's name in the command.
#         If no names are matched, it defaults to the passed character.
#         Args:
#             hint: A hint about the role of character we're looking for
#                   (e.g. "giver" or "recipent")
#             split_words: not needed for our GptParser
#             position: not needed for our GptParser
#         """

#         system_prompt = "Given a command, return the character who can be described as: \"{h}\". ".format(h=hint)
#         # Create an enumerated dict of the characters in the game

#         chars, curr_idx = self.get_characters_and_find_current(character)
#         if character:
#             system_prompt += f"Unless specified, assume \"{curr_idx}: {character.name}\" performs all actions.\nChoose from the following characters:\n"
#         else:
#             system_prompt += "Choose from the following characters:\n"
#         # Format the characters into a list structure for the system prompt
#         system_prompt += "{c}".format(c='\n'.join([str(i)+": "+str(c) for i, c in chars.items()]))

#         system_prompt += "\nYou must only return the single number whose corresponding character is performing the action.\n\
# If no command is given, return \"{curr_idx}: {character.name}\""
#         # if hint:
#         #     system_prompt += "As a hint, in the given command, the subject can be described as: \"{h}\". ".format(h=hint)
#         #     system_prompt += "If there are no good matches, the action is performed by the game player, so you should return 0.\n"
#         # else:
#         #     system_prompt += "If there are no good matches, the action is performed by the game player, so you should return 0.\n"

#         # create a new client
#         # client = OpenAI()

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt
#                 },
#                 {
#                     "role": "user",
#                     "content": "Command: {c}\nThe best character match is number: ".format(c=command)
#                 },
#             ],
#             temperature=0,
#             max_tokens=10,
#             top_p=0,
#             frequency_penalty=0,
#             presence_penalty=0
#         )

#         # Will probably need to do some parsing of the output here
#         char_idx = response.choices[0].message.content
#         try:
#             char_idx = self.extract_digit(char_idx)
#             char_idx = int(char_idx)
#         except Exception as e:
#             print("Couldn't match the following response to a number:")
#             print(char_idx)
#             print(e)

#         # print("Item system prompt: ", system_prompt)
#         print(f"GPTParse selected character: {char_idx}")
#         if char_idx not in chars:
#             print(f"no player with id {char_idx} in {str(chars)}")
#             return None
#         else:
#             name = chars[char_idx]
#             return self.game.characters[name]

#     def match_item(
#         self, command: str, item_dict: dict[str, Item], hint: str = None
#     ) -> Item:
#         """
#         Check whether the names of any of the items in this dictionary match the
#         command. If so, return Item, else return None.

#         Args:
#             item_dict: A map from item names to Items (could be a player's
#                        inventory or the items at a location)
#             hint: what kind of item we're looking for
#         """

#         system_prompt = "Given a command, return the item that is the direct object of the action.\nChoose from the following items:\n"
#         items = {i: it for i, it in enumerate(list(item_dict.keys()))}
#         system_prompt += "{c}".format(c=''.join([str(i)+": "+str(item)+"\n" for i, item in items.items()]))
#         system_prompt += """You must only return the single number whose corresponding item best matches the given command. \
# If there are no good matches, return '-999'\n"""
#         if hint:
#             system_prompt += "As a hint, in the given command, the item can be described as:\"{h}\".\n".format(h=hint)
#         else:
#             system_prompt += "\n"

#         # print("Item system prompt: ", system_prompt)
#         # client = OpenAI()

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt
#                 },
#                 {
#                     "role": "user",
#                     "content": "Command: {c}\n  The best item match is number: ".format(c=command)
#                 },
#             ],
#             temperature=0,
#             max_tokens=10,
#             top_p=0,
#             frequency_penalty=0,
#             presence_penalty=0
#         )

#         item_idx = response.choices[0].message.content
#         try:
#             item_idx = self.extract_digit(item_idx)
#             item_idx = int(item_idx)
#         except Exception as e:
#             print(e)

#         print(f"GPTParse selected item: {item_idx}")
#         if item_idx == -999:
#             return None
#         elif item_idx in items:
#             name = items[item_idx]
#             return item_dict[name]
#         else:
#             print(f'Item index {item_idx} not found in {str(items)}')

#     def get_direction(self, command: str, location: Location = None) -> str:
#         """
#         Return the direction from `location.connections` which the player
#         wants to travel to.
#         """
#         dirs = list(location.connections.keys())
#         names = [loc.name for loc in location.connections.values()]
#         connections = {i: dl for i, dl in enumerate(zip(dirs, names))}
#         print('Found connections: ', connections)

#         system_prompt = """
#         You must select the direction that best matches the description given in a command.
#         The possible directions to choose are:\n
#         """

#         system_prompt += "\n" + "{c}".format(c=''.join([str(i)+": "+str(d)+" or "+str(l)+"\n" for i, (d, l) in connections.items()]))

#         system_prompt += """\nYou must only return the single number whose corresponding direction best matches the given command.
#             If there are no good matches, return '-999'\n"""

#         # print("Direction system prompt: ", system_prompt)

#         # client = OpenAI()

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt
#                 },
#                 {
#                     "role": "user",
#                     "content": "Command: {c}\n  The best direction match is number:  ".format(c=command)
#                 }
#             ],
#             temperature=0,
#             max_tokens=100,
#             top_p=0,
#             frequency_penalty=0,
#             presence_penalty=0
#         )

#         dir_idx = response.choices[0].message.content
#         try:
#             dir_idx = self.extract_digit(dir_idx)
#             dir_idx = int(dir_idx)
#         except Exception as e:
#             print(e)
#         print(f"GPTParse selected direction: {dir_idx}")

#         if dir_idx in connections:
#             dir_name = connections[dir_idx][0]
#             return dir_name
#         else:
#             print(f'direction id "{dir_idx}" not in location connections: {connections}')
#             return None
