import json
import inspect
from collections import defaultdict, namedtuple
import os
from typing import TYPE_CHECKING, Literal
from numpy.random import permutation
import dill as pickle
import concurrent.futures

from .agent.memory_stream import MemoryType
from .things import Location, Character
from . import parsing, actions, blocks
from .utils.custom_logging import logger
from .agent.agent_cognition.vote import VotingSession, JuryVotingSession
from .assets.prompts import vote_prompt, world_info_prompt
from .utils.consts import get_output_logs_path
from .utils.general import create_dirs, get_logger_extras
from .gpt.gpt_helpers import GptCallHandler


class Game:
    """
    The Game class keeps track of the state of the world, and describes what
    the player sees as they move through different locations.

    Internally, we use a graph of Location objects and Item objects, which can
    be at a Location or in the player's inventory. Each locations has a set of
    exits which are the directions that a player can move to get to an
    adjacent location. The player can move from one location to another
    location by typing a command like "Go North".
    """

    def __init__(
        self,
        start_at: Location,
        player: Character,
        characters=None,
        custom_actions=None,
        verbose=False,
    ):
        """
        Initializes a game instance with a starting location, a player character, and optional non-player characters and
        custom actions. This constructor method sets up the game environment, including the player's starting position,
        game history, and the parser for handling commands.

        Args:
            start_at (Location): The starting location of the player in the game.
            player (Character): The player character controlled by the user.
            characters (list, optional): A list of additional characters (NPCs) to include in the game.
            custom_actions (list, optional): A list of custom actions to be added to the game's parser.

        Returns:
            None
        """

        self.start_at = start_at
        self.player = player
        self.verbose = verbose

        # Print the special commands associated with items in the game (helpful
        # for debugging and for novice players).
        self.give_hints = True

        # Records history of commands, states, and descriptions
        self.game_history = []

        self.game_over = False
        self.game_over_description = None

        # Add player to game and put them on starting point
        self.characters = {}
        self.add_character(player)
        self.start_at.add_character(player)
        self.start_at.has_been_visited = True

        # Add NPCs to game
        if characters:
            for c in characters:
                if isinstance(c, Character):
                    self.add_character(c)
                else:
                    err_msg = f"ERROR: invalid character ({c})"
                    raise Exception(err_msg)

        # Look up table for locations
        def location_map(location, acc):
            """
            Recursively builds a mapping of locations starting from a given location. This function populates an
            accumulator dictionary with location names as keys and their corresponding location objects as values,
            traversing through all connected locations.

            Args:
                location (Location): The starting location from which to build the mapping.
                acc (dict): The accumulator dictionary that stores the mapping of location names to location objects.

            Returns:
                dict: The updated accumulator dictionary containing the mapped locations.
            """

            # Store the current location in the accumulator dictionary
            acc[location.name] = location

            # Iterate through each connection of the current location
            for _, connection in location.connections.items():
                # Check if the connection's name is not already in the accumulator
                if connection.name not in acc:
                    # Recursively map the connected location and update the accumulator
                    acc = location_map(connection, acc)

            # Return the updated accumulator
            return acc

        # Initialize locations by mapping the starting location with an empty dictionary
        self.locations = location_map(self.start_at, {})

        # Initialize the parser with the current instance
        self.parser = parsing.Parser(self)

        # Add custom actions to the parser if provided
        if custom_actions:
            print("Adding custom actions")
            for ca in custom_actions:
                # Check if the custom action is a class and a subclass of actions.Action
                if inspect.isclass(ca) and issubclass(ca, actions.Action):
                    # Add the valid custom action to the parser
                    self.parser.add_action(ca)
                else:
                    # Raise an exception if the custom action is invalid
                    err_msg = f"ERROR: invalid custom action ({ca})"
                    raise Exception(err_msg)

        # Track seen locations to avoid duplicates
        seen_before = {}
        # Iterate through each location in the locations dictionary
        for name, location in self.locations.items():
            # Check if the location has blocks and hasn't been seen before
            if len(location.blocks) > 0 and name not in seen_before:
                # Add each block found in the location to the parser
                for b in location.blocks:
                    self.parser.add_block(b)
                # Mark the location as seen
                seen_before[name] = True

    def set_parser(self, parser: parsing.Parser):
        """
        Sets the parser for the game instance, allowing for command parsing and handling. This method updates the parser
        attribute with the provided parser instance, enabling the game to process user input effectively.

        Args:
            parser (parsing.Parser): The parser instance to be assigned to the game.

        Returns:
            None
        """

        self.parser = parser

    def game_loop(self):
        """
        Run the main game loop, processing player commands.

        This function continuously prompts the player for input and processes
        commands until the game is over. It starts by executing a default command
        to look around, then enters a loop to handle user input.

        Args:
            self: The instance of the class.

        Returns:
            None

        Raises:
            None

        Examples:
            game_loop()  # Starts the game loop for the player.
        """

        self.parser.parse_command("look")

        # Start an infinite loop to continuously prompt for user input
        while True:
            # Prompt the user for a command and store the input
            command = input("\n> ")

            # Parse the entered command using the parser
            self.parser.parse_command(command)

            # Check if the game is over after processing the command
            if self.is_game_over():
                # Exit the loop if the game is over
                break

    def is_won(self) -> bool:
        """
        Determine if the game has been won.

        This function checks the current state of the game to ascertain whether
        the player has achieved the winning condition. Currently, it always
        returns False, indicating that the game is not won.

        Args:
            self: The instance of the class.

        Returns:
            bool: False, indicating the game has not been won.

        Raises:
            None

        Examples:
            result = is_won()  # Returns False.
        """

        return False

    def is_game_over(self) -> bool:
        """
        Check if the game has ended.

        This function evaluates the current state of the game to determine if
        it is over. It checks for conditions such as whether the game over state
        has been set, if the player has died, or if the game has been won.

        Args:
            self: The instance of the class.

        Returns:
            bool: True if the game is over, otherwise False.

        Raises:
            None

        Examples:
            game_status = is_game_over()  # Returns True or False based on the game state.
        """

        # Something has set the game over state
        if self.game_over:
            return True
        # The player has died
        if self.player.get_property("is_dead"):
            self.game_over_description = "You have died. THE END"
            return True
        # Has the game has been won?
        return self.is_won()

    def add_character(self, character: Character):
        """
        Add a character to the game.

        This function adds a specified character to the game's character list
        using the character's name as the key. It allows for easy management and
        retrieval of characters within the game.

        Args:
            self: The instance of the class.
            character (Character): The character to be added to the game.

        Returns:
            None

        Raises:
            None

        Examples:
            add_character(new_character)  # Adds 'new_character' to the game's character list.
        """

        self.characters[character.name] = character

    def describe(self) -> str:
        """
        Generate a comprehensive description of the current game state.

        This function compiles and returns a detailed description of the player's
        current location, including exits, items, characters, and inventory. It
        provides a holistic view of the environment and available interactions.

        Args:
            self: The instance of the class.

        Returns:
            str: A formatted string containing the description of the current state.

        Raises:
            None

        Examples:
            current_description = describe()  # Retrieves the current game state description.
        """

        # Get the description of the current location and add a newline
        description = self.describe_current_location() + "\n"

        # Append the descriptions of exits and add a newline
        description += self.describe_exits() + "\n"

        # Append the descriptions of items and add a newline
        description += self.describe_items() + "\n"

        # Append the descriptions of characters and add a newline
        description += self.describe_characters() + "\n"

        # Append the description of the inventory without a newline
        description += self.describe_inventory()

        if self.verbose:
            print(f"total description: {description}")

        # Return the complete description of the current game state
        return description

    def describe_current_location(self) -> str:
        """
        Provide a description of the current location.

        This function generates and returns a detailed description of the
        player's current location within the game. It includes relevant
        information that helps the player understand their surroundings.

        Args:
            self: The instance of the class.

        Returns:
            str: A formatted string containing the description of the current location.

        Raises:
            None

        Examples:
            location_description = describe_current_location()  # Retrieves the description of the current location.
        """

        return f"location: {self.player.name} is at {self.player.location.description}"

    def describe_exits(self) -> str:
        """
        Generate a description of available exits from the current location.

        This function compiles a list of exits that the player can take from
        their current location, detailing the direction and the name of the
        connected location. It provides a clear overview of the possible paths
        the player can explore.

        Args:
            self: The instance of the class.

        Returns:
            str: A formatted string listing the available exits from the current location.

        Raises:
            None

        Examples:
            exit_description = describe_exits()  # Retrieves the description of exits from the current location.
        """

        # Initialize an empty list to store exit descriptions
        exits = []

        # Iterate over the keys (directions) in the player's current location's connections
        for direction in self.player.location.connections.keys():
            # Get the connected location for the current direction
            location = self.player.location.connections[direction]
            # Append a formatted string describing the exit to the exits list
            exits.append(f"{direction.capitalize()} to {location.name}")

        # Initialize the description string for exits
        description = "exits: "

        # Check if there are any exits available
        if exits:
            # If exits exist, add a message indicating possible directions from the current location
            description += (
                f"From {self.player.location.name}, {self.player.name} could go: "
            )
            # Iterate over the exits and append each exit description to the description string
            for exit in exits:
                description += f"{exit}, "

        if self.verbose:
            print(f"Exit description: {description}")

        # Return the final description of exits
        return description

    def describe_items(self) -> str:
        """
        Generate a description of items in the current location.

        This function compiles a list of items present in the player's current
        location, providing details about each item. If enabled, it also includes
        hints for special commands associated with the items, enhancing the player's
        understanding of their surroundings.

        Args:
            self: The instance of the class.

        Returns:
            str: A formatted string listing the items in the current location and any associated hints.

        Raises:
            None

        Examples:
            item_description = describe_items()  # Retrieves the description of items in the current location.
        """

        # Initialize the description string for items
        description = "items: "

        # Check if there are any items in the player's current location
        if len(self.player.location.items) > 0:
            # Append the player's name and a message indicating what they see
            description += f"{self.player.name} sees: "

            # Iterate through each item in the current location
            items = list(self.player.location.items.items())
            for i, (item_name, item) in enumerate(items):
                # Append the item's description to the overall description
                description += item.description

                # If hints are enabled, check for special commands associated with the item
                if self.give_hints:
                    if special_commands := item.get_command_hints():
                        # Start the hint section in the description
                        description += "(hint "

                        # Append each special command hint to the description
                        for cmd in special_commands:
                            description += f"{cmd}, "

                        # Close the hint section
                        description += ")"

                # Add a semicolon to separate items, but not after the last item
                if i < len(items) - 1:
                    description += "; "

        # Return the complete description of items in the current location
        return description

    def describe_characters(self) -> str:
        """
        Generate a description of characters in the current location.

        This function compiles a list of characters present in the player's
        current location, excluding the player themselves. It provides an overview
        of other characters that the player can interact with or observe.

        Args:
            self: The instance of the class.

        Returns:
            str: A formatted string listing the characters in the current location.

        Raises:
            None

        Examples:
            character_description = describe_characters()  # Retrieves the description of characters in the current
            location.
        """

        # Initialize the description string for characters
        description = "characters: "

        # Check if there are more than one character in the player's current location
        if len(self.player.location.characters) > 1:
            # Append the player's name and a message indicating what they see
            description += f"{self.player.name} sees characters: "

            # Create a list to store character names
            character_names = []

            # Iterate through each character in the current location
            for character_name in self.player.location.characters:
                # Skip the player themselves to avoid including them in the list
                if character_name == self.player.name:
                    continue

                # Retrieve the character object using their name
                character = self.player.location.characters[character_name]

                # Add the character's name to the list
                character_names.append(character.name)

            # Join the character names with commas and append to the description
            description += ", ".join(character_names)

        # Return the complete description of characters in the current location
        return description

    def describe_inventory(self) -> str:
        """
        Generate a description of the player's inventory.

        This function compiles a list of items currently held in the player's
        inventory, providing details about each item. It informs the player of
        what they have available for use or interaction within the game.

        Args:
            self: The instance of the class.

        Returns:
            str: A formatted string describing the contents of the player's inventory.

        Raises:
            None

        Examples:
            inventory_description = describe_inventory()  # Retrieves the description of the player's inventory.
        """

        # Initialize the description string for the inventory
        inventory_description = "inventory: "

        # Check if the player's inventory is empty
        if len(self.player.inventory) == 0:
            # Append a message indicating that the inventory is empty
            inventory_description += f"{self.player.name} has nothing in inventory."
            # Uncomment the following line to handle empty inventory logic
            # self.ok(empty_inventory, [], "Describe the player's inventory.")
        else:
            # Append a message indicating the contents of the player's inventory
            inventory_description += (
                f"In {self.player.name} inventory, {self.player.name} has: "
            )

            # Create a list to store item descriptions
            item_descriptions = []

            # Iterate through each item in the player's inventory
            for item_name in self.player.inventory:
                # Retrieve the item object using its name
                item = self.player.inventory[item_name]

                # Add the item's description to the list
                item_descriptions.append(item.description)

            # Join the item descriptions with commas and append to the inventory description
            inventory_description += ", ".join(item_descriptions)

        # Return the complete description of the player's inventory
        return inventory_description

    # The methods below read and write a game to JSON
    def to_primitive(self):
        """
        Convert the game state to a primitive data structure.

        This function creates a dictionary representation of the current game state,
        including details about the player, starting location, game history, and
        the status of the game. It also includes lists of characters, locations, and
        actions, making it suitable for serialization or further processing.

        Args:
            self: The instance of the class.

        Returns:
            dict: A dictionary containing the primitive representation of the game state.

        Raises:
            None

        Examples:
            game_data = to_primitive()  # Retrieves a dictionary representation of the current game state.
        """

        return {
            "player": self.player.name,
            "start_at": self.start_at.name,
            "game_history": self.game_history,
            "game_over": self.game_over,
            "game_over_description": self.game_over_description,
            "characters": [c.to_primitive() for c in self.characters.values()],
            "locations": [l.to_primitive() for l in self.locations.values()],
            "actions": sorted(list(self.parser.actions)),
        }

    @classmethod
    def default_actions(self):
        """
        Retrieve a dictionary of default actions available in the game.

        This class method scans the `actions` module for all classes that are
        subclasses of `actions.Action`, excluding the base class itself. It returns
        a dictionary mapping action names to their corresponding action classes.

        Args:
            self: The instance of the class.

        Returns:
            dict: A dictionary containing action names as keys and action classes as values.

        Raises:
            None

        Examples:
            actions_dict = default_actions()  # Retrieves a dictionary of default actions.
        """

        # Initialize an empty dictionary to store found actions
        actions_found = {}

        # Iterate through all members of the 'actions' module
        for member in dir(actions):
            # Retrieve the attribute corresponding to the member name
            attr = getattr(actions, member)

            # Check if the attribute is a class and a subclass of actions.Action,
            # while ensuring it is not the base class itself
            if (
                inspect.isclass(attr)
                and issubclass(attr, actions.Action)
                and not attr == actions.Action
            ):
                # Add the action name and class to the actions_found dictionary
                actions_found[attr.action_name()] = attr

        # Return the dictionary containing the found actions
        return actions_found

    @classmethod
    def default_blocks(self):
        """
        Retrieve a dictionary of default blocks available in the game.

        This class method scans the `blocks` module for all classes that are
        subclasses of `blocks.Block`, excluding the base class itself. It returns
        a dictionary mapping block names to their corresponding block classes.

        Args:
            self: The instance of the class.

        Returns:
            dict: A dictionary containing block names as keys and block classes as values.

        Raises:
            None

        Examples:
            blocks_dict = default_blocks()  # Retrieves a dictionary of default blocks.
        """

        # Initialize an empty dictionary to store found blocks
        blocks_found = {}

        # Iterate through all members of the 'blocks' module
        for member in dir(blocks):
            # Retrieve the attribute corresponding to the member name
            attr = getattr(blocks, member)

            # Check if the attribute is a class and a subclass of blocks.Block,
            # while ensuring it is not the base class itself
            if (
                inspect.isclass(attr)
                and issubclass(attr, blocks.Block)
                and not attr == blocks.Block
            ):
                # Add the block's name and class to the blocks_found dictionary
                blocks_found[attr.__name__] = attr

        # Return the dictionary containing the found blocks
        return blocks_found

    @classmethod
    def from_primitive(cls, data, custom_actions=None, custom_blocks=None):
        """
        Create an instance of the class from a primitive data structure.

        This class method reconstructs the game state from a provided dictionary,
        populating characters, locations, items, actions, and blocks. It performs
        multiple passes to ensure all relationships and properties are correctly
        established, and validates any custom actions or blocks provided.

        This complex method performs the huge job of converting a game from its
        primitive representation to fully formed python objects.

        There are three main parts to this method:

        1. Create skeletons for all characters and locations. Currently, items
           exist by being in a location or a character's inventory, and so this
           step also creates item skeletons. See the from_primitive methods for
           characters and locations for more.
        2. Replace fields in skeletons where an object's name exists with the
           actual objects. This step replaces fields where an object's name is
           stored instead of the actual object.
        3. Instantiate anything left that requires full object instances to
           work properly. Blocks require actual instances for everything.

        Once those steps are done, this method simply adds any remaining game
        fields to the game instance.

        Args:
            cls: The class itself.
            data (dict): A dictionary containing the primitive representation of the game state.
            custom_actions (list, optional): A list of custom action classes to include.
            custom_blocks (list, optional): A list of custom block classes to include.

        Returns:
            instance: An instance of the class populated with the reconstructed game state.

        Raises:
            Exception: If any custom actions or blocks are invalid, or if there are unmapped actions in the data.

        Examples:
            game_instance = from_primitive(data)  # Creates a game instance from the provided data.
        """

        # Define a named tuple to hold the skeleton context for characters, locations, and items
        SkeletonContext = namedtuple(
            "SkeletonContext", ["characters", "locations", "items"]
        )

        # FIRST PASS: Initialize characters, locations, and items from the provided data

        # Create a dictionary of characters by their names, converting each from primitive data
        characters = {
            c["name"]: Character.from_primitive(c) for c in data["characters"]
        }

        # Create a dictionary of locations by their names, converting each from primitive data
        locations = {l["name"]: Location.from_primitive(l) for l in data["locations"]}

        # Initialize an empty dictionary for items
        items = {}

        # Create a SkeletonContext instance to hold the characters, locations, and items
        context = SkeletonContext(characters, locations, items)

        # SECOND PASS: Establish relationships and properties for characters and locations

        # Process each character in the context
        for c in context.characters.values():
            # Set the character's location object based on their location name
            l = context.locations[c.location]
            c.location = l

            # Process each item in the character's inventory
            for item_name, item in c.inventory.items():
                # If the item has a location, update its location to the corresponding location object
                if hasattr(item, "location") and item.location:
                    l_obj = context.locations[item.location]
                    item.location = l_obj
                # If the item has an owner, update its owner to the corresponding character object
                elif hasattr(item, "owner") and item.owner:
                    c_obj = context.characters[item.owner]
                    item.owner = c_obj
                # Add the item to the context's items dictionary
                context.items[item_name] = item

        # Process each location in the context
        for l in context.locations.values():
            # Update each character in the location to the corresponding character object
            for char_name, c in l.characters.items():
                c_obj = context.characters[char_name]
                l.characters[char_name] = c_obj

            # Update each connection in the location to the corresponding location object
            for dir_name, connection in l.connections.items():
                c_obj = context.locations[connection]
                l.connections[dir_name] = c_obj

            # Process each item in the location
            for item_name, item in l.items.items():
                # If the item has a location, update its location to the corresponding location object
                if hasattr(item, "location") and item.location:
                    l_obj = context.locations[item.location]
                    item.location = l_obj
                # If the item has an owner, update its owner to the corresponding character object
                elif hasattr(item, "owner") and item.owner:
                    c_obj = context.characters[item.owner]
                    item.owner = c_obj
                # Add the item to the context's items dictionary
                context.items[item_name] = item

        # THIRD PASS: Handle actions and blocks

        # Retrieve the default actions available in the game
        action_map = cls.default_actions()

        # Validate any custom actions provided
        if custom_actions:
            for ca in custom_actions:
                # Check if the custom action is a valid subclass of actions.Action
                if inspect.isclass(ca) and issubclass(ca, actions.Action):
                    action_map[ca.action_name()] = ca
                else:
                    # Raise an error if the custom action is invalid
                    err_msg = f"ERROR: invalid custom action ({ca})"
                    raise Exception(err_msg)

        # Verify that all actions in the primitive data have corresponding actions in the action map
        action_names = list(action_map.keys())
        for action_name in data["actions"]:
            if action_name not in action_names:
                # Raise an error if an unmapped action is found
                err_msg = "".join(
                    [
                        f"ERROR: unmapped action ({action_name}) found in ",
                        "primitive data",
                    ]
                )
                raise Exception(err_msg)

        # Retrieve the default blocks available in the game
        block_map = cls.default_blocks()

        # Validate any custom blocks provided
        if custom_blocks:
            for cb in custom_blocks:
                # Check if the custom block is a valid subclass of blocks.Block
                if inspect.isclass(cb) and issubclass(cb, blocks.Block):
                    block_map[cb.__name__] = cb
                else:
                    # Raise an error if the custom block is invalid
                    err_msg = f"ERROR: invalid custom block ({cb})"
                    raise Exception(err_msg)

        # Instantiate all blocks for each location
        for l in context.locations.values():
            for direction, block_data in l.blocks.items():
                # Skip any blocks that have already been instantiated
                if isinstance(block_data, blocks.Block):
                    continue

                # Get the class type for the block based on its type
                cls_type = block_map[block_data["_type"]]
                del block_data["_type"]  # Remove the type key from block data

                # Create a property map to store properties of relevant items before installing the block
                prop_map = {}

                # Replace names in the primitive data with actual instances
                for param_name, param in block_data.items():
                    if param in context.items:
                        param_instance = context.items[param]
                    elif param in context.locations:
                        param_instance = context.locations[param]
                    block_data[param_name] = param_instance
                    prop_map[param_name] = (
                        param_instance.properties.copy()
                    )  # Store original properties

                # TODO: The prior version instantiated without assigning the location block dictionary value
                # Instantiate the block from primitive data and assign it to the location
                l.blocks[direction] = cls_type.from_primitive(block_data)

                # Restore properties found in the primitive data
                for param_name, param in block_data.items():
                    param.properties = prop_map[param_name]

        # Set the starting location and player character based on the primitive data
        start_at = context.locations[data["start_at"]]
        player = context.characters[data["player"]]

        # Create an instance of the class with the starting location and player
        instance = cls(start_at, player, custom_actions=action_map.values())
        instance.game_history = data["game_history"]  # Restore game history
        instance.game_over = data["game_over"]  # Restore game over state
        instance.game_over_description = data[
            "game_over_description"
        ]  # Restore game over description

        # Return the fully constructed instance
        return instance

    def to_json(self):
        """
        Convert the game state to a JSON string.

        This function transforms the current game state into a primitive data
        structure and then serializes it into a JSON format. It provides a
        convenient way to export the game state for storage or transmission.

        Args:
            self: The instance of the class.

        Returns:
            str: A JSON string representation of the game state.

        Raises:
            None

        Examples:
            json_data = to_json()  # Converts the game state to a JSON string.
        """

        data = self.to_primitive()
        return json.dumps(data)

    @classmethod
    def from_json(cls, data_json, **kw):
        """
        Create an instance of the class from a JSON string.

        This class method deserializes a JSON string into a primitive data
        structure and then reconstructs the game state by creating an instance
        of the class. It allows for easy loading of game state from a JSON format.

        Args:
            cls: The class itself.
            data_json (str): A JSON string representation of the game state.
            **kw: Additional keyword arguments to pass to the from_primitive method.

        Returns:
            instance: An instance of the class populated with the reconstructed game state.

        Raises:
            None

        Examples:
            game_instance = from_json(json_data)  # Creates a game instance from the provided JSON string.
        """

        data = json.loads(data_json)
        return cls.from_primitive(data, **kw)

    def save_game(self, filename):
        """
        Save the current game state to a file in JSON format.

        This function serializes the current game state into a JSON string and
        writes it to a specified file. It provides a way to persist the game state
        for later retrieval.

        Args:
            self: The instance of the class.
            filename (str): The name of the file where the game state will be saved.

        Returns:
            None

        Raises:
            None

        Examples:
            save_game("savefile.json")  # Saves the current game state to 'savefile.json'.
        """

        save_data = self.to_json()
        with open(filename, "w") as f:
            f.write(save_data)

    @classmethod
    def load_game(cls, filename, **kw):
        """
        Load a game state from a file in JSON format.

        This class method reads a JSON string from a specified file and reconstructs
        the game state by creating an instance of the class. It allows for easy
        retrieval of previously saved game states.

        Args:
            cls: The class itself.
            filename (str): The name of the file from which the game state will be loaded.
            **kw: Additional keyword arguments to pass to the from_json method.

        Returns:
            instance: An instance of the class populated with the loaded game state.

        Raises:
            None

        Examples:
            game_instance = load_game("savefile.json")  # Loads the game state from 'savefile.json'.
        """

        with open(filename, "r") as f:
            save_data = f.read()
            return cls.from_json(save_data, **kw)


# Override methods or implement a new class?
class SurvivorGame(Game):
    """
    A game class that simulates a survival competition among characters.

    This class extends the base Game class to implement specific mechanics for a
    survivor-style game, including character interactions, voting sessions, and
    tracking of game state. It manages the progression of rounds and ticks, as well
    as the end conditions for the game.

    Args:
        start_at (Location): The starting location of the player in the game.
        player (Character): The player character controlled by the user.
        characters (list, optional): A list of additional characters (NPCs) to include in the game.
        custom_actions (list, optional): A list of custom actions to be added to the game's parser.
        max_ticks (int, optional): The maximum number of ticks per round, defaulting to 10.
        num_finalists (int, optional): The number of finalists in the game, defaulting to 2.
        experiment_name (str, optional): The name of the experiment, defaulting to "exp1".
        experiment_id (int, optional): The ID of the experiment, defaulting to 1.
        end_state_check (Literal, optional): The condition for checking the end state, defaulting to "on_round".

    Returns:
        None
    """

    def __init__(
        self,
        start_at: Location,
        player: Character,
        characters=None,
        custom_actions=None,
        max_ticks: int = 10,
        num_finalists: int = 2,
        experiment_name: str = "exp1",
        experiment_id: int = 1,
        end_state_check: Literal["on_round", "on_tick", "on_action"] = "on_round",
    ):
        """
        Initializes a SurvivorGame instance with a starting location, a player character, and optional non-player
        characters and custom actions. This constructor method sets up the game environment, including game tracking
        variables, logging, and the initial state of the game.

        Args:
            start_at (Location): The starting location of the player in the game.
            player (Character): The player character controlled by the user.
            characters (list, optional): A list of additional characters (NPCs) to include in the game.
            custom_actions (list, optional): A list of custom actions to be added to the game's parser.
            max_ticks (int, optional): The maximum number of ticks per round, defaulting to 10.
            num_finalists (int, optional): The number of finalists in the game, defaulting to 2.
            experiment_name (str, optional): The name of the experiment, defaulting to "exp1".
            experiment_id (int, optional): The ID of the experiment, defaulting to 1.
            end_state_check (Literal, optional): The condition for checking the end state, defaulting to "on_round".

        Returns:
            None
        """

        # Call the initializer of the parent Game class with the starting location, player, characters, and custom
        # actions
        super().__init__(start_at, player, characters, custom_actions)

        # Initialize a custom logger ("gen_agents_global_logger") for the game with the experiment name and simulation
        # ID, which determine the saved file locations.
        game_logger = logger.CustomLogger(
            name="gen_agents_global_logger",
            experiment_name=experiment_name,
            simulation_id=experiment_id,
            logfile_prefix="sim",
            overwrite=True,
        )
        gpt_call_logger = logger.CustomLogger(
            name="gpt_call_logger",
            experiment_name=game_logger.get_experiment_name(),
            simulation_id=game_logger.get_simulation_id,
            logfile_prefix="gpt_calls",
            overwrite=False,
        )
        self.logger = game_logger.get_logger()
        self.gpt_call_logger = gpt_call_logger.get_logger()
        self.experiment_name = experiment_name  # Store the experiment name
        self.experiment_id = game_logger.get_simulation_id()  # Store the simulation ID

        # Store the original player ID for reference
        self.original_player_id = self.player.id

        # Game related tracking variables
        self.max_ticks_per_round = max_ticks
        self.round = 0
        self.tick = 0
        self.total_ticks = 0
        self.num_contestants = len(self.characters)
        self.end_state_check = end_state_check

        # Store end state variables:
        # Initialize a jury to hold exiled players who will cast the final vote
        self.jury = {}
        self.voting_history = defaultdict(lambda: defaultdict(list))
        self.num_finalists = num_finalists
        self.winner_declared = False

        # Log the starting locations of the characters for tracking purposes
        self._log_starting_locs()

    def update_world_info(self):
        """
        Updates the world information for the game, including details about the contestants and the current game state.
        This method constructs a dictionary of parameters that reflect the current status of the game and formats it
        into a world information string.

        Args:
            None

        Returns:
            None
        """

        # Construct a dictionary of parameters to represent the current game state
        params = {
            "contestant_count": len(
                self.characters
            ),  # Count the total number of contestants
            "contestant_names_locs": ", ".join(
                [
                    f"{c.name} who is at {c.location.name}"
                    for c in self.characters.values()
                    if c.id != self.player.id
                ]
            ),  # List names and locations of contestants excluding the player
            "n_finalists": self.num_finalists,  # Store the number of finalists in the game
            "rounds_until_finals": len(self.characters)
            - self.num_finalists,  # Calculate rounds remaining until finals
            "turns_left_this_round": self.max_ticks_per_round
            - (self.tick - 1),  # Calculate turns left in the current round
        }

        # Format the world information string using the constructed parameters
        self.world_info = world_info_prompt.world_info.format(**params)

    # Override game loop
    def game_loop(self):
        """
        Executes the main game loop, managing the progression of the game through rounds and ticks. This method handles
        character turns, goal setting, and checks for game-ending conditions, while also saving the game state and
        logging relevant data.

        Args:
            None

        Returns:
            None
        """

        # Start the game loop
        while True:
            # Iterate through the ticks for the current round
            for tick in range(self.max_ticks_per_round):
                self.tick = tick  # Update the current tick

                # Print the current round and tick for confirmation
                print(f"ROUND: {self.round}.{self.tick}")

                # Uncomment the following lines to handle voting sessions at the end of the round
                # if self.tick == (self.max_ticks_per_round - 1):
                #     self.handle_voting_sessions()

                # Set goals for all characters at the beginning of the round
                self.goal_setting_handler()

                # Reset the dialogue state for all characters
                self.reset_character_dialogue()

                # Iterate through characters in a random order for their turns
                for character in permutation(list(self.characters.values())):
                    print(f"It is: {character.name}'s turn")
                    self.turn_handler(character)  # Handle the character's turn

                    # Check if the game has ended after the character's action
                    if self.end_state_check == "on_action" and self.is_game_over():
                        self.save_end_game_data()
                        return  # Exit the game loop if the game is over

                # Update the total ticks that have occurred in the game
                self.total_ticks += 1

                # Check if the game has ended after the tick
                if self.end_state_check == "on_tick" and self.is_game_over():
                    self.save_end_game_data()
                    return  # Exit the game loop if the game is over

            # Check if the game has ended at the end of the round
            if self.end_state_check == "on_round" and self.is_game_over():
                self.save_end_game_data()
                return  # Exit the game loop if the game is over

            # Increment the round counter for the next iteration
            self.round += 1

            # Save the game results so far for later analysis
            self.save_end_game_data()

    def save_end_game_data(self):
        """
        Saves all necessary data when the game ends.
        """
        self.save_simulation_data()
        self._log_gpt_call_data()
        self.save_game("test_file.json")

    def reset_character_dialogue(self):
        """
        Resets the dialogue state for all characters in the game. This method sets each character's dialogue participant
        to None, effectively clearing any previous interactions.

        Args:
            None

        Returns:
            None
        """

        for c in self.characters.values():
            c.set_dialogue_participant(talked_to=None)

    def goal_setting_handler(self):
        """
        Handles the goal-setting process for all characters at the beginning of a round. This method updates the world
        information and prompts each character to generate their goals based on the current game state.

        Args:
            None

        Returns:
            None
        """

        # if it is the beginning of a round, everyone should make goals
        if self.tick == 0:
            # Update the world info with new tick, contestant counts, and non-player contestant names
            self.update_world_info()
            for character in self.characters.values():
                character.generate_goals(self)

        # # TODO: I need to test if the code below correctly gets goals in parallel
        # # if it is the beginning of a round, everyone should make goals
        # if self.tick == 0:

        #     self.update_world_info()

        #     def generate_goals_for_character(character):
        #         character.generate_goals(self)

        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         executor.map(generate_goals_for_character, self.characters.values())

    def turn_handler(self, character):
        """
        Handles the turn for a specified character during the game. This method sets the current player, updates the
        world information, and processes the character's actions, allowing for a maximum of three attempts to
        successfully enact a command.

        Args:
            character (Character): The character whose turn is being processed.

        Returns:
            None
        """

        # set the current player to the game's "player" for description purposes
        self.player = character

        # Update the world info with new tick, contestant counts, and non-player contestant names
        self.update_world_info()

        success = False
        # Only move on to the next character when current takes a successful action
        # But agent only gets three tries
        for _ in range(3):
            if character.id == self.original_player_id:
                # TODO: How do we integrate the ability for a human player to engage?
                command = character.engage(self)
            else:
                command = character.engage(self)

            if self._should_enact_command(command):
                success = self.parser.parse_command(command, character)
            else:
                # This is the end of round case when -999 is returned. I don't want to log that.
                break
            if success:
                self._log_action(character, command)
                break

    def is_game_over(self) -> bool:
        """
        Checks whether the game has ended by evaluating the game state. This method returns True if the game is marked
        as over or if the winning conditions have been met.

        Returns:
            bool: True if the game is over; otherwise, False.
        """

        return True if self.game_over else self.is_won()

    def is_won(self):
        """
        Checks whether the game has been won. For SurvivorWorld, the game is won
        once anyone has been voted the victor.
        """

        if self.winner_declared:
            print(
                (
                    f"""Congratulations!! {self.winner.name} won the game! """
                    """They're the ultimate Survivor. Jeff is so proud of u!"""
                )
            )
            return True
        return False

    def _should_enact_command(self, command):
        """
        Determines whether a command should be enacted based on its type and specific value. This method evaluates if
        the command is an integer or a string, returning False for the integer value -999 and raising an error for
        unsupported types.

        Args:
            command (Union[int, str]): The command to evaluate for enactment.

        Returns:
            bool: True if the command can be enacted; otherwise, False.

        Raises:
            ValueError: If the command is neither an integer nor a string.
        """

        # Check if the command is an integer
        if isinstance(command, int):
            # If the command is -999, it should not be enacted
            if command == -999:
                return (
                    False  # Return False to indicate the command should not be enacted
                )

        # Check if the command is a string
        elif isinstance(command, str):
            return True  # Return True to indicate the command can be enacted

        # If the command is neither an integer nor a string, raise an error
        else:
            raise ValueError(
                f"command: {command} must be str or int; got {type(command)}"
            )  # Raise an error with a descriptive message

    def view_character_locations(self):
        """
        Displays the current locations of all characters in the game. This method iterates through the characters and
        prints each character's name along with the name of the location they are currently in.

        Args:
            None

        Returns:
            None
        """

        for name, char in self.characters.items():
            print(f"{name} is in {char.location.name}\n")

    def handle_voting_sessions(self):
        """
        Manages the voting sessions in the game based on the current state of the characters. This method triggers a
        jury session if the number of characters matches the number of finalists, or initiates a voting session to
        exile a character if the end of a round is reached.

        Args:
            None

        Returns:
            None
        """

        # Check if the number of characters is equal to the number of finalists
        if len(self.characters) == self.num_finalists:
            # If true, trigger the jury session to determine the winner
            self.run_jury_session()

        # If the current tick is at the maximum for the round, initiate a voting session
        elif self.tick == (self.max_ticks_per_round - 1):
            # Run a voting session to exile a character from the game
            self.run_voting_session()

    def update_voting_history(self, session: "VotingSession"):
        """
        Updates the voting history for the current round based on the results from a voting session. This method records
        each character's vote and stores it in the voting history for the round.

        Args:
            session (VotingSession): The voting session containing the results to be recorded.

        Returns:
            None
        """

        for char in self.characters.values():
            record = session.record_vote(char)
            self.voting_history[self.round].update({char.name: record})

    def run_voting_session(self):
        """
        Conducts a voting session among the characters in the game to determine which character will be exiled. This
        method initializes a voting session, processes the votes, updates the voting history, manages the exile state,
        and logs the exiled character's status.

        Args:
            None

        Returns:
            None
        """

        # Initialize a new voting session with the current game instance and the characters as participants
        self.vote_session = VotingSession(
            game=self, participants=self.characters.values()
        )

        # Run the voting session to allow characters to cast their votes
        self.vote_session.run()

        # Read the results of the voting session to determine which character was exiled
        exiled = self.vote_session.read_votes()

        # Update the voting history with the results from the current voting session
        self.update_voting_history(session=self.vote_session)

        # Process the state of the game based on the exiled character
        self.update_exile_state(exiled)

        # Add the exiled character to the jury for final voting
        self.add_exiled_to_jury(exiled)

        # Log the details of the exiled character for tracking purposes
        self._log_exiled_player(exiled)

        # Print a message indicating that the character has been exiled and is now part of the jury
        print(f"{exiled.name} was exiled from the group and now sits on the jury.")

    def _log_exiled_player(self, exiled):
        """
        Logs the details of a player who has been exiled from the game. This method records the exiled player's name and
        their position among the remaining contestants in the voting session.

        Args:
            exiled (Character): The character that has been exiled from the game.

        Returns:
            None
        """

        # Count the number of contestants remaining in the game
        contestants_remaining = len(self.characters)

        # Create a message indicating that the exiled character has been exiled and their position
        message = f"{exiled.name} was exiled. Position: {contestants_remaining + 1}"

        # Log the exile of a character along with the generated message (logs a debug message to the game logger)
        self.vote_session.log_vote(exiled, message=message)

    def _log_gpt_call_data(self):
        """
        Logs the current counts of GPT calls and tokens processed during the game. This method retrieves logging extras,
        constructs messages for the current counts, and records them using the logger for debugging purposes.

        Args:
            None

        Returns:
            None
        """

        # Retrieve logging extras for the current context, with no specific character
        extras = get_logger_extras(self, character=None)

        # Set the type of log entry to "Calls" for tracking GPT call counts
        extras["type"] = "Calls"

        # Create a message containing the current count of GPT calls
        message = f"Current GPT calls count: {GptCallHandler.get_calls_count()}"

        # Log the message at the debug level, including the extras for context
        self.logger.debug(msg=message, extra=extras)

        # Update the type of log entry to "Tokens" for tracking GPT token counts
        extras["type"] = "Tokens"

        # Construct a message that includes the current counts of GPT input, GPT output, and embedding tokens processed
        message = (
            f"Current GPT input token count: {GptCallHandler.get_input_tokens_processed()}, "
            f"GPT output token count: {GptCallHandler.get_output_tokens_processed()}, "
            f"embeddings token count: {GptCallHandler.get_embedding_tokens_processed()}"
        )

        # Log the message at the debug level, including the extras for context
        self.logger.debug(msg=message, extra=extras)

    def _log_action(self, character, message):
        """
        Logs an action performed by a character during the game. This method constructs logging extras for the specified
        character and records the action message at the debug level for tracking purposes.

        Args:
            character (Character): The character performing the action.
            message (str): The message describing the action taken by the character.

        Returns:
            None
        """

        # Retrieve logging extras for the current context, including the specified character
        extras = get_logger_extras(self, character, include_gpt_call_id=True)

        # Set the type of log entry to "Act" to indicate an action has occurred
        extras["type"] = "Act"

        # Log the action message at the debug level, including the extras for context
        self.logger.debug(msg=message, extra=extras)

    def update_exile_state(self, exiled_agent):
        """
        Updates the state of the game by processing the effects of an exiled agent on all characters. This method
        manages the memories of each character regarding the exile, removes the exiled character from the game, and
        updates the jury with the exiled agent's information.

        Args:
            exiled_agent (Character): The character that has been exiled from the game.

        Returns:
            None
        """

        # Loop over all characters in the game
        for character in list(self.characters.values()):
            # Check if the current character is the one that was exiled
            if character == exiled_agent:
                # Add memory of the exile event for the exiled character, marking them for the jury
                self.add_exile_memory(
                    self.characters[character.name],
                    exiled_name=exiled_agent.name,
                    to_jury=True,
                )

                # Ensure the exiled character reflects on their actions and evaluates their goals one last time
                exiled_agent.engage(self)

                # Remove the exiled character from their current location
                character.location.remove_character(character)
                character.location = None  # Set the character's location to None

                # Remove the exiled character from the list of active characters
                _ = self.characters.pop(character.name)

            else:
                # For other characters, add memory of the exile event without marking them for the jury
                self.add_exile_memory(
                    self.characters[character.name],
                    exiled_name=exiled_agent.name,
                    to_jury=False,
                )

        # Loop over all characters in the jury
        for character in list(self.jury.values()):
            # Create a description of the exile event for the jury members
            description = f"{exiled_agent.name} was exiled and joins you on the jury to help decide the eventual game winner."

            # Extract keywords from the description for memory tracking
            desc_kwds = self.parser.extract_keywords(description)

            # Add the exile event memory to each jury member's memory
            character.memory.add_memory(
                self.round,
                tick=self.tick,
                description=description,
                keywords=desc_kwds,
                location=None,
                success_status=True,
                memory_importance=10,
                memory_type=MemoryType.ACTION.value,
                actor_id=character.id,
            )

    def add_exiled_to_jury(self, exiled):
        """
        Adds an exiled character to the jury for the game. This method updates the jury list by including the exiled
        character, allowing them to participate in the final voting process.

        Args:
            exiled (Character): The character that has been exiled and is to be added to the jury.

        Returns:
            None
        """

        # exile_key = f"{exiled.name}_{exiled.id}".replace(" ", "")
        self.jury.update({exiled.name: exiled})

    def add_exile_memory(self, character, exiled_name: str, to_jury: bool = False):
        """
        Records the memory of a character regarding the exile event. This method constructs a description based on
        whether the character was exiled or survived the vote, and
        updates the character's memory with relevant information.

        Args:
            character (Character): The character whose memory is being updated.
            exiled_name (str): The name of the character that has been exiled.
            to_jury (bool, optional): Indicates whether the character is being added to the jury; defaults to False.

        Returns:
            None
        """

        # Retrieve the vote count for the specified character from the voting session tally
        vote_count = self.vote_session.tally.get(character.name)

        # Get the total number of votes cast in the voting session
        vote_total = self.vote_session.tally.total()

        # Check if the character is being added to the jury
        if to_jury:
            # Create a description for the character being exiled and added to the jury
            description = "".join(
                [
                    f"{character.name} was exiled with {vote_count} votes of {vote_total}. ",
                    f"{character.name} will be added to a jury and will be able to cast a vote ",
                    "at the end of the game to determine the overall winner.",
                ]
            )

        else:
            # Create a description for the character who survived the vote
            description = "".join(
                [
                    f"{character.name} survived the vote. {character.name} received ",
                    f"{vote_count} out of {vote_total} votes. ",
                    f"{exiled_name} was exiled from the game but now sits on the final jury, ",
                    "where they will be allowed to cast a vote to help determine the game winner.",
                ]
            )

        # Extract keywords from the description for memory tracking
        desc_kwds = self.parser.extract_keywords(description)

        # Add the memory of the voting outcome to the character's memory
        character.memory.add_memory(
            self.round,
            tick=self.tick,
            description=description,
            keywords=desc_kwds,
            location=None,
            success_status=True,
            memory_importance=10,
            memory_type=MemoryType.ACTION.value,
            actor_id=character.id,
        )

    def run_jury_session(self):
        """
        Conducts a jury session to determine the winner of the game among the finalists. This method initializes a
        voting session with the jury members, processes the votes, updates the voting history, and logs the winner while
        also storing the winner's memory.

        Args:
            None

        Returns:
            None
        """

        # Create a list of all characters in the game to serve as finalists
        finalists = list(self.characters.values())

        # Initialize a new jury voting session with the current game instance, jury members, and finalists
        self.final_vote = JuryVotingSession(
            game=self, jury_members=list(self.jury.values()), finalists=finalists
        )

        # Run the jury voting session to allow jury members to cast their votes
        self.final_vote.run()

        # Determine the winner based on the results of the final vote
        winner = self.final_vote.determine_winner()

        # Update the voting history with the results from the final voting session
        self.update_voting_history(session=self.final_vote)

        # Store the winner of the game
        self.winner = winner

        # Set the flag to indicate that a winner has been declared
        self.winner_declared = True

        # Log the details of the finalists, including the winner
        self._log_finalists(winner=winner)

        # Add the winner's memory to all characters for future reference
        self._add_winner_memory()

    def _log_finalists(self, winner):
        """
        Logs the results of the finalists in the game, indicating the winner and finishing positions. This method
        iterates through all characters, recording a message for each character that reflects whether they won or lost
        the game.

        Args:
            winner (Character): The character who has won the game.

        Returns:
            None
        """

        # Iterate through all characters in the game
        for char in self.characters.values():
            # Check if the current character is the winner
            if char == winner:
                # Create a message indicating that the character won the game and their position
                message = f"{char.name} won the game. Position: 1"
            else:
                # TODO: Implement logic to rank non-winners based on their votes received in the future
                # Create a message indicating that the character lost the game and their position
                message = f"{char.name} lost the game. Position: 2"

            # Log the vote result for the character, including the message about their outcome
            self.vote_session.log_vote(char, message=message)

    def _add_winner_memory(self):
        """
        Records the memory of the winner for all characters in the game. This method constructs a description of the
        winning event, including the vote count, and updates the memory of each character with this information for
        future reference.

        Args:
            None

        Returns:
            None
        """

        # Retrieve the vote count for the winner from the final voting session tally
        vote_count = self.final_vote.tally.get(self.winner.name)

        # Get the total number of votes cast in the final voting session
        vote_total = self.final_vote.tally.total()

        # Create a description for the winner's memory using a predefined format
        description = vote_prompt.winner_memory_description.format(
            winner=self.winner.name, for_votes=vote_count, total_votes=vote_total
        )

        # Extract keywords from the description for memory tracking
        winner_kwds = self.parser.extract_keywords(description)

        # Combine all characters and jury members into a single list for memory updates
        everyone = list(self.characters.values()) + list(self.jury.values())

        # Iterate through each character and jury member to update their memory
        for c in everyone:
            c.memory.add_memory(
                round=self.round,  # Current round of the game
                tick=self.tick,  # Current tick of the game
                description=description,  # Description of the winner's memory
                keywords=winner_kwds,  # Keywords extracted from the description
                location=None,  # No specific location associated with this memory
                success_status=True,  # Indicate that the memory addition was successful
                memory_importance=10,  # Set the importance level of the memory
                memory_type=MemoryType.ACTION.value,  # Specify the type of memory
                actor_id=c.id,
            )  # ID of the character or jury member

    def _log_starting_locs(self):
        """
        Logs the starting locations of all characters in the game. This method iterates through each character,
        constructs a log message indicating their starting point, and records it using the logger for debugging
        purposes.

        Args:
            None

        Returns:
            None
        """

        # Iterate through all characters in the game
        for c in self.characters.values():
            # Retrieve logging extras for the current character
            extras = get_logger_extras(self, c)

            # Set the type of log entry to "Origin" to indicate the starting point of the character
            extras["type"] = "Origin"

            # Create a message indicating the starting location of the character
            message = f"Starting point: {c.location.name}"

            # Log the message at the debug level, including the extras for context
            self.logger.debug(msg=message, extra=extras)

    def save_simulation_data(self):
        """
        Saves the current simulation data, including voting history, character goals, and goal scores, to JSON files.
        This method organizes the data into specific directories based on the experiment name and ID, ensuring that all
        relevant information is stored for later analysis.

        Args:
            None

        Returns:
            None
        """

        # Get the output path for saving log files
        output_path = get_output_logs_path()

        # Create a directory path for the current experiment logs, including experiment name and ID
        experiment_dir = f"logs/{self.experiment_name}-{self.experiment_id}/"

        # Construct the file path for saving voting history in JSON format
        fp = os.path.join(
            output_path,
            experiment_dir,
            f"voting_history_{self.experiment_name}-{self.experiment_id}.json",
        )

        # Create necessary directories for the file path
        create_dirs(fp)

        # Save the voting history to the specified JSON file
        with open(fp, mode="w") as f:
            json.dump(
                self.voting_history, f, indent=4
            )  # Write the voting history with indentation for readability

        # Construct the file path for saving character goals in JSON format
        fp = os.path.join(
            output_path,
            experiment_dir,
            f"character_goals_{self.experiment_name}-{self.experiment_id}.json",
        )

        # Create necessary directories for the file path
        create_dirs(fp)

        # Save the goals of each character to the specified JSON file
        with open(fp, mode="w") as f:
            # Create a dictionary of character names and their goals, defaulting to "None" if no goals exist
            output = {
                name: c.get_goals() or "None" for name, c in self.characters.items()
            }

            # Add goals for jury members to the output dictionary
            for name, c in self.jury.items():
                output[name] = c.get_goals() or "None"

            # Write the character goals to the JSON file
            json.dump(output, f, indent=4)

        # Construct the file path for saving character goal scores in JSON format
        fp = os.path.join(
            output_path,
            experiment_dir,
            f"character_goal_scores_{self.experiment_name}-{self.experiment_id}.json",
        )

        # Create necessary directories for the file path
        create_dirs(fp)

        # Save the goal scores of each character to the specified JSON file
        with open(fp, mode="w") as f:
            # Create a dictionary of character names and their goal scores, defaulting to "None" if no scores exist
            output = {
                name: c.get_goal_scores() or "None"
                for name, c in self.characters.items()
            }

            # Add goal scores for jury members to the output dictionary
            for name, c in self.jury.items():
                output[name] = c.get_goal_scores() or "None"

            # Write the character goal scores to the JSON file
            json.dump(output, f, indent=4)


class BoardroomGame(Game):
    """
    Represents a game setup for the BoardroomGame.

    This class initializes a game with specified parameters, allowing for customization of player characters, actions,
    and game settings.

    Args:
        player (things.Character): The main character controlled by the player.
    """

    def __init__(
        self,
        start_at: Location,
        player: Character,
        characters=None,
        custom_actions=None,
        max_ticks: int = 1,
        # num_finalists: int = 2,
        experiment_name: str = "exp1",
        experiment_id: int = 1,
        end_state_check: Literal["on_round", "on_tick", "on_action"] = "on_round",
    ):
        """
        Initializes a SurvivorGame instance with a starting location, a player character, and optional non-player
        characters and custom actions. This constructor method sets up the game environment, including game tracking
        variables, logging, and the initial state of the game.

        Args:
            start_at (Location): The starting location of the player in the game.
            player (Character): The player character controlled by the user.
            characters (list, optional): A list of additional characters (NPCs) to include in the game.
            custom_actions (list, optional): A list of custom actions to be added to the game's parser.
            max_ticks (int, optional): The maximum number of ticks per round, defaulting to 10.
            num_finalists (int, optional): The number of finalists in the game, defaulting to 2.
            experiment_name (str, optional): The name of the experiment, defaulting to "exp1".
            experiment_id (int, optional): The ID of the experiment, defaulting to 1.
            end_state_check (Literal, optional): The condition for checking the end state, defaulting to "on_round".

        Returns:
            None
        """

        # Call the initializer of the parent SurvivorGame class with the starting location, player, characters, and custom
        # actions
        super().__init__(start_at, player, characters, custom_actions)

        # Initialize a custom logger ("gen_agents_global_logger") for the game with the experiment name and simulation
        # ID, which determine the saved file locations.
        game_logger = logger.CustomLogger(
            name="gen_agents_global_logger",
            experiment_name=experiment_name,
            simulation_id=experiment_id,
            logfile_prefix="sim",
            overwrite=True,
        )
        gpt_calls_logger = logger.CustomLogger(
            name="gpt_calls_logger",
            experiment_name=game_logger.get_experiment_name(),
            simulation_id=game_logger.get_simulation_id,
            logfile_prefix="gpt_calls",
            overwrite=False,
        )
        self.logger = game_logger.get_logger()
        self.gpt_calls_logger = gpt_calls_logger.get_logger()
        self.experiment_name = experiment_name  # Store the experiment name
        self.experiment_id = game_logger.get_simulation_id()  # Store the simulation ID

        # Store the original player ID for reference
        self.original_player_id = self.player.id

        # Game related tracking variables
        self.max_ticks_per_round = max_ticks
        self.round = 0
        self.tick = 0
        self.total_ticks = 0
        self.num_contestants = len(self.characters)
        self.end_state_check = end_state_check

        # Store end state variables:
        # Initialize a jury to hold exiled players who will cast the final vote
        self.jury = {}
        self.voting_history = defaultdict(lambda: defaultdict(list))
        self.num_finalists = num_finalists
        self.winner_declared = False

        # Log the starting locations of the characters for tracking purposes
        self._log_starting_locs()
