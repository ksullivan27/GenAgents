import json
import inspect
from collections import defaultdict, namedtuple
import os
from typing import TYPE_CHECKING, Literal
from numpy.random import permutation
import dill as pickle

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
    be at a Location or in the player's inventory.  Each locations has a set of
    exits which are the directions that a player can move to get to an
    adjacent location. The player can move from one location to another
    location by typing a command like "Go North".
    """

    def __init__(
        self,
        start_at: Location,
        player: Character,
        characters=None,
        custom_actions=None
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
            Recursively builds a mapping of locations starting from a given location. This function populates an accumulator dictionary with location names as keys and their corresponding location objects as values, traversing through all connected locations.

            Args:
                location (Location): The starting location from which to build the mapping.
                acc (dict): The accumulator dictionary that stores the mapping of location names to location objects.

            Returns:
                dict: The updated accumulator dictionary containing the mapped locations.
            """

            acc[location.name] = location
            for _, connection in location.connections.items():
                if connection.name not in acc:
                    acc = location_map(connection, acc)
            return acc

        self.locations = location_map(self.start_at, {})

        # Parser
        self.parser = parsing.Parser(self)

        # Add custom actions to parser
        if custom_actions:
            print("Adding custom actions")
            for ca in custom_actions:
                if inspect.isclass(ca) and issubclass(ca, actions.Action):
                    self.parser.add_action(ca)
                else:
                    err_msg = f"ERROR: invalid custom action ({ca})"
                    raise Exception(err_msg)

        # Visit each location and add any blocks found to parser
        seen_before = {}
        for name, location in self.locations.items():
            if len(location.blocks) > 0 and name not in seen_before:
                for b in location.blocks:
                    self.parser.add_block(b)
                    seen_before[name] = True

    def set_parser(self, parser: parsing.Parser):
        """
        Sets the parser for the game instance, allowing for command parsing and handling. This method updates the parser attribute with the provided parser instance, enabling the game to process user input effectively.

        Args:
            parser (parsing.Parser): The parser instance to be assigned to the game.

        Returns:
            None
        """

        self.parser = parser

    def game_loop(self):
        """
        A simple loop that starts the game, loops over commands from the user,
        and then stops if the game's state says the game is over.
        """
        self.parser.parse_command("look")

        while True:
            command = input("\n> ")
            self.parser.parse_command(command)
            if self.is_game_over():
                break

    def is_won(self) -> bool:
        """
        A conditional check intended for subclasses to use for defining the
        game's winning conditions.
        """
        return False

    def is_game_over(self) -> bool:
        """
        A conditional check that determines if the game is over. By default it
        checks if the player has died or won.
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
        Puts characters in the game
        """
        self.characters[character.name] = character

    def describe(self) -> str:
        """
        Describe the current game state by first describing the current
        location, then listing any exits, and then describing any objects
        in the current location.
        """
        description = self.describe_current_location() + "\n"
        description += self.describe_exits() + "\n"
        description += self.describe_items() + "\n"
        description += self.describe_characters() + "\n"
        description += self.describe_inventory() 
        # print(f"total description: {description}")
        return description

    def describe_current_location(self) -> str:
        """
        Describe the current location by printing its description field.
        """
        loc_description = f"location: {self.player.name} is at {self.player.location.description}"
        # print(f"location description: {loc_description}")
        return loc_description

    def describe_exits(self) -> str:
        """
        List the directions that the player can take to exit from the current
        location.
        """
        exits = []
        for direction in self.player.location.connections.keys():
            location = self.player.location.connections[direction]
            exits.append(direction.capitalize() + " to " + location.name)
        description = "exits: "
        if len(exits) > 0:
            description += f"From {self.player.location.name} {self.player.name} could go: "
            for exit in exits:
                description += exit + ", "
        # print(f"Exit description: {description}")
        return description

    def describe_items(self) -> str:
        """
        Describe what items are in the current location.
        """
        description = "items: "
        if len(self.player.location.items) > 0:
            description += f"{self.player.name} sees:"
            for item_name in self.player.location.items:
                item = self.player.location.items[item_name]
                description += item.description 
                if self.give_hints:
                    special_commands = item.get_command_hints()
                    if special_commands:
                        description += "(hint "
                        for cmd in special_commands:
                            description += cmd + ", "
                        description += ")"
                    description += "; "
        return description

    def describe_characters(self) -> str:
        """
        Describe what characters are in the current location.
        """
        description = "characters: "
        if len(self.player.location.characters) > 1:
            description += f"{self.player.name} sees characters: "
            for character_name in self.player.location.characters:
                if character_name == self.player.name:
                    continue
                character = self.player.location.characters[character_name]
                # TODO: may want to change this to just the character name for ease of parsing later
                description += character.name + ", "
        return description

    def describe_inventory(self) -> str:
        """
        Describes the player's inventory.
        """
        inventory_description = "inventory: "
        if len(self.player.inventory) == 0:
            inventory_description += f"{self.player.name} has nothing in inventory."
            # self.ok(empty_inventory, [], "Describe the player's inventory.")
        else:
            # descriptions = []  # JD logical issue?
            inventory_description += f"In {self.player.name} inventory, {self.player.name} has: "
            for item_name in self.player.inventory:
                item = self.player.inventory[item_name]
                d = "{item_description}, "
                inventory_description += d.format(
                    # item=item_name, 
                    item_description=item.description
                )
        return inventory_description

    # The methods below read and write a game to JSON
    def to_primitive(self):
        """
        Serialize a game to json.
        """
        data = {
            "player": self.player.name,
            "start_at": self.start_at.name,
            "game_history": self.game_history,  # TODO this is empty?
            "game_over": self.game_over,
            "game_over_description": self.game_over_description,
            "characters": [c.to_primitive() for c in self.characters.values()],
            "locations": [l.to_primitive() for l in self.locations.values()],
            "actions": sorted([a for a in self.parser.actions]),
        }
        return data

    @classmethod
    def default_actions(self):
        """
        Generates a dictionary of all actions packaged as part of this library
        """
        actions_found = {}
        for member in dir(actions):
            attr = getattr(actions, member)
            if inspect.isclass(attr) and issubclass(attr, actions.Action):
                # dont include base class
                if not attr == actions.Action:
                    actions_found[attr.action_name()] = attr
        return actions_found

    @classmethod
    def default_blocks(self):
        """
        Generates as dictionary of all blocks packaged as part of this library
        """
        blocks_found = {}
        for member in dir(blocks):
            attr = getattr(blocks, member)
            if inspect.isclass(attr) and issubclass(attr, blocks.Block):
                # dont include base class
                if not attr == blocks.Block:
                    # if this changes, also adjust _type in blocks.Block
                    blocks_found[attr.__name__] = attr
        return blocks_found

    @classmethod
    def from_primitive(cls, data, custom_actions=None, custom_blocks=None):
        """
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
        """
        SkeletonContext = namedtuple(
            "SkeletonContext", ["characters", "locations", "items"]
        )

        # FIRST PASS

        characters = {
            c["name"]: Character.from_primitive(c) for c in data["characters"]
        }
        locations = {l["name"]: Location.from_primitive(l) for l in data["locations"]}
        items = {}
        context = SkeletonContext(characters, locations, items)

        # SECOND PASS

        # Characters
        for c in context.characters.values():
            # locations
            l = context.locations[c.location]
            c.location = l
            # inventory
            for item_name, item in c.inventory.items():
                if hasattr(item, "location") and item.location:
                    l_obj = context.locations[item.location]
                    item.location = l_obj
                elif hasattr(item, "owner") and item.owner:
                    c_obj = context.characters[item.owner]
                    item.owner = c_obj
                context.items[item_name] = item

        # Locations
        for l in context.locations.values():
            # characters
            for char_name, c in l.characters.items():
                c_obj = context.characters[char_name]
                l.characters[char_name] = c_obj
            # connections
            for dir_name, connection in l.connections.items():
                c_obj = context.locations[connection]
                l.connections[dir_name] = c_obj
            # items
            for item_name, item in l.items.items():
                if hasattr(item, "location") and item.location:
                    l_obj = context.locations[item.location]
                    item.location = l_obj
                elif hasattr(item, "owner") and item.owner:
                    c_obj = context.characters[item.owner]
                    item.owner = c_obj
                context.items[item_name] = item

        # THIRD PASS

        # Actions
        action_map = cls.default_actions()

        # Validate custom actions
        if custom_actions:
            for ca in custom_actions:
                if inspect.isclass(ca) and issubclass(ca, actions.Action):
                    action_map[ca.action_name()] = ca
                else:
                    err_msg = f"ERROR: invalid custom action ({ca})"
                    raise Exception(err_msg)

        # verify all commands from primitive data have an associated action
        action_names = list(action_map.keys())
        for action_name in data["actions"]:
            if action_name not in action_names:
                err_msg = "".join(
                    [
                        f"ERROR: unmapped action ({action_name}) found in ",
                        "primitive data",
                    ]
                )
                raise Exception(err_msg)

        # Blocks
        block_map = cls.default_blocks()

        # Validate custom blocks
        if custom_blocks:
            for cb in custom_blocks:
                if inspect.isclass(cb) and issubclass(cb, blocks.Block):
                    block_map[cb.__name__] = cb
                else:
                    err_msg = f"ERROR: invalid custom block ({cb})"
                    raise Exception(err_msg)

        # Instantiate all blocks for all locations
        for l in context.locations.values():
            for direction, block_data in l.blocks.items():
                # it is possible for two locations to have the same block, so
                # skip any that have already been instantiated
                if isinstance(block_data, blocks.Block):
                    continue
                cls_type = block_map[block_data["_type"]]
                del block_data["_type"]
                # we will copy the properties of relevant items before we
                # install the block, so we can restore them after
                prop_map = {}
                # replace thing names in primitive with thing instances
                for param_name, param in block_data.items():
                    if param in context.items:
                        param_instance = context.items[param]
                    elif param in context.locations:
                        param_instance = context.locations[param]
                    block_data[param_name] = param_instance
                    prop_map[param_name] = param_instance.properties.copy()
                instance = cls_type.from_primitive(block_data)
                # restore properties found in primitive data
                for param_name, param in block_data.items():
                    param.properties = prop_map[param_name]

        start_at = context.locations[data["start_at"]]
        player = context.characters[data["player"]]

        instance = cls(start_at, player, custom_actions=action_map.values())
        instance.game_history = data["game_history"]
        instance.game_over = data["game_over"]
        instance.game_over_description = data["game_over_description"]

        return instance

    def to_json(self):
        """
        Creates a JSON version of a game's primitive data.
        """
        data = self.to_primitive()
        data_json = json.dumps(data)
        return data_json

    @classmethod
    def from_json(cls, data_json, **kw):
        """
        Goes from JSON into actual game instances.
        """
        data = json.loads(data_json)
        instance = cls.from_primitive(data, **kw)
        return instance

    def save_game(self, filename):
        """
        Converts a game's state to JSON and then saves it to a file
        """
        save_data = self.to_json()
        with open(filename, 'w') as f:
            f.write(save_data)

    @classmethod
    def load_game(cls, filename, **kw):
        """
        Reads a file with a game's state stored as JSON and converts it to a
        game instance.
        """
        with open(filename, 'r') as f:
            save_data = f.read()
            return cls.from_json(save_data, **kw)


# Override methods or implement a new class?
class SurvivorGame(Game):
    """
    Represents a Survivor game that extends the base game functionality with specific mechanics for a competitive
    environment. This class manages the game state, player interactions, voting sessions, and the overall flow of the
    game, including tracking contestants and determining a winner.

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

    def __init__(self, 
                 start_at: Location, 
                 player: Character, 
                 characters=None, 
                 custom_actions=None, 
                 max_ticks: int = 10, 
                 num_finalists: int = 2,
                 experiment_name: str = "exp1",
                 experiment_id: int = 1,
                 end_state_check: Literal["on_round", "on_tick", "on_action"] = "on_round"):
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

        super().__init__(start_at, player, characters, custom_actions)
        game_logger = logger.CustomLogger(experiment_name=experiment_name, sim_id=experiment_id)
        self.logger = game_logger.get_logger()
        self.experiment_name = experiment_name
        self.experiment_id = game_logger.get_simulation_id()
        
        self.original_player_id = self.player.id
        
        # Game related tracking variables
        self.max_ticks_per_round = max_ticks
        self.round = 0
        self.tick = 0
        self.total_ticks = 0
        self.num_contestants = len(self.characters)
        self.end_state_check = end_state_check
        
        # Store end state variables: 
        # Exiled players in jury cast the final vote
        self.jury = {}
        self.voting_history = defaultdict(lambda: defaultdict(list))
        self.num_finalists = num_finalists
        self.winner_declared = False

        # Log the starting loctions of the characters
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

        params = {"contestant_count": len(self.characters),
                  "contestant_names_locs": ", ".join([f"{c.name} who is at {c.location.name}" 
                                                      for c in self.characters.values() 
                                                      if c.id != self.player.id]),
                  "n_finalists": self.num_finalists,
                  "rounds_until_finals": len(self.characters) - self.num_finalists,
                  "turns_left_this_round": self.max_ticks_per_round - (self.tick - 1)}
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

        while True:
            for tick in range(self.max_ticks_per_round):
                self.tick = tick

                # Confirming Round increments
                print(f"ROUND: {self.round}.{self.tick}")
                
                # If this is the end of the round, vote
                # if self.tick == (self.max_ticks_per_round - 1):
                #     self.handle_voting_sessions()
                
                # Set goals for all characters at beginning of round
                self.goal_setting_handler()

                self.reset_character_dialogue()

                for character in permutation(list(self.characters.values())):  # random permuted ordering, not based on character initiative
                    print(f"It is: {character.name}'s turn")
                    self.turn_handler(character)

                    # EXPLORATION: check if game ended
                    if self.end_state_check == "on_action" and self.is_game_over():
                        return 

                # Update the total ticks that have occurred in the game.
                self.total_ticks += 1

                # EXPLORATION: check if game ended
                if self.end_state_check == "on_tick" and self.is_game_over():
                    break
            
            # NOTE: this placement allows agents to reflect prior to voting.
            
            if self.end_state_check == "on_round" and self.is_game_over():
                break

            # Increment the rounds
            self.round += 1

            # save game results so far
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
            for character in self.characters.values():
                # Update the world info with new tick, contestant counts, and non-player contestant names
                self.update_world_info()
                character.generate_goals(self)

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

        if self.game_over:
            return True
        return self.is_won()
    
    def is_won(self):
        """
        Checks whether the game has been won. For SurvivorWorld, the game is won
        once any has been voted the victor.
        """
        if self.winner_declared:
            print(f"Congratulations!! {self.winner.name} won the game! They're the ultimate Survivor. Jeff is so proud of u")
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

        if isinstance(command, int):
            if command == -999:
                return False
        elif isinstance(command, str):
            return True
        else:
            raise ValueError(f"command: {command} must be str or int; got {type(command)}")

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

        if len(self.characters) == self.num_finalists:
            # This should trigger the end of game
            self.run_jury_session()
        # If we've reached the end of a round, run a voting session to exile someone.
        elif self.tick == (self.max_ticks_per_round - 1):
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

        self.vote_session = VotingSession(game=self, 
                                          participants=self.characters.values())
        self.vote_session.run()
        exiled = self.vote_session.read_votes()
        self.update_voting_history(session=self.vote_session)
        self.update_exile_state(exiled)
        self.add_exiled_to_jury(exiled)
        self._log_exiled_player(exiled)
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

        contestants_remaining = len(self.characters)
        message = f"{exiled.name} was exiled. Position: {contestants_remaining + 1}"
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

        extras = get_logger_extras(self, character=None)
        extras["type"] = "Calls"
        message = f"Current GPT calls count: {GptCallHandler.get_calls_count()}"
        self.logger.debug(msg=message, extra=extras)

        extras["type"] = "Tokens"
        message = f"Current GPT tokens count: {GptCallHandler.get_tokens_processed()}"
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

        extras = get_logger_extras(self, character)
        extras["type"] = "Act"
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

        # Loop over them
        for character in list(self.characters.values()):
            # Pass appropriate memories to each agent
            if character == exiled_agent:
                self.add_exile_memory(self.characters[character.name],
                                      exiled_name=exiled_agent.name, 
                                      to_jury=True)
                # Make sure they do one final reflection and goal evaluation
                exiled_agent.engage(self)
                # remove the agent that was exiled
                character.location.remove_character(character)
                character.location = None
                _ = self.characters.pop(character.name)

            else:
                self.add_exile_memory(self.characters[character.name],
                                      exiled_name=exiled_agent.name,
                                      to_jury=False)
        
        for character in list(self.jury.values()):
            description = f"{exiled_agent.name} was exiled and joins you on the jury to help decide the eventual game winner."
            desc_kwds = self.parser.extract_keywords(description)
            character.memory.add_memory(self.round,
                                        tick=self.tick, 
                                        description=description, 
                                        keywords=desc_kwds, 
                                        location=None, 
                                        success_status=True,
                                        memory_importance=10, 
                                        memory_type=MemoryType.ACTION.value,
                                        actor_id=character.id)
        
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
        Records the memory of a character regarding the exile event, detailing the voting outcome and the character's
        status. This method constructs a description based on whether the character was exiled or survived the vote, and
        updates the character's memory with relevant information.

        Args:
            character (Character): The character whose memory is being updated.
            exiled_name (str): The name of the character that has been exiled.
            to_jury (bool, optional): Indicates whether the character is being added to the jury; defaults to False.

        Returns:
            None
        """

        vote_count = self.vote_session.tally.get(character.name)
        vote_total = self.vote_session.tally.total()
        if to_jury:
            description = "".join([
                f"{character.name} was exiled with {vote_count} votes of {vote_total}. ",
                f"{character.name} will be added to a jury and will be able to cast a vote ",
                "at the end of the game to determine the overall winner."
            ])
            
        else:
            description = "".join([
                f"{character.name} survived the vote. {character.name} recieved ",
                f"{vote_count} out of {vote_total} votes. ",
                f"{exiled_name} was exiled from the game but now sits on the final jury. ",
                "They will be allowed to cast a vote to help determine the game winner."
            ])

        desc_kwds = self.parser.extract_keywords(description)
        character.memory.add_memory(self.round,
                                    tick=self.tick, 
                                    description=description, 
                                    keywords=desc_kwds, 
                                    location=None, 
                                    success_status=True,
                                    memory_importance=10, 
                                    memory_type=MemoryType.ACTION.value,
                                    actor_id=character.id)
        
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

        finalists = list(self.characters.values())
        self.final_vote = JuryVotingSession(game=self,
                                            jury_members=list(self.jury.values()), 
                                            finalists=finalists)
        self.final_vote.run()
        winner = self.final_vote.determine_winner()
        self.update_voting_history(session=self.final_vote)
        self.winner = winner
        self.winner_declared = True
        self._log_finalists(winner=winner)
        self._add_winner_memory()

    def _log_finalists(self, winner):
        """
        Logs the results of the finalists in the game, indicating the winner and their position. This method iterates
        through all characters, recording a message for each character that reflects whether they won or lost the game.

        Args:
            winner (Character): The character who has won the game.

        Returns:
            None
        """

        for char in self.characters.values():
            if char == winner:
                message = f"{char.name} won the game. Position: 1"
            else:
                # TODO: should eventually make this rank non-winners based on their votes received
                message = f"{char.name} lost the game. Position: 2"
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

        vote_count = self.final_vote.tally.get(self.winner.name)
        vote_total = self.final_vote.tally.total()
        description = vote_prompt.winner_memory_description.format(winner=self.winner.name,
                                                                   for_votes=vote_count,
                                                                   total_votes=vote_total)
        winner_kwds = self.parser.extract_keywords(description)

        # Pass this memory to all characters
        everyone = list(self.characters.values()) + list(self.jury.values())
        for c in everyone:
            c.memory.add_memory(round=self.round,
                                tick=self.tick, 
                                description=description, 
                                keywords=winner_kwds, 
                                location=None, 
                                success_status=True,
                                memory_importance=10, 
                                memory_type=MemoryType.ACTION.value,
                                actor_id=c.id)
            
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

        for c in self.characters.values():
            extras = get_logger_extras(self, c)
            extras["type"] = "Origin"
            message = f"Starting point: {c.location.name}"
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

        output_path = get_output_logs_path()
        experiment_dir = f"logs/{self.experiment_name}-{self.experiment_id}/"
        fp = os.path.join(output_path, experiment_dir, f"voting_history_{self.experiment_name}-{self.experiment_id}.json")
        create_dirs(fp)

        # Save voting history
        with open(fp, mode="w") as f:
            json.dump(self.voting_history, f, indent=4)

        # Save goal scores and goals
        fp = os.path.join(output_path, experiment_dir, f"character_goals_{self.experiment_name}-{self.experiment_id}.json")
        create_dirs(fp)
        
        with open(fp, mode="w") as f:
            output = {}
            for name, c in self.characters.items():
                output[name] = c.get_goals() or "None"

            for name, c in self.jury.items():
                output[name] = c.get_goals() or "None"
            json.dump(output, f, indent=4)

        fp = os.path.join(output_path, experiment_dir, f"character_goal_scores_{self.experiment_name}-{self.experiment_id}.json")
        create_dirs(fp)

        with open(fp, mode="w") as f:
            output = {}
            for name, c in self.characters.items():
                output[name] = c.get_goal_scores() or "None"

            for name, c in self.jury.items():
                output[name] = c.get_goal_scores() or "None"
            json.dump(output, f, indent=4)
