# local imports
from text_adventure_games.agent.memory_stream import MemoryType
from text_adventure_games.things.characters import Character
from . import base
from ..things import Item
from text_adventure_games.utils.general import enumerate_dict_options, get_logger_extras
import random


class Search_Idol(base.Action):
    """
    Represents an action to search for an idol in the game. This action typically requires a tool to be successful and
    involves checking various conditions before attempting to find the idol.

    Args:
        game: The game instance in which the action is being performed.
        command (str): The command issued by the player to search for the idol.
        character (Character): The character performing the search action.

    Attributes:
        valid_idol_locations (list): Locations where an idol can be found.
        command (str): The command issued by the player.
        character (Character): The character performing the action.
        location: The current location of the character.
        tool_required: Indicates if a tool is required to search.
        tool: The tool being used for the search, if any.
        clue: A clue item that may assist in the search.

    Methods:
        check_preconditions() -> bool:
            Checks if the preconditions for searching for an idol are met, including location, tool availability, and
            specific conditions for certain locations.

        apply_effects():
            Attempts to find an idol based on randomized success and updates the character's inventory and score if
            successful.
    """

    # The name of the action that players can perform to search for an idol.
    ACTION_NAME = "search idol"

    # A brief description of the action, indicating that it involves looking for an idol and typically requires a tool
    # for success.
    ACTION_DESCRIPTION = (
        "Look for an idol. Typically requires a tool in order to be successful."
    )

    # Alternative names or phrases that can be used to invoke the search idol action.
    ACTION_ALIASES = ["look for idol", "search for idol", "find idol"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes the Search_Idol action with the game context, command, and character. This constructor sets up the
        necessary properties for the action, including valid idol locations, the required tool, and any clues available
        to the character.

        Args:
            game: The game instance in which the action is being performed.
            command (str): The command issued by the player to search for the idol.
            character (Character): The character performing the search action.

        Attributes:
            valid_idol_locations (list): Locations where an idol can be found.
            command (str): The command issued by the player.
            character (Character): The character performing the action.
            location: The current location of the character.
            tool_required: Indicates if a tool is required to search.
            tool: The tool being used for the search, if any.
            clue: A clue item that may assist in the search.
        """

        # Call the initializer of the parent class to set up the game context.
        super().__init__(game)

        # Create a list of valid locations where an idol can be found based on the game's locations.
        self.valid_idol_locations = [
            loc for loc in game.locations.values() if loc.get_property("has_idol")
        ]

        # Store the command issued by the player for searching the idol.
        self.command = command

        # Assign the character performing the action to an instance variable.
        self.character = character

        # Get the current location of the character.
        self.location = self.character.location

        # Check if a tool is required for searching in the current location.
        self.tool_required = self.location.get_property("tool_required")

        # Initialize the tool variable to indicate no tool is selected initially.
        self.tool = False

        # Check if the command includes "machete" and attempt to match it with the character's items.
        if " machete" in command:
            self.tool = self.parser.match_item(
                "machete", self.parser.get_items_in_scope(self.character)
            )

        # Check if the command includes "stick" and attempt to match it with the character's items.
        if " stick" in command:
            self.tool = self.parser.match_item(
                "stick", self.parser.get_items_in_scope(self.character)
            )

        # Attempt to find an "idol clue" item in the character's scope for exploration purposes.
        self.clue = self.parser.match_item(
            "idol clue", self.parser.get_items_in_scope(self.character)
        )

    def _log_found_idol(self, message):
        """
        Logs a debug message when an idol is found during the search action. This method captures additional context
        about the game and character, allowing for better tracking of events related to idol discovery.

        Args:
            message (str): The message to be logged, detailing the discovery of the idol.

        Returns:
            None
        """

        # Retrieve additional logging context related to the current game and character.
        extras = get_logger_extras(self.game, self.character)

        # Set the type of log entry to "Idol" to categorize the log message.
        extras["type"] = "Idol"

        # Log the debug message along with the extra context for tracking the idol discovery event.
        self.game.logger.debug(msg=message, extra=extras)

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the character to successfully search for an idol. This method evaluates
        various factors such as the presence of an idol in the location, the character's current position, and the
        availability of required tools.

        Preconditions:
        * The character must be at the jungle
        * The character must have a machete in their inventory

        Returns:
            bool: True if all preconditions are met for a successful search, False otherwise. The method also handles
            logging and messaging for any failures encountered during the checks.
        """

        # Check if the current location has an idol; if not, prepare a failure message.
        if not self.location.get_property("has_idol"):
            description = f"You look around, but cannot find an idol in the {self.location.name}. "

            # If an idol has been found previously, append the found message to the description.
            if self.location.get_property("idol_found"):
                description += self.location.get_property("found_message")
            else:
                description += "This area seems unlikely to have one."

            # Log the failure reason and notify the parser of the failure.
            self.parser.fail(self.command, description, self.character)
            return False

        # Check if the character is in the correct location for the search.
        if not self.location.here(self.character):
            print("DEBUG: idol search failure due to character in wrong place")
            self.parser.fail(
                self.command,
                "You get the feeling there won't be an idol at this location",
                self.character,
            )
            return False

        # Check if a tool is required and if the character has one.
        if not self.tool and self.tool_required:
            print("DEBUG: idol search failure due to lack of proper tools")
            no_tool = f"{self.character.name} looks around the {self.location.name} "

            # If the location is valid for finding an idol, append the search failure message.
            if self.location.id in [loc.id for loc in self.valid_idol_locations]:
                no_tool += self.location.get_property("search_fail")
            else:
                no_tool += "and sense that there probably is no idol hidden here."

            # Notify the parser of the failure due to lack of tools.
            self.parser.fail(self.command, no_tool, self.character)
            return False

        # Special condition for the "rocky shore" location, where searches fail every other round.
        if self.location.name == "rocky shore":
            # Fail the search on even-numbered rounds.
            if self.game.round % 2 == 0:
                dangerous = "".join(
                    [
                        f"{self.character.name} looks out at the {self.location.name} ",
                        self.location.get_property("search_fail"),
                        f"{self.character.name} could not search the rocky shore on an even numbered round. ",
                        "What pattern does the failure message above suggest is the correct time to search for this idol?",
                    ]
                )
                print(
                    "DEBUG: idol search failure due to an even round at the rocky shore"
                )
                self.parser.fail(self.command, dangerous, self.character)

            return False

        # If all preconditions are met, return True to indicate the search can proceed.
        return True

    def apply_effects(self):
        """
        Applies the effects of the idol search action, determining whether the character successfully finds an idol. The
        outcome is influenced by a random number and the character's previous search attempts, with successful searches
        resulting in the addition of an idol to the character's inventory and updates to their score.

        Returns:
            bool: True if the search was unsuccessful, indicating that the character can attempt to search again;
            otherwise, returns True after successfully finding an idol and updating the game state.
        """

        """
        Effects:
        * Randomized success of finding an idol
        * If found, adds idol to player inventory
        * Player is immune until next round.
        """

        # Generate a random number to determine the success of the idol search.
        random_number = random.random()

        # Calculate a threshold padding based on the number of previous idol searches by the character.
        threshold_pad = self.character.get_idol_searches() * 0.1

        # Log the generated random number and the threshold padding for debugging purposes.
        print("Search random number: ", random_number)
        print("Searcher odds padding: ", threshold_pad)

        # Check if the random number indicates a successful search for the idol.
        if random_number < (0.7 + threshold_pad) or (
            random_number < (0.8 + threshold_pad) and self.clue
        ):
            # Create a new idol item and add it to the character's inventory.
            idol = Item(
                "idol", "an immunity idol", "THIS IDOL SCORES POINTS FOR YOUR TEAM!"
            )
            idol.add_command_hint("keep it a secret from your enemies!")

            # Update the character's inventory and set their immunity status.
            self.character.add_to_inventory(idol)
            self.character.set_property("immune", True)

            # Update the location properties to reflect that the idol has been found.
            self.location.set_property("has_idol", False)
            self.location.set_property("idol_found", True)

            # Calculate the value of the idol based on the game's total ticks.
            idol_value = 100 - self.game.total_ticks

            # Update the character's score with the value of the found idol.
            self.character.update_score(idol_value)

            # Log the successful search for debugging purposes.
            print(
                f"DEBUG: idol search successful! {self.character.name} found it at the {self.location.name}"
            )
            self._log_found_idol(
                message=f"Found idol at: {self.location.name}; worth points: {idol_value}"
            )
        else:
            # Prepare a failure message if the search was unsuccessful.
            description = "".join(
                [
                    "You look around for an idol but found nothing. It seems like this is the correct way to search. ",
                    "You sense it should be nearby and you can keep on trying! You might have better luck next time!",
                ]
            )

            # Notify the parser of the failure and increment the character's search attempts.
            self.parser.fail(self.command, description, self.character)
            self.character.increment_idol_search()
            return True

        # Create a description of the successful search for the idol.
        found = "".join(
            [
                "{character_name} searches around in the {location} and ",
                "finds an idol! They have scored {value} points for your team!",
            ]
        )
        description = found.format(
            character_name=self.character.name,
            location=self.location.name,
            value=idol_value or (100 - self.game.total_ticks),
        )

        # Extract keywords from the description for memory tracking.
        idol_kwds = self.parser.extract_keywords(description)

        # Add a memory entry for each character in the game regarding the idol discovery.
        for c in list(self.game.characters.values()):
            c.memory.add_memory(
                round=self.game.round,
                tick=self.game.tick,
                description=description,
                keywords=idol_kwds,
                location=self.character.location.name,
                success_status=True,
                memory_importance=10,
                memory_type=MemoryType.ACTION.value,
                actor_id=self.character.id,
            )

        # Return True to indicate that the effects have been applied successfully.
        return True


class Read_Clue(base.Action):
    """
    Represents an action to read a clue that provides details about the idol's location. This action allows the
    character to examine the clue and gain insights that may assist in their search for the idol.

    Args:
        game: The game instance in which the action is being performed.
        command (str): The command issued by the player to read the clue.
        character (Character): The character performing the action.

    Attributes:
        command (str): The command issued by the player.
        character (Character): The character performing the action.
        clue: The clue item that the character is attempting to read.

    Methods:
        check_preconditions() -> bool:
            Checks if the clue is available for the character to read.

        apply_effects():
            Processes the effects of reading the clue, updating the character's memory with the clue's content and
            logging the action.
    """

    # The name of the action that players can perform to read a clue about the idol's location.
    ACTION_NAME = "read clue"

    # A brief description of the action, indicating that it involves examining a clue for details on the idol's
    # whereabouts.
    ACTION_DESCRIPTION = "Examine the clue for details on the idol's location."

    # Alternative names or phrases that can be used to invoke the read clue action.
    ACTION_ALIASES = ["examine clue", "read clue", "read idol clue"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes the Read_Clue action with the game context, command, and character. This constructor sets up the
        necessary properties for the action, including the command issued by the player and the clue item that the
        character is attempting to read.

        Args:
            game: The game instance in which the action is being performed.
            command (str): The command issued by the player to read the clue.
            character (Character): The character performing the action.

        Attributes:
            command (str): The command issued by the player.
            character (Character): The character performing the action.
            clue: The clue item that the character is attempting to read.
        """

        # Call the initializer of the parent class to set up the game context.
        super().__init__(game)

        # Store the command issued by the player for reading the clue.
        self.command = command

        # Assign the character performing the action to an instance variable.
        self.character = character

        # Attempt to match the "idol clue" item from the character's current scope of items.
        self.clue = self.parser.match_item(
            "idol clue", self.parser.get_items_in_scope(self.character)
        )

    def _log_clue(self, game, character, message):
        """
        Logs a debug message related to the clue reading action. This method captures additional context about the game
        and character, allowing for better tracking of events associated with clue interactions.

        Args:
            game: The game instance in which the action is being performed.
            character: The character involved in the clue reading action.
            message (str): The message to be logged, detailing the clue reading event.

        Returns:
            None
        """

        # Retrieve additional logging context related to the current game and character.
        extras = get_logger_extras(game, character)

        # Set the type of log entry to "Clue" to categorize the log message appropriately.
        extras["type"] = "Clue"

        # Log the debug message along with the extra context for tracking the clue reading event.
        game.logger.debug(msg=message, extra=extras)

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the character to read the clue. This method verifies if the clue is
        available at the current location, ensuring that the character can proceed with the action.

        Preconditions:
        * The character must have the clue nearby

        Returns:
            bool: True if the clue is present and the character can read it; False otherwise, indicating that the action
            cannot proceed.
        """

        # Check if the clue is not present; if it is missing, notify the parser of the failure.
        if not self.clue:
            self.parser.fail(
                self.command, "There is no idol clue at this location", self.character
            )
            return False

        # The following line is commented out; it could be used for debugging to confirm the clue's details.
        # print("Clue was found: ", self.clue.name, ". Description: ", self.clue.description)

        # Return True to indicate that the preconditions for reading the clue are satisfied.
        return True

    def apply_effects(self):
        """
        Processes the effects of reading the idol clue, updating the character's memory with the clue's content. This
        method constructs a description of the action, logs the event, and allows the character to retain information
        about the clue for future reference.

        Effects:
        * Let agent know details about the idol

        Returns:
            bool: True, indicating that the effects of reading the clue have been successfully applied.
        """

        # Construct a detailed message about the character reading the idol clue, including the clue's content.
        d = "".join(
            [
                "{character_name} reads the idol clue to themself:\n",
                self.clue.get_property("clue_content"),
                "\nTo share this information with your teammate, you must talk to them.",
            ]
        )

        # Format the message with the character's name.
        description = d.format(character_name=self.character.name)

        # Summarize and score the action based on the description, character, and command.
        action_statement, action_importance, action_keywords = (
            self.parser.summarise_and_score_action(
                description=description,
                thing=self.character,
                command=self.command,
                needs_summary=False,
            )
        )

        # Add a memory entry for the character, capturing the details of the action performed.
        self.character.memory.add_memory(
            round=self.game.round,
            tick=self.game.tick,
            description=action_statement,
            keywords=action_keywords,
            location=self.character.location.name,
            success_status=True,
            memory_importance=action_importance,
            memory_type=MemoryType.ACTION.value,
            actor_id=self.character.id,
        )

        # The following line is commented out; it could be used to confirm the action was successful.
        # self.parser.ok(self.command, description, self.character)

        # Log the event of the character reading the clue for tracking purposes.
        self._log_clue(
            self.game, self.character, f"{self.character.name} read the clue."
        )

        # Return True to indicate that the effects have been successfully applied.
        return True
