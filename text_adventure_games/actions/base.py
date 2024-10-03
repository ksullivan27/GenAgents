from ..things import Thing, Character, Item, Location
import re


class Action:
    """
    Represents an action that can be performed in the game. This class provides methods to check preconditions and apply
    effects of actions within the game context.

    In the game, rather than allowing players to do anything, we have a specific set of Actions they can do. The Action
    class checks preconditions (the set of conditions that must be true in order for the action to occur), and applies
    the effects of the action by updating the state of the world. Different actions have different arguments, so we
    subclass Action to create new actions.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action.
        ACTION_ALIASES (list[str]): A list of aliases for the action.

    Args:
        game: The game instance that this action is associated with.

    Methods:
        check_preconditions: Validates the state for applying the action.
        apply_effects: Applies the action and modifies the game state.
        __call__: Executes the action if preconditions are met.
        action_name: Returns the action name used for routing command strings.
        at: Checks if a specified thing is at a given location.
        has_connection: Verifies if the location has an exit in a specified direction.
        is_blocked: Checks if movement in a specified direction is blocked.
        property_equals: Validates if a thing has a specific property value.
        has_property: Checks if a thing possesses a specified property.
        loc_has_item: Determines if a location contains a specific item.
        is_in_inventory: Checks if a character has a specific item in their inventory.
        was_matched: Verifies if a thing was matched by the game's parser.
    """

    # The name of the action, intended to be a string that identifies the action.
    ACTION_NAME: str = None

    # A description of the action, providing details about its purpose or effect.
    ACTION_DESCRIPTION: str = None

    # A list of alternative names or aliases for the action, allowing for varied command inputs.
    ACTION_ALIASES: list[str] = None

    def __init__(self, game):
        """
        Initializes an instance of the action with the associated game context. This constructor sets up the game and
        its parser for use within the action.

        Args:
            game: The game instance that this action will interact with.
        """

        # Assign the game instance to the current object's game attribute
        self.game = game

        # Set the parser attribute to the game's parser for command processing
        self.parser = game.parser

    def check_preconditions(self) -> bool:
        """
        Checks whether the preconditions for applying the action are met. This method is intended to be overridden in
        subclasses to implement specific precondition logic.

        Returns:
            bool: Always returns False, indicating that the preconditions are not met by default.
        """

        return False

    def apply_effects(self):
        """
        Applies the effects of the action within the game context. This method is responsible for executing the action's
        intended effects and returning a confirmation of the outcome.

        Returns:
            bool: Returns a success message indicating that there is "no effect" from the action.
        """

        return self.parser.ok("no effect")

    def __call__(self):
        """
        Executes the action by first checking its preconditions. If the preconditions are met, it applies the effects of
        the action; otherwise, it returns False.

        Returns:
            bool: Returns the result of applying the effects if preconditions are satisfied, otherwise returns False.
        """

        return self.apply_effects() if self.check_preconditions() else False

    @classmethod
    def action_name(cls):
        """
        Determines the action name used for routing command strings in the game. This class method returns the action
        name in lowercase, either from a predefined attribute or derived from the class name.

        This method plays a crucial role in how command strings are routed to actual action names. This method provides
        the key used in the game's dict of actions.

        Returns:
            str: The action name in lowercase, either from ACTION_NAME or generated from the class name.
        """

        # Check if ACTION_NAME is defined and is a string; if so, return it in lowercase.
        if cls.ACTION_NAME and isinstance(cls.ACTION_NAME, str):
            return cls.ACTION_NAME.lower()

        # Get the class name and remove underscores for formatting.
        cls_name = cls.__name__
        cls_name = cls_name.replace("_", "")

        # Split the class name into words based on uppercase letters and create a list of words.
        words = re.sub(r"([A-Z])", r" \1", cls_name).split()

        # Join the words into a single string, converting them to lowercase.
        return " ".join([w.lower() for w in words])

    ###
    # Preconditions - these functions are common preconditions.
    # They handle the error messages sent to the parser.
    ###

    def at(self, thing: Thing, location: Location, describe_error: bool = True) -> bool:
        """
        Checks if a specified thing is located at a given location. This method verifies the presence of the thing at
        the location and can provide an error message if it is not found.

        Args:
            thing (Thing): The object to check for at the location.
            location (Location): The location where the thing is expected to be.
            describe_error (bool, optional): Indicates whether to describe the error if the thing is not at the
            location. Defaults to True.

        Returns:
            bool: Returns True if the thing is at the location, otherwise returns False.
        """

        # loc_has_item checks if item.name in location.items.
        # self.items is a dictionary mapping item names to Item objects present in this location. Characters are stored
        # in Characters and items are stored in Items. Both have their self.location set to the current location.
        # .here(thing) checks if a specified thing lists this as its current location (if thing.location == self)

        # Check if the specified thing is not present at the given location.
        if not location.here(thing):
            # If describe_error is True, create an error message indicating the thing is not at the location.
            if describe_error:
                message = "{name} is not at {loc}".format(
                    name=thing.name.capitalize(), loc=location.name
                )
                # Report the failure to the parser with the generated error message.
                self.parser.fail(
                    f"Check {thing.name} is in {location.name}", message, thing
                )
            # Return False since the thing is not at the location.
            return False
        else:
            # Return True since the thing is confirmed to be at the location.
            return True

    def has_connection(
        self, location: Location, direction: str, describe_error: bool = True
    ) -> bool:
        """
        Checks if a specified location has an exit in a given direction. This method verifies the existence of a
        connection and can provide an error message if the connection is not found.

        Args:
            location (Location): The location to check for the connection.
            direction (str): The direction to verify for an exit.
            describe_error (bool, optional): Indicates whether to describe the error if the connection is not found.
            Defaults to True.

        Returns:
            bool: Returns True if the location has an exit in the specified direction, otherwise returns False.
        """

        # Check if the specified direction is not present in the location's connections.
        if direction not in location.connections:  # JD logical change
            # If describe_error is True, create an error message indicating the absence of an exit in the specified
            # direction.
            if describe_error:
                m = "{location_name} does not have an exit '{direction}'"
                message = m.format(
                    location_name=location.name.capitalize(), direction=direction
                )
                # Report the failure to the parser with the generated error message.
                self.parser.fail(
                    f"go {direction} from {location.name}", message, location
                )
            # Return False since there is no exit in the specified direction.
            return False
        else:
            # Return True since an exit in the specified direction exists.
            return True

    def is_blocked(
        self, location: Location, direction: str, describe_error: bool = True
    ) -> bool:
        """
        Checks if movement in a specified direction is blocked at a given location. This method verifies the blockage
        status and can provide an error message if movement is not possible.

        Args:
            location (Location): The location to check for blockage.
            direction (str): The direction to verify for blockage.
            describe_error (bool, optional): Indicates whether to describe the error if movement is blocked. Defaults to
            True.

        Returns:
            bool: Returns True if movement in the specified direction is blocked, otherwise returns False.
        """

        # Check if movement in the specified direction is blocked at the given location.
        if location.is_blocked(direction):
            # Get the description of the blockage for the specified direction.
            message = location.get_block_description(direction)
            # If describe_error is True, report the blockage to the parser with an error message.
            if describe_error:
                self.parser.fail(
                    f"go {direction} from {location.name}", message, location
                )
            # Return True since movement in the specified direction is blocked.
            return True
        else:
            # Return False since movement in the specified direction is not blocked.
            return False

    def property_equals(
        self,
        thing: Thing,
        property_name: str,
        property_value: str,
        error_message: str = None,
        display_message_upon: bool = False,
        describe_error: bool = True,
    ) -> bool:
        """
        Checks whether a specified property of a given thing matches an expected value. This method can provide error
        messages based on the comparison result and can optionally display messages when the property value matches.

        Args:
            thing (Thing): The object whose property is being checked.
            property_name (str): The name of the property to verify.
            property_value (str): The expected value of the property.
            error_message (str, optional): A custom error message to display if the property does not match. Defaults to
            None.
            display_message_upon (bool, optional): Indicates whether to display an error message when the property value
            matches and to avoid reporting an error message when the property values doesn't match. Defaults to False.
            describe_error (bool, optional): Indicates whether to describe the error regardless of the match. Defaults
            to True.

        Returns:
            bool: Returns True if the property matches the expected value, otherwise returns False.
        """

        # Check if the specified property of the thing does not match the expected value.
        if thing.get_property(property_name) != property_value:
            # TODO: I think display_message_upon should be removed.
            # If display_message_upon is False, proceed to handle the error message.
            if not display_message_upon:
                # If no custom error message is provided, create a default error message.
                if not error_message:
                    error_message = "{name}'s {property_name} is not {value}".format(
                        name=thing.name.capitalize(),
                        property_name=property_name,
                        value=property_value,
                    )
                # If describe_error is True, report the failure to the parser with the error message.
                if describe_error:
                    self.parser.fail(
                        f"Check {thing.name} property value", error_message, thing
                    )
            # Return False since the property value does not match the expected value.
            return False
        else:
            # TODO: I think display_message_upon should be removed.
            # If the property value matches the expected value and display_message_upon is True.
            if display_message_upon:
                # If no custom error message is provided, create a default success message.
                if not error_message:
                    error_message = "{name}'s {property_name} is {value}".format(
                        name=thing.name.capitalize(),
                        property_name=property_name,
                        value=property_value,
                    )
                # If describe_error is True, report the success to the parser with the message.
                if describe_error:
                    # TODO: Why are we reporting failure when the property matched – this is a success
                    self.parser.fail(
                        f"Check {thing.name} property value", error_message, thing
                    )
            # Return True since the property value matches the expected value.
            return True

        # ### UPDATED VERSION (TEST THAT THIS RUNS BEFORE IMPLEMENTING) ###
        # #TODO: Update the method signature and docstrings to reflect the removal of display_message_upon,
        # # Check if the specified property of the thing does not match the expected value.
        # if thing.get_property(property_name) != property_value:
        #     # If describe_error is True, report the failure to the parser with the error message.
        #     if describe_error:
        #         # If no custom error message is provided, create a default error message.
        #         if not error_message:
        #             error_message = "{name}'s {property_name} is not {value}".format(
        #                 name=thing.name.capitalize(),
        #                 property_name=property_name,
        #                 value=property_value,
        #             )
        #         # Report the error.
        #         self.parser.fail(
        #             f"Check {thing.name} property value", error_message, thing
        #         )
        #     # Return False since the property value does not match the expected value.
        #     return False
        # # Return True since the property value matches the expected value.
        # return True

    def has_property(
        self,
        thing: Thing,
        property_name: str,
        error_message: str = None,
        display_message_upon: bool = False,
        describe_error: bool = True,
    ) -> bool:
        """
        Checks whether a specified thing has a given property. This method can provide error messages based on the
        presence of the property and can optionally display messages when the property is found.

        Args:
            thing (Thing): The object whose property is being checked.
            property_name (str): The name of the property to verify.
            error_message (str, optional): A custom error message to display if the property is not present. Defaults to
            None.
            display_message_upon (bool, optional): Indicates whether to display a message when the property is found.
            Defaults to False.
            describe_error (bool, optional): Indicates whether to describe the error if the property is not present.
            Defaults to True.

        Returns:
            bool: Returns True if the property is present, otherwise returns False.
        """

        # Check if the specified property of the thing is not present.
        if not thing.get_property(property_name):
            # TODO: I think display_message_upon should be removed.
            # If display_message_upon is False, proceed to handle the error message.
            if not display_message_upon:
                # If no custom error message is provided, create a default error message indicating the property is
                # False.
                if not error_message:
                    error_message = "{name} {property_name} is False".format(
                        name=thing.name.capitalize(), property_name=property_name
                    )
                # If describe_error is True, report the failure to the parser with the error message.
                if describe_error:
                    self.parser.fail(
                        f"Check for {thing.name} property", error_message, thing
                    )
            # Return False since the property is not present.
            return False
        else:
            # TODO: I think display_message_upon should be removed.
            # If the property is present and display_message_upon is True.
            if display_message_upon:
                # If no custom error message is provided, create a default success message indicating the property is
                # True.
                if not error_message:
                    error_message = "{name} {property_name} is True".format(
                        name=thing.name.capitalize(), property_name=property_name
                    )
                # If describe_error is True, report the success to the parser with the message.
                if describe_error:
                    self.parser.fail(
                        f"Check for {thing.name} property", error_message, thing
                    )
            # Return True since the property is present.
            return True

        # ### UPDATED VERSION (TEST THAT THIS RUNS BEFORE IMPLEMENTING) ###
        # # TODO: Update the method signature and docstrings to reflect the removal of display_message_upon,
        # # Check if the specified property of the thing is present.
        # if not thing.get_property(property_name):
        #     # If describe_error is True, report the failure to the parser with the error message.
        #     if describe_error:
        #         # If no custom error message is provided, create a default error message.
        #         if not error_message:
        #             error_message = "{name} {property_name} is False".format(
        #                 name=thing.name.capitalize(), property_name=property_name
        #             )
        #         # Report the error.
        #         self.parser.fail(
        #             f"Check for {thing.name} property", error_message, thing
        #         )
        #     # Return False since the property is not present.
        #     return False
        # # Return True since the property is present.
        # return True

    def loc_has_item(
        self, location: Location, item: Item, describe_error: bool = True
    ) -> bool:
        """
        Checks if a specified item is present in a given location. This method verifies the existence of the item and
        can provide an error message if the item is not found. This method has a similar functionality to "at", but it
        checks for items that have multiple locations like doors.

        Args:
            location (Location): The location to check for the item.
            item (Item): The item to verify its presence in the location.
            describe_error (bool, optional): Indicates whether to describe the error if the item is not found. Defaults
            to True.

        Returns:
            bool: Returns True if the item is present in the location, otherwise returns False.
        """

        # Check if the item's name is present in the list of items at the specified location.
        if item.name in location.items:
            # Return True since the item is found in the location.
            return True

        # If the item is not found and describe_error is True, create an error message.
        if describe_error:
            message = "{loc} does not have {item}".format(
                loc=location.name, item=item.name
            )
            # Report the failure to the parser with the generated error message.
            self.parser.fail(f"Get {item.name}", message, location)

        # Return False since the item is not present in the location.
        return False

    def is_in_inventory(
        self, character: Character, item: Item, describe_error: bool = True
    ) -> bool:
        """
        Checks if a specified item is present in a character's inventory. This method verifies the item's presence and
        can provide an error message if the item is not found.

        Args:
            character (Character): The character whose inventory is being checked.
            item (Item): The item to verify its presence in the character's inventory.
            describe_error (bool, optional): Indicates whether to describe the error if the item is not found. Defaults
            to True.

        Returns:
            bool: Returns True if the item is present in the character's inventory, otherwise returns False.
        """

        # Check if the specified item is not present in the character's inventory.
        if not character.is_in_inventory(item):
            # If the item is not found and describe_error is True, create an error message.
            if describe_error:
                message = "{name} does not have {item_name}".format(
                    name=character.name.capitalize(), item_name=item.name
                )
                # Report the failure to the parser with the generated error message.
                self.parser.fail("check inventory", message, character)
            # Return False since the item is not present in the inventory.
            return False
        else:
            # Return True since the item is confirmed to be in the character's inventory.
            return True

    def was_matched(
        self,
        character: Character,
        thing: Thing,
        error_message: str = None,
        describe_error: bool = True,
    ) -> bool:
        """
        Checks if a specified thing was matched by the game's parser. This method verifies the presence of the thing and
        can provide an error message if the thing is not found.

        Args:
            character (Character): The character associated with the command being checked.
            thing (Thing): The thing to verify if it was matched.
            error_message (str, optional): A custom error message to display if the thing is not matched. Defaults to
            None.
            describe_error (bool, optional): Indicates whether to describe the error if the thing is not matched.
            Defaults to True.

        Returns:
            bool: Returns True if the thing was matched, otherwise returns False.
        """

        # Check if the specified thing is None, indicating it was not matched by the parser.
        if thing is None:
            # If no custom error message is provided, set a default error message.
            if not error_message:
                error_message = "Something was not matched by the game's parser."
            # If describe_error is True, report the failure to the parser with the error message.
            if describe_error:
                self.parser.fail("Unknown command", error_message, character)
            # Return False since the thing was not matched.
            return False
        else:
            # Return True since the thing was successfully matched.
            return True


class ActionSequence(Action):
    """
    Represents a sequence of actions to be performed in order. This class allows multiple actions to be executed
    sequentially, with each action separated by commas.

    Example: get pole, go out, south, catch fish with pole

    Attributes:
        ACTION_NAME (str): The name of the action sequence.
        ACTION_DESCRIPTION (str): A description of the action sequence.

    Args:
        game: The game instance that this action sequence is associated with.
        command (str): A string representing the sequence of commands to be executed.
        character (Character, optional): The character that will perform the actions. Defaults to None.

    Methods:
        check_preconditions: Checks if the preconditions for executing the action sequence are met.
        apply_effects: Executes the sequence of commands and returns the results of the actions.
    """

    # Define the name of the action
    ACTION_NAME = "sequence"

    # Provide a description of the action's functionality
    ACTION_DESCRIPTION = "Perform multiple actions in order, separated by commas"

    def __init__(self, game, command: str, character: Character = None):
        """
        Initializes an instance of the ActionSequence class with the specified game context, command string, and
        optional character. This constructor sets up the necessary attributes for executing a sequence of actions.

        Args:
            game: The game instance that this action sequence is associated with.
            command (str): A string representing the sequence of commands to be executed.
            character (Character, optional): The character that will perform the actions. Defaults to None.
        """

        # Call the initializer of the parent class (Action) to set up the game context.
        super().__init__(game)

        # Store the command string representing the sequence of actions to be executed.
        self.command = command

        # Store the character that will perform the actions, if provided; otherwise, it defaults to None.
        self.character = character

    def check_preconditions(self) -> bool:
        """
        Checks whether the preconditions for executing the action are met. This method currently always returns True,
        indicating that there are no preconditions to satisfy.

        Returns:
            bool: Always returns True.
        """

        return True

    def apply_effects(self):
        """
        Executes the sequence of commands defined in the action. This method parses each command, executes it, and
        collects the responses to determine if all actions were successful.

        Returns:
            bool: Returns True if all commands were successfully executed, otherwise returns False.
        """

        # Initialize a list to store the responses from executing each command.
        responses = []

        # Split the command string by commas to get individual commands and iterate over them.
        for cmd in self.command.split(","):
            # Remove any leading or trailing whitespace from the command.
            cmd = cmd.strip()
            # Parse/execute each command and append the result to the responses list.
            responses.append(self.parser.parse_command(cmd, self.character))

        # Return True if all commands were successfully executed; otherwise, return False.
        return all(responses)


class Quit(Action):
    """
    Represents the action of quitting the game. This class allows the player to terminate or exit the current game
    session, providing an option to leave or stop the game.

    Attributes:
        ACTION_NAME (str): The name of the quit action.
        ACTION_DESCRIPTION (str): A description of the quit action.
        ACTION_ALIASES (list[str]): A list of alternative names for the quit action.

    Args:
        game: The game instance that this quit action is associated with.
        command (str): The command string associated with the quit action.
        character (Character, optional): The character that is quitting the game. Defaults to None.

    Methods:
        check_preconditions: Checks if the preconditions for quitting the game are met.
        apply_effects: Executes the quit action, terminating the game session if it is not already over.
    """

    # The name of the action that allows the player to quit the game.
    ACTION_NAME = "quit"

    # A description of the quit action, explaining its purpose and functionality.
    ACTION_DESCRIPTION = "Quit the game. Terminate or exit the current session, also mentioned as leaving or stopping"

    # A list of alternative names or aliases for the quit action, allowing for varied command inputs.
    ACTION_ALIASES = ["q"]

    def __init__(self, game, command: str, character: Character = None):
        """
        Initializes an instance of the Quit action with the specified game context, command string, and optional
        character. This constructor sets up the necessary attributes for executing the quit action.

        Args:
            game: The game instance that this quit action is associated with.
            command (str): The command string associated with the quit action.
            character (Character, optional): The character that is quitting the game. Defaults to None.
        """

        # Call the initializer of the parent class (Action) to set up the game context.
        super().__init__(game)

        # Store the command string representing the quit action to be executed.
        self.command = command

        # Store the character that is associated with the quit action, if provided; otherwise, it defaults to None.
        self.character = character

    def check_preconditions(self) -> bool:
        """
        Checks whether the preconditions for executing the quit action are met. This method currently always returns
        True, indicating that there are no preconditions to satisfy for quitting the game.

        Returns:
            bool: Always returns True.
        """

        return True

    def apply_effects(self):
        """
        Executes the effects of quitting the game by terminating the current game session. This method sets the game
        state to indicate that the game is over and provides feedback to the player.

        Returns:
            bool: Returns True if the game was successfully terminated, otherwise returns False if the game had already
            ended.
        """

        # Check if the game is not already marked as over.
        if not self.game.game_over:
            # Set the game state to indicate that the game is now over.
            self.game.game_over = True

            # If no game over description has been set, provide a default description.
            if not self.game.game_over_description:
                self.game.game_over_description = "The End"

            # Send a success message to the parser with the command and game over description.
            self.parser.ok(
                self.command, self.game.game_over_description, self.character
            )
            # Return True indicating that the game was successfully terminated.
            return True

        # If the game is already over, report a failure to the parser with an appropriate message.
        self.parser.fail(self.command, "Game already ended.", self.character)
        # Return False since the game was not terminated because it was already over.
        return False


class Describe(Action):
    """
    Represents the action of describing the current location in the game. This class allows the player to receive
    details about their surroundings, often requested with terms like 'observe' or 'survey'.

    Attributes:
        ACTION_NAME (str): The name of the describe action.
        ACTION_DESCRIPTION (str): A description of the describe action.
        ACTION_ALIASES (list[str]): A list of alternative names for the describe action.

    Args:
        game: The game instance that this describe action is associated with.
        command (str): The command string associated with the describe action.
        character (Character, optional): The character that is performing the describe action. Defaults to None.

    Methods:
        check_preconditions: Checks if the preconditions for describing the location are met.
        apply_effects: Executes the describe action and provides details about the current location.
    """

    # The name of the action that allows the player to describe the current location.
    ACTION_NAME = "describe"

    # A description of the describe action, explaining its purpose and functionality.
    ACTION_DESCRIPTION = (
        "Describe the current location by providing details about the surroundings. It is also commonly "
        + "requested with terms like 'observe' or 'survey'."
    )

    # A list of alternative names or aliases for the describe

    def __init__(self, game, command: str, character: Character = None):
        """
        Initializes an instance of the Describe action with the specified game context, command string, and optional
        character. This constructor sets up the necessary attributes for executing the describe action.

        Args:
            game: The game instance that this describe action is associated with.
            command (str): The command string associated with the describe action.
            character (Character, optional): The character that is performing the describe action. Defaults to None.
        """

        # Call the initializer of the parent class (Action) to set up the game context.
        super().__init__(game)

        # Store the command string representing the describe action to be executed.
        self.command = command

        # Store the character that is associated with the describe action, if provided; otherwise, it defaults to None.
        self.character = character

    def check_preconditions(self) -> bool:
        """
        Checks whether the preconditions for executing the describe action are met. This method currently always
        returns True, indicating that there are no preconditions to satisfy for describing the location.

        Returns:
            bool: Always returns True.
        """

        return True

    def apply_effects(self):
        """
        Executes the effects of the describe action by providing details about the current location. This method sends
        the description to the parser and indicates that the action was successfully applied.

        Returns:
            bool: Always returns True, indicating that the describe action was executed successfully.
        """

        # Send a success message to the parser with the command, the description of the current location, and the
        # character.
        self.parser.ok(self.command, self.game.describe(), self.character)

        # Return True to indicate that the describe action was executed successfully.
        return True
