# local imports

circular_import_prints = False

if circular_import_prints:
    print("Importing Actions Locations")

if circular_import_prints:
    print(f"\t{__name__} calling imports for Base")
from . import base

if circular_import_prints:
    print(f"\t{__name__} calling imports for Character")
from ..things import Character  # Item  # , Location


class Go(base.Action):
    """
    Go class represents an action that allows a character to move in a specified direction or towards a location. It
    handles the command parsing, checks preconditions for movement, and applies the effects of the action to update the
    game state.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A brief description of what the action does.
        ACTION_ALIASES (list): Alternative phrases that can trigger the action.

    Args:
        game: The game instance where the action takes place.
        command (str): The command string that triggers the action.
        character (Character): The character that will perform the action.

    Methods:
        check_preconditions() -> bool:
            Validates if the character can perform the action based on their location and available exits.

        apply_effects() -> bool:
            Moves the character to a new location based on the specified direction and updates the game state.
    """

    ACTION_NAME = "go"
    ACTION_DESCRIPTION = (
        """Go, move, continue, head in a specified compass direction or towards a location, """
        """including all cardinal directions and entrances/exits."""
    )
    ACTION_ALIASES = [
        "north",
        "n",
        "south",
        "s",
        "east",
        "e",
        "west",
        "w",
        "out",
        "in",
        "up",
        "down",
    ]

    def __init__(
        self,
        game,
        command: str,
        character: Character,
        # location: Location, direction: str
    ):
        """
        Initializes a new action for a character in the game based on the provided command.

        This constructor sets up the character's current location and the direction derived from the command. It also
        initializes the command that will be executed.

        Args:
            game: The game instance that this action is associated with.
            command (str): The command input that triggers the action.
            character (Character): The character that will perform the action.

        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Assign the provided character to the instance variable
        self.character = (
            character
        )

        # Set the current location of the character
        self.location = self.character.location

        # Determine the direction based on the command and the character's current location
        self.direction = self.parser.get_direction(command, self.location)

        # Store the command that triggered this action
        self.command = command

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions for the character to perform an action in the current location.

        This method verifies if the character is at the correct location, if there is a valid exit in the specified
        direction, and if the path is not blocked. It returns True if all conditions are met; otherwise, it triggers a
        failure message and returns False.

        Preconditions:
        * The character must be at the location.
        * The location must have an exit in the specified direction
        * The direction must not be blocked

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the character is at the current location
        if not self.location.here(self.character):
            # Format a message indicating the character is not at the expected location
            message = "{name} is not at {location_name}".format(
                name=self.character.capitalize(),
                location_name=self.location.name.capitalize(),
            )
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, message, self.character)
            return False  # Return False if the character is not at the location

        # Check if there is a valid connection in the specified direction
        if not self.location.get_connection(self.direction):
            # Format a message indicating there is no exit in the specified direction
            d = "{location_name} does not have an exit '{direction}'"
            message = d.format(
                location_name=self.location.name.capitalize(), direction=self.direction
            )
            # self.parser.fail(message)  # (commented out) Original failure message
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, message, self.character)
            return False  # Return False if there is no valid exit

        # Check if the path in the specified direction is blocked
        if self.location.is_blocked(self.direction):
            # Get the description of the blockage in the specified direction
            description = self.location.get_block_description(self.direction)
            if not description:
                # Format a message indicating the direction is blocked if no description is available
                d = "{location_name} is blocked towards {direction}"
                description = d.format(
                    location_name=self.location.name.capitalize(),
                    direction=self.direction,
                )
            # self.parser.fail(message)  # (commented out) Original failure message
            # Trigger a failure with the command and the blockage description
            self.parser.fail(self.command, description, self.character)
            return False  # Return False if the path is blocked

        # Return True if all preconditions are satisfied
        return True

    def apply_effects(self):
        """
        Applies the effects of the character's action, moving them to a new location and updating game state.

        This method handles the movement of the character from their current location to a new one based on the
        specified direction. It also checks for game-ending conditions and provides feedback to the player if necessary.

        Returns:
            bool: True if the action results in a game over, False otherwise.
        """

        # Check if the character is the main player
        is_main_player = self.character == self.game.player

        # Move the character from their current location
        from_loc = self.location
        # If the character is in the current location, remove them
        if self.character.name in from_loc.characters:
            from_loc.remove_character(self.character)

        # Move the character to the new location based on the specified direction
        to_loc = self.location.connections[self.direction]
        to_loc.add_character(self.character)

        # Mark the location as visited if the character is the main player
        if is_main_player:
            self.has_been_visited = True

        # Check if the new location triggers a game over condition
        if to_loc.get_property("game_over") and is_main_player:
            # Set the game state to over and store the description of the location
            self.game.game_over = True
            self.game.game_over_description = to_loc.description
            # Provide feedback to the player about the game over condition
            self.parser.ok(self.command, to_loc.description, self.character)
            return True  # Return True if the game is over
        else:
            # Create a Describe action to provide details about the new location
            action = base.Describe(
                self.game, command=self.command, character=self.character
            )
            return action()  # Execute the action and return its result
