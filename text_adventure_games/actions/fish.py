# local imports
from text_adventure_games.things.characters import Character
from . import base
from . import preconditions as P
from ..things import Item


class Catch_Fish(base.Action):
    """
    Represents a fishing action that allows a character to catch fish using a fishing pole.

    This class manages the logic for catching fish, including checking preconditions, applying effects, and generating
    descriptive messages about the action. It ensures that the character is at a pond, has a fishing pole, and that fish
    are present in the pond before allowing the action to proceed.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A brief description of the action.
        ACTION_ALIASES (list): Alternative names for the action.

    Args:
        game: The game instance in which the action takes place.
        command (str): The command string representing the action to be performed.
        character (Character): The character who will perform the action.

    Methods:
        check_preconditions: Verifies if the action can be performed based on the current state.
        apply_effects: Applies the effects of catching a fish on the character and the pond.
    """

    # Define the name of the action
    ACTION_NAME = "catch fish"

    # Provide a description of the action's purpose
    ACTION_DESCRIPTION = "Catch fish with a pole. Generally, catch an aquatic animal or creature with a rod."

    # Define alternative names for the action
    ACTION_ALIASES = ["go fishing"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a Catch_Fish action for a character in the game.

        This constructor sets up the command and character for the fishing action, determines if the character has a
        fishing pole, and initializes a fish item that can be caught. It also checks if the character is at a pond and
        adds the fish to the pond if applicable.

        Args:
            game: The game instance in which the action takes place.
            command (str): The command string representing the action to be performed.
            character (Character): The character who will perform the action.

        Returns:
            None
        """

        # Call the constructor of the parent class with the game instance
        super().__init__(game)

        # (Commented out) Retrieve the character associated with the command
        # self.character = self.parser.get_character(command)

        # Store the command string for the action
        self.command = command

        # Store the character instance that will perform the action
        self.character = character

        # Set the pond location where the character is currently located
        self.pond = self.character.location

        # Initialize the pole variable to track if the character has a fishing pole
        self.pole = False

        # Check if the command includes "pole" or "rod" to determine if the character has a fishing pole
        if " pole" in command or " rod" in command:
            self.pole = self.parser.match_item(
                "pole", self.parser.get_items_in_scope(self.character)
            )

        # Create a new fish item with a description and properties
        fish = Item("fish", "a dead fish", "IT SMELLS TERRIBLE.")
        fish.add_command_hint("eat fish")  # Add a command hint for eating the fish
        fish.set_property("is_food", True)  # Set the fish as food
        fish.set_property(
            "taste", "disgusting! It's raw! And definitely not sashimi-grade!"
        )  # Set the taste property of the fish

        # Check if the character is at a pond and if the pond has fish
        if self.pond and self.pond.has_property("has_fish"):
            self.pond.set_property(
                "has_fish", True
            )  # Set the pond to indicate it has fish
            self.pond.add_item(fish)  # Add the fish item to the pond

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions for performing the catch fish action.

        This method verifies if the character is at a pond, if the pond has fish, and if the pond is a valid body of
        water for fishing. It returns True if all conditions are met; otherwise, it triggers a failure with an
        appropriate message.

        Preconditions:
        * There must be a pond
        * The character must be at the pond
        * The character must have a fishing pole in their inventory

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the character is at a pond and if the pond has fish
        if self.pond and not self.pond.has_property("has_fish"):
            self.parser.fail(
                self.command,
                f"{self.character.name} tried to fish in {self.pond.name}. Fish are not found here.",  # Set failure message if no fish are found
                self.character,
            )
            return False  # Return False if there are no fish in the pond

        # Check if the pond is a valid body of water for fishing
        if not self.was_matched(
            self.character, self.pond, "There's no body of water here."
        ):
            self.parser.fail(
                self.command,
                f"{self.character.name} tried to fish in {self.pond.name}, which might not be a body of water",  # Set failure message if the location is not a body of water
                self.character,
            )
            return False  # Return False if the pond is not valid

        # Check if the pond has fish available for catching
        if not self.pond.get_property("has_fish"):
            self.parser.fail(
                self.command,
                "The body of water has no fish.",
                self.character,  # Set failure message if the pond has no fish
            )
            return False  # Return False if there are no fish in the pond

        # (Commented out) Check if the character has a fishing pole in their inventory
        # if not self.character.is_in_inventory(self.pole):
        #     return False

        # All preconditions are satisfied, return True
        return True

    def apply_effects(self):
        """
        Applies the effects of catching a fish on the character and the pond.

        This method checks if the character has a fishing pole; if not, it generates a failure message indicating that
        the character cannot catch fish without one. If the character has a pole, it retrieves the fish from the pond,
        updates the pond's state, and adds the fish to the character's inventory, while also providing a descriptive
        message about the successful catch.

        Effects:
        * Creates a new item for the fish
        * Adds the fish to the character's inventory
        * Sets the 'has_fish' property of the pond to False.

        Returns:
            bool: True after successfully applying the effects of the action, False if the action fails.
        """

        # Check if the character has a fishing pole
        if not self.pole:
            # Create a message indicating the character is trying to catch fish without a pole
            no_pole = "".join(
                [
                    f"{self.character.name} reaches into the pond and tries to ",
                    "catch a fish with their hands, but the fish are too fast. ",
                    "Try to specify that you want to catch fish with the fishing pole.",
                ]
            )
            self.parser.fail(
                self.command, no_pole, self.character
            )  # Trigger failure with the message
            return False  # Return False if the character does not have a pole

        # Attempt to retrieve the fish item from the pond
        if fish := self.pond.get_item("fish"):
            self.pond.set_property(
                "has_fish", False
            )  # Update the pond to indicate it no longer has fish
            self.pond.remove_item(fish)  # Remove the fish item from the pond
            self.character.add_to_inventory(
                fish
            )  # Add the fish to the character's inventory

        # Create a description of the successful catch
        d = "".join(
            [
                f"{self.character.name} dips their hook into the pond and ",
                "catches a fish. It might be good to eat!",
            ]
        )
        description = d.format(
            character_name=self.character.name
        )  # Format the description with the character's name

        # Notify the parser of the successful action with the generated description
        # self.parser.ok(description)  # (Commented out) Previous notification
        self.parser.ok(
            self.command, description, self.character
        )  # Notify the parser of the successful catch
        return (
            True  # Return True to indicate the effects have been successfully applied
        )
