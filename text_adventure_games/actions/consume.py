# local imports
from ..things.characters import Character
from . import base
from . import preconditions as P


class Eat(base.Action):
    """
    Represents an action for a character to consume food items for nourishment.

    This action checks if the character can eat a specified food item and applies the effects of eating,
    including removing the item from inventory and updating the character's hunger status.

    Args:
        game: The game instance in which the action is being performed.
        command (str): The command issued by the player to perform the eat action.
        character (Character): The character that is attempting to eat the food item.

    Returns:
        None

    Raises:
        None

    Examples:
        eat_action = Eat(game_instance, "eat apple", character_instance)
        if eat_action.check_preconditions():
            eat_action.apply_effects()
    """

    # Define the name of the action
    ACTION_NAME = "eat"

    # Provide a description of the action's purpose
    ACTION_DESCRIPTION = "Ingest food items for nourishment."

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a ConsumeAction instance for a character in the game.

        This constructor sets up the command and character for the action, and attempts to match an item based on the
        command and the items available to the character.

        Args:
            game: The game instance in which the action takes place.
            command (str): The command string representing the action to be performed.
            character (Character): The character who will perform the action.

        Returns:
            None
        """

        # Call the constructor of the parent class with the game instance
        super().__init__(game)

        # Store the command string for the action
        self.command = command

        # Store the character instance that will perform the action
        self.character = character

        # Match the item based on the command and the items available to the character
        self.item = self.parser.match_item(
            command, self.parser.get_items_in_scope(self.character)
        )

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions for performing the consume action.

        This method verifies if the item can be consumed by checking if it was matched, if it is food, and if the
        character has it in their inventory. It returns True if all conditions are met, otherwise it triggers a failure
        with an appropriate message.

        Preconditions:
        * There must be a matched item
        * The item must be food
        * The food must be in character's inventory

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the item was matched with the character
        if not self.was_matched(self.character, self.item):
            return False  # Return False if the item was not matched

        # Check if the item is classified as food
        elif not self.item.get_property("is_food"):
            description = "That's not edible."  # Set failure message for non-food items
            self.parser.fail(
                self.command, description, self.character
            )  # Trigger failure
            return False  # Return False for non-food items

        # Check if the character has the item in their inventory
        elif not self.character.is_in_inventory(self.item):
            description = "You don't have it."  # Set failure message for missing item
            self.parser.fail(
                self.command, description, self.character
            )  # Trigger failure
            return False  # Return False if the item is not in inventory

        # All preconditions are satisfied, return True
        return True

    def apply_effects(self):
        """
        Applies the effects of consuming an item on the character.

        This method updates the character's state by removing the consumed item from their inventory, setting their
        hunger status, and generating a descriptive message about the action. It also checks if the item has any taste
        or if it is poisonous, updating the character's status accordingly.

        Effects:
        * Removes the food from the inventory so that it has been consumed.
        * Causes the character's hunger to end
        * Describes the taste (if the "taste" property is set)
        * If the food is poisoned, it causes the character to die.

        Returns:
            bool: True after successfully applying the effects of the action.
        """

        # Remove the consumed item from the character's inventory
        self.character.remove_from_inventory(self.item)

        # Set the character's hunger status to false
        self.character.set_property("is_hungry", False)

        # Create a description of the action performed by the character
        description = "{name} eats the {food}.".format(
            name=self.character.name.capitalize(), food=self.item.name
        )

        # Check if the item has a taste property and append it to the description
        if self.item.get_property("taste"):
            description += " It tastes {taste}".format(
                taste=self.item.get_property("taste")
            )

        # Check if the item is poisonous
        if self.item.get_property("is_poisonous"):
            # Set the character's status to dead if the item is poisonous
            self.character.set_property("is_dead", True)
            # Append a message indicating the character died from the poisonous food
            description += " The {food} is poisonous. {name} died.".format(
                food=self.item.name, name=self.character.name.capitalize()
            )

        # Notify the parser of the successful action with the generated description
        # self.parser.ok(description)
        self.parser.ok(self.command, description, self.character)

        # Return True to indicate the effects have been successfully applied
        return True


class Drink(base.Action):
    """
    Represents a drink action that allows a character to consume a liquid.

    This class handles the logic for drinking a liquid, including checking preconditions, applying effects, and
    generating descriptive messages about the action. It ensures that the item is drinkable, available in the
    character's inventory, and manages the consequences of consuming the drink, such as thirst status and potential
    poisoning.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A brief description of the action.

    Args:
        game: The game instance in which the action takes place.
        command (str): The command string representing the action to be performed.
        character (Character): The character who will perform the action.

    Methods:
        check_preconditions: Verifies if the action can be performed based on the current state.
        apply_effects: Applies the effects of drinking the liquid on the character.
    """

    # Define the name of the action
    ACTION_NAME = "drink"

    # Provide a description of the action's purpose
    ACTION_DESCRIPTION = "Drink a liquid."

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a Drink action for a character in the game.

        This constructor sets up the command and character for the drinking action, and attempts to match a drink item
        based on the command and the items available to the character.

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

        # Match the item based on the command and the items available to the character
        self.item = self.parser.match_item(
            command, self.parser.get_items_in_scope(self.character)
        )

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions for performing the drink action.

        This method verifies if the drink item can be consumed by checking if it was matched, if it is drinkable, and
        if the character has it in their inventory. It returns True if all conditions are met; otherwise, it triggers a
        failure with an appropriate message.

        Preconditions:
        * There must be a matched item
        * The item must be a drink
        * The drink must be in character's inventory

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the item was matched with the character
        if not self.was_matched(self.character, self.item):
            return False  # Return False if the item was not matched

        # Check if the item is classified as a drink
        elif not self.item.get_property("is_drink"):
            description = (
                "That's not drinkable."  # Set failure message for non-drinkable items
            )
            self.parser.fail(
                self.command, description, self.character
            )  # Trigger failure
            return False  # Return False for non-drinkable items

        # Check if the character has the item in their inventory
        elif not self.character.is_in_inventory(self.item):
            description = "You don't have it."  # Set failure message for missing item
            # self.parser.fail(description)  # (Commented out) Previous failure trigger
            self.parser.fail(
                self.command, description, self.character
            )  # Trigger failure
            return False  # Return False if the item is not in inventory

        # All preconditions are satisfied, return True
        return True

    def apply_effects(self):
        """
        Applies the effects of consuming a drink on the character.

        This method updates the character's state by removing the consumed drink from their inventory, setting their
        thirst status, and generating a descriptive message about the action. It also checks for additional properties
        of the drink, such as taste, poison, and alcohol content, and updates the character's status accordingly.

        Effects:
        * Removes the drink from the inventory so that it has been consumed.
        * Causes the character's thirst to end
        * Describes the taste (if the "taste" property is set)
        * If the drink is poisoned, it causes the character to die.

        Returns:
            bool: True after successfully applying the effects of the action.
        """

        # Remove the consumed drink from the character's inventory
        self.character.remove_from_inventory(self.item)

        # Set the character's thirst status to false
        self.character.set_property("is_thirsty", False)

        # Create a description of the action performed by the character
        description = "{name} drinks the {drink}.".format(
            name=self.character.name.capitalize(), drink=self.item.name
        )

        # Notify the parser of the successful action with the generated description
        # self.parser.ok(description)  # (Commented out) Previous notification
        self.parser.ok(self.command, description, self.character)

        # Check if the item has a taste property and append it to the description
        if self.item.get_property("taste"):
            description = " It tastes {taste}".format(
                taste=self.item.get_property("taste")
            )
            # self.parser.ok(description)  # (Commented out) Previous notification
            self.parser.ok(self.command, description, self.character)

        # Check if the item is poisonous
        if self.item.get_property("is_poisonous"):
            # Set the character's status to dead if the item is poisonous
            self.character.set_property("is_dead", True)
            # Append a message indicating the character died from the poisonous drink
            description = "The {drink} is poisonous. {name} died.".format(
                drink=self.item.name, name=self.character.name.capitalize()
            )
            # self.parser.ok(description)  # (Commented out) Previous notification
            self.parser.ok(self.command, description, self.character)

        # Check if the item is alcoholic
        if self.item.get_property("is_alcohol"):
            # Set the character's status to indicate they are drunk
            self.character.set_property("is_drink", True)
            # Append a message indicating the character is now drunk
            description = "{name} is now drunk from {drink}.".format(
                drink=self.item.name, name=self.character.name.capitalize()
            )
            # self.parser.ok(description)  # (Commented out) Previous notification
            self.parser.ok(self.command, description, self.character)

        # Return True to indicate the effects have been successfully applied
        return True


class Light(base.Action):
    """
    Represents a light action that allows a character to ignite a flammable item or turn on a light.

    This class handles the logic for lighting an item, including checking preconditions, applying effects, and
    generating descriptive messages about the action. It ensures that the item is lightable, available in the
    character's inventory, and manages the state of the item after it has been lit.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A brief description of the action.

    Args:
        game: The game instance in which the action takes place.
        command (str): The command string representing the action to be performed.
        character (Character): The character who will perform the action.

    Methods:
        check_preconditions: Verifies if the action can be performed based on the current state.
        apply_effects: Applies the effects of lighting the item on the character and the item itself.
    """

    # Define the name of the action
    ACTION_NAME = "light"

    # Provide a description of the action's purpose
    ACTION_DESCRIPTION = "Ignite something flammable like a lamp or a candle. Also includes turning on a light."

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a Light action for a character in the game.

        This constructor sets up the command and character for the lighting action, and attempts to match a lightable
        item based on the command and the items available to the character.

        Effects:
        * Removes the drink from the inventory so that it has been consumed.
        * Causes the character's thirst to end
        * Describes the taste (if the "taste" property is set)
        * If the drink is poisoned, it causes the character to die.

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

        # Match the item based on the command and the items available to the character
        self.item = self.parser.match_item(
            command, self.parser.get_items_in_scope(self.character)
        )

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions for performing the light action.

        This method verifies if the item can be lit by checking if it was matched, if it is in the character's
        inventory, if it is lightable, and if it is already lit. It returns True if all conditions are met; otherwise,
        it triggers a failure with an appropriate message.

        Preconditions:
        * There must be a matched item
        * The item must be in character's inventory
        * The item must be lightable

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the item was matched with the character
        if not self.was_matched(self.character, self.item):
            return False  # Return False if the item was not matched

        # Check if the item is in the character's inventory
        if not self.is_in_inventory(self.character, self.item):
            return False  # Return False if the item is not in inventory

        # Check if the item is lightable
        if not self.item.get_property("is_lightable"):
            description = "That's not something that can be lit."  # Set failure message for non-lightable items
            self.parser.fail(
                self.command, description, self.character
            )  # Trigger failure
            return False  # Return False for non-lightable items

        # Check if the item is already lit
        if self.item.get_property("is_lit"):
            description = (
                "It is already lit."  # Set failure message for already lit items
            )
            self.parser.fail(
                self.command, description, self.character
            )  # Trigger failure
            return False  # Return False if the item is already lit

        # All preconditions are satisfied, return True
        return True

    def apply_effects(self):
        """
        Applies the effects of lighting an item on the character.

        This method updates the state of the item to indicate that it is now lit and generates a descriptive message
        about the action performed by the character. It notifies the parser of the successful action and returns True
        to indicate that the effects have been applied.

        Effects:
        * Changes the state to lit

        Returns:
            bool: True after successfully applying the effects of the action.
        """

        # Set the item's property to indicate that it is now lit
        self.item.set_property("is_lit", True)

        # Create a description of the action performed by the character
        description = "{name} lights the {item}. It glows.".format(
            name=self.character.name, item=self.item.name
        )

        # Notify the parser of the successful action with the generated description
        # self.parser.ok(description)  # (Commented out) Previous notification
        self.parser.ok(self.command, description, self.character)

        # Return True to indicate the effects have been successfully applied
        return True
