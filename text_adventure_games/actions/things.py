print("Importing Things")

# local imports
print(f"\t{__name__} calling imports for Base")
from . import base
from . import preconditions as P
print(f"\t{__name__} calling imports for Drink and Eat")
from .consume import Drink, Eat
print(f"\t{__name__} calling imports for Character")
from ..things import Character


class Get(base.Action):
    """
    Represents an action for a character to acquire an item from their current location.

    This action allows a character to pick up an item and add it to their inventory, ensuring that the necessary
    conditions are met before the action is executed.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action's purpose.
        ACTION_ALIASES (list): Alternative names for the action.

    Args:
        game: The game instance that this action is associated with.
        command (str): The command input that triggers the action.
        character (Character): The character that will perform the action.
    """

    # The name of the action that represents acquiring an item.
    ACTION_NAME = "get"

    # A description of the action, explaining its purpose and functionality.
    ACTION_DESCRIPTION = (
        "Acquire, get, take, pick up an item for personal use and add to inventory."
    )

    # A list of alternative names or synonyms for the action, allowing for varied command inputs.
    ACTION_ALIASES = ["take", "collect", "pick up"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a new action for a character involving an item based on the provided command.

        This constructor sets up the command to be executed, assigns the character performing the action, and identifies
        the item in the character's current location. It ensures that the action is properly linked to the game instance
        and the relevant item.

        Args:
            game: The game instance that this action is associated with.
            command (str): The command input that triggers the action.
            character (Character): The character that will perform the action.
        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Set the command that will be executed for this action
        self.command = command

        # self.character = self.parser.get_character(command)  # (commented out) Original character retrieval

        # Assign the provided character to the instance variable
        self.character = character

        # Set the current location of the character
        self.location = self.character.location

        # Match the item specified in the command with the items in the character's current location
        self.item = self.parser.match_item(command, self.location.items)

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the character to interact with an item in the current location.

        This method verifies that the item is present, the character is in the correct location, and that the item is
        accessible. It returns True if all conditions are met; otherwise, it triggers appropriate failure messages and
        returns False.

        Preconditions:
        * The item must be matched.
        * The character must be at the location
        * The item must be at the location
        * The item must be gettable

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the item matches the expected criteria for interaction
        if not self.was_matched(self.character, self.item, "I don't see it."):
            # Format a message indicating the item is not found in the current location
            message = f"I don't see this item in {self.location.name}."
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, message, self.character)
            return False  # Return False if the item is not matched

        # Check if the character is in the correct location
        if not self.location.here(self.character):
            # Format a message indicating the character is not in the expected location
            message = "{name} is not in {loc}.".format(
                name=self.character.name, loc=self.location.name
            )
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, message, self.character)
            return False  # Return False if the character is not in the location

        # Check if the item is present in the current location
        if not self.location.here(self.item):
            # Format a message indicating the item is not found in the current location
            message = "There is no {name} in {loc}.".format(
                name=self.item.name, loc=self.location.name
            )
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, message, self.character)
            return False  # Return False if the item is not in the location

        # Check if the item is accessible (gettable)
        if self.item and not self.item.get_property("gettable"):
            # Format a message indicating the item is not accessible
            message = "{name} is not {property_name}.".format(
                name=self.item.name.capitalize(), property_name="gettable"
            )
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, message, self.character)
            return False  # Return False if the item is not gettable

        # Return True if all preconditions are satisfied
        return True

    def apply_effects(self):
        """
        Applies the effects of the action by transferring an item from the location to the character's inventory.

        This method removes the specified item from the current location and adds it to the character's inventory,
        providing feedback about the action. It returns True to indicate that the effects were successfully applied.

        Returns:
            bool: True if the effects were successfully applied.
        """

        # Remove the specified item from the current location
        self.location.remove_item(self.item)

        # Add the item to the character's inventory
        self.character.add_to_inventory(self.item)

        # Create a description of the action indicating the character has obtained the item
        description = "{character_name} got the {item_name}.".format(
            character_name=self.character.name, item_name=self.item.name
        )

        # self.parser.ok(description)  # (commented out) Original success message

        # Provide feedback to the player about the action performed
        self.parser.ok(self.command, description, self.character)

        # Return True to indicate that the effects were successfully applied
        return True


class Drop(base.Action):
    """
    Represents an action to drop or remove an item from the character's possession.

    This class handles the logic for dropping an item, including checking preconditions to ensure the item is in the
    character's inventory and applying the effects of the action by moving the item to the current location. It provides
    feedback to the player about the action performed.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action's purpose.
        ACTION_ALIASES (list): Alternative names for the action.

    Args:
        game: The game instance that this action is associated with.
        command (str): The command input that triggers the action.
        character (Character): The character that will perform the action.
    """

    # The name of the action, indicating the primary function of dropping an item
    ACTION_NAME = "drop"

    # A description of the action's purpose, explaining what it does
    ACTION_DESCRIPTION = "Drop or remove an item from possession, alternatively described as discarding or eliminating."

    # A list of alternative names or phrases that can be used to refer to this action
    ACTION_ALIASES = ["toss", "get rid of"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a new action for dropping an item from the character's inventory.

        This constructor sets up the command to be executed, assigns the character performing the action, and identifies
        the item to be dropped from the character's inventory. It ensures that the action is properly linked to the game
        instance and the relevant item.

        Args:
            game: The game instance that this action is associated with.
            command (str): The command input that triggers the action.
            character (Character): The character that will perform the action.
        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Set the command that will be executed for this action
        self.command = command

        # self.character = self.parser.get_character(command)  # (commented out) Original character retrieval

        # Assign the provided character to the instance variable
        self.character = character

        # Set the current location of the character
        self.location = self.character.location

        # Match the item specified in the command with the items in the character's inventory
        self.item = self.parser.match_item(command, self.character.inventory)

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the character to drop an item from their inventory.

        This method verifies that the item to be dropped is present in the character's inventory and that it matches the
        expected criteria for interaction. It returns True if all conditions are met; otherwise, it triggers a failure
        message and returns False.

        Returns:
            bool: True if all preconditions are satisfied, False otherwise.
        """

        # Check if the item matches the expected criteria for interaction
        if not self.was_matched(self.character, self.item, "I don't see it."):
            return False  # Return False if the item is not matched

        # Check if the character has the item in their inventory
        if not self.character.is_in_inventory(self.item):
            # Format a message indicating the character does not have the item
            d = "{character_name} does not have the {item_name}."
            description = d.format(
                character_name=self.character.name, item_name=self.item.name
            )
            # Trigger a failure with the command and the message
            self.parser.fail(self.command, description, self.character)
            return False  # Return False if the item is not in the inventory

        # Return True if all preconditions are satisfied
        return True

    def apply_effects(self):
        """
        Applies the effects of the drop action by transferring an item from the character's inventory to the current
        location.

        This method removes the specified item from the character's inventory and adds it to the current location,
        providing feedback about the action performed. It returns True to indicate that the effects were successfully
        applied.

        Returns:
            bool: True if the effects were successfully applied.
        """

        # Remove the specified item from the character's inventory
        self.character.remove_from_inventory(self.item)

        # Set the item's location to the current location
        self.item.location = self.location

        # Add the item to the current location
        self.location.add_item(self.item)

        # Create a description of the action indicating the character has dropped the item
        d = "{character_name} dropped the {item_name} in the {location}."
        description = d.format(
            character_name=self.character.name.capitalize(),
            item_name=self.item.name,
            location=self.location.name,
        )

        # self.parser.ok(description)  # (commented out) Original success message

        # Provide feedback to the player about the action performed
        self.parser.ok(self.command, description, self.character)

        # Return True to indicate that the effects were successfully applied
        return True


class Inventory(base.Action):
    """
    Represents an action to check the character's personal inventory of items.

    This class handles the logic for displaying the items currently held by the character, providing feedback about the
    inventory's contents or indicating if it is empty. It ensures that the action is properly linked to the game
    instance and the relevant character.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action's purpose.
        ACTION_ALIASES (list): Alternative names for the action.

    Args:
        game: The game instance that this action is associated with.
        command (str): The command input that triggers the action.
        character (Character): The character whose inventory will be checked.
    """

    # The name of the action, indicating the primary function of checking the inventory
    ACTION_NAME = "inventory"

    # A description of the action's purpose, explaining what it does
    ACTION_DESCRIPTION = (
        """Check personal inventory, a list of items currently held and """
        """often referred to as checking belongings."""
    )

    # A list of alternative names or abbreviations that can be used to refer to this action
    ACTION_ALIASES = ["i"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a new action for checking the character's inventory.

        This constructor sets up the command to be executed and assigns the character whose inventory will be checked.
        It ensures that the action is properly linked to the game instance and the relevant character.

        Args:
            game: The game instance that this action is associated with.
            command (str): The command input that triggers the action.
            character (Character): The character whose inventory will be checked.
        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Set the command that will be executed for this action
        self.command = command

        # self.character = self.parser.get_character(command)  # (commented out) Original character retrieval

        # Assign the provided character to the instance variable
        self.character = character

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the inventory action to be executed.

        This method verifies that the character is not None, ensuring that there is a valid character associated with
        the action. It returns True if the precondition is met; otherwise, it returns False.

        Returns:
            bool: True if the preconditions are satisfied, False otherwise.
        """

        return self.character is not None

    def apply_effects(self):
        """
        Applies the effects of the inventory action by providing a summary of the character's current items.

        This method checks if the character's inventory is empty and generates a corresponding message. If items are
        present, it lists each item in the inventory, providing feedback to the player about what the character
        currently possesses.

        Returns:
            bool: True to indicate that the effects were successfully applied.
        """

        # Check if the character's inventory is empty
        if len(self.character.inventory) == 0:
            # Set the description indicating the inventory is empty
            description = f"{self.character.name}'s inventory is empty."
        else:
            # Set the initial description indicating the inventory contains items
            description = f"{self.character.name}'s inventory contains:\n"
            # Iterate through each item in the character's inventory
            for item_name in self.character.inventory:
                item = self.character.inventory[item_name]
                # Append the item's description to the inventory list
                description += "* {item}\n".format(item=item.description)

        # self.parser.ok(description)  # (commented out) Original success message

        # Provide feedback to the player about the current inventory
        self.parser.ok(self.command, description, self.character)

        # Return True to indicate that the effects were successfully applied
        return True


class Examine(base.Action):
    """
    Represents an action to closely inspect or look at an object or item in the game.

    This class handles the logic for examining an item, providing detailed information about it based on the player's
    command. It ensures that the action is properly linked to the game instance and the relevant character, and it
    retrieves the appropriate item to be examined.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action's purpose.
        ACTION_ALIASES (list): Alternative names for the action.

    Args:
        game: The game instance that this action is associated with.
        command (str): The command input that triggers the action.
        character (Character): The character performing the action.
    """

    # The name of the action, indicating the primary function of examining an item
    ACTION_NAME = "examine"

    # A description of the action's purpose, explaining what it does
    ACTION_DESCRIPTION = (
        """Closely inspect or look at an object/item to learn more about it, """
        """including examining or scrutinizing."""
    )

    # A list of alternative names or phrases that can be used to refer to this action
    ACTION_ALIASES = ["look at", "x"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a new action for examining an item in the game.

        This constructor sets up the command to be executed, assigns the character performing the action, and identifies
        the item to be examined based on the command. It ensures that the action is properly linked to the game instance
        and retrieves the relevant item within the character's scope.

        Args:
            game: The game instance that this action is associated with.
            command (str): The command input that triggers the action.
            character (Character): The character who will perform the examination.
        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Set the command that will be executed for this action
        self.command = command

        # self.character = self.parser.get_character(command)  # (commented out) Original character retrieval

        # Assign the provided character to the instance variable
        self.character = character

        # Match the item specified in the command with the items in the character's scope
        self.matched_item = self.parser.match_item(
            command, self.parser.get_items_in_scope(self.character)
        )

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the examine action to be executed.

        This method verifies that the character is not None, ensuring that there is a valid character associated with
        the action. It returns True if the precondition is met; otherwise, it returns False.

        Returns:
            bool: True if the preconditions are satisfied, False otherwise.
        """

        return self.character is not None

    def apply_effects(self):
        """
        Applies the effects of the examine action by providing detailed information about the matched item.

        This method checks if an item has been matched for examination and retrieves its description or examine text to
        present to the player. If no item is matched, it informs the player that there is nothing special to see.

        Returns:
            bool: True to indicate that the effects were successfully applied.
        """

        # Check if an item has been matched for examination
        if self.matched_item:
            # If the matched item has specific examine text, display it
            if self.matched_item.examine_text:
                self.parser.ok(
                    self.command, self.matched_item.examine_text, self.character
                )
            else:
                # If no specific examine text, display the item's general description
                self.parser.ok(
                    self.command, self.matched_item.description, self.character
                )
        else:
            # If no item is matched, inform the player that there is nothing special to see
            self.parser.ok(
                self.command, "You don't see anything special.", self.character
            )

        # Return True to indicate that the effects were successfully applied
        return True


class Give(base.Action):
    """
    Represents an action to give or transfer an item from one character to another.

    This class handles the logic for transferring an item, ensuring that the giver has the item in their inventory and
    is in the same location as the recipient. It also checks if the recipient has specific needs, such as hunger or
    thirst, and allows them to consume the item if applicable.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action's purpose.
        ACTION_ALIASES (list): Alternative names for the action.

    Args:
        game: The game instance that this action is associated with.
        command (str): The command input that triggers the action.
        character (Character): The character performing the action of giving.
    """

    # The name of the action, indicating the primary function of giving an item
    ACTION_NAME = "give"

    # A description of the action's purpose, explaining what it does
    ACTION_DESCRIPTION = (
        """Give or transfer something (an item for example) to another individual. """
        """Also referred to as handing over."""
    )

    # A list of alternative names or phrases that can be used to refer to this action
    ACTION_ALIASES = ["hand", "deliver", "offer"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a new action for giving an item from one character to another.

        This constructor sets up the command to be executed, identifies the giver and recipient characters, and
        determines the item to be given based on the command. It ensures that the action is properly linked to the game
        instance and retrieves the relevant characters and item involved in the transfer.

        Args:
            game: The game instance that this action is associated with.
            command (str): The command input that triggers the action.
            character (Character): The character who is giving the item.
        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Set the command that will be executed for this action
        self.command = command

        # self.character = self.parser.get_character(command)  # (commented out) Original character retrieval

        # List of keywords that indicate the action of giving
        give_words = ["give", "hand"]

        # Initialize variables to hold parts of the command
        command_before_word = ""
        command_after_word = command

        # Split the command to identify the giver and recipient based on the keywords
        for word in give_words:
            if word in command:
                parts = command.split(
                    word, 1
                )  # Split the command at the first occurrence of the word
                command_before_word = parts[0]  # Text before the action word
                command_after_word = parts[1]  # Text after the action word
                break

        # self.giver = self.parser.get_character(command_before_word)  # (commented out) Original giver retrieval
        # Assign the character who is giving the item
        self.giver = character

        # Identify the recipient character based on the command after the action word
        self.recipient = self.parser.get_character(command_after_word, character=None)

        # Match the item specified in the command with the items in the giver's inventory
        self.item = self.parser.match_item(command, self.giver.inventory)

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the give action to be executed.

        This method verifies that the item to be given is present and matched with the giver, that the giver has the
        item in their inventory, and that the giver is in the same location as the recipient. It returns True if all
        conditions are met; otherwise, it returns False.

        Preconditions:
        * The item must be in the giver's inventory
        * The character must be at the same location as the recipient

        Returns:
            bool: True if the preconditions are satisfied, False otherwise.
        """

        # Check if the item to be given is matched with the giver
        if not self.was_matched(self.giver, self.item, "I don't see it."):
            return False  # Return False if the item is not matched

        # Check if the giver has the item in their inventory
        if not self.giver.is_in_inventory(self.item):
            return False  # Return False if the item is not in the giver's inventory

        # Return True if the giver is in the same location as the recipient
        return bool(self.giver.location.here(self.recipient))

    def apply_effects(self):
        """
        Applies the effects of the give action by transferring an item from the giver to the recipient.

        This method removes the specified item from the giver's inventory and adds it to the recipient's inventory,
        providing feedback about the transfer. If the recipient has specific needs, such as hunger or thirst, they will
        consume the item if it is food or drink, respectively.

        Returns:
            bool: True to indicate that the effects were successfully applied.
        """

        # Remove the specified item from the giver's inventory
        self.giver.remove_from_inventory(self.item)

        # Add the item to the recipient's inventory
        self.recipient.add_to_inventory(self.item)

        # Create a description of the action indicating the item has been given
        description = "{giver} gave the {item_name} to {recipient}".format(
            giver=self.giver.name.capitalize(),
            item_name=self.item.name,
            recipient=self.recipient.name.capitalize(),
        )

        # Provide feedback to the player about the item transfer
        self.parser.ok(self.command, description, self.giver)

        # Check if the recipient is hungry and the item is food
        if self.recipient.get_property("is_hungry") and self.item.get_property(
            "is_food"
        ):
            # Create a command for the recipient to eat the food item
            command = "{name} eat {food}".format(
                name=self.recipient.name, food=self.item.name
            )
            # Create an Eat action for the recipient
            eat = Eat(self.game, command, self.recipient)
            eat()  # Execute the eat action

        # Check if the recipient is thirsty and the item is a drink
        if self.recipient.get_property("is_thisty") and self.item.get_property(
            "is_drink"
        ):
            # Create a command for the recipient to drink the drink item
            command = "{name} drink {drink}".format(
                name=self.recipient.name, drink=self.item.name
            )
            # Create a Drink action for the recipient
            drink = Drink(self.game, command, self.recipient)
            drink()  # Execute the drink action

        # Return True to indicate that the effects were successfully applied
        return True


class Unlock_Door(base.Action):
    """
    Represents an action to unlock a door that is currently locked.

    This class handles the logic for unlocking a door using a key, ensuring that the necessary conditions are met before
    the action is applied. It retrieves the relevant key and door from the character's scope and updates the door's
    state to unlocked if the action is successful.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A description of the action's purpose.

    Args:
        game: The game instance that this action is associated with.
        command: The command input that triggers the action.
        character: The character attempting to unlock the door.
    """

    # The name of the action, indicating the primary function of unlocking a door
    ACTION_NAME = "unlock door"

    # A description of the action's purpose, explaining what it does
    ACTION_DESCRIPTION = (
        "Unlock a door that is currently locked so that it may be opened"
    )

    def __init__(self, game, command, character):
        """
        Initializes a new action for unlocking a door using a key.

        This constructor sets up the command to be executed, assigns the character attempting the action, and identifies
        the key and door within the character's scope. It ensures that the action is properly linked to the game
        instance and retrieves the relevant items needed for the unlocking process.

        Args:
            game: The game instance that this action is associated with.
            command: The command input that triggers the action.
            character: The character attempting to unlock the door.
        """

        # Call the initializer of the parent class with the game instance
        super().__init__(game)

        # Set the command that will be executed for this action
        self.command = command

        # self.character = self.parser.get_character(command)  # (commented out) Original character retrieval

        # Assign the provided character to the instance variable
        self.character = character

        # Match the key specified in the command with the items in the character's scope
        self.key = self.parser.match_item(
            "key", self.parser.get_items_in_scope(self.character)
        )

        # Match the door specified in the command with the items in the character's scope
        self.door = self.parser.match_item(
            "door", self.parser.get_items_in_scope(self.character)
        )

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for the unlock door action to be executed.

        This method verifies that the door is present, that it is locked, and that a key is available to unlock it. It
        returns True if all conditions are met; otherwise, it returns False.

        Returns:
            bool: True if the preconditions are satisfied, False otherwise.
        """

        return bool(self.door and self.door.get_property("is_locked") and self.key)

    def apply_effects(self):
        """
        Applies the effects of the unlock door action by changing the state of the door.

        This method unlocks the specified door and provides feedback to the player about the successful action. It
        returns True to indicate that the effects were successfully applied.

        Returns:
            bool: True to indicate that the effects were successfully applied.
        """

        # Set the property of the door to indicate that it is now unlocked
        self.door.set_property("is_locked", False)

        # self.parser.ok("Door is unlocked")  # (commented out) Original success message

        # Provide feedback to the player about the door being unlocked
        self.parser.ok(self.command, "Door is unlocked", self.character)

        # Return True to indicate that the effects were successfully applied
        return True
