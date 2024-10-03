# local imports
from text_adventure_games.things.characters import Character
from . import base
from .things import Drop
from . import preconditions as P


class Attack(base.Action):
    """
    Represents an action to attack a target using a weapon.

    This class handles the logic for initiating an attack in the game, including determining the attacker, victim, and
    weapon involved. It checks preconditions for a valid attack and applies the effects of the attack if successful.

    Args:
        game: The game instance in which the action takes place.
        command (str): The command string that triggers the attack.
        character (Character): The character initiating the attack.

    Returns:
        None

    Raises:
        None

    Examples:
        attack_action = Attack(game_instance, "attack goblin", player_character)
    """

    # Define the name of the action for the fight module.
    ACTION_NAME = "attack"

    # Provide a description of the action, explaining its purpose and context within the game.
    ACTION_DESCRIPTION = (
        """Attack someone or something with a weapon or initiate combat or physical confrontation, also """
        """known as striking or hitting."""
    )

    # List alternative names or synonyms for the action that can be used in commands.
    ACTION_ALIASES = ["hit"]

    def __init__(self, game, command: str, character: Character):
        """
        Initializes a fight action in the game.

        This constructor sets up the fight action by parsing the command to identify the attacker, victim, and weapon
        involved in the action. It processes the command to extract relevant information based on predefined attack
        words.

        Args:
            game: The game instance in which the action takes place.
            command (str): The command string that describes the fight action.
            character (Character): The character initiating the fight.

        Returns:
            None
        """

        # Call the initializer of the parent class with the game instance to set up the context.
        super().__init__(game)

        # Store the command string that describes the fight action.
        self.command = command

        # Initialize a list of keywords that indicate an attack action.
        attack_words = ["attack", "hit", "strike", "punch", "thwack"]

        # Initialize variables to hold the parts of the command before and after the attack word.
        command_before_word = ""
        command_after_word = command

        # Iterate through the list of attack words to find one present in the command.
        for word in attack_words:
            if word in command:
                # Split the command into two parts: before and after the attack word.
                parts = command.split(word, 1)
                command_before_word = parts[0]  # Text preceding the attack word.
                command_after_word = parts[1]  # Text following the attack word.
                break  # Exit the loop after finding the first matching attack word.

        # Assign the character initiating the attack to the attacker variable.
        self.attacker = character

        # Retrieve the victim character based on the command after the attack word.
        self.victim = self.parser.get_character(command_after_word, character=None)

        # Match the weapon from the command using the attacker's inventory.
        self.weapon = self.parser.match_item(command, self.attacker.inventory)

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions for executing a fight action.

        This method verifies that the attacker and victim are valid, that the attacker is in the same location as the
        victim, and that the attacker possesses a valid weapon. It ensures that the victim is not unconscious or dead
        before allowing the action to proceed.

        Preconditions:
        * There must be an attacker and a victim
        * They must be in the same location
        * There must be a matched weapon
        * The attacker must have the weapon in their inventory
        * The weapon have the property 'is_weapon'
        * The victim must not already be dead or unconscious

        Returns:
            bool: True if all preconditions are met, False otherwise.
        """

        # Check if the attacker is valid by matching the attacker with themselves.
        if not self.was_matched(self.attacker, self.attacker):
            description = "The attacker couldn't be found."
            # Report failure if the attacker is not found.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # Check if the victim is valid by matching the attacker with the victim.
        if not self.was_matched(self.attacker, self.victim):
            description = f"{self.victim} could not be found"
            # Report failure if the victim is not found.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # Verify that the victim is in the same location as the attacker.
        if not self.attacker.location.here(self.victim):
            description = (
                f"""{self.attacker.name} tried to attack {self.victim.name} but {self.victim.name} """
                f"""is NOT found at {self.attacker.location}"""
            )
            # Report failure if the victim is not in the attacker's location.
            self.parser.fail(description)
            return False

        # Check if the attacker has a valid weapon.
        if not self.was_matched(
            self.attacker,
            self.weapon,
            error_message="{name} doesn't have a weapon.".format(
                name=self.attacker.name
            ),
            describe_error=False,
        ):
            # Report failure if the attacker does not have a weapon.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # Verify that the weapon is in the attacker's inventory.
        if not self.attacker.is_in_inventory(self.weapon):
            description = f"{self.attacker.name} doesn't have the {self.weapon.name}."
            # Report failure if the weapon is not found in the attacker's inventory.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # Check if the weapon is classified as a weapon.
        if not self.weapon.get_property("is_weapon"):
            description = "{item} is not a weapon".format(item=self.weapon.name)
            # Report failure if the item is not a weapon.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # Check if the victim is unconscious.
        if self.victim.get_property("is_unconscious"):
            description = "{name} is already unconscious".format(name=self.victim.name)
            # Report failure if the victim is already unconscious.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # Check if the victim is dead.
        if self.victim.get_property("is_dead"):
            description = "{name} is already dead".format(name=self.victim.name)
            # Report failure if the victim is already dead.
            self.parser.fail(self.command, description, self.attacker)
            return False

        # All preconditions are met; return True.
        return True

    def apply_effects(self):
        """
        Applies the effects of an attack action in the game.

        This method processes the outcome of an attack by the attacker on the victim, including handling weapon
        properties and the state of the victim. It updates the game state based on the attack's effects, such as
        knocking the victim unconscious and managing the weapon's condition.

        Effects:
        * If the victim is not invulerable to attacks
        ** Knocks the victim unconscious
        ** The victim drops all items in their inventory
        * If the weapon is fragile then it breaks

        Returns:
            bool: True if the effects were successfully applied, False otherwise.
        """

        # Create a description of the attack action, including the attacker, victim, and weapon used.
        description = "{attacker} attacked {victim} with the {weapon}.".format(
            attacker=self.attacker.name,
            victim=self.victim.name,
            weapon=self.weapon.name,
        )

        # Log the successful attack action in the parser.
        self.parser.ok(self.command, description, self.attacker)

        # Check if the weapon is fragile and handle its breakage.
        if self.weapon.get_property("is_fragile"):
            description = "The fragile weapon broke into pieces."
            # Remove the fragile weapon from the attacker's inventory.
            self.attacker.remove_from_inventory(self.weapon)
            # Log the breakage of the weapon.
            self.parser.ok(self.command, description, self.attacker)

        # Check if the victim is invulnerable to the attack.
        if self.victim.get_property("is_invulerable"):
            description = "The attack has no effect on {name}.".format(
                name=self.victim.name
            )
            # Log that the attack had no effect on the invulnerable victim.
            self.parser.ok(self.command, description, self.attacker)
        else:
            # The victim is knocked unconscious as a result of the attack.
            self.victim.set_property("is_unconscious", True)
            description = "{name} was knocked unconscious.".format(
                name=self.victim.name.capitalize()
            )
            # Log the unconscious state of the victim.
            self.parser.ok(self.command, description, self.attacker)

            # The victim drops their inventory items upon being knocked unconscious.
            items = list(self.victim.inventory.keys())
            for item_name in items:
                item = self.victim.inventory[item_name]
                command = "{victim} drop {item}".format(
                    victim=self.victim.name, item=item_name
                )
                # Create a Drop action for each item the victim drops.
                drop = Drop(self.game, command)
                # Check preconditions for the drop action and apply effects if valid.
                if drop.check_preconditions():
                    drop.apply_effects()

        # Return True indicating that the effects of the attack were successfully applied.
        return True
