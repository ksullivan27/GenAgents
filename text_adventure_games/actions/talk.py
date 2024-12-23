from text_adventure_games.things.characters import Character
from . import base
from .things import Drop
from . import preconditions as P
from text_adventure_games.managers import Dialogue
from collections.abc import Iterable

circular_import_prints = False

from typing import (
    TYPE_CHECKING,
    Union,
    Set,
)  # Allows conditional imports for type hints

if TYPE_CHECKING:
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game


class Talk(base.Action):
    """
    Talk class represents an action to initiate a dialogue between characters in a game. It manages the command parsing
    and checks necessary preconditions before starting the conversation.

    Attributes:
        ACTION_NAME (str): The name of the action.
        ACTION_DESCRIPTION (str): A brief description of the action.
        ACTION_ALIASES (list): Alternative phrases that can trigger the action.

    Args:
        game: The game instance where the action takes place.
        command (str): The command string that triggers the dialogue.
        character (Character): The character initiating the dialogue.

    Methods:
        check_preconditions() -> bool:
            Validates if the dialogue can proceed based on character availability and location.

        apply_effects() -> bool:
            Initiates the dialogue and updates the participants' dialogue history.
    """

    # The name of the action that players can use to initiate a conversation.
    ACTION_NAME = "talk to"

    # A brief description of what the action does.
    ACTION_DESCRIPTION = "Start a dialogue with someone or multiple people"

    # A list of alternative phrases that can also trigger the action.
    ACTION_ALIASES = [
        "talk with",
        "chat with",
        "speak with",
        "go talk to",
        "address",
        "start a conversation with",
        "discuss",
    ]

    def __init__(self, game: "Game", command: str, character: Character, talking_to: Union[Set[Character], None]=None,
                 dialogue_duration: Union[int, None]=None, max_iterations: Union[int, None]=None):
        """
        Initializes a Talk action, setting up the command and identifying the characters involved in the dialogue. This
        constructor processes the command to determine the character initiating the talk and the character being
        addressed.

        Args:
            game (Game): The game instance where the action is taking place.
            command (str): The command string that triggers the dialogue.
            character (Character): The character initiating the dialogue.
            talking_to (Union[Set[Character], None]): The character(s) who is/are being addressed in the dialogue.
                                                      Defaults to None.
            dialogue_duration (Union[int, None]): The duration of the dialogue. Defaults to None.
            max_iterations (Union[int, None]): The maximum number of iterations for the dialogue. Defaults to None.

        Attributes:
            command (str): The command string provided for the action.
            starter (Character): The character who is starting the dialogue.
            talking_to (Set[Character]): The character(s) who is/are being addressed in the dialogue.
            participants (list): A list containing both the starter and the talking-to characters.
        """

        # Call the initializer of the parent class to set up the game context.
        super().__init__(game)

        # Store the command string provided for the action.
        self.command = command

        # Store the dialogue duration provided for the action.
        self.dialogue_duration = dialogue_duration

        # Store the maximum number of iterations provided for the action.
        self.max_iterations = max_iterations

        # The character parameter is currently not used; it may be utilized in future implementations.
        # self.character = character

        # List of keywords that indicate a dialogue action.
        talk_words = ["talk", "chat", "dialogue", "speak", 'discuss', 'address']

        # Initialize variables to hold parts of the command.
        command_before_word = ""
        command_after_word = command

        # Iterate through the talk words to find the first occurrence in the command.
        for word in talk_words:
            if word in command:
                # Split the command at the first occurrence of the talk word to isolate the following text.
                parts = command.split(word, 1)
                command_after_word = parts[
                    1
                ]  # Update command_after_word to the text after the talk word.
                break

        # Set the character initiating the dialogue.
        self.starter = character

        if talking_to is None:
            # Retrieve the character(s) being addressed based on the command context.
            self.talking_to = set(self.parser.get_character(command_after_word, character=None, allow_multiple=True))
        else:
            self.talking_to = set(list(talking_to))

        # Create a list of participants in the dialogue, including both the starter and the talking-to character.
        self.participants = {self.starter} | self.talking_to

    def check_preconditions(self) -> bool:
        """
        Checks the preconditions necessary for initiating a dialogue between characters. This method ensures that both
        the initiating character and the character being addressed are valid and available for conversation.

        Preconditions:
        * There must be a starter and a talking_to
        * They must be in the same location
        * Talking-to character must be available to talk (TODO)

        Returns:
            bool: True if all preconditions are met, otherwise False.

        Raises:
            Fail: If any of the preconditions are not satisfied, an error is reported through the parser with a
            descriptive message.
        """

        # Check if the character being talked to is None, indicating they cannot be found.
        if self.talking_to is None:
            description = (
                f"The character {self.starter.name} tried talking to couldn't be found."
            )
            # Report the failure to the parser with a descriptive message.
            self.parser.fail(self.command, description, self.starter)
            return False

        # Verify that the starter character is valid and can be found.
        if not self.was_matched(self.starter, self.starter):
            description = "The character starting the dialogue couldn't be found."
            # Report the failure to the parser if the starter is not found.
            self.parser.fail(self.command, description, self.starter)
            return False

        # Verify that the talking-to character is valid and can be found.
        for talking_to in self.talking_to:
            if not self.was_matched(self.starter, talking_to):
                description = f"{talking_to.name} could not be found."
                # Report the failure to the parser if the talking-to character is not found.
                self.parser.fail(self.command, description, self.starter)
                return False

        # Check if the talked-to character is in the same location as the starter character.
        for talking_to in self.talking_to:
            if not self.starter.location.here(talking_to):
                description = (
                    f"{self.starter.name} tried to talk to {talking_to.name} but {talking_to.name} "
                    f"is NOT found at {self.starter.location}"
                )
                # Report the failure to the parser if the characters are not in the same location.
                self.parser.fail(self.command, description, self.starter)
                return False

        # Check if the talked-to characters are the same as the last dialogue targets.
        last_dialogue_targets = self.starter.get_last_dialogue_targets()

        if self.talking_to == last_dialogue_targets:
            description = (
                f"{self.starter.name} just spoke with {", ".join([person.name for person in self.talking_to])} last turn. "
                "You must wait a while to talk to them again."
            )
            # Report the failure to the parser for talking to the same character(s) too soon.
            self.parser.fail(self.command, description, self.starter)
            return False

        # If all checks pass, return True indicating that preconditions are met.
        return True

    def apply_effects(self):
        """
        Applies the effects of the talk action by initiating a dialogue between the participants. This method creates a
        dialogue instance, processes the conversation, and updates the participants' dialogue history.

        Effects:
        ** Starts a dialogue

        Returns:
            bool: Always returns True, indicating that the dialogue has been successfully initiated.

        Raises:
            None: This method does not raise exceptions but reports the outcome of the dialogue through the parser.
        """

        # Create a new Dialogue instance with the current game context, participants, command, and dialogue duration.
        dialogue = Dialogue(
            self.game,
            self.starter,
            self.participants,
            self.command,
            dialogue_duration=self.dialogue_duration,
            max_iterations=self.max_iterations,
        )

        # Start the dialogue loop, which handles the conversation and returns the dialogue history.
        dialogue_history = dialogue.dialogue_loop()

        # Register the talked-to character(s) as dialogue participants for each character.
        for character in self.participants:
            character.set_dialogue_participants(self.participants.difference({character}))

        # Report the successful execution of the command along with the dialogue history to the parser.
        self.parser.ok(self.command, dialogue_history, self.starter)

        # Indicate that the effects of the action have been successfully applied.
        return True


# class TalkWithEveryone(base.Action):
#     """Action to initiate or continue a dialogue with all characters present.

#     This class allows a character to engage in conversation with all other characters in the same location.
#     It manages the dialogue process and ensures that all participants are valid and available for interaction.

#     Attributes:
#         ACTION_NAME (str): The name of the action that players can use to initiate a conversation.
#         ACTION_DESCRIPTION (str): A brief description of what the action does.
#         ACTION_ALIASES (list): A list of alternative phrases that can also trigger the action.
#         command (str): The command string provided for the action.
#         starter (Character): The character who is starting the dialogue.
#         talked_to (Character): The character who is being addressed in the dialogue.
#         participants (list): A list containing both the starter and the talking-to characters.

#     Args:
#         game: The game instance where the action takes place.
#         command (str): The command string that triggers the dialogue.
#         character (Character): The character initiating the dialogue.

#     Methods:
#         check_preconditions() -> bool:
#             Validates if the dialogue can proceed based on character availability and location.

#         apply_effects() -> bool:
#             Initiates the dialogue and updates the participants' dialogue history.
#     """

#     # The name of the action that players can use to initiate a conversation.
#     ACTION_NAME = "talk to"

#     # A brief description of what the action does.
#     ACTION_DESCRIPTION = "Start or continue the dialogue with everyone"

#     # A list of alternative phrases that can also trigger the action.
#     ACTION_ALIASES = [
#         "talk with everyone",
#         "chat with everyone",
#         "speak with everyone",
#         "go talk to everyone",
#         "address everyone",
#         "start a conversation with everyone",
#         "talk with all",
#         "chat with all",
#         "speak with all",
#         "go talk to all",
#         "address all",
#         "start a conversation with all",
#         "talk with group",
#         "chat with group",
#         "speak with group",
#         "go talk to group",
#         "address group",
#         "start a conversation with group",
#         "talk with participants",
#         "chat with participants",
#         "speak with participants",
#         "go talk to participants",
#         "address participants",
#         "start a conversation with participants",
#     ]

#     def __init__(self, game, command: str, character: Character):
#         """
#         Initializes a TalkWithEveryone action, setting up the command and identifying the characters involved in the
#         dialogue. This constructor processes the command to determine the character initiating the talk and the
#         character being addressed.

#         Args:
#             game: The game instance where the action is taking place.
#             command (str): The command string that triggers the dialogue.
#             character (Character): The character initiating the dialogue.

#         Attributes:
#             command (str): The command string provided for the action.
#             starter (Character): The character who is starting the dialogue.
#             talked_to (Character): The character who is being addressed in the dialogue.
#             participants (list): A list containing both the starter and the talking-to characters.
#         """

#         # Call the initializer of the parent class to set up the game context.
#         super().__init__(game)

#         # Store the command string provided for the action.
#         self.command = command

#         # The character parameter is currently not used; it may be utilized in future implementations.
#         # self.character = character

#         # Set the character initiating the dialogue.
#         self.starter = character

#         self.talked_to = list(character.location.characters.keys())

#         # Create a list of participants in the dialogue, including both the starter and everyone else in the room.
#         self.participants = [self.starter, *self.talked_to]

#     def check_preconditions(self) -> bool:
#         """
#         Checks the preconditions necessary for initiating a dialogue between characters. This method ensures that both
#         the initiating character and the character being addressed are valid and available for conversation.

#         Preconditions:
#         * There must be a starter and a talked_to
#         * They must be in the same location
#         * Talking-to character must be available to talk (TODO)

#         Returns:
#             bool: True if all preconditions are met, otherwise False.

#         Raises:
#             Fail: If any of the preconditions are not satisfied, an error is reported through the parser with a
#             descriptive message.
#         """

#         # Verify that the starter character is valid and can be found.
#         if not self.was_matched(self.starter, self.starter):
#             description = "The character starting the dialogue couldn't be found."
#             # Report the failure to the parser if the starter is not found.
#             self.parser.fail(self.command, description, self.starter)
#             return False

#         for talked_to in self.talked_to:
#             # Check if the character being talked to is None, indicating they cannot be found.
#             if talked_to is None:
#                 description = f"The character {self.starter.name} tried talking to couldn't be found."
#                 # Report the failure to the parser with a descriptive message.
#                 self.parser.fail(self.command, description, self.starter)
#                 return False

#             # Verify that the talking-to character is valid and can be found.
#             if not self.was_matched(self.starter, talked_to):
#                 description = f"{talked_to.name} could not be found."
#                 # Report the failure to the parser if the talking-to character is not found.
#                 self.parser.fail(self.command, description, self.starter)
#                 return False

#             # Check if the talking-to character is in the same location as the starter character.
#             if not self.starter.location.here(talked_to):
#                 description = (
#                     f"""{self.starter.name} tried to talk to {talked_to.name} but {talked_to.name} """
#                     """is NOT found at {self.starter.location}"""
#                 )
#                 # Report the failure to the parser if the characters are not in the same location.
#                 self.parser.fail(self.command, description, self.starter)
#                 return False

#         # Check if the talking-to characters are the same as the last dialogue targets.
#         if set(self.talked_to) == set(self.starter.get_last_dialogue_targets()):
#             description = (
#                 f"""{self.starter.name} just spoke with {", ".join([person.name for person in self.talked_to]) if
#                 isinstance(self.talked_to, Iterable) else self.talked_to.name} last turn. You must wait a while to """
#                 """talk to them again."""
#             )
#             # Report the failure to the parser for talking to the same character(s) too soon.
#             self.parser.fail(self.command, description, self.starter)
#             return False

#         # If all checks pass, return True indicating that preconditions are met.
#         return True

#     def apply_effects(self):
#         """Apply dialogue effects in the current game context.

#         This method initiates a dialogue between the starter character and other participants,
#         managing the conversation flow and updating the dialogue participants accordingly.
#         It also reports the outcome of the dialogue to the parser.

#         Args:
#             self: The instance of the class.

#         Returns:
#             bool: True if the effects of the action have been successfully applied.
#         """

#         # Create a new Dialogue instance with the current game context, participants, and command.
#         dialogue = Dialogue(self.game, self.participants, self.command)

#         # Start the dialogue loop, which handles the conversation and returns the dialogue history.
#         dialogue_history = dialogue.dialogue_loop()

#         # Create a copy of participants excluding the starter character.
#         participants_minus_starter = self.participants.copy()
#         participants_minus_starter.remove(self.starter)

#         # Register the remaining participants as dialogue participants for the starter character.
#         self.starter.set_dialogue_participants(participants_minus_starter)

#         # Iterate through each participant to set their dialogue participants.
#         for participant in participants_minus_starter:
#             # Create a temporary list of participants excluding the current participant.
#             temp_participants = self.participants.copy()
#             temp_participants.remove(participant)

#             # Set the dialogue participants for the current participant.
#             participant.set_dialogue_participants(temp_participants)

#         # Report the successful execution of the command along with the dialogue history to the parser.
#         self.parser.ok(self.command, dialogue_history, self.starter)

#         # Indicate that the effects of the action have been successfully applied.
#         return True
