"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent_cognition/impressions.py
Description: defines how agents store interpersonal impressions and theory-of-mind of other characters. 
             This assumes that "set_impression" will be called at least once at the end of a round.
             So, in the case that this agent has already made an impression about a person, only memories from 
             the last round should be reasoned over; other memories are theoretically already encoded in the agent's 
             impressions of the target.

             However, if a player comes across a new target, all relevant memories to the target will be pulled.

"""

circular_import_prints = False

if circular_import_prints:
    print("Importing Impressions")

# Importing the inspect module, which provides useful functions to get information about live objects
import inspect

from collections import (
    defaultdict,
)  # Import defaultdict for creating dictionaries with default values
from typing import (
    TYPE_CHECKING,
)  # Import TYPE_CHECKING for type hinting without runtime overhead

# local imports
from text_adventure_games.assets.prompts import (
    impressions_prompts as ip,
)  # Import prompts for impressions

if circular_import_prints:
    print(f"\t{__name__} calling imports for GptHelpers")
from text_adventure_games.gpt.gpt_helpers import (  # Import helper functions for GPT interactions
    limit_context_length,  # Function to limit the context length of prompts
    get_token_remainder,  # Function to get the remaining tokens available
    get_prompt_token_count,  # Function to count tokens in a prompt
    context_list_to_string,  # Function to convert a list of contexts to a string
    GptCallHandler,  # Class to handle GPT calls
)

if circular_import_prints:
    print(f"\t{__name__} calling imports for Prompt Classes")
from ...assets.prompts import prompt_classes

if circular_import_prints:
    print(f"\t{__name__} calling imports for Retrieve")
from .retrieve import Retrieve  # Import Retrieve class for agent cognition

if circular_import_prints:
    print(f"\t{__name__} calling imports for General")
from text_adventure_games.utils.general import (
    get_logger_extras,
)  # Import utility for logging extras

if circular_import_prints:
    print(f"\t{__name__} calling imports for Consts")
# Import the get_models_config function from the consts module in the utils package.
# This function is used to retrieve the configuration for different models used in the game.
from text_adventure_games.utils.consts import get_models_config

if circular_import_prints:
    print(f"{__name__} calling imports for MemoryType")
from ..memory_stream import MemoryType

if TYPE_CHECKING:  # Check if type checking is enabled
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game  # Import Game class for type hints

    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Character")
    from text_adventure_games.things import (
        Character,
    )  # Import Character class for type hints


class Impressions:
    """
    Manages and generates impressions of target characters based on interactions and memories.
    This class allows for the creation, updating, and retrieval of impressions, utilizing a GPT model for generation.

    Attributes:
        impressions (defaultdict): A dictionary storing impressions keyed by target agent identifiers.
        name (str): The name of the agent.
        id (int): The unique identifier of the agent.
        token_offset (int): Offset for token calculations in GPT prompts.
        offset_pad (int): Additional padding for token limits.

    Class Variables:
        gpt_handler (GptCallHandler): Handler for GPT calls to generate impressions.

    Args:
        name (str): The name of the agent.
        id (int): The unique identifier of the agent.
    """

    gpt_handler = None  # Class-level attribute to store the shared GPT handler
    # Define the parameters for the GPT model configuration
    model_params = {
        "max_output_tokens": 512,  # Maximum number of tokens for the output
        "temperature": 1,  # Sampling temperature for randomness in output
        "top_p": 1,  # Nucleus sampling parameter
        "max_retries": 5,  # Maximum number of retries for API calls
    }

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """

        if circular_import_prints:
            print(f"-\tImpressions Module is initializing GptCallHandler")

        # Initialize the GPT handler if it hasn't been set up yet
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(
                model_config_type="impressions", **cls.model_params
            )

    def __init__(self, character: "Character"):
        """
        Initializes an Impressions object to manage character impressions.
        This constructor sets up the necessary attributes for storing impressions and configuring the GPT handler.

        The structure of this dict is:
        top level keys are strings of the target agent "{name}_{id}"
        The inner dict contains the "impression": text describing the agent's impression of a target agent,
                                    "round": round in which the impression was made,  # probably mostly for logging?
                                    "tick": tick on which the impression was made,  # probably mostly for logging?
                                    "creation": the total ticks at time of creation.

        Args:
            character (Character): The character associated with this impressions object.
        """

        if circular_import_prints:
            print(f"-\tInitializing Impressions")

        # Initialize a dictionary to store impressions, with default values as empty dictionaries
        self.impressions = defaultdict(dict)

        # Set the character associated with this impressions object
        self.character = character

        # Initialize the GPT handler if it hasn't been set up yet
        Impressions.initialize_gpt_handler()

        # Define the token offset for GPT prompts, accounting for variable tokens
        self.token_offset = 50

        # Set an additional padding for token calculations
        self.offset_pad = 5

    def _log_impression(self, game, target, message):
        """
        Logs an impression message for a specific character about a target character in the game.
        This method retrieves additional logging information and sends a debug log with the impression details.

        Args:
            game (Game): The current game object where the impression is being logged.
            target (Character): The character the impression is being made about.
            message (str): The impression message to be logged.

        Returns:
            None
        """

        # Retrieve additional logging information specific to the game and character
        extras = get_logger_extras(game, self.character, include_gpt_call_id=True)

        extras["type"] = "Impressions"

        # Add the target character to the extras dictionary
        extras["target"] = f"{target.name}"

        # Log the impression message at the debug level, including the extra information
        game.logger.debug(msg=message, extra=extras)

    def _get_impression(self, target: "Character", str_only=True):
        """
        Retrieves the impression of a specified target character.
        This method can return either the impression text or the full impression data based on the provided flag.

        Args:
            target (Character): The target character whose impression is being retrieved.
            str_only (bool): If True, returns only the impression text; if False, returns the full impression data.

        Returns:
            str or dict: The impression text if str_only is True, the full impression data if str_only is False, or None
            if no impression exists.
        """

        # # Debugging: Print the type of keys in impressions
        # print(
        #     "IMPRESSIONS KEYS TYPES (_get_impression)",
        #     [type(key) for key in self.impressions.keys()],
        # )  # Print types of keys
        # # Check if the target has __hash__ and __eq__ methods defined
        # print("TARGET ID (_get_impression)", target.id)  # Print the ID of the target
        # print("IMPRESSIONS IDs (_get_impression)", [key.id for key in self.impressions.keys()])

        # Retrieve the impression for the target character using a formatted key
        impression = self.impressions.get(target, None)

        # # Debugging: Check if the target is in the keys of the impressions
        # print("Is target in impressions keys?", target in self.impressions)
        # print("IMPRESSION (_get_impression)", impression)

        # If an impression exists and only the string is requested, return the impression text
        if impression and str_only:
            return impression["impression"]

        # If an impression exists and the full data is requested, return the entire impression
        elif impression:
            return impression

        # If no impression exists, return None
        else:
            return None

    def get_impressions(self, as_str: bool = True, prefix: str = "") -> str | dict:
        """
        Retrieves all impressions for the character.

        This method returns a list of impression messages for all characters if `as_str` is True,
        or a dictionary mapping character names to their respective impressions if `as_str` is False.

        Args:
            as_str (bool): A flag indicating whether to return impressions as a string (True) or as a dictionary (False).
            prefix (str): A prefix to add to the beginning of each impression message.

        Returns:
            str | dict: A string containing all impressions joined by new lines if `as_str` is True,
                        or a dictionary mapping character names to their impressions if `as_str` is False.
        """
        # Initialize an empty list to store impressions for all characters
        all_impressions = []

        # Initialize a dictionary to store impressions if as_str is False
        impressions_dict = {}

        # Iterate through all impressions stored in the dictionary
        for target, impression in self.impressions.items():
            # If as_str is True, format the impression message
            if as_str:
                formatted_impression = f"{prefix}{target.name}: {impression['impression']}".strip()
                all_impressions.append(formatted_impression)
            else:
                # Extract the character name from the key and map it to the impression
                # character_name = target.split('_')[0]  # Assuming key format is "{name}_{id}"
                impressions_dict[target] = impression['impression']

        # Return the list of impressions, or the dictionary if as_str is False
        return "\n".join(all_impressions) if as_str else impressions_dict

    def get_multiple_impressions(self, character_list) -> list:
        """
        Retrieves impressions for multiple characters, excluding the current character.
        This method constructs a list of impression messages that describe the agent's thoughts and relationships with
        each character.

        Args:
            character_list (list[Character]): A list of character objects for which impressions are to be retrieved.

        Returns:
            list: A list of strings containing the impressions for each character, or "None" if no impression exists.
        """

        # Initialize an empty list to store impressions for the characters
        char_impressions = []

        # Iterate through each character in the provided character list
        for char in character_list:
            # print("GETTING IMPRESSION FOR", char.name, char)

            # print("CHARACTER IMPRESSIONS", self.impressions)
            # Skip the current character to avoid self-impression
            if char is self.character:
                continue

            # Create a string that introduces the character's impression
            char_impression = (
                f"Your theory of mind of and relationship with {char.name}:\n"
            )

            # Append the impression text for the character, or "None" if no impression exists
            char_impression += self._get_impression(char) or "None\n"

            # print("CHAR IMPRESSION FOR", char.name, char_impression)

            # Add the constructed impression string to the list
            char_impressions.append(char_impression)

        # Return the list of character impressions
        return char_impressions

    def update_impression(self, game: "Game", target: "Character") -> None:
        """
        Conditionally updates the impression of a target character based on the game's state.
        This method checks if the impression needs to be updated based on its age and the current game ticks, and
        triggers an update if necessary.

        Args:
            game (Game): The current game object containing game state information.
            target (Character): The target character whose impression is being updated.

        Returns:
            None
        """

        # print("[IMPRESSIONS] UPDATING IMPRESSION FOR:", target.name)

        # Get the total number of ticks that have occurred in the game
        total_ticks = game.total_ticks

        # Initialize a flag to determine if the impression should be updated
        should_update = False

        # Retrieve the current impression for the target character, if it exists
        if impression := self._get_impression(target, str_only=False):
            # Get the age of the impression based on its creation time
            impression_age = impression["creation"]

            # Check if the impression is older than the maximum allowed ticks per round
            if (total_ticks - impression_age) > game.max_ticks_per_round:
                should_update = True
        else:
            # If no impression exists, set the flag to update
            should_update = True

        # Ensure that an impression is not made until at least half a round has passed
        if should_update: # and total_ticks > game.max_ticks_per_round / 2:
            # Trigger the process to set a new impression for the target character
            self.set_impression(game, target)

    def set_impression(self, game: "Game", target: "Character") -> None:
        """
        Creates and logs an impression of a target character based on the current game context.
        This method generates prompts for the GPT model, retrieves the impression, and updates the internal record of
        impressions.

        Args:
            game (Game): The current game object providing context for the impression.
            target (Character): The target character for whom the impression is being created.

        Returns:
            None
        """

        # print("SETTING IMPRESSION FOR:", target.name)

        # Generate the system and user prompts for the GPT model based on the game context and characters
        system, user = self.build_impression_prompts(game, target)

        # Retrieve the impression from the GPT model using the generated prompts
        impression_dict = self.gpt_generate_impression(system, user, game, target)

        impression_str = ""
        for key, value in impression_dict.items():
            impression_str += f"\n- {key}: {value}"

        # print("GPT GENERATED IMPRESSION:")
        # print(impression_str)

        # Update the internal impressions dictionary with the new impression details
        self.impressions.update(
            {
                target: {
                    "impression": impression_str,  # The generated impression text
                    "round": game.round,  # The current round of the game
                    "tick": game.tick,  # The current tick of the game
                    "creation": game.total_ticks,  # The total ticks at the time of impression creation
                }
            }
        )

        # Summarize and score the impression, obtaining keywords and importance
        _, importance, ref_kwds = game.parser.summarize_and_score_action(
            description=impression_str,
            thing=self.character,
            needs_summary=False,
            needs_score=True,
        )

        self.character.memory.add_memory(
            game.round,
            game.tick,
            description=impression_str,
            keywords=ref_kwds,
            location=self.character.location.name,
            success_status=True,
            memory_importance=importance,
            memory_type=MemoryType.IMPRESSION.value,
            actor_id=self.character.id,
        )

    def gpt_generate_impression(self, system_prompt, user_prompt, game, target) -> str:
        """
        Generates an impression of a target character using the GPT model based on provided prompts.
        This method handles potential errors related to token limits and adjusts the token offset accordingly.

        System prompt uses: world info, agent personal summary, and the target's name.
        User prompt uses: target's name, a list of memories about the target, and the existing impression of the target.

        Args:
            system_prompt (str): The system prompt containing contextual information for the GPT model.
            user_prompt (str): The user prompt that specifies the target character and relevant details.
            game (Game): The current game object providing context for the impression.
            target (Character): The target character for whom the impression is being generated.

        Returns:
            dict: The generated impression of the target character, or triggers a re-evaluation if a token limit error
            occurs.
        """

        # Generate an impression using the GPT handler with the provided system and user prompts
        impression = self.gpt_handler.generate(
            system=system_prompt,
            user=user_prompt,
            character=self.character,
            response_format=prompt_classes.Impressions,
            game=game,
        )

        # Check if the result is a tuple, indicating a potential error due to exceeding token limits
        if isinstance(impression, tuple):
            # Unpack the tuple to get the success status and the token difference
            success, token_difference = impression

            # Update the token offset to account for the token difference and add padding
            self.token_offset = token_difference + self.offset_pad

            # Increase the offset padding for future calculations
            self.offset_pad += 2 * self.offset_pad

            # Trigger a re-evaluation of the impression for the last target character
            return self.set_impression(self.game, target)

        # Convert the parsed goals to a dictionary
        impressions_dict = {}
        impressions_dict["Key Strategies"] = impression.key_strategies
        impressions_dict["Probable Next Moves"] = impression.probable_next_moves
        impressions_dict["Impressions of Me"] = impression.impressions_of_me
        impressions_dict["Information to Keep Secret"] = impression.information_to_keep_secret

        # Log the impression in the game's logger for record-keeping
        self._log_impression(game, target, impressions_dict)

        self.add_to_memory(game, target, impressions_dict)

        # Return the generated impression if no errors occurred
        return impressions_dict

    def add_to_memory(self, game: "Game", target: "Character", impressions_dict: dict) -> None:
        """
        Add the generated impression to the character's memory.

        This method iterates through the impressions dictionary and formats each key-value pair
        to include the target character's name. It then summarizes and scores the action described
        in the impression, adding the summarized memory to the character's memory.

        Args:
            game (Game): The current game object providing context for the impression.
            target (Character): The target character for whom the impression is being generated.
            impressions_dict (dict): A dictionary containing the impressions with keys as categories
                                     and values as the corresponding impressions.

        Returns:
            None: This method modifies the character's memory in place and does not return a value.
        """
        # Iterate through each key-value pair in the impressions dictionary
        for key, value in impressions_dict.items():

            # Format the key to include the target's name
            key = target.name + "'s perceived " + key.lower() + ": "

            # Summarize and score the action described in the statement, obtaining keywords and importance
            _, importance, ref_kwds = game.parser.summarize_and_score_action(
                description=key + value,
                thing=self.character,
                needs_summary=False,
                needs_score=True,
            )

            # Add the summarized memory to the character's memory with relevant details
            self.character.memory.add_memory(
                game.round,
                game.tick,
                value,  # TODO: Consider using key + value here
                ref_kwds,
                self.character.location.name,
                success_status=True,
                memory_importance=importance,
                memory_type=MemoryType.IMPRESSION.value,
                actor_id=self.character.id,
            )

    def build_impression_prompts(self, game, target):
        """
        Constructs the system and user prompts required for generating an impression of a target character.
        This method combines contextual information from the game and characters to create prompts for the GPT model.

        Args:
            game: The current game object providing context for the impression.
            target: The target character for whom the impression is being generated.

        Returns:
            tuple: A tuple containing the system prompt and the user prompt for the GPT model.
        """

        # Generate the system prompt and count the number of tokens used in the system prompt
        system_prompt, sys_token_count = self.build_system_prompt(game, target.name)

        # Calculate the total number of tokens consumed by adding the system token count and the current token offset
        consumed_tokens = sys_token_count + self.token_offset

        # Create the user prompt using the game context, target, and the total consumed tokens
        user_prompt = self.build_user_message(
            game, target, consumed_tokens=consumed_tokens
        )

        # Return both the system prompt and user prompt for further processing
        return system_prompt, user_prompt

    def build_system_prompt(self, game, target_name):
        """
        Constructs the system prompt for the GPT model using information about the character and the target.
        This method combines standard character information with a specific prompt format to create a comprehensive
        system prompt.

        Args:
            game: The current game object providing context for the prompt.
            target_name: The name of the target character for whom the impression is being generated.

        Returns:
            tuple: A tuple containing the constructed system prompt and the token count of the prompt.
        """

        # Import Character from text_adventure_games.things.characters (inside the function to avoid circular imports)

        if circular_import_prints:
            print(f"\t{__name__} interior calling imports for Character")
        from text_adventure_games.things.characters import Character

        # Retrieve the standard information of the character, excluding perceptions
        system_prompt = self.character.get_standard_info(
            game, include_perceptions=False
        )

        # Append the formatted prompt for generating impressions, including the target character's name
        system_prompt += "\n\nCURRENT TASK:\n" + ip.gpt_impressions_prompt.format(target_name=target_name)

        # Calculate the number of tokens in the constructed system prompt
        sys_tkn_count = get_prompt_token_count(
            system_prompt, role="system", tokenizer=game.parser.tokenizer
        )

        # Return the constructed system prompt along with its token count
        return system_prompt, sys_tkn_count

    def build_user_message(self, game, target, consumed_tokens=0) -> str:
        """
        Constructs a user message for the GPT model that includes the target character's name and relevant memories.
        This method gathers the agent's current impression of the target and any pertinent memories to create a
        comprehensive message.

        Args:
            game: The current game object providing context for the message.
            target: The target character for whom the message is being constructed.
            consumed_tokens (int, optional): The number of tokens already consumed in the prompt. Defaults to 0.

        Returns:
            str: The constructed user message for the GPT model.
        """

        # Import Character from text_adventure_games.things.characters (inside the function to avoid circular imports)

        if circular_import_prints:
            print(f"\t{__name__} interior calling imports for Character")
        from text_adventure_games.things.characters import Character

        current_tom_primer = "\n\nCurrent theory of mind for {t} ".format(
            t=target.name
        )

        memories_primer = "\n\nMemories to consider in developing a theory of mind for {t}:\n".format(
            t=target.name
        )

        # Create a list containing a string that identifies the target person by name
        always_included = [
            "Target person: {t}".format(t=target.name),
            current_tom_primer,
            memories_primer,
            "in chronological order:\n",
            "in order from least to most relevant:\n",
        ]

        # Calculate the token count for the always included string, considering it as a user role prompt
        always_included_count = get_prompt_token_count(
            always_included,
            role="user",
            pad_reply=True,
            tokenizer=game.parser.tokenizer,
        )

        # Retrieve the current impression of the target character from the agent's memory
        target_impression = self._get_impression(target)

        # Initialize a list to hold relevant memories
        if target_impression:
            # If an impression exists, retrieve memory nodes from the last round that mention the target character's
            # name
            memory_ids = self.character.memory.get_observations_by_round(game.round)
            nodes = [self.character.memory.get_observation(m_id) for m_id in memory_ids]

            # Filter the nodes to create a context list based on the target character's keywords
            context_list = [
                node.node_description
                for node in nodes
                if target.name in node.node_keywords
            ]
            self.chronological = True  # Set the chronological flag to True
            target_impression_tkns = get_prompt_token_count(
                target_impression
            )  # Count tokens in the target impression
        else:
            # If no impression exists, retrieve the most relevant memories about the target character
            # TODO: Consider improving this query for better relevance
            context_list = Retrieve.retrieve(
                game=game,
                character=self.character,
                query=f"I want to remember everything I know about {target.name}",
                n=-1,
                memory_types=[
                    MemoryType.ACTION,
                    MemoryType.DIALOGUE,
                    MemoryType.REFLECTION,
                    MemoryType.PERCEPT,
                    
                ],
            )
            self.chronological = False  # Set the chronological flag to False
            target_impression_tkns = 0  # No tokens to count for the impression

        # Calculate the number of available tokens for the message based on the model's context limit
        available_tokens = get_token_remainder(
            self.gpt_handler.model_context_limit,
            consumed_tokens,
            target_impression_tkns,
            always_included_count,
            self.gpt_handler.max_output_tokens,  # Max output tokens set by user
        )

        # Limit the context list to fit within the available token count
        if context_list:
            context_list = limit_context_length(
                context_list,
                max_tokens=available_tokens,
                tokenizer=game.parser.tokenizer,
            )

        # Initialize the message with the always included string
        message = always_included[0]

        # If an impression exists, append the current theory of mind to the message
        if target_impression:
            message += always_included[1] + (always_included[3] if self.chronological else always_included[4]) + target_impression

        # If there are relevant memories, append them to the message
        if context_list:
            memory_str = context_list_to_string(context_list, sep="\n")
            message += "\n\nMemories to consider in developing a theory of mind for {t}:\n{m}".format(
                t=target.name, m=memory_str
            )

            message += always_included[2] + memory_str

        # Return the constructed message
        return message
