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
from text_adventure_games.gpt.gpt_helpers import (  # Import helper functions for GPT interactions
    limit_context_length,  # Function to limit the context length of prompts
    get_token_remainder,  # Function to get the remaining tokens available
    get_prompt_token_count,  # Function to count tokens in a prompt
    context_list_to_string,  # Function to convert a list of contexts to a string
    GptCallHandler,  # Class to handle GPT calls
)
from text_adventure_games.agent.agent_cognition import (
    retrieve,
)  # Import retrieve function for agent cognition
from text_adventure_games.utils.general import (
    get_logger_extras,
)  # Import utility for logging extras

# Import the get_models_config function from the consts module in the utils package.
# This function is used to retrieve the configuration for different models used in the game.
from text_adventure_games.utils.consts import get_models_config

if TYPE_CHECKING:  # Check if type checking is enabled
    from text_adventure_games.games import Game  # Import Game class for type hints
    from text_adventure_games.things import (
        Character,
    )  # Import Character class for type hints

IMPRESSION_MAX_OUTPUT = 512  # Define maximum output length for impressions


class Impressions:
    """
    Manages and generates impressions of target characters based on interactions and memories.
    This class allows for the creation, updating, and retrieval of impressions, utilizing a GPT model for generation.

    Attributes:
        impressions (defaultdict): A dictionary storing impressions keyed by target agent identifiers.
        name (str): The name of the agent.
        id (int): The unique identifier of the agent.
        last_target (Character): The last target character for which an impression was created.
        gpt_handler (GptCallHandler): Handler for GPT calls to generate impressions.
        token_offset (int): Offset for token calculations in GPT prompts.
        offset_pad (int): Additional padding for token limits.

    Args:
        name (str): The name of the agent.
        id (int): The unique identifier of the agent.
    """

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

        # Initialize a dictionary to store impressions, with default values as empty dictionaries
        self.impressions = defaultdict(dict)

        # Set the character associated with this impressions object
        self.character = character

        # Initialize the last target character to None
        self.last_target = None

        # Set up the GPT call handler for generating impressions
        self.gpt_handler = self._set_up_gpt()

        # Define the token offset for GPT prompts, accounting for variable tokens
        self.token_offset = 50

        # Set an additional padding for token calculations
        self.offset_pad = 5

    def _set_up_gpt(self):
        """
        Configures and initializes the GPT call handler with specified parameters.
        This method sets up the necessary model parameters for generating impressions using the GPT model.

        Returns:
            GptCallHandler: An instance of the GPT call handler configured with the specified parameters.
        """

        # Define the parameters for the GPT model configuration
        model_params = {
            "api_key_org": "Helicone",  # Organization API key for authentication
            "model": get_models_config()["impressions"][
                "model"
            ],  # Specify the GPT model version to use
            "max_tokens": IMPRESSION_MAX_OUTPUT,  # Maximum number of tokens for the output
            "temperature": 1,  # Sampling temperature for randomness in output
            "top_p": 1,  # Nucleus sampling parameter
            "max_retries": 5,  # Maximum number of retries for API calls
        }

        # Return an instance of the GptCallHandler initialized with the specified parameters
        return GptCallHandler(**model_params)

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

        # Retrieve the impression for the target character using a formatted key
        impression = self.impressions.get(f"{target.name}_{target.id}", None)

        # If an impression exists and only the string is requested, return the impression text
        if impression and str_only:
            return impression["impression"]

        # If an impression exists and the full data is requested, return the entire impression
        elif impression:
            return impression

        # If no impression exists, return None
        else:
            return None

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
            # Skip the current character to avoid self-impression
            if char is self.character:
                continue

            # Create a string that introduces the character's impression
            char_impression = (
                f"Your theory of mind of and relationship with {char.name}:\n"
            )

            # Append the impression text for the character, or "None" if no impression exists
            char_impression += self._get_impression(char) or "None\n"

            # Add the constructed impression string to the list
            char_impressions.append(char_impression)

        # Return the list of character impressions
        return char_impressions

    def update_impression(
        self, game: "Game", target: "Character"
    ) -> None:
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
        if should_update and total_ticks > game.max_ticks_per_round / 2:
            # Trigger the process to set a new impression for the target character
            self.set_impression(game, target)

    def set_impression(
        self, game: "Game", target: "Character"
    ) -> None:
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

        # Store the target character for which the impression is being set
        self.last_target = target

        # Generate the system and user prompts for the GPT model based on the game context and characters
        system, user = self.build_impression_prompts(game, target)

        # Retrieve the impression from the GPT model using the generated prompts
        impression = self.gpt_generate_impression(system, user, target)

        # Log the generated impression for debugging purposes (commented out)
        # print(f"{self.character.name} impression of {target.name}: {impression}")

        # Log the impression in the game's logger for record-keeping
        self._log_impression(game, target, impression)

        # Update the internal impressions dictionary with the new impression details
        self.impressions.update(
            {
                f"{target.name}_{target.id}": {
                    "impression": impression,  # The generated impression text
                    "round": game.round,  # The current round of the game
                    "tick": game.tick,  # The current tick of the game
                    "creation": game.total_ticks,  # The total ticks at the time of impression creation
                }
            }
        )

    def gpt_generate_impression(self, system_prompt, user_prompt) -> str:
        """
        Generates an impression of a target character using the GPT model based on provided prompts.
        This method handles potential errors related to token limits and adjusts the token offset accordingly.

        System prompt uses: world info, agent personal summary, and the target's name.
        User prompt uses: target's name, a list of memories about the target, and the existing impression of the target.

        Args:
            system_prompt (str): The system prompt containing contextual information for the GPT model.
            user_prompt (str): The user prompt that specifies the target character and relevant details.

        Returns:
            str: The generated impression of the target character, or triggers a re-evaluation if a token limit error
            occurs.
        """

        # Generate an impression using the GPT handler with the provided system and user prompts
        impression = self.gpt_handler.generate(system=system_prompt, user=user_prompt, character=self.character)

        # Check if the result is a tuple, indicating a potential error due to exceeding token limits
        if isinstance(impression, tuple):
            # Unpack the tuple to get the success status and the token difference
            success, token_difference = impression

            # Update the token offset to account for the token difference and add padding
            self.token_offset = token_difference + self.offset_pad

            # Increase the offset padding for future calculations
            self.offset_pad += 2 * self.offset_pad

            # Trigger a re-evaluation of the impression for the last target character
            return self.set_impression(self.game, self.last_target)

        # Return the generated impression if no errors occurred
        return impression

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
        system_prompt, sys_token_count = self.build_system_prompt(
            game, target.name
        )

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

        # Retrieve the standard information of the character, excluding perceptions
        system_prompt = self.character.get_standard_info(game, include_perceptions=False)

        # Append the formatted prompt for generating impressions, including the target character's name
        system_prompt += ip.gpt_impressions_prompt.format(target_name=target_name)

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

        # Create a list containing a string that identifies the target person by name
        always_included = ["Target person: {t}\n\n".format(t=target.name)]

        # Calculate the token count for the always included string, considering it as a user role prompt
        always_included_count = get_prompt_token_count(
            always_included[0],
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
            context_list = retrieve.retrieve(
                game,
                self.character,
                n=-1,
                query=f"I want to remember everything I know about {target.name}",
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
            ordering = (
                "in chronological order"
                if self.chronological
                else "in order from least to most relevant"
            )
            message += "Current theory of mind for {t} {o}:\n{i}\n\n".format(
                t=target.name, o=ordering, i=target_impression
            )

        # If there are relevant memories, append them to the message
        if context_list:
            memory_str = context_list_to_string(context_list, sep="\n")
            message += "Memories to consider in developing a theory of mind for {t}:\n{m}".format(
                t=target.name, m=memory_str
            )

        # Return the constructed message
        return message
