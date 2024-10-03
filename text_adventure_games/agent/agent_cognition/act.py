"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent_cognition/act.py
Description: defines how agents select an action given their perceptions and memory
"""

import contextlib

# Steps to choosing an action:
# 1. perceive environment (perceive) -- already put into memory
# 2. collect goals, world info, character relationships (retrieve)
# 3. get a list of the currently available actions (game.actions)
# 4. Ask GPT to pick an option
# 5. Parse and return

from typing import TYPE_CHECKING

# local imports
from text_adventure_games.gpt.gpt_helpers import (
    limit_context_length,
    get_prompt_token_count,
    get_token_remainder,
    context_list_to_string,
    GptCallHandler,
)
from text_adventure_games.utils.general import enumerate_dict_options, get_logger_extras
from .retrieve import retrieve
from text_adventure_games.assets.prompts import act_prompts as ap

if TYPE_CHECKING:
    from text_adventure_games.games import Game
    from text_adventure_games.things import Character


class Act:
    """
    Manages the actions of a character within a game environment.
    It facilitates the interaction between the character and the game by generating actions based on prompts.

    Args:
        game: The game instance that the character is part of.
        character: The character instance that will perform actions in the game.

    Methods:
        act: Generates and executes an action for the character based on the current game state.
        generate_action: Creates an action based on system and user prompts, handling token limits.
        build_messages: Constructs the system and user messages for the action generation.
        build_system_message: Creates the system prompt for agent actions and returns its token count.
        build_user_message: Forms the user message based on the game state and character context.
        get_user_token_limits: Calculates the token limits for impressions and memories based on available tokens.
    """

    def __init__(self, game, character):
        """
        Initializes an Act instance for managing character actions in a game.
        This constructor sets up the game context, character, and necessary configurations for action generation.

        Args:
            game: The game instance that the character is part of.
            character: The character instance that will perform actions in the game.
        """

        # Assign the provided game instance to the instance variable for later use.
        self.game = game

        # Assign the provided character instance to the instance variable for later use.
        self.character = character

        # Initialize the GPT handler by calling the setup method, which configures the model parameters.
        self.gpt_handler = self._set_up_gpt()

        # Initialize the token offset to zero, which will be used to manage token limits during action generation.
        self.token_offset = 0

        # Set the offset padding to 5, which will be used to adjust token calculations.
        self.offset_pad = 5

    def _set_up_gpt(self):
        """
        Configures and initializes the GPT handler with the necessary model parameters.
        This method sets up the API key, model type, and various settings to control the behavior of the GPT model.

        Returns:
            GptCallHandler: An instance of the GptCallHandler configured with the specified model parameters.
        """

        model_params = {
            "api_key_org": "Helicone",
            "model": "gpt-4",
            "max_tokens": 100,
            "temperature": 1,
            "top_p": 1,
            "max_retries": 5,
        }

        return GptCallHandler(**model_params)

    # def _log_action(self, game, character, message):
    #     extras = get_logger_extras(game, character)
    #     extras["type"] = "Act"
    #     game.logger.debug(msg=message, extra=extras)

    def act(self):
        """
        Generates and executes an action for the character based on the current game state.
        This method constructs the necessary prompts, retrieves the action to take, and outputs the chosen action.

        Returns:
            str: The action that the character has chosen to take in the game.
        """

        system_prompt, user_prompt = self.build_messages()

        # print("act system:", system_prompt, sep='\n')
        # print("-" * 50)
        # print("act user:", user_prompt, sep='\n')

        action_to_take = self.generate_action(system_prompt, user_prompt)

        # self._log_action(self.game, self.character, action_to_take)
        print(f"{self.character.name} chose to take action: {action_to_take}")
        return action_to_take

    def generate_action(self, system_prompt, user_prompt):
        """
        Generates an action for the character based on the provided system and user prompts.
        This method interacts with the GPT handler to produce a response and manages token limits to ensure valid
        requests.

        Args:
            system_prompt (str): The prompt that provides context for the action generation.
            user_prompt (str): The prompt that reflects the user's current situation and choices.

        Returns:
            str: The generated action for the character, or recursively calls the act method if token limits are
            exceeded.

        Raises:
            ValueError: If the response from the GPT handler is invalid or cannot be processed.
        """

        # Uncomment the following line to set up the OpenAI client with the specified organization.
        # client = set_up_openai_client("Helicone")

        # Generate a response from the GPT handler using the provided system and user prompts.
        response = self.gpt_handler.generate(system=system_prompt, user=user_prompt)

        # Check if the response is a tuple, indicating a potential error related to token limits.
        if isinstance(response, tuple):
            # Unpack the tuple to get the success status and the token difference.
            success, token_difference = response
            print(
                f"Action prompts exceeded token limit of model by {token_difference} tokens."
            )

            # Update the token offset to account for the exceeded limit and add padding for future calculations.
            self.token_offset = token_difference + self.offset_pad
            self.offset_pad += 2 * self.offset_pad

            # Recursively call the act method to attempt generating an action again with the updated token limits.
            return self.act(self.game, self.character)

        # Return the generated action if no errors occurred.
        return response

    def build_messages(self):
        """
        Constructs the system and user messages required for action generation.
        This method combines the system prompt with the user prompt, taking into account the current token usage.

        Returns:
            tuple: A tuple containing the system message and the user message.
        """

        # Build the system message and retrieve the token count associated with it.
        system_msg, sys_token_count = self.build_system_message()

        # Calculate the total number of tokens consumed by adding the system token count to the current token offset.
        consumed_tokens = sys_token_count + self.token_offset

        # Generate the user message based on the total consumed tokens.
        user_msg = self.build_user_message(consumed_tokens=consumed_tokens)

        # Return both the system message and the user message as a tuple.
        return system_msg, user_msg

    def build_system_message(self) -> str:
        """
        Constructs the system message that provides context for the character's actions.
        This method gathers relevant information about the character and the game, and formats it into a structured
        message.

        Returns:
            tuple: A tuple containing the system message as a string and the token count of the message.
        """

        # Initialize an empty string to build the system message.
        system = ""

        # Append the standard information of the character to the system message.
        system += self.character.get_standard_info(self.game)

        # Add predefined system message segments to provide context for actions.
        system += ap.action_system_mid
        system += ap.action_system_end

        # Retrieve the available game actions from the game parser.
        game_actions = self.game.parser.actions

        # Generate a string of action choices, using an inverted argument to reflect the game's action structure.
        choices_str, _ = enumerate_dict_options(
            game_actions, names_only=True, inverted=True
        )
        system += choices_str

        # Calculate the token count for the constructed system message to manage token limits.
        sys_token_count = get_prompt_token_count(
            content=system, role="system", pad_reply=False
        )

        # Return the complete system message and its token count as a tuple.
        return system, sys_token_count

    def build_user_message(self, consumed_tokens: int):
        """
        Constructs the user message that provides context for the character's actions.
        This method includes relevant memories, characters in view, and the game goal to guide the character's
        decision-making.

        Args:
            consumed_tokens (int): The number of tokens already consumed by previous messages.

        Returns:
            str: The constructed user message that incorporates memories, characters, and objectives.
        """

        # Check if the game has a method to retrieve the basic game goal and set the goal reminder accordingly.
        if hasattr(self.game, "get_basic_game_goal"):
            goal_reminder = self.game.get_basic_game_goal()
        else:
            # Default goal reminder if the game goal method is not available.
            goal_reminder = "Complete the objective of the game as quickly as you can."

        # Retrieve the characters currently in view for the user.
        chars_in_view = self.character.get_characters_in_view(self.game)

        # Prepare a list of always included information for the user message.
        always_included = [
            "\nThese are select MEMORIES in ORDER from LEAST to MOST RELEVANT:\n",
            f"In this location, you see: {', '.join([c.name for c in chars_in_view])}\n",
            ap.action_incentivize_exploration,
            goal_reminder,
            "Given the above information and others present here, what would you like to do?",
        ]

        # Calculate the token count for the always included content to manage token limits.
        always_included_tokens = get_prompt_token_count(
            content=always_included,
            role="user",
            pad_reply=True,
            tokenizer=self.game.parser.tokenizer,
        )

        # Determine the available tokens for the user based on the model's limits and consumed tokens.
        user_available_tokens = get_token_remainder(
            self.gpt_handler.model_context_limit,
            self.gpt_handler.max_tokens,
            consumed_tokens,
            always_included_tokens,
        )

        # Calculate the limits for impressions and memories based on the available tokens.
        imp_limit, mem_limit = self.get_user_token_limits(
            user_available_tokens, props=(0.33, 0.66)
        )

        # Initialize an empty string for user messages and a token count variable.
        user_messages = ""
        tok_count = 0

        # Attempt to retrieve impressions of characters in view and limit their token count.
        with contextlib.suppress(AttributeError):
            impressions = self.character.impressions.get_multiple_impressions(
                chars_in_view
            )
            impressions, tok_count = limit_context_length(
                history=impressions,
                max_tokens=imp_limit,
                tokenizer=self.game.parser.tokenizer,
                return_count=True,
            )
            # Append the impressions to the user message.
            user_messages += context_list_to_string(impressions)
        # Retrieve all relevant memories related to the current situation.
        memories_list = retrieve(
            self.game, self.character, query=None, n=40
        )  # Consider whether to limit the number of memories retrieved.

        # Calculate the remaining tokens available for memories.
        memory_available_tokens = get_token_remainder(user_available_tokens, tok_count)

        # Limit the memories list to the available token count.
        memories_list = limit_context_length(
            memories_list,
            max_tokens=memory_available_tokens,
            tokenizer=self.game.parser.tokenizer,
        )

        # Construct the final user message by adding always included content and memories.
        user_messages += always_included[0]
        user_messages += context_list_to_string(context=memories_list, sep="\n")
        user_messages += "\n".join(always_included[1:])

        # Return the constructed user message.
        return user_messages

    def get_user_token_limits(self, remainder, props):
        """
        Calculates the token limits for impressions and memories based on the available token remainder.
        This method uses specified ratios to determine how many tokens can be allocated to each category.

        Args:
            remainder (int): The total number of tokens available for use.
            props (tuple): A tuple containing the ratios for impressions and memories.

        Returns:
            tuple: A tuple containing the calculated token limits for impressions and memories.
        """

        # Unpack the provided ratios for impressions and memories from the props tuple.
        ratio_impressions, ratio_memories = props

        # Calculate the number of tokens available for impressions based on the total remainder and the impression
        # ratio.
        remaining_tokens_impressions = int(remainder * ratio_impressions)

        # Calculate the number of tokens available for memories based on the total remainder and the memory ratio.
        remaining_tokens_memories = int(remainder * ratio_memories)

        # Return the calculated token limits for impressions and memories as a tuple.
        return remaining_tokens_impressions, remaining_tokens_memories
