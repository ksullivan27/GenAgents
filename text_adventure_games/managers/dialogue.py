circular_import_prints = False

if circular_import_prints:
    print("Importing Dialogue Revised")

from typing import (
    TYPE_CHECKING,
    Union,
    List,
    Set,
    Literal,
    Tuple,
)  # Allows conditional imports for type hints

import tiktoken  # Importing the tiktoken library for tokenization tasks

import numpy as np

import heapq

import logging

import random

if circular_import_prints:
    print(f"{__name__} calling imports for GptHelpers")
from text_adventure_games.gpt.gpt_helpers import (  # Importing helper functions for GPT interactions
    limit_context_length,  # Function to limit the context length for GPT
    get_prompt_token_count,  # Function to count tokens in a prompt
    GptCallHandler,  # Class to handle GPT calls
)

if circular_import_prints:
    print(f"{__name__} calling imports for Dialogue Prompt")
from text_adventure_games.assets.prompts import (
    dialogue_prompt as dp,
)  # Importing the dialogue prompt for use in the game

if circular_import_prints:
    print(f"{__name__} calling imports for General")
from ..utils.general import (
    set_up_openai_client,
)  # Importing a utility function to set up the OpenAI client

if circular_import_prints:
    print(f"{__name__} calling imports for Retrieve")
from ..agent.agent_cognition.retrieve import (
    Retrieve,
)  # Importing a function to retrieve information for agent cognition

if circular_import_prints:
    print(f"{__name__} calling imports for MemoryType")
from ..agent.memory_stream import MemoryType

if circular_import_prints:
    print(f"{__name__} calling imports for MemoryStream")
from ..agent.memory_stream import MemoryStream

if TYPE_CHECKING:
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game

    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Character")
    from text_adventure_games.things.characters import Character

if circular_import_prints:
    print(f"\t{__name__} calling imports for Prompt Classes")
from ..assets.prompts import prompt_classes


ACTION_MAX_OUTPUT = 100  # Constant defining the maximum output length for actions


class Dialogue:
    """
    Manages the dialogue between two characters in a text adventure game.
    It facilitates interactions, updates character instructions, and handles GPT responses.

    Args:
        game (Game): The game instance managing the overall game state.
        conversation_initiator (Character): The character who started the conversation.
        participants (Set[Character]): Set of dialogue participants.
        command (str): The initial command or intent of the dialogue.
        dialogue_duration (Union[int, None]): The duration of the dialogue.

    Attributes:
        logger (Logger): Class-level attribute to store the logger.
        gpt_handler (GptCallHandler): Class-level attribute to store the shared GPT handler.
        model_params (dict): Parameters for the GPT model configuration.
    """

    logger = None  # Class-level attribute to store the logger
    gpt_handler = None  # Class-level attribute to store the shared GPT handler
    model_params = {
        "max_output_tokens": 3000,
        "temperature": 1,
        "top_p": 1,
        "max_retries": 5,
    }

    @classmethod
    def initialize_gpt_handler(cls) -> None:
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.

        Returns:
            None: This method does not return a value.
        """
        if circular_import_prints:
            print(f"-\tGoals Module is initializing GptCallHandler")

        # Initialize the GPT handler if it hasn't been set up yet
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(
                model_config_type="dialogue", **cls.model_params
            )

    def __init__(
        self,
        game: "Game",
        conversation_initiator: "Character",
        participants: Set["Character"],
        command: str,
        dialogue_duration: Union[int, None] = None,
        max_iterations: Union[int, None] = None,
    ) -> None:
        """
        Initializes a Dialogue instance for managing interactions between game characters. This constructor sets up the
        necessary attributes, including the game context, participants, and command, while also preparing the dialogue
        history and character instructions.

        Args:
            game (Game): The game context in which the dialogue occurs.
            conversation_initiator (Character): The character who started the conversation.
            participants (Set[Character]): A sorted list of characters participating in the dialogue, ordered by
            initiative.
            command (str): The initial command or intent that starts the dialogue.
            dialogue_duration (Union[int, None]): The duration of the dialogue.
            max_iterations (Union[int, None]): The maximum number of iterations for the dialogue.

        Returns:
            None: This method does not return a value.
        """
        if circular_import_prints:
            print(f"-\tInitializing Goals")

        # Initialize the GPT handler if it hasn't been set up yet
        Dialogue.initialize_gpt_handler()

        if Dialogue.logger is None:
            Dialogue.logger = logging.getLogger("dialogue")

        # Assign the game instance to the instance variable
        self.game = game

        # Initialize the end dialogue flag
        self.end_dialogue = False

        # Set padding for token limits
        self.offset_pad = 5

        # Store the maximum number of iterations provided for the action
        self.max_iterations = max_iterations

        # Define the maximum token limit for the model based on the GPT handler
        self.model_context_limit = Dialogue.gpt_handler.model_context_limit

        # Define the adjusted context limit (used to calculate memory and dialogue token limits)
        # By having a separate context limit, we can artificially limit the context to avoid bumping up against the
        # model's true context limit (if token counts are off for some reason)
        self.working_context_limit = self.model_context_limit

        # Store the participants in the dialogue
        self.participants = participants

        # Store the conversation initiator
        self.conversation_initiator = conversation_initiator

        # Initialize a dictionary to hold system prompt components for characters
        self.sys_prompt_components = {}

        # Initialize a dictionary to hold memory and dialogue token limits for characters
        self.token_limits = {}

        # Store the initial command for the dialogue
        self.command = command

        # Create a list of all characters mentioned in the conversation (useful in the future if getting impressions
        # only for mentioned characters)
        self.characters_mentioned = [
            character.name for character in self.participants
        ]  # Characters mentioned so far in the dialogue

        # Store the original dialogue duration
        self.dialogue_duration = float(dialogue_duration)

        # Store the remaining time for the dialogue
        self.remaining_time = float(dialogue_duration)

        # Store the characters who are ready to leave the dialogue
        self.ready_to_leave = set()

        # Get the initial command
        initial_command = (
            self.command
            + ("." if not self.command.endswith(".") else "")
            + " Included in the conversation are: "
            + ", ".join(self.characters_mentioned)
        )

        # Initialize the responses' keywords and embeddings with the initial command (used to retrieve memories)
        self.responses_keywords_embeddings = Retrieve.get_query_keywords_and_embeddings(
            game=self.game, query=initial_command
        )

        # Initialize the short-term dialogue memory token limit
        # TODO: Experiment with different short-term memory limits (e.g. 15 minutes)
        self.dialogue_short_term_memory_token_limit = np.min(
            [
                (200 * 4),  # tokens per minute x minutes of short-term memory
                self.working_context_limit,  # maximum token limit
            ]
        )

        # Initialize the memory-dialogue ratio (used to calculate memory and dialogue token limits)
        self.memory_dialogue_ratio = 1

        # dict[str, Union[OrderedDict[str, Tuple[Tuple(float, int), np.ndarray]], Dict[str, np.ndarray]]]
        # {
        #     "keywords": {
        #         keyword_type: {keyword: embedding},
        #         ...
        #     },
        #     "embeddings": {query: ((recency score, importance score), embedding)},
        # }
        # Modify the "embeddings" to map to tuples with the first index being a tuple representing the recency (larger
        # numbers mean more recent) and importance scores (larger numbers mean more important)
        # – set the initial command's importance scores to 10
        for query, embedding in self.responses_keywords_embeddings[
            "embeddings"
        ].items():
            self.responses_keywords_embeddings["embeddings"][query] = (
                (self.dialogue_duration - self.remaining_time, 10),
                embedding,
            )

        # Initialize the dialogue history with a starting message
        self.dialogue_history = [
            f"BEGINNING OF DIALOGUE: {self.conversation_initiator.name} wants to {self.command}{"." if not self.command.endswith(".") else ""}"
        ]

        # Initialize the dialogue history summaries
        self.dialogue_history_summaries = [
            (
                (self.dialogue_duration - self.remaining_time, 10),
                f"{self.conversation_initiator.name} started the dialogue to {self.command}{"." if not self.command.endswith(".") else ""}",
            )
        ]

        # Initialize the dialogue history messages dictionary
        self.dialogue_history_messages_dict = {
            character.name: {
                "messages": [
                    {
                        "role": "user",
                        "content": f"BEGINNING OF DIALOGUE: {self.conversation_initiator.name} wants to {self.command}{"." if not self.command.endswith(".") else ""}",
                    }
                ],
                "token_count": [
                    get_prompt_token_count(
                        content=f"BEGINNING OF DIALOGUE: {self.conversation_initiator.name} wants to {self.command}{"." if not self.command.endswith(".") else ""}",
                        role="user",
                        pad_reply=False,
                    )
                ],
            }
            for character in self.participants
        }

        # TODO: Set this to 10 minutes (currently 3 for testing)
        self.update_cognitive_functions_every = 3 * 200  # 3 minutes

        # Calculate the token count since the last cognitive functions update occurred (it happens every 10 minutes)
        self.token_count_since_last_cognitive_update = get_prompt_token_count(
            content=self.dialogue_history, role=None, pad_reply=False
        )

        # Create a dialogue queue to manage the order of dialogue turns
        self.dialogue_queue = DialogueQueue(game, self.participants, decay_rate=0.9)

        # Iterate over each participant to set up their instructions
        for participant in self.participants:
            # Initialize a dictionary for the character's system prompt components using their name as the key
            self.sys_prompt_components[participant.name] = dict()

            # Initialize a dictionary for the character's memory and dialogue token limits
            self.token_limits[participant.name] = {"memory": None, "dialogue": None}

            # Update the character's prompts – intro and memories
            self.update_sys_message_components(
                character=participant, update_intro=True, update_memories=True
            )

    ###################### UPDATE FUNCTIONS: memories, system instructions, and dialogue history #######################

    def update_intro(
        self, character: "Character", forced_to_speak: bool = False
    ) -> None:
        """
        Update the intro for a specified character – who they are; their goals, perceptions, and impressions; and the
        overall dialogue response format instructions.

        This function updates the intro for a character by combining their
        standard information (plus goals, perceptions, and impressions) with dialogue instructions
        based on other participants in the game. It also calculates the token count for the system
        prompt and updates the character's information in the system dictionary.

        Args:
            character (Character): The character for whom the intro is being updated.
            forced_to_speak (bool): Whether the character is forced to speak (impacts the dialogue instructions)

        Returns:
            None: This function updates the character's intro in place and does not return a value.
        """
        ### REQUIRED START TO SYSTEM PROMPT (CAN'T TRIM) ###
        # Retrieve the standard information for the specified character
        intro = character.get_standard_info(
            game=self.game,
            include_goals=True,
            include_perceptions=True,
            include_impressions=True,
        )  # TODO: In the future, modify to only get impressions of mentioned characters. In the meantime, this is fine
        # Since all characters are participants in the boardroom dialogue

        # Construct a string of dialogue instructions by joining the names of other participants
        other_characters_list = [
            x.name for x in self.participants if x.name != character.name
        ]
        if len(other_characters_list) == 2:
            other_characters = " and ".join(other_characters_list)
        elif len(other_characters_list) > 2:
            other_characters = (
                ", ".join(other_characters_list[:-1])
                + ", and "
                + other_characters_list[-1]
            )
        else:
            other_characters = ""
            print(
                "DEBUG: There must be at least two unique participants in the dialogue."
            )
            # # Commenting this out in case players (validly) leaving causes an error:
            # ValueError("There must be at least two unique participants in the dialogue.")

        # Append the formatted dialogue system prompt to the intro
        intro += (
            "\n\nCURRENT TASK:\n"
            + dp.gpt_dialogue_system_prompt_intro.format(
                other_characters=other_characters,
                conversation_initiator=self.conversation_initiator.name,
                max_score=GptCallHandler.max_importance_score,
            )
            + (dp.gpt_speak_or_listen_prompt if not forced_to_speak else "")
            + dp.gpt_dialogue_system_prompt_outputs.format(
                max_score=GptCallHandler.max_importance_score
            )
            + (
                dp.gpt_end_dialogue_prompt.format(
                    conversation_initiator=self.conversation_initiator.name
                )
                if self.remaining_time <= 0
                and character.name == self.conversation_initiator.name
                else ""
            )
            + dp.gpt_prompt_conclusion
        )

        # Calculate the token count for the constructed system prompt intro
        intro_token_count = get_prompt_token_count(
            content=intro, role="system", pad_reply=True
        )

        # Update the character's intro information in the system dictionary with the token count and intro string
        self.sys_prompt_components[character.name]["intro"] = (
            intro,
            intro_token_count + self.offset_pad,
        )

    def update_memories(self, character: "Character") -> None:
        """
        Update the memories for a specified character.

        This function retrieves the most relevant memories for the character and updates
        the character's "memories" in the character system dictionary.

        Args:
            character (Character): The character whose memories are being updated.

        Returns:
            None: This function updates the character's memory data in place (in the system's character data) and does
                  not return a value.
        """
        # Always include the memories prompt in the memories retrieval
        always_included = ["\n\nMEMORIES (in chronological order):"]

        # Calculate the token count for the always included string
        always_included_tokens = get_prompt_token_count(
            always_included, role=None, pad_reply=False
        )

        # Calculate the token count for the used tokens
        available_tokens = (
            self.token_limits[character.name]["memory"]
            - always_included_tokens
            - 10  # tokens for padding
        )

        # Retrieve the most relevant memories for the character based on the previous responses
        if memories := Retrieve.retrieve(
            game=self.game,
            character=character,
            query=self.responses_keywords_embeddings,
            sort_nodes=False,
            weighted=True,
            max_tokens=available_tokens,
            prepend="\n- ",
            memory_types=[
                MemoryType.ACTION,
                MemoryType.REFLECTION,
                MemoryType.PERCEPT,
            ],
        ):

            # Prepend the always included string to the context list
            memories = always_included + memories

            # Convert the list of memories into a single formatted string
            memories_str = "".join(memories)

            # Calculate the token count for the memories string
            memories_token_count = get_prompt_token_count(
                content=memories, role=None, pad_reply=False
            )

            # Update the character's memories in the system's character data
            self.sys_prompt_components[character.name]["memories"] = (
                memories_str,
                memories_token_count,
            )
        else:
            # If no memories are found, set an empty string
            self.sys_prompt_components[character.name]["memories"] = ("", 0)

    def update_sys_message_components(
        self,
        character: "Character",
        update_intro: bool = True,
        update_memories: bool = True,
        forced_to_speak: bool = False,
    ) -> None:
        """
        Update the system message components for a specified character.

        This method updates the character's introductory information, calculates the available tokens for memory
        and dialogue – validating the context weights, and updates the character's memories based on the available
        tokens.

        Args:
            character (Character): The character whose system instruction is to be updated.
            update_intro (bool): Whether to update the introductory information.
            update_memories (bool): Whether to update the memories.
            forced_to_speak (bool): Whether the character is forced to speak (impacts the dialogue instructions)

        Returns:
            None: This method does not return a value. It modifies the internal state of the system instructions and
            memories for the specified character.

        Raises:
            ValueError: If the available tokens are insufficient for the required operations.
        """
        # Update the intro
        if update_intro:
            self.update_intro(character=character, forced_to_speak=forced_to_speak)

            # Calculate the available tokens
            available_tokens = (
                self.working_context_limit
                - self.sys_prompt_components[character.name]["intro"][
                    1
                ]  # includes the response format token count
                - Dialogue.gpt_handler.max_output_tokens
            )

            # Calculate the memory and dialogue token limits (based on the available tokens and memory-dialogue ratio)
            memory_token_limit, dialogue_token_limit = (
                self.calculate_memory_and_dialogue_token_limits(available_tokens)
            )

            # Validate the context weights (adjusting if necessary)
            memory_token_limit, dialogue_token_limit = self.validate_context_weights(
                character=character,
                memory_token_limit=memory_token_limit,
                dialogue_token_limit=dialogue_token_limit,
                available_tokens=available_tokens,
            )

            # Update the character's memory token limit
            self.token_limits[character.name]["memory"] = memory_token_limit

            # Update the character's dialogue token limit
            self.token_limits[character.name]["dialogue"] = dialogue_token_limit

        # Update the memories
        if update_memories:
            self.update_memories(character=character)

    def update_dialogue_history(
        self, speaker: "Character", response_dict: dict
    ) -> None:
        """
        Updates the dialogue history with the latest response from a speaker.

        This method processes the response from the specified speaker, updating:
            - the dialogue history
            - the dialogue history summaries
            - the dialogue history messages dictionaries
            - the speaker's memory (with their response)

        Args:
            speaker (Character): The character object who is speaking.
            response_dict (dict): A dictionary containing the response details, including:
                - response (str): The full response from the speaker.
                - response_summary (str): A summary of the response.
                - response_splits (list[ResponseComponent]): A list of response components to be processed individually.

        Returns:
            None: This method does not return a value. It modifies the internal state of the dialogue history and memory
            structures.
        """
        # Initialize the maximum memory importance score
        # The highest importance score from the current response is used to score the importance of the overall summary
        # of the response, which is a separate text from the response itself
        max_memory_importance_score = 0

        # For each response split
        for response_split in response_dict["response_splits"]:

            # # TODO: Modify summarize_and_score_action to score multiple separate texts at once, and use this instead
            # # since it's more impartial (the system prompt is designed to be impartial)
            # summary, importance, ref_kwds = self.game.parser.summarize_and_score_action(
            #     description=response_split["component"],
            #     thing=self.character,
            #     needs_summary=False,
            #     needs_score=False,
            # )

            # Update the maximum memory importance score
            max_memory_importance_score = max(
                max_memory_importance_score, response_split["importance_score"]
            )

            # Retrieve the query keywords and embeddings for the response component
            query_keywords_embeddings = Retrieve.get_query_keywords_and_embeddings(
                game=self.game, query=response_split["component"]
            )

            # Combine the 'keywords' dictionaries
            for keyword_type, keywords_dict in query_keywords_embeddings[
                "keywords"
            ].items():
                if keyword_type not in self.responses_keywords_embeddings["keywords"]:
                    self.responses_keywords_embeddings["keywords"][keyword_type] = {}
                self.responses_keywords_embeddings["keywords"][keyword_type].update(
                    keywords_dict
                )

            # Combine the 'embeddings' default dictionaries
            for query, embedding in query_keywords_embeddings["embeddings"].items():
                self.responses_keywords_embeddings["embeddings"][query] = (
                    (
                        self.dialogue_duration - self.remaining_time,
                        response_split["importance_score"],
                    ),
                    embedding,
                )

            # TODO: Instead of repeatedly storing the response components in the query_keywords_embeddings,
            # dialogue_history_messages_dict, and memories, try having the query_keywords_embeddings and
            # dialogue_history_messages_dict values map to their respective memories directly (remember, add_memory
            # returns the assigned memory node id).

            # Add the response to the character's memories
            speaker.memory.add_memory(
                round=self.game.round,
                tick=self.game.tick,
                description=response_split["component"],
                keywords=query_keywords_embeddings["keywords"],
                location=speaker.location.name,
                success_status=True,
                memory_importance=response_split["importance_score"],
                memory_type=MemoryType.RESPONSE.value,
                actor_id=speaker.id,
            )

        # Add the response to the dialogue history
        self.dialogue_history.append(f"{speaker.name}: {response_dict['response']}")

        # Add the response summary to the dialogue history summaries
        self.dialogue_history_summaries.append(
            (
                (
                    self.dialogue_duration - self.remaining_time,
                    max_memory_importance_score,
                ),
                response_dict["response_summary"],
            )
        )

        # Add the response and token counts to the players' dialogue history messages dictionaries
        for character in self.participants:
            if character.name != speaker.name:
                self.dialogue_history_messages_dict[character.name]["messages"].append(
                    {
                        "role": "user",
                        "content": f"{speaker.name}: {response_dict['response']}",
                    }
                )
                self.dialogue_history_messages_dict[character.name][
                    "token_count"
                ].append(
                    get_prompt_token_count(
                        content=f"{speaker.name}: {response_dict['response']}",
                        role="user",
                        pad_reply=False,
                    )
                )
            else:
                self.dialogue_history_messages_dict[character.name]["messages"].append(
                    {
                        "role": "assistant",
                        "content": response_dict["response"],
                    }
                )
                self.dialogue_history_messages_dict[character.name][
                    "token_count"
                ].append(
                    get_prompt_token_count(
                        content=response_dict["response"],
                        role="assistant",
                        pad_reply=False,
                    )
                )

            # Add the response summary to the character's memories
            character.memory.add_memory(
                round=self.game.round,
                tick=self.game.tick,
                description=response_dict["response_summary"],
                keywords=query_keywords_embeddings["keywords"],
                location=speaker.location.name,
                success_status=True,
                memory_importance=max_memory_importance_score,
                memory_type=MemoryType.DIALOGUE.value,
                actor_id=speaker.id,
            )

    ###################### GET FUNCTIONS: dialogue history, system message, and dialogue messages ######################

    def get_short_and_long_term_dialogue_history(
        self, character: "Character"
    ) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]] | None]:
        """
        Retrieve both dialogue short-term (verbatim) and long-term (summarized) histories for a specified character,
        with the former containing the most recent responses and the latter encompassing older responses.

        This method calculates the cumulative token count of the dialogue history messages
        in reverse order and identifies the oldest message still within the dialogue short-term memory
        token limit. If no messages exceed the limit, the entire dialogue history for the character
        is returned with None for the summaries. If some messages exceed the limit, the non-exceeding
        dialogue history messages and their token counts are returned along with summaries (for the exceeding
        messages) and their token counts.

        Args:
            character (Character): The character whose dialogue history is being summarized.

        Returns:
            tuple[tuple[list[str], list[int]], tuple[list[str], list[int]] | None]: A tuple of tuples. The first
            tuple contains a list of dialogue short-term memory messages and their token counts. The second tuple
            contains a list of dialogue long-term memory summaries and their token counts. If the dialogue history
            messages exceed the short-term memory token limit, then the second tuple is None.
        """
        # Check that the dialogue history messages, summaries, and token counts are the same size
        if len(self.dialogue_history_messages_dict[character.name]["messages"]) != len(
            self.dialogue_history_messages_dict[character.name]["token_count"]
        ) or len(
            self.dialogue_history_messages_dict[character.name]["messages"]
        ) != len(
            self.dialogue_history_summaries
        ):
            raise ValueError(
                "Dialogue history messages, summaries, and token counts must be the same size."
            )

        # Reverse the dialogue history token counts
        reversed_counts = np.array(
            self.dialogue_history_messages_dict[character.name]["token_count"]
        )[::-1]

        # Calculate the cumulative sum
        cumulative_sum = np.cumsum(reversed_counts)

        # Find the index of the oldest message where the cumulative sum is less than or equal to the limit
        valid_indices = np.where(
            cumulative_sum <= self.dialogue_short_term_memory_token_limit
        )[0]

        # If no valid indices exist, return the entire dialogue history and None for the summaries
        if valid_indices.size == 0:
            return (
                (
                    self.dialogue_history_messages_dict[character.name]["messages"],
                    cumulative_sum[-1],
                ),
                None,
            )

        # Get the oldest (leftmost after reversing) valid index that is within the limit
        oldest_index = valid_indices[-1]

        # Get the reduced dialogue history messages and summaries
        reduced_dialogue_messages = self.dialogue_history_messages_dict[character.name][
            "messages"
        ][-oldest_index:]
        reduced_dialogue_summaries = [
            x[1] for x in self.dialogue_history_summaries[0 : -oldest_index - 1]
        ]

        # Calculate the token counts for the reduced dialogue history messages and summaries
        reduced_dialogue_messages_tokens = cumulative_sum[oldest_index]
        reduced_dialogue_summaries_tokens = get_prompt_token_count(
            content=reduced_dialogue_summaries, role=None, pad_reply=False
        )

        # Return a tuple. The first index is a tuple of a sublist of dialogue history messages and their token counts.
        # The second tuple contains a sublist of dialogue history summaries and their token counts.
        return (
            (reduced_dialogue_messages, reduced_dialogue_messages_tokens),
            (reduced_dialogue_summaries, reduced_dialogue_summaries_tokens),
        )

    def get_sys_message(self, character: "Character") -> tuple[list[dict], int]:
        """
        Retrieve the system message for a specified character.

        This function accesses the character's system information and returns a tuple containing the system message and
        the system message token count.

        Args:
            character (Character): The character for which to retrieve the system instruction.

        Returns:
            tuple[list[dict], int]: A tuple where the first element is the system message and the second element is the
            system message token count.
        """
        # Get this character's dictionary of system prompt components
        char_sys_prompt_comps = self.sys_prompt_components[character.name]

        # Construct the system message
        system_message = [
            {
                "role": "system",
                "content": char_sys_prompt_comps["intro"][0]
                + char_sys_prompt_comps["memories"][0],
            }
        ]

        # Calculate the token count for the system message
        system_message_token_count = (
            char_sys_prompt_comps["intro"][1]
            + char_sys_prompt_comps["memories"][1]
            + self.offset_pad
        )

        # Return a tuple containing the system message and the system message token count
        return (system_message, system_message_token_count)

    def get_dialogue_messages(
        self,
        character: "Character",
    ) -> tuple[list[dict], int]:
        """
        Retrieve the dialogue messages for a specified character.

        This function constructs the dialogue messages including an initial summary 'user' message (long-term memory)
        for older dialogue responses (if any) as well as 'user' and 'assistant' messages (short-term memory) for the
        more recent dialogue responses. Both are constructed to ensure they fit within the model's allotted context
        limits.

        Args:
            character (Character): The character whose dialogue messages are being retrieved.

        Returns:
            tuple[list[dict], int]: A tuple containing the combined messages and their token count.
        """
        minutes = int(self.remaining_time)  # Convert remaining time to integer minutes
        seconds = int(
            (self.remaining_time - minutes) * 60
        )  # Calculate remaining seconds

        if minutes < 0 or seconds < 0:
            time = f"TIME REMAINING IN MEETING: The meeting has exceeded its scheduled time by {abs(minutes)} minutes and {abs(seconds):02d} seconds.\n\n"
        else:
            time = f"TIME REMAINING IN MEETING: {minutes} minutes and {seconds:02d} seconds\n\n"

        # Always include the summary of older dialogue history
        always_included = [
            "SUMMARY OF OLDER DIALOGUE HISTORY:",
            time,
            "THE CONVERSATION CONTINUES, WITH THE MOST RECENT RESPONSES BELOW:",
        ]

        # Calculate the token count for the always included string
        always_included_tokens = get_prompt_token_count(
            always_included, role=None, pad_reply=False
        )

        # # Calculate the available tokens
        # available_tokens = (
        #     self.working_context_limit
        #     - self.characters_system[character.name]["intro"][
        #         1
        #     ]  # includes the response format token count
        #     - Dialogue.gpt_handler.max_output_tokens
        #     - always_included_tokens
        # )

        # Get the messages and summaries of the dialogue history for the character
        (dialogue_messages, dialogue_messages_tokens), (
            dialogue_summaries,
            dialogue_summaries_tokens,
        ) = self.get_short_and_long_term_dialogue_history(character=character)

        # Calculate the token count for the dialogue messages
        dialogue_messages_token_count = np.sum(
            [
                get_prompt_token_count(
                    content=message["content"], role=message["role"], pad_reply=False
                )
                + self.offset_pad
                for message in dialogue_messages
            ]
        )

        # If there are dialogue summaries
        if dialogue_summaries:
            # limit the context length of the dialogue summaries (if necessary)
            dialogue_summaries = limit_context_length(
                history=["\n- " + x for x in dialogue_summaries],
                max_tokens=self.token_limits[character.name]["dialogue"]
                - dialogue_messages_token_count
                - always_included_tokens
                - self.offset_pad,
            )

        # Join the limited dialogue summaries into a single string
        dialogue_str = (
            (
                always_included[0] + "".join(dialogue_summaries) + "\n\n"
                if dialogue_summaries
                else ""
            )
            + always_included[1]
            + always_included[2]
        )

        # Insert into the dialogue messages a new user dialogue message containing the dialogue summaries
        # and a description explaining that the conversation continues
        dialogue_messages.insert(0, {"role": "user", "content": dialogue_str})

        # Return a tuple containing the combined messages and their token count
        return (dialogue_messages, dialogue_messages_token_count)

    ############################################## GPT RESPONSE FUNCTION ###############################################

    def get_gpt_response(
        self, character: "Character", forced_dialogue: bool = False
    ) -> dict:
        """
        Generate a response from the GPT model for a specified character.

        This function retrieves the context messages – the system and dialogue (user and assistant) messages – checks if
        the combined token count exceeds the model's context limit, and updates the character's token limits if
        necessary. It then generates a response from the GPT model and handles any potential errors related to token
        limits.

        Args:
            character (Character): The character for whom the GPT response is being generated.
            forced_dialogue (bool, optional): Whether the dialogue is forced (they must speak). Defaults to False.

        Returns:
            dict: The generated response from the GPT model, or None if an error occurs.
        """
        # Get the system message and its token count
        system_messages, system_messages_token_count = self.get_sys_message(
            character=character,
        )

        # Get the dialogue messages and their token count
        dialogue_messages, dialogue_messages_token_count = self.get_dialogue_messages(
            character=character
        )

        # Combine the system messages and dialogue messages
        combined_messages = system_messages + dialogue_messages

        # if the sum of the system message and dialogue messages token counts exceeds the max tokens
        if (
            system_messages_token_count + dialogue_messages_token_count
            >= self.model_context_limit
        ):
            self.handle_token_limit_error(
                difference=system_messages_token_count
                + dialogue_messages_token_count
                - self.model_context_limit,
                character=character,
                forced_dialogue=forced_dialogue,
                system_messages_token_count=system_messages_token_count,
                dialogue_messages_token_count=dialogue_messages_token_count,
            )

        # get GPT's response
        response = Dialogue.gpt_handler.generate(
            messages=combined_messages,
            character=character,
            game=self.game,
            response_format=(
                (
                    prompt_classes.DialogueInitiator
                    if character == self.conversation_initiator and self.remaining_time < 0
                    else prompt_classes.Dialogue
                )
                if not forced_dialogue
                else (
                    prompt_classes.ForcedDialogueInitiator
                    if character == self.conversation_initiator and self.remaining_time < 0
                    else prompt_classes.ForcedDialogue
                )
            ),
        )

        # TODO: Implement the self.offset_pad for each message?
        if isinstance(response, tuple):
            # This occurs when there was a Bad Request Error cause for exceeding token limit
            success, token_difference = response
            self.handle_token_limit_error(
                token_difference,
                character=character,
                forced_dialogue=forced_dialogue,
                system_messages_token_count=system_messages_token_count,
                dialogue_messages_token_count=dialogue_messages_token_count,
            )

        # Convert the parsed goals to a dictionary
        response_dict = {}
        response_dict["respond_bool"] = not (
            hasattr(response, "speak_or_listen")
            and response.speak_or_listen == "listen"
        )
        response_dict["response"] = response.response
        response_dict["response_summary"] = response.response_summary
        response_dict["response_splits"] = (
            [
                {
                    "component": response.response_splits[i].component,
                    "importance_score": response.response_splits[i].importance_score,
                }
                for i in range(len(response.response_splits))
            ]
            if response.response_splits
            else []
        )

        # Subtract the token count of the response from the remaining time
        self.remaining_time -= (1 / 200) * get_prompt_token_count(
            content=response.response, role=None, pad_reply=False
        )

        # If the allotted time is up and the character is ready to leave, add them to the ready to leave set
        if self.remaining_time <= 0 and response.leave_dialogue:
            self.ready_to_leave.add(character)

        # If the conversation is not time-constrained and the character is ready to leave, remove them from the
        # participants
        elif self.remaining_time is None and response.leave_dialogue:
            self.participants.remove(character)

        # If the character is the conversation initiator and the response indicates the dialogue should end, set the end
        # dialogue flag
        if (
            character == self.conversation_initiator
            and (response.end_dialogue if hasattr(response, "end_dialogue") else False)
            and self.remaining_time <= 0
        ):
            self.end_dialogue = True

        # print(f"\n{'*' * 25} DIALOGUE SYSTEM INSTRUCTION {'*' * 25}\n")
        # print(self.game.parser.wrap_text(system_instruction_str))

        # print(f"\n{'*' * 25} DIALOGUE USER INSTRUCTION {'*' * 25}\n")
        # print(self.game.parser.wrap_text(user_instruction_str))

        # print(f"\n{'*' * 25} DIALOGUE RESPONSE {'*' * 25}\n")
        # print(self.game.parser.wrap_text(response))

        return response_dict

    ############################################## DIALOGUE LOOP FUNCTIONS #############################################

    def is_dialogue_over(self) -> bool:
        """
        Determine if the dialogue has concluded.

        This function checks the number of participants in the dialogue (excluding those who are ready to leave).
        If there is one or no participant, it indicates that the dialogue is over.

        Args:
            None

        Returns:
            bool: True if the dialogue is over, otherwise False.
        """
        return (
            len(self.participants.difference(self.ready_to_leave)) <= 0
            or self.end_dialogue
            or self.remaining_time <= -10
        )

    def dialogue_loop(self) -> list:
        """
        Manage the interactive dialogue loop among participants.

        This function initiates a dialogue session, allowing each participant to respond
        in turn while monitoring for new characters mentioned and potential conversation
        termination conditions. It continues the dialogue until a specified number of
        iterations is reached or if the conversation ends prematurely.

        Returns:
            list: The complete dialogue history recorded during the session.
        """
        if circular_import_prints:
            print(f"{__name__} calling imports for Parser")
        from ..parsing import Parser

        # Initialize a counter to limit the duration of the dialogue loop
        i = (
            self.max_iterations if self.max_iterations is not None else np.inf
        )  # Counter to avoid dialogue dragging on for too long

        print(
            f"\n{'~'*((120 - len(' DIALOGUE LOOP STARTED ')) // 2)} DIALOGUE LOOP STARTED {'~'*((120 - len(' DIALOGUE LOOP STARTED ')) // 2)}"
        )

        speaker = self.conversation_initiator
        response = -999
        forced_to_speak = True
        response_dict = {}

        # Begin the dialogue loop, allowing for a maximum of max_iterations iterations (unless None)
        while i > 0:
            # print("\nRESPONSE DICT:", response_dict)
            # print("RESPONSE:", response)
            if response != -999:
                # Get the next speaker in the dialogue
                speaker, forced_to_speak = self.dialogue_queue.get_next_speaker(
                    speaker=speaker, last_response=response_dict["response_splits"]
                )

                print("New Speaker:", speaker.name)
                print(f"Forced to speak: {forced_to_speak}")

            # # Retrieve the last line of dialogue to analyze for new character mentions
            # last_line = self.dialogue_history[-1]
            # # Extract keywords from the last line, specifically looking for character names
            # keywords = self.game.parser.extract_keywords(last_line).get(
            #     "characters", None
            # )

            # Update the memories of the current speaker
            self.update_sys_message_components(
                character=speaker,
                update_intro=True,
                update_memories=True,
                forced_to_speak=forced_to_speak,
            )

            # old_response_dict = response_dict

            # Get the response from the GPT model for the current character
            response_dict = self.get_gpt_response(
                speaker, forced_dialogue=forced_to_speak
            )

            if response_dict["respond_bool"]:
                # Format the response to include the speaker's name
                response = f"{speaker.name}: {response_dict['response']}"
                print(
                    f"\n- {Parser.wrap_text(response)}"
                )  # Print the speaker's response
                self.update_dialogue_history(
                    speaker=speaker, response_dict=response_dict
                )  # Add the response to the dialogue history
                self.game.save_dialogue_data(response)

            else:
                print("CHOSE NOT TO SPEAK")
                response_dict["response_splits"] = None
                # response_dict = old_response_dict
                continue

            # Update the token count since the last cognitive functions update
            self.token_count_since_last_cognitive_update += get_prompt_token_count(
                content=response, role=None, pad_reply=False
            )

            if (
                self.token_count_since_last_cognitive_update
                >= self.update_cognitive_functions_every
            ):

                print("\n~ UPDATING COGNITIVE FUNCTIONS ~")
                self.game.update_cognitive_functions(
                    update_round=True,
                    update_impressions=True,
                    evaluate_goals=True,
                    update_reflections=True,
                    update_perceptions=False,
                    update_goals=True,
                )

                self.game.tick = 0
                self.token_count_since_last_cognitive_update = 0

            # Check if the dialogue is over
            if self.is_dialogue_over():
                print(
                    f"\n{'~'*((120 - len(' THE DIALOGUE IS OVER ')) // 2)} THE DIALOGUE IS OVER {'~'*((120 - len(' THE DIALOGUE IS OVER ')) // 2)}"
                )
                break  # Exit the while loop if the dialogue is over

            if self.max_iterations is not None:
                i -= 1  # Decrement the counter to eventually end the loop
                print(f"Decrementing the counter to {i}")

        print(
            f"\n{'~'*((120 - len(' DIALOGUE LOOP ENDED ')) // 2)} DIALOGUE LOOP ENDED {'~'*((120 - len(' DIALOGUE LOOP ENDED ')) // 2)}"
        )

        # Return the complete dialogue history recorded during the session
        return self.dialogue_history

    ##################################### TOKEN LIMIT ERROR and VALIDATION HANDLING ####################################

    def handle_token_limit_error(
        self,
        difference: int,
        character: "Character",
        forced_dialogue: bool,
        system_messages_token_count: int,
        dialogue_messages_token_count: int,
    ) -> dict:
        """
        Handle token limit errors by adjusting the offset pad or working context limit based on the token count
        difference.

        This function checks if the difference in token count is less than 200 tokens. If it is, it increases the offset
        pad. If the difference is greater than or equal to 200 tokens, it reduces the working context limit by 3%. It
        also logs the adjustments made.

        Args:
            difference (int): The difference in token count that triggered the error handling.

        Returns:
            dict: The response dictionary obtained from the GPT model after handling the token limit error.
        """
        # If the adjusted context limit is less than 200 tokens (not missing by much)
        if difference < 200:
            # Double the offset pad
            self.offset_pad += 2 * self.offset_pad

            # Log the error
            Dialogue.logger.warning(
                "Token count exceeds the model context limit. System messages token count: "
                + str(system_messages_token_count)
                + ". Dialogue messages token count: "
                + str(dialogue_messages_token_count)
                + f". Doubling the offset pad to {self.offset_pad}."
            )
        else:
            # reduce the working context limit by 3%
            self.working_context_limit *= 0.97

            # Log the error
            Dialogue.logger.warning(
                "Token count exceeds the model context limit. System messages token count: "
                + str(system_messages_token_count)
                + ". Dialogue messages token count: "
                + str(dialogue_messages_token_count)
                + f". Reducing the working context limit to {self.working_context_limit}."
            )

        return self.get_gpt_response(
            character=character, forced_dialogue=forced_dialogue
        )

    def calculate_memory_and_dialogue_token_limits(
        self, available_tokens: int
    ) -> tuple[int, int]:
        """
        Calculates the memory and dialogue token limits based on the available tokens and memory-dialogue ratio.

        Args:
            available_tokens (int): The total number of available tokens to allocate between memory and dialogue.

        Returns:
            Tuple[int, int]: A tuple containing the calculated memory token limit and dialogue token limit.
        """

        # print("CALCULATING MEMORY AND DIALOGUE TOKEN LIMITS")
        # print(f"Available tokens: {available_tokens}")
        # print(f"Memory-dialogue ratio: {self.memory_dialogue_ratio}")

        # Calculate the memory and dialogue token limits using the memory-dialogue ratio
        memory_token_limit = available_tokens * (
            self.memory_dialogue_ratio / (1 + self.memory_dialogue_ratio)
        )
        dialogue_token_limit = available_tokens * (1 / (1 + self.memory_dialogue_ratio))

        return memory_token_limit, dialogue_token_limit

    def validate_context_weights(
        self,
        character: "Character",
        memory_token_limit: int,
        dialogue_token_limit: int,
        available_tokens: int,
    ) -> tuple[int, int]:
        """
        Validates the context weights (adjusting if necessary) based on the available tokens and token limits for
        memories and dialogue history. Also adjusts the short-term memory token limit if necessary.

        At minimum, this method ensures that there are enough tokens allocated:
        - 1000 tokens (750 words) for dialogue short-term messages
        - 1000 tokens (750 words) for dialogue long-term summaries
        - 1000 tokens (750 words) for memories

        Args:
            character (Character): The character for whom the token limits are being validated.
            memory_token_limit (int): The available token limit for memories.
            dialogue_token_limit (int): The available token limit for dialogue history.
            available_tokens (int): The total available tokens.

        Returns:
            tuple[int, int]: The updated memory_token_limit and dialogue_token_limit.
        """

        # print("\nVALIDATING CONTEXT WEIGHTS")
        # print(f"Character: {character.name}")
        # print(f"Memory token limit: {memory_token_limit}")
        # print(f"Dialogue token limit: {dialogue_token_limit}")
        # print(f"Available tokens: {available_tokens}")

        ### MINIMUM TOKEN LIMITS ###

        # Scale the minimum token limits by the GPT response format scaler (conservative estimate)
        gpt_response_format_scaler = 1.25

        # The effective minimum token limits for the dialogue short-term and long-term memories
        min_dialogue_st_token_limit = 1000 * gpt_response_format_scaler
        min_dialogue_lt_token_limit = 1000 * gpt_response_format_scaler

        # The effective minimum token limit for the dialogue history
        min_dialogue_token_limit = (
            min_dialogue_st_token_limit + min_dialogue_lt_token_limit
        )

        # The effective minimum token limit for the memories
        min_memories_token_limit = 1000 * gpt_response_format_scaler

        ### SHORT-TERM MEMORY TOKEN LIMIT RECALCULATION ###

        def recalculate_st_token_limit(available_tokens: int) -> None:
            """
            Recalculates the short-term memory token limit based on the current memory-dialogue ratio and given
            available tokens. Adjusts the short-term memory token limit if necessary.

            Args:
                available_tokens (int): The total available tokens.

            Returns:
                None
            """

            # Calculate the memory and dialogue token limits
            memory_token_limit, dialogue_token_limit = (
                self.calculate_memory_and_dialogue_token_limits(available_tokens)
            )

            # If the dialogue short-term memory token limit is greater than the minimum dialogue short-term memory token
            # limit and the remaining dialogue token limit (for the dialogue long-term memory) is less than the minimum
            # dialogue long-term memory token limit
            if (
                self.dialogue_short_term_memory_token_limit
                > min_dialogue_st_token_limit
                and (dialogue_token_limit - self.dialogue_short_term_memory_token_limit)
                < min_dialogue_lt_token_limit
            ):
                # Adjust the short-term memory token limit to equal the remaining dialogue token limit (we know there's
                # enough tokens between the two, so give the dialogue long-term memory its bare minimum by removing
                # this quantity of tokens from the full dialogue token limit when assigning the short-term memory
                # token limit)
                self.dialogue_short_term_memory_token_limit = (
                    dialogue_token_limit - min_dialogue_lt_token_limit
                )

        ### MEMORIES TOKEN LIMIT RECALCULATION ###

        # If there's enough context for memories and dialogue history
        if (
            memory_token_limit >= min_memories_token_limit
            and dialogue_token_limit >= min_dialogue_token_limit
        ):
            # If the dialogue short-term memory token limit is sufficient, but the dialogue long-term memory token limit
            # is not
            if (
                self.dialogue_short_term_memory_token_limit
                >= min_dialogue_st_token_limit
                and (dialogue_token_limit - self.dialogue_short_term_memory_token_limit)
                < min_dialogue_lt_token_limit
            ):
                # Adjust the short-term memory token limit accordingly
                self.dialogue_short_term_memory_token_limit = (
                    dialogue_token_limit - min_dialogue_lt_token_limit
                )
            # Return the updated memory and dialogue token limits
            return self.calculate_memory_and_dialogue_token_limits(available_tokens)

        # Otherwise, if there's not enough context for memories
        elif memory_token_limit < min_memories_token_limit:
            # If the total token limit is insufficient
            if (
                dialogue_token_limit + memory_token_limit
                < min_dialogue_token_limit + min_memories_token_limit
            ):
                raise ValueError(
                    "Ran out of context for the dialogue history. Reduce the amount of information in the system prompt. It is far too long."
                )
            # Adjust the memory-dialogue ratio (increase memory ratio)
            self.memory_dialogue_ratio = min_memories_token_limit / (
                dialogue_token_limit - (min_memories_token_limit - memory_token_limit)
            )
            # Recalculate the short-term memory token limit
            recalculate_st_token_limit(available_tokens)
            # Return the updated memory and dialogue token limits
            return self.calculate_memory_and_dialogue_token_limits(available_tokens)

        # Otherwise, there's not enough context for the dialogue history
        else:
            # If the total dialogue and memory token limit is sufficient
            if (
                dialogue_token_limit + memory_token_limit
                >= min_dialogue_token_limit + min_memories_token_limit
            ):
                # Adjust memory-dialogue ratio (decrease memory ratio)
                self.memory_dialogue_ratio = (
                    memory_token_limit
                    - (min_dialogue_token_limit - dialogue_token_limit)
                ) / min_dialogue_token_limit
                # Recalculate the short-term memory token limit
                recalculate_st_token_limit(available_tokens)
                # Return the updated memory and dialogue token limits
                return self.calculate_memory_and_dialogue_token_limits(available_tokens)
            # Otherwise, there's not enough context for both memories and dialogue history
            else:
                raise ValueError(
                    "Ran out of context for the dialogue history. Reduce the amount of information in the system prompt. It is far too long."
                )

    ###################################### TRIVIAL GET FUNCTIONS: dialogue history #####################################

    def get_dialogue_history_list(self) -> list:
        """
        Retrieve the list of dialogue history.

        This function returns the stored dialogue history, which contains the
        interactions that have occurred in the conversation. It provides access
        to the dialogue data for further processing or analysis.

        Args:
            None

        Returns:
            list: The list of dialogue history entries.
        """
        return self.dialogue_history

    def get_dialogue_history(self) -> str:
        """
        Retrieve the dialogue history as a formatted string.

        This function concatenates the entries in the dialogue history into a single
        string, with each entry separated by a newline. It provides a readable format
        of the conversation for display or logging purposes.

        Args:
            None

        Returns:
            str: A string representation of the dialogue history, with entries separated by newlines.
        """
        return "\n".join(self.dialogue_history)


class DialogueQueue:
    """
    A class to manage the dialogue queue for participants in a game.

    Attributes:
        game (Game): The game instance.
        participants (set): A set of Character objects participating in the dialogue.
        priority_queue (PriorityQueue): An instance of PriorityQueue to manage speaking order.
        listener_scores (dict): A dictionary to hold scores for each listener.
        memory_types (list): A list containing different types of memory used for scoring.
        original_top_scoring_listener (Character or None): The original top-scoring listener.
    """

    def __init__(
        self, game: "Game", participants: set, decay_rate: float = 0.9
    ) -> None:
        """
        Initializes the DialogueQueue with game, participants, and decay rate.

        Args:
            game (Game): The game instance.
            participants (set): A set of Character objects in the dialogue.
            decay_rate (float): The rate at which listener scores decay. Default is 0.9.

        Returns:
            None: This method does not return a value.
        """
        self.game = game
        self.participants = participants
        self.priority_queue = PriorityQueue(decay_rate=decay_rate)
        self.original_top_scoring_listener = None
        self.memory_types = [
            MemoryType.PERSONA,
            MemoryType.RESPONSE,
            MemoryType.REFLECTION,
            MemoryType.IMPRESSION,
            MemoryType.GOAL,
        ]

        # Calculate final listener scores based on weighted memory types
        self.persona_weight = 0.25
        self.response_weight = 0.375
        self.reflection_weight = 0.125
        self.impression_weight = 0.125
        self.goal_weight = 0.125

        self.percentile_threshold = 0.75  # This should be a parameter

    def get_next_speaker(
        self,
        speaker: "Character",
        last_response: Union[dict[str, Union[str, int]], list[str], str, None],
    ) -> Tuple["Character", bool]:
        """
        Get the next speaker in the dialogue priority queue.

        Args:
            speaker (Character): The current speaker.
            last_response (str or None): The last response from the GPT model.

        Returns:
            Tuple[Character, bool]: The next speaker (Character) and a boolean indicating if the speaker must speak
            (everyone else declined and we are back to the original top scorer).
        """
        if last_response is None or last_response == []:
            # print("THE PREVIOUS PERSON CHOSE NOT TO SPEAK")
            if not self.priority_queue.heap:  # Check if the heap is empty
                print(
                    "NOBODY ELSE WANTS TO SPEAK, SO WE'RE GOING BACK TO THE ORIGINAL TOP SCORER"
                )
                return self.original_top_scoring_listener, True
            print("CHECKING IF SOMEONE ELSE WANTS TO SPEAK")
            return self.priority_queue.get_next_speaker(), False

        if isinstance(last_response, list) and all(
            isinstance(item, dict) for item in last_response
        ):
            query_keywords_embeddings = Retrieve.get_query_keywords_and_embeddings(
                game=self.game,
                query=[split["component"] for split in last_response],
                scores=[(0, split["importance_score"]) for split in last_response],
            )
        else:
            # Retrieve query keywords and embeddings based on the last response
            query_keywords_embeddings = Retrieve.get_query_keywords_and_embeddings(
                game=self.game, query=last_response
            )

        # print("QUERY", query_keywords_embeddings)
        listener_scores = dict()  # Dictionary to hold listener scores

        # Calculate priority scores for each listener
        for listener in self.participants.difference(set([speaker])):
            # Get the priority scores for the listener (dicts mapping persona, response, reflection, impression, and
            # goal to dicts mapping recency, importance, and relevance scores to lists of memory node ID scores – order
            # is preserved)

            # The priority_scores structure is as follows:
            # {persona:
            #     {recency: [memory node ID scores],
            #      importance: [memory node ID scores],
            #      relevance: [memory node ID scores]
            #     },
            #  response:
            #     {recency: [memory node ID scores],
            #      importance: [memory node ID scores],
            #      relevance: [memory node ID scores]
            #     },
            #  reflection:
            #     {recency: [memory node ID scores],
            #      importance: [memory node ID scores],
            #      relevance: [memory node ID scores]
            #     },
            #  impression:
            #     {recency: [memory node ID scores],
            #      importance: [memory node ID scores],
            #      relevance: [memory node ID scores]
            #     },
            #  goal:
            #     {recency: [memory node ID scores],
            #      importance: [memory node ID scores],
            #      relevance: [memory node ID scores]
            #     }
            # }
            # It maps each memory type to a dict with recency, importance, and relevance scores, each of which maps to a
            # list of memory node ID scores

            priority_scores = Retrieve.priority_scores(
                game=self.game,
                character=listener,
                query=query_keywords_embeddings,
                standardize=False,
                weighted=True,
            )
            listener_scores[listener] = priority_scores
            # print("LISTENER", listener.name, "priority_scores", priority_scores)

        # Normalize the priority scores across all listeners for each memory type and score type (in-place)
        self._normalize_listener_scores(listener_scores)

        # Collapse the normalized scores into weighted averages for each listener and memory type (in-place)
        listener_scores = self._collapse_listener_scores(listener_scores)

        # print(
        #     "\nVALUES 1",
        #     [(listener.name, scores) for listener, scores in listener_scores.items()],
        # )

        # Initialize a dictionary to hold the means for each memory type
        memory_type_means = {memory_type: [] for memory_type in self.memory_types}

        # Calculate the mean score for each memory type above a percentile threshold
        for listener_priority_scores in listener_scores.values():
            for memory_type in self.memory_types:
                if memory_type in listener_priority_scores:
                    scores = listener_priority_scores[memory_type]
                    mean_score = self.calculate_mean_above_percentile(scores)
                    listener_priority_scores[memory_type] = mean_score
                    memory_type_means[memory_type].append(mean_score)

        # print(
        #     "VALUES 2",
        #     [(listener.name, scores) for listener, scores in listener_scores.items()],
        # )

        # Normalize the memory type means across listeners
        memory_type_means = self.normalize_means_across_listeners(
            listener_scores, memory_type_means
        )

        # Calculate the final score for each listener
        for listener_name, listener_priority_scores in listener_scores.items():
            listener_scores[listener_name] = self.calculate_final_score(
                listener_priority_scores
            )

        print("\nNEW LISTENER SCORES")
        for listener, score in listener_scores.items():
            print(listener.name + ":", round(score, 2))

        # Update the priority queue with the new listener scores
        self.priority_queue.update_listeners(listener_scores)

        # Print the player names and scores in the dialogue queue
        print("\nCurrent Dialogue Queue:")
        for listener, score in self.priority_queue.listener_map.items():
            print(f"{listener.name}: {round(score, 2)}")

        self.original_top_scoring_listener = self.priority_queue.get_next_speaker()

        return self.original_top_scoring_listener, False  # Return the next speaker

        # The priority_scores structure could be as follows:
        # {persona:
        #     {recency: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      importance: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      relevance: np.ndarray of shape (n,) or (n, 1) or (n, 2)
        #     },
        #  response:
        #     {recency: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      importance: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      relevance: np.ndarray of shape (n,) or (n, 1) or (n, 2)
        #     },
        #  reflection:
        #     {recency: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      importance: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      relevance: np.ndarray of shape (n,) or (n, 1) or (n, 2)
        #     },
        #  impression:
        #     {recency: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      importance: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      relevance: np.ndarray of shape (n,) or (n, 1) or (n, 2)
        #     },
        #  goal:
        #     {recency: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      importance: np.ndarray of shape (n,) or (n, 1) or (n, 2),
        #      relevance: np.ndarray of shape (n,) or (n, 1) or (n, 2)
        #     }
        # }
        # It maps each memory type to a dict with recency, importance, and relevance scores, each of which maps to
        # either a numpy array of shape (n,), (n, 1), or (n, 2)

    def _normalize_listener_scores(
        self, listener_scores: dict[str, dict[str, dict[str, np.ndarray]]]
    ) -> None:
        """
        Standardize the priority scores across all listeners for each memory type and score type.

        This method adjusts the scores for each listener by standardizing them based on the mean and standard deviation
        of all scores for a given memory type and score type. If the standard deviation is zero, scores are set to zero
        to avoid division by zero errors. If scores are arrays of shape (n, 2), standardization is done for each index
        in the array separately.

        Args:
            listener_scores (dict): A dictionary where keys are listeners and values are dictionaries containing
                                    priority scores for each memory type and score type.

        Returns:
            None
        """

        # print("NORMALIZING LISTENER SCORES", listener_scores)

        # Iterate over each memory type (e.g., persona, response, reflection, etc.)
        for memory_type in self.memory_types:
            # Iterate over each score type (e.g., recency, importance, relevance)
            for score_type in ["recency", "importance", "relevance"]:
                # Collect all scores for the current memory type and score type across all listeners
                all_scores = []
                for listener, scores in listener_scores.items():
                    if memory_type in scores and score_type in scores[memory_type]:
                        all_scores.append(scores[memory_type][score_type])

                if not all_scores:
                    continue  # Skip to the next score type if no scores are present

                # Check the shape of the scores to determine how to standardize
                all_scores = np.concatenate(all_scores)
                # print("-"*100)
                # print("\nMEMORY TYPE", memory_type, "SCORE TYPE", score_type)
                # print("ALL SCORES (norm)", all_scores)

                if all_scores.ndim == 2 and all_scores.shape[1] == 2:
                    # Standardize each index of the array separately
                    for index in range(2):
                        index_scores = all_scores[:, index]
                        mean_value = np.mean(index_scores)
                        std_value = np.std(index_scores)

                        # Standardize scores for each listener
                        for listener, scores in listener_scores.items():
                            if (
                                memory_type in scores
                                and score_type in scores[memory_type]
                            ):
                                if std_value > 0:
                                    scores[memory_type][score_type][:, index] = (
                                        scores[memory_type][score_type][:, index]
                                        - mean_value
                                    ) / std_value
                                else:
                                    scores[memory_type][score_type][:, index] = 0
                else:
                    # Standardize non-tuple scores
                    mean_value = np.mean(all_scores)
                    std_value = np.std(all_scores)

                    # Standardize scores for each listener
                    for listener, scores in listener_scores.items():
                        if memory_type in scores and score_type in scores[memory_type]:
                            if std_value > 0:  # Avoid division by zero
                                scores[memory_type][score_type] = (
                                    scores[memory_type][score_type] - mean_value
                                ) / std_value
                            else:
                                scores[memory_type][score_type] = np.zeros_like(
                                    scores[memory_type][score_type]
                                )

    def _collapse_listener_scores(
        self, listener_scores: dict[str, dict[str, dict[str, np.ndarray]]]
    ) -> dict[str, dict[str, list[float]]]:
        """
        Collapse the normalized scores into weighted averages for each listener and memory type.

        This method calculates a weighted average list of recency, importance, and relevance scores for each memory type
        using predefined alpha values for each score type. If scores are arrays of shape (n, 1) or (n, 2), the numbers
        in the rows of the array are averaged before processing. The function handles cases where not all memory types
        are present for each listener.

        Args:
            listener_scores (dict): A dictionary where keys are listeners and values are dictionaries containing
                                    normalized priority scores for each memory type and score type.

        Returns:
            dict: A dictionary mapping each listener (str) to another dictionary mapping memory types (str) to their
                  weighted average scores lists.
        """

        # print("COLLAPSING LISTENER SCORES", listener_scores)

        collapsed_scores = {}
        # Iterate over each listener
        for listener, scores in listener_scores.items():
            collapsed_scores[listener] = {}
            # Iterate over each memory type
            for memory_type in self.memory_types:
                if memory_type in scores:
                    # Initialize a list to accumulate weighted scores
                    weighted_scores = []
                    # Determine the number of scores to process
                    num_scores = len(scores[memory_type].get("recency", []))
                    # Iterate over each index of the scores
                    for i in range(num_scores):
                        # Initialize weighted sum for the current index
                        weighted_sum = 0.0
                        # Iterate over each score type
                        for score_type, alpha in zip(
                            ["recency", "importance", "relevance"],
                            [
                                MemoryStream.recency_alpha,
                                MemoryStream.importance_alpha,
                                MemoryStream.relevance_alpha,
                            ],
                        ):
                            if score_type in scores[memory_type]:
                                score_values = scores[memory_type][score_type]
                                if i < len(score_values):
                                    score_value = score_values[i]
                                    # print("-"*100)
                                    # print("memory_type", memory_type, "score_type", score_type, "index", i)
                                    # print("SCORE VALUE", score_value.ndim, score_value.shape, score_value)
                                    # If the score is an array of shape (n, 2), average the numbers in the array
                                    if (
                                        score_value.ndim == 2
                                        and score_value.shape[1] == 2
                                        or score_value.shape == (2,)
                                    ):
                                        # print("AVERAGING SCORE VALUE", score_value)
                                        score_value = np.mean(score_value)
                                        # print("AVERAGED SCORE VALUE", score_value)
                                    # Add the weighted score to the weighted sum
                                    weighted_sum += alpha * score_value
                        # Append the weighted sum to the weighted scores list
                        weighted_scores.append(weighted_sum)
                    # Store the weighted scores list in the collapsed scores dictionary
                    collapsed_scores[listener][memory_type] = weighted_scores
        # print("COLLAPSED SCORES", collapsed_scores)
        return collapsed_scores

    def calculate_final_score(self, listener_priority_scores: dict) -> float:
        """
        Calculate the final score for a listener based on weighted memory types.

        Args:
            listener_priority_scores (dict): A dictionary mapping memory types to scores.

        Returns:
            float: The final weighted score for the listener.
        """
        total_weight = 0
        final_score = 0

        if MemoryType.PERSONA in listener_priority_scores:
            final_score += (
                listener_priority_scores[MemoryType.PERSONA] * self.persona_weight
            )
            total_weight += self.persona_weight

        if MemoryType.RESPONSE in listener_priority_scores:
            final_score += (
                listener_priority_scores[MemoryType.RESPONSE] * self.response_weight
            )
            total_weight += self.response_weight

        if MemoryType.REFLECTION in listener_priority_scores:
            final_score += (
                listener_priority_scores[MemoryType.REFLECTION] * self.reflection_weight
            )
            total_weight += self.reflection_weight

        if MemoryType.IMPRESSION in listener_priority_scores:
            final_score += (
                listener_priority_scores[MemoryType.IMPRESSION] * self.impression_weight
            )
            total_weight += self.impression_weight

        if MemoryType.GOAL in listener_priority_scores:
            final_score += listener_priority_scores[MemoryType.GOAL] * self.goal_weight
            total_weight += self.goal_weight

        # Adjust the final score to ensure the total weight sums to 1
        return final_score / total_weight if total_weight > 0 else 0

    def calculate_mean_above_percentile(self, scores: list) -> float:
        """
        Calculate the mean of scores above a given percentile.

        Args:
            scores (list): A list of scores (float or int).

        Returns:
            float: The mean of scores above the percentile threshold.
        """
        if not scores:
            return 0.0
        threshold_value = np.percentile(scores, self.percentile_threshold * 100)
        filtered_scores = [score for score in scores if score > threshold_value]
        return np.mean(filtered_scores) if filtered_scores else 0.0

    def normalize_means_across_listeners(
        self, listener_scores: dict, memory_type_means: dict
    ) -> None:
        """
        Normalize the means for each memory type across all listeners.

        Args:
            listener_scores (dict): A dictionary mapping listeners to their priority scores.
            memory_type_means (dict): A dictionary mapping memory types to lists of mean scores.

        Returns:
            dict: A dictionary mapping memory types to their normalized means.
        """

        # Initialize a dictionary to hold the means for each memory type
        memory_type_means = {memory_type: [] for memory_type in self.memory_types}

        for memory_type, means in memory_type_means.items():
            if means:  # Check if there are means to normalize
                mean_value = np.mean(means)
                std_value = np.std(means)
                for listener_priority_scores in listener_scores.values():
                    if memory_type in listener_priority_scores:
                        if std_value > 0:  # Avoid division by zero
                            listener_priority_scores[memory_type] = (
                                listener_priority_scores[memory_type] - mean_value
                            ) / std_value
                        else:
                            listener_priority_scores[memory_type] = (
                                0  # Assign zero if no variation
                            )

        return memory_type_means


class PriorityQueue:
    """
    A class to manage a priority queue of listeners based on their scores.

    Attributes:
        heap (list): A list to maintain the heap structure for the priority queue.
        listener_map (dict): A dictionary mapping listener names to their scores.
        decay_rate (float): The rate at which scores decay over time.
    """

    def __init__(self, decay_rate: float = 0.9):
        """
        Initializes the PriorityQueue with a specified decay rate.

        Args:
            decay_rate (float): The decay rate for listener scores. Default is 0.9.
        """
        self.heap = []
        self.listener_map = {}
        self.decay_rate = decay_rate

    def pop(self) -> "Character":
        """
        Removes and returns the listener with the highest score from the queue.

        Returns:
            Character: The listener with the highest score.
        """
        return heapq.heappop(self.heap)[-1]

    def add_listener(self, listener: "Character", new_score: float) -> None:
        """
        Adds or updates a listener in the heap with decay logic.

        Args:
            listener (Character): The listener to add or update.
            new_score (float): The new score for the listener.
        """
        # Check if player is already in the heap (exists in listener_map)
        if listener in self.listener_map:
            old_score = self.listener_map[listener]

            if old_score > new_score:
                # Apply exponential decay to the old score
                updated_score = old_score * self.decay_rate
            else:
                # Use the new score as it is higher
                updated_score = new_score

            # Update the listener score in the map
            self.listener_map[listener] = updated_score
        else:
            # If listener is not in the heap, simply add them with their new score
            self.listener_map[listener] = new_score
            updated_score = new_score

        # Rebuild the heap to reflect the updated scores
        self._rebuild_heap()

    def update_listeners(self, listeners_scores_dict: dict) -> None:
        """
        Updates multiple listeners in the heap based on their new scores.

        Args:
            listeners_scores_dict (dict): A dictionary mapping listener names to their new scores.
        """
        for listener, new_score in listeners_scores_dict.items():
            # Check if player is already in the heap (exists in listener_map)
            if listener in self.listener_map:

                # Apply exponential decay to the old score
                old_score = self.listener_map[listener] * self.decay_rate

                if old_score > new_score:
                    # Use the decayed, old score as it is higher
                    updated_score = old_score
                else:
                    # Use the new score as it is higher
                    updated_score = new_score

                # Update the listener score in the map
                self.listener_map[listener] = updated_score
            else:
                # If listener is not in the heap, simply add them with their new score
                self.listener_map[listener] = new_score

        # Rebuild the heap to reflect the updated scores
        self._rebuild_heap()

    def _rebuild_heap(self) -> None:
        """
        Rebuilds the heap based on the updated scores in the listener_map.
        """
        # Clear the heap
        self.heap.clear()

        # Push all listeners from the listener_map back into the heap
        for listener, score in self.listener_map.items():
            # Generate a random number to break ties when scores are equal
            random_tiebreaker = random.random()
            # Push (-score, random_tiebreaker, listener) onto the heap
            heapq.heappush(self.heap, (-score, random_tiebreaker, listener))

    def get_next_speaker(self) -> "Character":
        """
        Retrieves the next speaker with the highest score from the heap.

        Returns:
            Character: The next speaker.
        """
        # Pop the listener with the highest score (smallest negative value)
        score, _, listener = heapq.heappop(self.heap)
        self.listener_map.pop(listener)
        return listener  # Return the Character object
