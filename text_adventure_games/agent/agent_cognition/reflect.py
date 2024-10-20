"""
Authors: Sam Thudium and Kyle Sullivan

File: agent_cognition/reflect.py
Description: defines how agents reflect upon their past experiences
"""

# Stages of reflection
# 1. self-evaluation of actions:
#   - what was the quality of your actions?
#   - could your actions have been performed more efficiently and if so why?
# 2. Strategic reflection: relate progress toward goals to the game’s end state
#   - sub-routine that triggers the Goals module for analysis of goals
#       - give previous goal(s)
#       - return Updated or same goals
# 4. Interpersonal reflection:
#   - here is how you feel about Person A: <summary of relationship>
#   - Given your experiences of them: <memories>
#   - Would you update your personal understanding and feelings toward Person A?

# What memories should be reflected upon?
# 0. For all types: Just give all memories from the last round + reflection nodes + goals
#   - Would need to store a cache of these in the memory
# 1a. self-eval should focus on the agent's actions
#   - this means we need to distinguish between self and others' actions
# 2a. strategic reflections should use agent goals and memories relevant to them
#   -
# 3a. Need just goals from the last round
# 4a. see 4

from typing import TYPE_CHECKING, Dict
import json
import logging

# Local imports for memory management, prompts, and GPT helper functions
from text_adventure_games.agent.memory_stream import MemoryType
from text_adventure_games.assets.prompts import reflection_prompts as rp
from text_adventure_games.utils.consts import get_models_config
from text_adventure_games.gpt.gpt_helpers import (
    limit_context_length,
    get_prompt_token_count,
    get_token_remainder,
    GptCallHandler,
)

# Importing OrderedSet for maintaining unique ordered collections
from ordered_set import OrderedSet
from . import retrieve  # Importing the retrieve module from the current package

# Type checking imports for better IDE support and type hints
if TYPE_CHECKING:
    from text_adventure_games.games import Game  # Game class for type hints
    from text_adventure_games.things import Character  # Character class for type hints


class Reflect:
    """
    A class to manage and perform reflection operations for characters in the game.

    This class provides methods to facilitate different types of reflections, such as reflecting on past actions,
    goals, and relationships. It utilizes a shared GPT handler to generate reflections based on the character's
    memories and the current game state.
    """
    
    # Constants for reflection output limits and retry attempts
    REFLECTION_MAX_OUTPUT = 512  # Maximum output length for reflections
    REFLECTION_MAX_MEMORIES = 25  # Maximum number of memories to retrieve
    REFLECTION_RETRIES = 5  # Number of retries for reflection generation

    # Shared GptCallHandler for all instances
    gpt_handler = None

    # Model parameters for the GPT handler
    model_params = {
        "api_key_org": "Helicone",
        "model": get_models_config()["reflect"]["model"],
        "max_tokens": REFLECTION_MAX_OUTPUT,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_retries": REFLECTION_RETRIES,
    }
    
    # Set up the logger at the module level
    logger = logging.getLogger("agent_cognition")

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(**cls.model_params)

    @classmethod
    def reflect(cls, game: "Game", character: "Character"):
        """
        Perform a complete reflection; this is composed of sub-types of reflection:
        1. reflect on actions from past round (inspired by CLIN)
        2. reflect on goals
        3. reflect on relationships
        """
        cls.generalize(game, character)
        # cls.reflect_on_goals(game, character)
        # cls.reflect_on_relationships(game, character)

    @classmethod
    def generalize(cls, game: "Game", character: "Character"):
        """
        Generates new generalizations based on the character's memories and impressions.
        
        This method retrieves relevant memories and impressions from the character's past actions and interactions,
        categorizes them into reflections and observations, and constructs a user prompt for the GPT model to 
        generate new insights. The process involves calculating token counts to ensure the prompt fits within 
        the model's context limits and managing retries for generating responses.

        Returns:
            None
        """
        # Set the max number of memories to retrieve for each query
        memories_per_retrieval = cls.REFLECTION_MAX_MEMORIES

        # Construct the system prompt
        system_prompt = (
            character.get_standard_info(game, include_goals=True, include_perceptions=False)
            + rp.gpt_generalize_prompt
        )

        # Calculate the token count for the system prompt
        system_prompt_token_count = get_prompt_token_count(
            content=system_prompt, role="system", pad_reply=False
        )

        # Initialize token count and impressions list
        impressions_token_count = 0
        impressions = []

        # If the character uses impressions, retrieve them from other characters
        if character.use_impressions:
            impressions = character.impressions.get_multiple_impressions(
                game.characters.values()
            )
            impressions_token_count = get_prompt_token_count(
                content=impressions, role="user", pad_reply=True
            )

        # Initialize a list to hold relevant memories
        relevant_memories = []

        # Retrieve relevant memories based on predefined query questions
        for question in rp.memory_query_questions:
            memories = retrieve.retrieve(
                game=game,
                character=character,
                query=question,
                n=memories_per_retrieval,
                include_idx=True,
            )
            relevant_memories.extend(memories)

        # Remove duplicates from the relevant memories and sort them by length
        relevant_memories = list(set(relevant_memories))
        relevant_memories.sort(key=len)

        # Calculate the token count for the relevant memories
        relevant_memories_token_count = get_prompt_token_count(
            content=relevant_memories, role=None, pad_reply=False
        )

        # Prepare a primer for relevant reflections and memories
        relevant_memories_primer = [
            "\nRelevant Reflections:\n",
            "\nRelevant Memories:\n",
            "None\n",
        ]

        # Calculate the token count for the relevant memories primer
        rel_mem_primer_token_count = get_prompt_token_count(
            content=relevant_memories_primer, role=None, pad_reply=False
        )

        # Prepare the insight question prompt
        insight_q_prompt = ["\n" + rp.insight_question]
        # Calculate the token count for the insight question prompt
        insight_q_token_count = get_prompt_token_count(
            content=insight_q_prompt, role=None, pad_reply=False
        )

        # Calculate the number of available tokens for the GPT model
        available_tokens = get_token_remainder(
            cls.gpt_handler.model_context_limit,
            cls.gpt_handler.max_output_tokens,
            system_prompt_token_count,
            impressions_token_count,
            rel_mem_primer_token_count,
            insight_q_token_count,
        )

        # Process relevant memories while there are still tokens available
        while relevant_memories_token_count > 0:
            # Limit the relevant memories to fit within the available token count
            relevant_memories_limited = limit_context_length(
                relevant_memories,
                max_tokens=available_tokens,
                tokenizer=game.parser.tokenizer,
                keep_most_recent=False,
            )

            # Break the loop if there are no memories to process
            if not relevant_memories_limited:
                break

            # Categorize memories into reflections and observations
            categorized_memories = {"reflections": [], "observations": []}
            for full_memory in relevant_memories_limited:
                idx, memory_desc = full_memory.split(".", 1)
                idx = int(idx)
                memory_desc = memory_desc.strip()
                memory_type = character.memory.get_observation_type(idx)
                key = (
                    "reflections"
                    if memory_type.value == MemoryType.REFLECTION.value
                    else "observations"
                )
                categorized_memories[key].append(
                    full_memory if key == "reflections" else memory_desc
                )

            # Use the relevant memories primer if no reflections or observations are found
            reflections_lmtd = categorized_memories["reflections"] or [
                relevant_memories_primer[2]
            ]
            observations_lmtd = categorized_memories["observations"] or [
                relevant_memories_primer[2]
            ]

            # Construct the user prompt by combining impressions, memories, and insight questions
            user_prompt_list = (
                impressions
                + [relevant_memories_primer[0]]
                + reflections_lmtd
                + [relevant_memories_primer[1]]
                + observations_lmtd
                + insight_q_prompt
            )

            # Join the user prompt list into a single string
            user_prompt_str = "".join(user_prompt_list)

            success = False
            # Attempt to generate a response from the GPT model
            while not success:
                try:
                    response = cls.gpt_handler.generate(
                        system=system_prompt, user=user_prompt_str
                    )
                    # Parse the response as JSON to extract new generalizations
                    new_generalizations = json.loads(response)
                except json.JSONDecodeError as e:
                    cls.logger.error(f"JSON decoding error: {e}")
                    continue  # Retry if JSON decoding fails
                else:
                    success = True
                    # Add the new generalizations to the character's memory
                    cls.add_generalizations_to_memory(game, character, new_generalizations)

            # Update the relevant memories to exclude those already reflected upon
            relevant_memories = list(
                OrderedSet(relevant_memories) - OrderedSet(relevant_memories_limited)
            )
            # Recalculate the token count for the remaining relevant memories
            relevant_memories_token_count = get_prompt_token_count(
                content=relevant_memories, role=None, pad_reply=False
            )

    @classmethod
    def add_generalizations_to_memory(cls, game: "Game", character: "Character", generalizations: Dict):
        """
        Parses the GPT-generated generalizations dictionary to extract and add new reflections
        to the character's memory. The structure of the generalizations depends on the prompt used.

        Args:
            generalizations (Dict): A dictionary containing generalizations generated by the GPT model.

        Returns:
            None: This method does not return a value.
        """
        # Add new generalizations for the specified character in the given game
        cls.add_new_generalizations(game, character, generalizations)

        # Update existing generalizations for the specified character in the given game
        cls.update_existing_generalizations(game, character, generalizations)

    @classmethod
    def add_new_generalization_helper(cls, game: "Game", character: "Character", generalization: Dict):
        """
        Add a new generalization to the character's memory.

        This function attempts to extract a statement from the provided generalization
        and adds it to the character's memory if the statement is valid. If the generalization
        is malformed (i.e., missing the statement), it is skipped without any changes.

        Args:
            generalization (Dict): A dictionary containing the generalization data,
                                   including a "statement" key.

        Returns:
            None: This function does not return a value.

        Raises:
            KeyError: If the generalization does not contain the "statement" key,
                       the function will skip adding it to memory.
        """
        try:
            # Attempt to extract the "statement" from the generalization dictionary
            desc = generalization["statement"]
        except KeyError:
            # If the "statement" key is missing, this indicates a malformed reflection; log the error and skip processing
            cls.logger.error(f"Malformed generalization: {generalization}")
            pass
        else:
            # Summarize and score the action described in the statement, obtaining keywords and importance
            _, ref_importance, ref_kwds = game.parser.summarize_and_score_action(
                description=desc, thing=character, needs_summary=False
            )

            # Add the summarized memory to the character's memory with relevant details
            character.memory.add_memory(
                game.round,
                game.tick,
                desc,
                ref_kwds,
                character.location.name,
                success_status=True,
                memory_importance=ref_importance,
                memory_type=MemoryType.REFLECTION.value,
                actor_id=character.id,
            )

    @classmethod
    def add_new_generalizations(cls, game: "Game", character: "Character", generalizations: Dict):
        """
        Add new generalizations as memories to the character's memory.

        This method extracts new generalizations from the provided dictionary and attempts to add each one
        to the character's memory. If the input is malformed or missing the expected structure, it logs an error.

        Args:
            generalizations (Dict): A dictionary containing generalization data, expected to have a key "new"
                                    that maps to a list of new generalizations.

        Returns:
            None: This function does not return a value.
        """
        try:
            # Attempt to extract the list of new generalizations from the generalizations dictionary
            new_gens = generalizations["new"]
        except (KeyError, TypeError) as e:
            # If the "new" key is missing or generalizations is not a dictionary, log the error and skip processing
            # TODO: maybe build in some retry logic?
            cls.logger.error(f"Error processing new generalizations: {e}")
            pass
        else:
            # Iterate over each new generalization and add it to the character's memory
            for ref in new_gens:
                cls.add_new_generalization_helper(game, character, ref)

    @classmethod
    def update_existing_generalizations(cls, game: "Game", character: "Character", generalizations: Dict):
        """
        Update existing generalizations in the character's memory based on the provided data.

        This method attempts to find and replace the descriptions of reflection nodes that have been updated
        by GPT. It processes the input dictionary to extract updated generalizations and updates the corresponding
        memory nodes in the character's memory.

        Args:
            generalizations (Dict): A dictionary containing updated generalization data, expected to have a key
                                    "updated" that maps to a list of updated generalizations.

        Returns:
            None: This function does not return a value.
        """
        try:
            # Attempt to extract the list of updated generalizations from the generalizations dictionary
            updated_gens = generalizations["updated"]
        except KeyError:
            # If the "updated" key is missing, log the error and skip processing
            # TODO: again, do we want retry logic for reflections if GPT got JSON structure wrong?
            cls.logger.error(
                "Error processing updated generalizations: 'updated' key missing in generalizations dictionary"
            )
            pass
        except TypeError:
            # If generalizations is not a dictionary, log the error and skip processing
            # TODO: again, do we want retry logic for reflections if GPT got JSON structure wrong?
            cls.logger.error(
                "Error processing updated generalizations: Invalid generalizations structure (not a dictionary)"
            )
            pass
        else:
            # Iterate over each updated generalization
            for ref in updated_gens:
                try:
                    # Attempt to extract the index and statement from the updated generalization
                    prev_idx = int(ref["index"])
                    statement = ref["statement"]
                except KeyError:
                    # If the index or statement key is missing, log the error and create a new reflection
                    cls.logger.error(f"Missing required key in updated generalization: {ref}")
                    cls.add_new_generalization_helper(game, character, ref)
                    continue
                except ValueError:
                    # If the index cannot be converted to an integer, log the error and create a new reflection
                    cls.logger.error(f"Invalid index format in updated generalization: {ref}")
                    cls.add_new_generalization_helper(game, character, ref)
                    continue
                except TypeError:
                    # If there's a type mismatch, log the error and create a new reflection
                    cls.logger.error(f"Type mismatch in updated generalization: {ref}")
                    cls.add_new_generalization_helper(game, character, ref)
                    continue
                else:
                    # Get the type of memory associated with the previous index
                    memory_type = character.memory.get_observation_type(prev_idx)
                    if memory_type and memory_type.value != MemoryType.REFLECTION.value:
                        # If the existing memory is not a reflection, create a new reflection
                        cls.add_new_generalization_helper(game, character, ref)
                    else:
                        # Score the action described in the statement, obtaining keywords and importance
                        _, ref_importance, ref_kwds = (
                            game.parser.summarize_and_score_action(
                                description=statement, thing=character, needs_summary=False
                            )
                        )

                        # Update the existing memory node with the new statement and round/tick information
                        _ = character.memory.update_node(
                            node_id=prev_idx,
                            node_round=game.round,
                            node_tick=game.tick,
                            node_description=statement,
                            node_importance=ref_importance,
                            node_keywords=ref_kwds,
                        )
                        # Update the embedding of the memory node with the new description
                        _ = character.memory.update_node_embedding(
                            node_id=prev_idx, new_description=statement
                        )
