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

circular_import_prints = False

if circular_import_prints:
    print("Importing Reflect")

from typing import TYPE_CHECKING, Dict
import json
import logging

if circular_import_prints:
    print(f"\t{__name__} calling imports for General")
from text_adventure_games.utils.general import (
    enumerate_dict_options,
)

# Local imports for memory management, prompts, and GPT helper functions
from text_adventure_games.assets.prompts import reflection_prompts as rp

if circular_import_prints:
    print(f"\t{__name__} calling imports for Consts")
from text_adventure_games.utils.consts import get_models_config

if circular_import_prints:
    print(f"\t{__name__} calling imports for GptHelpers")
from text_adventure_games.gpt.gpt_helpers import (
    limit_context_length,
    get_prompt_token_count,
    get_token_remainder,
)

# Importing OrderedSet for maintaining unique ordered collections
from ordered_set import OrderedSet

if circular_import_prints:
    print(f"\t{__name__} calling imports for Retrieve")
# Importing the Retrieve class from the current package
from .retrieve import Retrieve

if circular_import_prints:
    print(f"\t{__name__} calling imports for MemoryType")
from text_adventure_games.agent.memory_stream import MemoryType

if circular_import_prints:
    print(f"\t{__name__} calling Type Checking imports for GptCallHandler")
from text_adventure_games.gpt.gpt_helpers import GptCallHandler

# Type checking imports for better IDE support and type hints
if TYPE_CHECKING:
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game  # Game class for type hints

    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Character")
    from text_adventure_games.things import Character  # Character class for type hints


class Reflect:
    """
    A class to manage and perform reflection operations for characters in the game.

    This class offers methods to facilitate various types of reflections, including reflections on past actions,
    goals, and relationships. It leverages a shared GPT handler to generate reflections based on the character's
    memories and the current state of the game.

    Class Variables:
        REFLECTION_MAX_MEMORIES (int): Maximum number of memories to retrieve.
        gpt_handler (GptCallHandler): Shared GptCallHandler for all instances.
        model_params (dict): Model parameters for the GPT handler, including API key, model type, and token limits.
        logger (Logger): Logger instance for logging reflection-related events.

    Class Methods:
        initialize_gpt_handler(): Initializes the shared GptCallHandler if it has not been created yet.
        reflect(game: Game, character: Character): Executes a comprehensive reflection process.
        generalize(game: Game, character: Character): Generates new generalizations based on the character's memories
        and impressions.
        add_generalizations_to_memory(game: Game, character: Character, generalizations: Dict): Parses and adds new
        reflections to the character's memory.
        add_new_generalization_helper(game: Game, character: Character, generalization: Dict): Adds a new generalization
        to the character's memory.
        add_new_generalizations(game: Game, character: Character, generalizations: Dict): Adds new generalizations as
        memories to the character's memory.
        update_existing_generalizations(game: Game, character: Character, generalizations: Dict): Updates existing
        generalizations in the character's memory based on the provided data.
    """

    REFLECTION_MAX_MEMORIES = 25  # Maximum number of memories to retrieve

    # Shared GptCallHandler for all instances
    gpt_handler = None
    # Model parameters for the GPT handler
    model_params = {
        # "max_output_tokens": 512,
        # "temperature": 1,
        # "top_p": 1,
        # "frequency_penalty": 0,
        # "presence_penalty": 0,
        # "max_retries": 5,
    }

    # Set up the logger at the module level
    logger = logging.getLogger("agent_cognition")

    memory_query_keywords_and_embeddings = None

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """

        if circular_import_prints:
            print(f"-\tReflect Module is initializing GptCallHandler")

        # Initialize the GPT handler if it hasn't been set up yet
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(
                model_config_type="reflect", **cls.model_params
            )

    # @classmethod
    # def get_memory_query_keywords_and_embeddings(cls, game: "Game"):
    #     if cls.memory_query_keywords_and_embeddings_list is None:
    #         cls.memory_query_keywords_and_embeddings_list = [
    #             Retrieve.get_query_keywords_and_embeddings(game=game, query=q)
    #             for q in rp.memory_query_questions
    #         ]

    @classmethod
    def get_memory_query_keywords_and_embeddings(cls, game: "Game"):
        if cls.memory_query_keywords_and_embeddings is None:
            cls.memory_query_keywords_and_embeddings = (
                Retrieve.get_query_keywords_and_embeddings(
                    game=game, query=rp.memory_query_questions
                )
            )

    @classmethod
    def reflect(cls, game: "Game", character: "Character"):
        """
        Executes a comprehensive reflection process, which includes:
        1. Reflecting on actions from the previous round (inspired by CLIN)
        2. Reflecting on goals
        3. Reflecting on relationships
        """
        # Initialize the GPT handler if it hasn't been set up yet
        cls.initialize_gpt_handler()
        cls.get_memory_query_keywords_and_embeddings(
            game
        )  # Get the memory query keywords and embeddings
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
            character.get_standard_info(
                game, include_goals=True, include_perceptions=True
            )
            + "\n\nCURRENT TASK:\n"
            + rp.gpt_generalize_prompt
        )

        # Calculate the token count for the system prompt
        system_prompt_token_count = get_prompt_token_count(
            content=system_prompt, role="system", pad_reply=False
        )

        # print("-" * 100)
        # print(character.name, "BEGINS REFLECTION GENERALIZATION")
        # print("OBSERVATIONS (reflect)")
        # for obs in character.memory.observations:
        #     print("-", obs.node_id, obs.node_type, obs.node_round, obs.node_description)

        # Retrieve relevant memories based on predefined query questions
        relevant_memories = Retrieve.retrieve(
            game=game,
            character=character,
            query=cls.memory_query_keywords_and_embeddings,
            n=memories_per_retrieval,
            # memory_lookback=-1,
            # round=game.round,
            include_idx=True,
            prepend="\n- ",
            memory_types=[
                MemoryType.ACTION,
                MemoryType.DIALOGUE,
                # MemoryType.REFLECTION,
                MemoryType.PERCEPT,
            ],
        )
        
        # print("RELEVANT MEMORIES (reflect)", relevant_memories)

        # Retrieve all reflections from the previous three rounds
        memories = Retrieve.retrieve(
            game=game,
            character=character,
            query=cls.memory_query_keywords_and_embeddings,
            n=memories_per_retrieval,
            memory_lookback=-1,
            round=game.round - 3,
            include_idx=True,
            prepend="\n- ",
            memory_types=[
                MemoryType.REFLECTION,
            ],
        )
        
        # print("MEMORIES (reflect)", memories)

        # Create a set of memories for quick lookup
        memories_set = set(memories)

        # Filter relevant_memories to remove any duplicates found in memories
        relevant_memories = [mem for mem in relevant_memories if mem not in memories_set]

        # Extend relevant_memories with memories, preserving order
        relevant_memories.extend(memories)

        # print("RELEVANT MEMORIES (reflect)", relevant_memories)

        # Calculate the token count for the relevant memories
        relevant_memories_token_count = get_prompt_token_count(
            content=relevant_memories, role=None, pad_reply=False
        )

        # Prepare a primer for relevant reflections and memories
        relevant_memories_primer = [
            "\n\nRelevant Reflections:",
            "\n\nRelevant Memories:",
            "\nNone\n",
            "\nNone\n",
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
            rel_mem_primer_token_count,
            insight_q_token_count,
        )

        # print("RELEVANT MEMORIES TOKEN COUNT (reflect)", relevant_memories_token_count)
        # print("RELEVANT MEMORIES COUNT (reflect)", len(relevant_memories))
        # print("RELEVANT MEMORIES ID'S (reflect)", [mem.split(".", 1)[0].replace("-", "").strip() for mem in relevant_memories])

        # Process relevant memories while there are still tokens available
        while relevant_memories_token_count > 0:

            # Limit the relevant memories to fit within the available token count
            relevant_memories_limited = limit_context_length(
                relevant_memories,
                max_tokens=available_tokens,
                tokenizer=game.parser.tokenizer,
                keep_most_recent=False,
            )

            # FOR DEBUGGING (DELETE)
            relevant_memories_limited_token_count = get_prompt_token_count(
                content=relevant_memories_limited, role=None, pad_reply=False
            )

            # print("LIMITED RELEVANT MEMORIES TOKEN COUNT (reflect)", relevant_memories_limited_token_count)
            # print("LIMITED RELEVANT MEMORIES COUNT (reflect)", len(relevant_memories_limited))
            # print("LIMITED RELEVANT MEMORIES ID'S (reflect)", [mem.split(".", 1)[0].replace("-", "").strip() for mem in relevant_memories_limited])

            # print("REFLECT MEMORIES LIMITED")
            # for mem in relevant_memories_limited:
            #     print(mem)

            # Break the loop if there are no memories to process
            if not relevant_memories_limited:
                break

            # Categorize memories into reflections and observations
            categorized_memories = {"reflections": [], "observations": []}
            for full_memory in relevant_memories_limited:
                # print("\nFULL MEMORY", full_memory)
                idx, memory_desc = full_memory.split(".", 1)
                idx = int(
                    idx.replace("-", "").strip()
                )  # Adding and removing for accurate token counts
                memory_desc = memory_desc.strip()
                # print("IDX", idx)
                # print("MEMORY DESC", memory_desc)
                memory_type = character.memory.get_observation_type(idx)
                # print("MEMORY TYPE", memory_type, memory_type == MemoryType.REFLECTION, memory_type.value == MemoryType.REFLECTION.value)
                key = (
                    "reflections"
                    if memory_type.value == MemoryType.REFLECTION.value
                    else "observations"
                )
                # print("KEY", key)

                # Prepend the memory description with a tab and dash for better formatting
                memory_desc = "\n- " + memory_desc

                categorized_memories[key].append(
                    full_memory  # if key == "reflections" else memory_desc
                )

            # Use the relevant memories primer if no reflections or observations are found
            reflections_lmtd = categorized_memories["reflections"] or [
                relevant_memories_primer[2]
            ]

            # print("REFLECTIONS LIMITED")
            # for ref in reflections_lmtd:
            #     print("-", ref)

            observations_lmtd = categorized_memories["observations"] or [
                relevant_memories_primer[3]
            ]

            # # Remove the leading newline from the first element of the lists
            # if reflections_lmtd:
            #     reflections_lmtd[0] = reflections_lmtd[0].strip()
            # if observations_lmtd:
            #     observations_lmtd[0] = observations_lmtd[0].strip()

            # Construct the user prompt by combining impressions, memories, and insight questions
            user_prompt_list = (
                [relevant_memories_primer[0]]
                + reflections_lmtd
                + [relevant_memories_primer[1]]
                + observations_lmtd
                + insight_q_prompt
            )

            # Join the user prompt list into a single string
            user_prompt_str = "".join(user_prompt_list)

            # print("~" * 100)
            # print("USER PROMPT (reflect):\n", user_prompt_str)
            # print("~" * 100)

            success = False
            # Attempt to generate a response from the GPT model
            while not success:
                try:
                    response = cls.gpt_handler.generate(
                        system=system_prompt,
                        user=user_prompt_str,
                        character=character,
                        game=game,
                    )
                    # Parse the response as JSON to extract new generalizations
                    new_generalizations = json.loads(response)
                except json.JSONDecodeError as e:
                    cls.logger.error(f"JSON decoding error: {e}")
                    continue  # Retry if JSON decoding fails
                else:
                    success = True
                    # Add the new generalizations to the character's memory
                    cls.add_generalizations_to_memory(
                        game, character, new_generalizations
                    )

            # Update the relevant memories to exclude those already reflected upon
            relevant_memories = list(
                OrderedSet(relevant_memories) - OrderedSet(relevant_memories_limited)
            )

            # Recalculate the token count for the remaining relevant memories
            relevant_memories_token_count = get_prompt_token_count(
                content=relevant_memories, role=None, pad_reply=False
            )

            # print("RELEVANT MEMORIES (after)", OrderedSet(relevant_memories))
            # print("RELEVANT MEMORIES TOKEN COUNT (after)", relevant_memories_token_count)
            # print("RELEVANT MEMORIES COUNT (after)", len(relevant_memories))
            # print("RELEVANT MEMORIES ID'S (after)", [mem.node_id for mem in relevant_memories])

            # print("-" * 100)

    @classmethod
    def add_generalizations_to_memory(
        cls, game: "Game", character: "Character", generalizations: Dict
    ):
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
    def add_new_generalization_helper(
        cls, game: "Game", character: "Character", generalization: Dict
    ):
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
    def add_new_generalizations(
        cls, game: "Game", character: "Character", generalizations: Dict
    ):
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
    def update_existing_generalizations(
        cls, game: "Game", character: "Character", generalizations: Dict
    ):
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
                    cls.logger.error(
                        f"Missing required key in updated generalization: {ref}"
                    )
                    cls.add_new_generalization_helper(game, character, ref)
                    continue
                except ValueError:
                    # If the index cannot be converted to an integer, log the error and create a new reflection
                    cls.logger.error(
                        f"Invalid index format in updated generalization: {ref}"
                    )
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
                                description=statement,
                                thing=character,
                                needs_summary=False,
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
