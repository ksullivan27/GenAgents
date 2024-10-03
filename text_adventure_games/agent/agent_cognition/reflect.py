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

# Local imports for memory management, prompts, and GPT helper functions
from text_adventure_games.agent.memory_stream import MemoryType
from text_adventure_games.assets.prompts import reflection_prompts as rp
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

# Constants for reflection output limits and retry attempts
REFLECTION_MAX_OUTPUT = 512  # Maximum output length for reflections
REFLECTION_RETRIES = 5  # Number of retries for reflection generation


def reflect(game: "Game", character: "Character"):
    """
    Perform a complete reflection; this is composed of _ sub-types of reflection:
    1. reflect on actions from past round (inspired by CLIN)
    2. reflect on goals
    3. reflect on relationships

    Args:
        game (Game): _description_
        character (Character): _description_
    """
    generalize(game, character)
    # reflect_on_goals(game, character)
    # reflect_on_relationships(game, character)


# def generalize(game, character):
#     """
#     Reflection upon understanding of the world

#     Args:
#         game (_type_): _description_
#         character (_type_): _description_
#     """
#     # 1. Get MemoryType.REFLECTION nodes
#     # 2. Get nodes from the current round
#     # 3. Generalize new observations with old reflections to update/add

#     # Get an enumerated list of the action nodes for this character in this round
#     this_round_mem_ids = character.memory.get_observations_by_round(game.round)
#     this_round_mem_desc = character.memory.get_enumerated_description_list(this_round_mem_ids, as_type="str")

#     gpt_generalize(game, character, this_round_mem_desc)


def generalize(game, character):
    """Generates new generalizations based on the character's memories and impressions.

    This function retrieves relevant memories and impressions from the character,
    constructs a prompt for the GPT model, and processes the model's response to
    update the character's memory with new generalizations.

    Args:
        game: The current game instance containing the game state.
        character: The character whose memories and impressions are being processed.

    Returns:
        None: This function does not return a value.

    Raises:
        JSONDecodeError: If the response from the GPT model cannot be parsed as JSON.
    """

    model_params = {
        "api_key_org": "Helicone",
        "model": "gpt-4",
        "max_tokens": REFLECTION_MAX_OUTPUT,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_retries": 5,
    }

    # Initialize the GPT call handler with the specified model parameters
    gpt_handler = GptCallHandler(**model_params)

    # Set the number of memories to retrieve for each query
    memories_per_retrieval = 25

    # Construct the system prompt by combining character information and the generalization prompt
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
        # Calculate the token count for the impressions
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
        gpt_handler.model_context_limit,
        gpt_handler.max_tokens,
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
                response = gpt_handler.generate(
                    system=system_prompt, user=user_prompt_str
                )
                # Parse the response as JSON to extract new generalizations
                new_generalizations = json.loads(response)
            except json.JSONDecodeError:
                continue  # Retry if JSON decoding fails
            else:
                success = True
                # Add the new generalizations to the character's memory
                add_generalizations_to_memory(game, character, new_generalizations)

        # Update the relevant memories to exclude those already reflected upon
        relevant_memories = list(
            OrderedSet(relevant_memories) - OrderedSet(relevant_memories_limited)
        )
        # Recalculate the token count for the remaining relevant memories
        relevant_memories_token_count = get_prompt_token_count(
            content=relevant_memories, role=None, pad_reply=False
        )


def add_generalizations_to_memory(
    game: "Game",
    character: "Character",
    generalizations: Dict,
):
    """
    Parse the gpt-generated generalizations dict for new reflections.
    The structure of this obviously depends on the prompt used.

    Args:
        generalizations (Dict): _description_

    Returns:
        None
    """

    # Add new generalizations for the specified character in the given game
    add_new_generalizations(game, character, generalizations)

    # Update existing generalizations for the specified character in the given game
    update_existing_generalizations(game, character, generalizations)


def add_new_generalization_helper(
    game: "Game", character: "Character", generalization: Dict
):
    """Add a new generalization to the character's memory.

    This function attempts to extract a statement from the provided generalization
    and adds it to the character's memory if the statement is valid. If the generalization
    is malformed (i.e., missing the statement), it is skipped without any changes.

    Args:
        game (Game): The game instance that contains the current game state.
        character (Character): The character to which the generalization will be added.
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
        # If the "statement" key is missing, this indicates a malformed reflection; skip processing
        pass
    else:
        # Summarize and score the action described in the statement, obtaining keywords and importance
        # ref_kwds = game.parser.extract_keywords(desc)  # (Commented out) Extract keywords from the description
        # ref_importance = gpt_get_action_importance(desc)  # (Commented out) Get the importance of the action
        _, ref_importance, ref_kwds = game.parser.summarise_and_score_action(
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


def add_new_generalizations(
    game: "Game", character: "Character", generalizations: Dict
):
    """
    Add new generalizations as memories.

    Args:
        game (Game): _description_
        character (Character): _description_
        generalizations (Dict): _description_
    """

    try:
        # Attempt to extract the list of new generalizations from the generalizations dictionary
        new_gens = generalizations["new"]
    except (KeyError, TypeError):
        # If the "new" key is missing or generalizations is not a dictionary, skip processing
        # TODO: maybe build in some retry logic?
        pass
    else:
        # Iterate over each new generalization and add it to the character's memory
        for ref in new_gens:
            add_new_generalization_helper(
                game=game, character=character, generalization=ref
            )


def update_existing_generalizations(
    game: "Game", character: "Character", generalizations: Dict
):
    """
    Find the appropriate reflection nodes that GPT updated and replace the description

    Args:
        character (Character): _description_
        generalizations (Dict): _description_
    """

    try:
        # Attempt to extract the list of updated generalizations from the generalizations dictionary
        updated_gens = generalizations["updated"]
    except (KeyError, TypeError):
        # If the "updated" key is missing or generalizations is not a dictionary, skip processing
        # TODO: again, do we want retry logic for reflections if GPT got JSON structure wrong?
        pass
    else:
        # Iterate over each updated generalization
        for ref in updated_gens:
            try:
                # Attempt to extract the index and statement from the updated generalization
                prev_idx = int(ref["index"])
                statement = ref["statement"]
            except (KeyError, ValueError, TypeError):
                # If the index is missing or invalid, create a new reflection instead
                add_new_generalization_helper(
                    game=game, character=character, generalization=ref
                )
                continue
            else:
                # Get the type of memory associated with the previous index
                memory_type = character.memory.get_observation_type(prev_idx)
                if memory_type and memory_type.value != MemoryType.REFLECTION.value:
                    # If the existing memory is not a reflection, create a new reflection
                    add_new_generalization_helper(
                        game=game, character=character, generalization=ref
                    )
                else:
                    # Update the existing memory node with the new statement and round/tick information
                    _ = character.memory.update_node(
                        node_id=prev_idx,
                        node_round=game.round,
                        node_tick=game.tick,
                        node_description=statement,
                    )
                    # Update the embedding of the memory node with the new description
                    _ = character.memory.update_node_embedding(
                        node_id=prev_idx, new_description=statement
                    )
