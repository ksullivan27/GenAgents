"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent_cognition/act.py
Description: defines how agents select an action given their perceptions and memory
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# local imports
if TYPE_CHECKING:
    from text_adventure_games.games import Game
    from text_adventure_games.things.characters import Character
from text_adventure_games.utils.general import (combine_dicts_helper,
                                                get_text_embedding)

# initially focus on the people that are around the current character

# memory ranking:
# recency: gamma^(curr_idx - retrieved idx) --> will need to do linear conversion (similar to min/max scaling) to [0,1]
# importance: interpreted by GPT --> rescale to [0, 1]
# Relevance: cosine similarity --> Could probably just take the absolute value here since it is [-1, 1]

# What is the query for the action selection?
# goals, surroundings, num ticks left to vote?
# Goals should be generated based on the game information and decomposed into a series of sub-tasks

# what is the query for dialogue?
# initial is the dialogue command, subsequent is the last piece of dialogue

def retrieve(game: "Game", character: "Character", query: str = None, n: int = -1, include_idx=False):
    """
    Retrieve relevant memory nodes for a given character based on a query.
    This function gathers keywords, ranks memory nodes, and returns a list of descriptions or indexed descriptions.

    Using character goals, current perceptions, and possibly additional inputs,
    parse these for keywords, get a list of memory nodes based on the keywords,
    then calculate the retrieval score for each and return a ranked list of memories

    Args:
        game (Game): The game context in which the character exists.
        character (Character): The character whose memory is being queried.
        query (str, optional): An optional search query to refine the memory retrieval. Defaults to None.
        n (int, optional): The maximum number of memory nodes to return. Defaults to -1, which returns all.
        include_idx (bool, optional): If True, includes the index of each memory node in the output. Defaults to False.

    Returns:
        list or None: A list of memory node descriptions or indexed descriptions, or None if no relevant memory nodes
        are found.
    """
    
    # TODO: refine the inputs used to assess keywords for memory retrieval
    # TODO: WHAT IS THE QUERY STRING FOR RELEVANCY (COS SIM)?

    # Gather keywords for searching relevant memory nodes based on the game and character context
    search_keys = gather_keywords_for_search(game, character, query)

    # Retrieve memory node IDs that are relevant to the gathered search keys
    memory_node_ids = get_relevant_memory_ids(search_keys, character)

    # If no relevant memory node IDs are found, return None
    if len(memory_node_ids) == 0:
        return None

    # TODO: Determine how many memory nodes should be returned; default behavior is to return all
    ranked_memory_ids = rank_nodes(character, memory_node_ids, query)

    # If a positive integer is specified, limit the number of returned memory nodes
    # Use negative indexing to select the last 'n' nodes, as they are sorted by relevancy
    if n > 0:
        ranked_memory_ids = ranked_memory_ids[-n:]

    # Check if the index should be included in the output
    # If not, return a list of memory node descriptions as strings
    if not include_idx:
        return [f"{character.memory.observations[t[0]].node_description}\n" for t in ranked_memory_ids]
    else:
        # If including index, return a list of indexed memory node descriptions
        return [f"{t[0]}. {character.memory.observations[t[0]].node_description}" for t in ranked_memory_ids]


def rank_nodes(character, node_ids, query):
    """
    Rank memory nodes based on recency, importance, and relevance to a given query. 
    This function calculates scores for each node and returns them sorted by their total score. It's a wrapper for the
    component scores that sum to define total node score

    Args:
        character: The character whose memory is being evaluated.
        node_ids: A list of memory node IDs to be ranked.
        query: The search query used to assess relevance.

    Returns:
        list: A sorted list of tuples containing node IDs and their corresponding scores.
    """

    # Calculate the recency score for each memory node based on the character's memory
    recency = calculate_node_recency(character, node_ids)

    # Calculate the importance score for each memory node based on the character's memory
    importance = calculate_node_importance(character, node_ids)

    # Calculate the relevance score for each memory node in relation to the provided query
    relevance = calculate_node_relevance(character, node_ids, query)

    # Scale the raw scores by the character's memory weights for recency, importance, and relevance
    # Currently, all weights are set to 1
    recency = character.memory.recency_alpha * recency
    importance = character.memory.importance_alpha * importance
    relevance = character.memory.relevance_alpha * relevance

    # Calculate the total score by summing the scaled scores
    total_score = recency + importance + relevance

    # Combine node IDs with their corresponding total scores into tuples
    node_scores = zip(node_ids, list(total_score))

    # Return the sorted list of node scores, ordered by score in ascending order
    return sorted(node_scores, key=lambda x: x[1])


def calculate_node_recency(character, memory_ids):
    """
    Calculate the recency scores for a list of memory nodes based on their age using an exponential decay assumption.
    This function determines how recent each memory node is relative to the most recent observation.

    Args:
        character: The character whose memory is being evaluated.
        memory_ids: A list of memory node IDs for which recency scores are calculated.

    Returns:
        list: A list of normalized recency scores for the specified memory nodes, scaled between 0 and 1.
    """

    # The most recent memory node is represented by the last index, which corresponds to the total number of
    # observations made
    latest_node = character.memory.num_observations

    # Calculate the "age" of each memory node by taking the difference between the latest observation index and each
    # relevant node ID
    # This results in a list of recency scores, where more recent nodes have higher scores
    recency = [character.memory.gamma ** (latest_node - i) for i in memory_ids]

    # Normalize the recency scores to a range between 0 and 1 for consistent scaling
    return minmax_normalize(recency, 0, 1)


def calculate_node_importance(character, memory_ids):
    """
    Calculate the importance scores for a list of memory nodes. 
    This function retrieves the importance values of specified memory nodes and normalizes them for consistent scaling.

    Args:
        character: The character whose memory is being evaluated.
        memory_ids: A list of memory node IDs for which importance scores are calculated.

    Returns:
        list: A list of normalized importance scores for the specified memory nodes, scaled between 0 and 1.
    """

    # Retrieve the importance scores for each memory node specified by the memory IDs
    importances = [character.memory.observations[i].node_importance for i in memory_ids]

    # Normalize and return the importance scores to a range between 0 and 1 for consistent scaling
    return minmax_normalize(importances, 0, 1)


def calculate_node_relevance(character, memory_ids, query):
    """
    Calculate the relevance scores of memory nodes based on their similarity to a given query. 
    This function uses embeddings to assess how closely related each memory node is to the query or to default queries
    if none is provided.

    Args:
        character: The character whose memory is being evaluated.
        memory_ids: A list of memory node IDs for which relevance scores are calculated.
        query: An optional search query to determine relevance. If not provided, default queries are used.

    Returns:
        list: A list of normalized relevance scores for the specified memory nodes, scaled between 0 and 1.
    """

    # Retrieve the embeddings for each memory node specified by the memory IDs
    memory_embeddings = [character.memory.get_embedding(i) for i in memory_ids]

    # Check if a query is provided to determine relevance
    if query:
        # If a query is passed, compute the embedding for the query and use it to rank node relevance
        query_embedding = get_text_embedding(query).reshape(1, -1)
        relevances = cosine_similarity(memory_embeddings, query_embedding).flatten()
    else:
        # If no query is passed, use default queries (e.g., persona, goals) to assess relevance
        # Calculate the raw relevance scores based on the default embeddings
        default_embeddings = character.memory.get_query_embeddings()
        raw_relevance = cosine_similarity(memory_embeddings, default_embeddings)
        # Take the maximum relevance score from the default queries for each memory node
        relevances = np.max(raw_relevance, axis=1)

    # Normalize the relevance scores to a range between 0 and 1 for consistent scaling
    return minmax_normalize(relevances, 0, 1)


def get_relevant_memory_ids(seach_keys, character):
    """
    Retrieve a list of memory node IDs that are relevant to the provided search keywords. 
    This function aggregates node IDs based on keyword types and their associated search words from the character's
    memory.

    Args:
        seach_keys: A dictionary where keys are keyword types and values are lists of search words.
        character: The character whose memory is being queried for relevant node IDs.

    Returns:
        list: A list of unique memory node IDs that match the search keywords.
    """

    # Initialize an empty list to store relevant memory node IDs
    memory_ids = []

    # Iterate over each keyword type and its associated search words in the search keys
    for kw_type, search_words in seach_keys.items():
        # For each search word, retrieve the corresponding node IDs from the character's memory
        for w in search_words:
            node_ids = character.memory.keyword_nodes[kw_type][w]
            # Extend the memory_ids list with the retrieved node IDs
            memory_ids.extend(node_ids)

    # Return a list of unique memory node IDs by converting the list to a set and back to a list
    return list(set(memory_ids))


def gather_keywords_for_search(game, character, query):
    """
    Collect keywords for searching based on the character's recent memories, goals, and an optional query. 
    This function extracts relevant keywords from the character's observations, current goals, and the provided query to
    facilitate memory retrieval.

    Args:
        game: The game context that provides parsing capabilities for extracting keywords.
        character: The character whose memories and goals are being analyzed for keyword extraction.
        query: An optional search query from which to extract additional keywords.

    Returns:
        dict: A dictionary of keywords gathered from the character's memories, goals, and the query.
    """

    # Initialize an empty dictionary to store keywords for retrieval
    retrieval_kwds = {}

    # 1. Gather keywords from the last 'n' memories, simulating "short term memory"
    for node in character.memory.observations[-character.memory.lookback:]:
        # Extract keywords from the node's description and combine them into the retrieval dictionary
        if node_kwds := game.parser.extract_keywords(node.node_description):
            retrieval_kwds = combine_dicts_helper(existing=retrieval_kwds, new=node_kwds)

    # 2. Gather keywords from the character's current goals
    # TODO: Confirm how goals are stored and if any parsing is needed to convert them to a string
    prev_round = max(0, game.round - 1)  # Get the previous round number
    try:
        # Attempt to retrieve the current goals for the previous round as a string
        current_goals = character.goals.get_goals(round=prev_round, as_str=True)
    except AttributeError:
        # If the goals cannot be accessed, set current_goals to None
        current_goals = None

    # If current goals are available, extract keywords from them and combine with existing keywords
    if current_goals:
        if goal_kwds := game.parser.extract_keywords(current_goals):
            retrieval_kwds = combine_dicts_helper(retrieval_kwds, goal_kwds)

    # 3. Gather keywords from the provided query, if any
    if query:
        if query_kwds := game.parser.extract_keywords(query):
            retrieval_kwds = combine_dicts_helper(retrieval_kwds, query_kwds)

    # TODO: Consider adding more sources for keywords if necessary

    # Return the dictionary of gathered keywords for retrieval
    return retrieval_kwds


def minmax_normalize(lst, target_min: int, target_max: int):
    """
    Normalize a list of values to a specified range using min-max normalization. 
    This function scales the input values to fit within the target minimum and maximum, handling edge cases such as
    empty lists and non-numeric values.

    Args:
        lst: A list of numeric values to be normalized.
        target_min (int): The minimum value of the target range.
        target_max (int): The maximum value of the target range.

    Returns:
        np.ndarray: An array of normalized values scaled to the specified range.
    """

    # Attempt to find the minimum and maximum values in the list
    try:
        min_val = min(lst)  # Get the minimum value
        max_val = max(lst)  # Get the maximum value
    except TypeError:
        # If a TypeError occurs (e.g., due to non-numeric values), use numpy functions to find min and max, ignoring
        # NaNs
        try:
            min_val = np.nanmin(lst)  # Get the minimum value, ignoring NaNs
            max_val = np.nanmax(lst)  # Get the maximum value, ignoring NaNs
        except TypeError:
            # If another TypeError occurs, replace non-numeric values with 0 and then find min and max
            fixed_list = [x if x else 0 for x in lst]  # Replace None or invalid values with 0
            min_val = np.nanmin(fixed_list)  # Get the minimum value, ignoring NaNs
            max_val = np.nanmax(fixed_list)  # Get the maximum value, ignoring NaNs

    # Calculate the range of the values
    range_val = max_val - min_val

    # If there is no variance in the values (all values are the same), return a list of 0.5 for each element
    if range_val == 0:
        return [0.5] * len(lst)

    # Attempt to normalize the values to the specified range
    try:
        out = [((x - min_val) * (target_max - target_min) / range_val + target_min) for x in lst]
    except TypeError:
        # If there are None values in the list, replace them with the midpoint value of the range
        mid_val = (max_val + min_val) / 2  # Calculate the midpoint
        tmp = [x or mid_val for x in lst]  # Replace None values with the midpoint
        out = [((x - min_val) * (target_max - target_min) / range_val + target_min) for x in tmp]  # Normalize the adjusted list

    # Return the normalized values as a numpy array
    return np.array(out)


# def cosine_similarity(x, query):
#     """
#     Get the (normalized) cosine similarity between two vectors

#     Args:
#         x (np.array): vector of interest
#         query (np.array): reference vector

#     Returns:
#         float: the similarity between vectors
#     """
#     return np.dot(x, query) / (np.norm(x) * np.norm(query))
