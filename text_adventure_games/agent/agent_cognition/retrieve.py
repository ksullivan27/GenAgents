"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent_cognition/act.py
Description: defines how agents select an action given their perceptions and memory
"""

# TODO: Check where retrieve is called, trying to get a dict from the queries to their embeddings as input to reduce
# the number of calls to GPT.

from __future__ import annotations  # Enables postponed evaluation of type annotations

print("Importing Retrieve")

from typing import TYPE_CHECKING, Union, List  # Allows conditional imports for type hints
import numpy as np  # Imports NumPy for numerical operations
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # Imports cosine similarity function for measuring similarity between vectors

# local imports
if (
    TYPE_CHECKING
):  # Ensures that the following imports are only evaluated during type checking
    print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game  # Imports Game class for type hinting

    print(f"\t{__name__} calling Type Checking imports for Character")
    from text_adventure_games.things.characters import (
        Character,
    )

    print(f"\t{__name__} calling Type Checking imports for MemoryType")
    from text_adventure_games.agent.memory_stream import MemoryType


print(f"\t{__name__} calling imports for General")
from text_adventure_games.utils.general import (  # Imports utility functions for general use
    combine_dicts_helper,  # Function to combine dictionaries
    get_text_embedding,  # Function to obtain text embeddings
)

print(f"\t{__name__} calling imports for Consts")
from text_adventure_games.utils.consts import get_models_config

# Importing GptCallHandler to interface with OpenAI client
print(f"\t{__name__} calling imports for GptHelpers")
from ...gpt.gpt_helpers import GptCallHandler


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


# Constants for reflection output limits and retry attempts
RETRIEVE_MAX_OUTPUT = 512  # Maximum output length for reflections
RETRIEVE_MAX_MEMORIES = 25  # Maximum number of memories to retrieve
RETRIEVE_RETRIES = 3  # Number of retries for reflection generation


class Retrieve:
    gpt_handler = None  # Class-level attribute to store the shared GPT handler
    model_params = {
        # "max_output_tokens": 512,
        # "temperature": 1,
        # "top_p": 1,
        # "frequency_penalty": 0,
        # "presence_penalty": 0,
        # "max_retries": RETRIEVE_RETRIES,
    }

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """

        print(f"-\tRetrieve Module is initializing GptCallHandler")

        # Initialize the GPT handler if it hasn't been set up yet
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(
                model_config_type="retrieve", **cls.model_params
            )

    @classmethod
    def retrieve(
        cls,
        game: "Game",
        character: "Character",
        query: Union[dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]], list[str], str] = None,
        n: int = -1,
        include_idx=False,
        percentile: float = 0.75,
        method: str = "mean",
        threshold: float = 0.5,
        memory_type: list[str] = ["ACTION", "DIALOGUE", "REFLECTION", "PERCEPT"],
    ):
        """
        Retrieve relevant memory nodes for a given character based on a query.
        This function gathers keywords, gets all memories associated with them, ranking these memory nodes and returning a
        list of descriptions or indexed descriptions.

        Using character goals, current perceptions, and possibly additional inputs, parse these for keywords, get a list of
        memory nodes based on the keywords, then calculate the retrieval score for each and return a ranked list of
        memories.

        Args:
            game (Game): The game context in which the character exists.
            character (Character): The character whose memory is being queried.
            query (Union[dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]], list[str], str],
            optional): An optional search query to refine the memory retrieval. Defaults to None.
            n (int, optional): The maximum number of memory nodes to return. Defaults to -1, which returns all.
            include_idx (bool, optional): If True, includes the index of each memory node in the output. Defaults to
                                          False.
            percentile (float, optional): The percentile threshold for filtering memory scores. Defaults to 0.75.
            method (str, optional): The method to use for aggregation ('mean' or 'median'). Defaults to 'mean'.
            threshold (float, optional): The cosine similarity threshold to consider a keyword as a match (default is
                                         0.9).
            memory_type (list[str], optional): A list of memory types to consider during retrieval. Defaults to
                                               ["ACTION", "DIALOGUE", "REFLECTION", "PERCEPT"].

        Returns:
            list or None: A list of memory node descriptions or indexed descriptions, or None if no relevant memory
                          nodes are found.
        """

        print("-\tInitializing Retrieve")

        # Initialize the GPT handler if it hasn't been set up yet
        cls.initialize_gpt_handler()

        # Gather keywords for searching relevant memory nodes based on the game and character context.
        # This gets the keywords associated with recent memories and goals, along with the optional query.
        keywords_to_embeddings_dict = Retrieve._gather_keywords_for_search(game, character, query)

        # Retrieve memory node IDs that are relevant to the gathered search keys
        memory_node_ids = Retrieve._get_relevant_memory_ids(
            keywords_to_embeddings_dict,
            character,
            memory_type,
            threshold=threshold,
        )

        # If no relevant memory node IDs are found, return None
        if len(memory_node_ids) == 0:
            return None

        # Rank the memory nodes based on recency, importance, and relevance
        ranked_memory_ids = Retrieve._rank_nodes(
            character, memory_node_ids, query, percentile, method
        )

        # If a positive integer is specified, limit the number of returned memory nodes
        # Use negative indexing to select the last 'n' nodes, as they are sorted by relevancy
        if n > 0:
            ranked_memory_ids = ranked_memory_ids[-n:]

        # Check if the index should be included in the output
        # If not, return a list of memory node descriptions as strings
        if not include_idx:
            return [
                f"{character.memory.observations[t[0]].node_description}\n"
                for t in ranked_memory_ids
            ]
        else:
            # If including index, return a list of indexed memory node descriptions
            return [
                f"{t[0]}. {character.memory.observations[t[0]].node_description}"
                for t in ranked_memory_ids
            ]

    @classmethod
    def _rank_nodes(
        cls,
        character: "Character",
        node_ids: List[int],
        query: Union[dict[str, dict[str, dict[str, np.ndarray]]], list[str], str],
        percentile: float = 0.75,
        method: str = "mean",
    ) -> List[tuple]:
        """
        Rank memory nodes based on recency, importance, and relevance to a given query.
        This function calculates scores for each node and returns them sorted by their total score. It's a wrapper for
        the component scores that sum to define total node score.

        Args:
            character (Character): The character whose memory is being evaluated.
            node_ids (List[int]): A list of memory node IDs to be ranked.
            query (Union[dict[str, dict[str, dict[str, np.ndarray]]], list[str], str]): The search query or queries used to
                                                                                   assess relevance.
            percentile (float): The percentile threshold to filter relevance scores (default is 0.75).
            method (str): The method to use for aggregation of relevance scores ('mean' or 'median', default is 'mean').

        Returns:
            List[tuple]: A sorted list of tuples containing node IDs and their corresponding scores.
        """

        # Calculate the recency score for each memory node based on the character's memory
        recency = Retrieve._calculate_node_recency(character, node_ids)

        # Calculate the importance score for each memory node based on the character's memory
        importance = Retrieve._calculate_node_importance(character, node_ids)

        # Calculate the relevance score for each memory node in relation to the provided query
        relevance = Retrieve._calculate_node_relevance(
            character, node_ids, query, percentile, method
        )

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

    @classmethod
    def _calculate_node_recency(cls, character, memory_ids):
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
        return Retrieve._minmax_normalize(recency, 0, 1)

    @classmethod
    def _calculate_node_importance(cls, character, memory_ids):
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
        importances = [
            character.memory.observations[i].node_importance for i in memory_ids
        ]

        # Normalize and return the importance scores to a range between 0 and 1 for consistent scaling
        return Retrieve._minmax_normalize(importances, 0, 1)

    @classmethod
    def _calculate_node_relevance(
        cls,
        character: "Character",
        memory_ids: list[int],
        query: Union[
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ],
        percentile: float = 0.75,
        method: str = "mean",
    ):
        """
        Calculate the relevance scores of memory nodes based on their similarity to a given query.
        This function uses embeddings to assess how closely related each memory node is to the query or to default
        queries if none is provided.

        Args:
            character (Character): The character whose memory is being evaluated.
            memory_ids (list[int]): A list of memory node IDs for which relevance scores are calculated.
            query (Union[dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]], list[str], str]):
                An optional search query to determine relevance. If not provided, default queries are used.
            percentile (float): The percentile threshold to filter relevance scores (default is 0.75).
            method (str): The method to use for aggregation of relevance scores ('mean' or 'median', default is 'mean').

        Returns:
            list: A list of normalized relevance scores for the specified memory nodes, scaled between 0 and 1.
        """

        # Retrieve the embeddings for each memory node specified by the memory IDs
        memory_embeddings = [character.memory.get_embedding(i) for i in memory_ids]

        # Check if a query is provided to determine relevance
        if query:
            # If a query is a string, compute its embeddings
            if isinstance(query, str):
                query_embeddings = [Retrieve.gpt_handler.generate_embeddings(query)]
            # If a query is a list of strings, compute the embeddings for each string
            elif isinstance(query, list):
                query_embeddings = [Retrieve.gpt_handler.generate_embeddings(q) for q in query]
            # If a query is a dictionary with embeddings, extract the embeddings
            elif isinstance(query, dict) and 'embeddings' in query:
                query_embeddings = [q_dict["embeddings"] for q_dict in query.values()]
            else:
                # raise an error if the query is not in the correct format
                raise ValueError(f"Query is not in the correct format: {query}")
            # Compute the cosine similarity between the memory embeddings and the query embeddings
            relevances = cosine_similarity(memory_embeddings, query_embeddings)
        else:
            # If no query is passed, use default queries (e.g., persona, goals) to assess relevance
            default_embeddings = character.memory.get_query_embeddings()
            relevances = cosine_similarity(memory_embeddings, default_embeddings)

        # Get aggregated relevance scores across all queries
        scored_relevances = Retrieve._score_cos_sim(
            relevances, percentile=percentile, method=method
        )

        # Normalize the relevance scores to a range between 0 and 1 for consistent scaling
        return Retrieve._minmax_normalize(scored_relevances, 0, 1)

    @classmethod
    def _get_relevant_memory_ids(
        cls,
        search_keys: dict[str, dict[str, np.ndarray]],
        character: "Character",
        memory_type: list[str],
        threshold: float = 0.5,
    ) -> list[int]:
        """
        Retrieve a list of memory node IDs that are relevant to the provided search keywords. This function aggregates
        node IDs based on keyword types and their associated search words from the character's memory.

        Args:
            search_keys (dict[str, dict[str, np.ndarray]]): A dictionary where keys are keyword types and values
                                                            are dictionaries of keywords mapped to their embeddings.
            character (Character): The character whose memory is being queried for relevant node IDs.
            memory_type (list[str]): A list of memory types to consider during retrieval.
            threshold (float): The cosine similarity threshold to consider a keyword as a match (default is 0.5).

        Returns:
            list[int]: A list of unique memory node IDs that match the search keywords.
        """

        # Initialize an empty list to store relevant memory node IDs
        memory_ids = []

        # Collect all keywords and their associated node IDs from all keyword types
        all_keywords = []
        all_node_ids = []
        for kw_type_dict in character.memory.keyword_nodes.values():
            for keyword, node_ids in kw_type_dict.items():
                all_keywords.append(keyword)
                all_node_ids.append(node_ids)

        # Flatten the search keys dictionary to get all embeddings
        search_word_embeddings = [embedding for kw_embedding_dict in search_keys.values() for embedding in kw_embedding_dict.values()]

        # Get the embeddings of all keywords
        keyword_embeddings = [
            character.memory.keyword_embeddings[kw] for kw in all_keywords
        ]

        # Compute cosine similarity between all search words and all keywords
        similarities = cosine_similarity(search_word_embeddings, keyword_embeddings)

        # Iterate over each search word and its corresponding similarity scores
        for search_word_similarities in similarities:
            for i, similarity in enumerate(search_word_similarities):
                if similarity >= threshold:
                    # Filter node IDs to include only those with specific valid memory types
                    memory_ids.extend(
                        node_id
                        for node_id in all_node_ids[i]
                        if character.memory.get_observation(node_id)
                        and character.memory.get_observation(node_id).node_type
                        in memory_type
                    )

        # Return a list of unique memory node IDs by converting the list to a set and back to a list
        return list(set(memory_ids))

    @classmethod
    def _gather_keywords_for_search(
        cls,
        game,
        character,
        query: Union[
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ] = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Collect keywords for searching based on the character's recent memories, goals, and an optional query.
        This function extracts relevant keywords from the character's observations, current goals, and the provided
        query to facilitate memory retrieval.

        Args:
            game (Game): The game context that provides parsing capabilities for extracting keywords.
            character (Character): The character whose memories and goals are being analyzed for keyword extraction.
            query (Union[dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]], list[str], str],
            optional): An optional search query from which to extract additional keywords.

        Returns:
            dict[str, dict[str, np.ndarray]]: A dictionary of keyword types mapping to dictionaries of keywords and
                                              their embeddings.
        """

        # Initialize an empty dictionary to store keywords and their embeddings for retrieval
        retrieval_kwds_with_embeddings = {}

        ### 1. Gather keywords from the last 'n' memories, simulating "short term memory" ###

        # Iterate over the last 'n' memories
        for node in character.memory.observations[-character.memory.lookback :]:
            # Extract keywords from the node's description
            if node_kwds := game.parser.extract_keywords(node.node_description):
                # Iterate over the extracted keyword types
                for kw_type, keywords in node_kwds.items():
                    if kw_type not in retrieval_kwds_with_embeddings:
                        retrieval_kwds_with_embeddings[kw_type] = {}
                    # Iterate over the keywords
                    for keyword in keywords:
                        # Check if the keyword has an associated embedding in the memory stream
                        if not (keyword_embedding := character.memory.get_keyword_embeddings(keyword)):
                            # If not, generate an embedding for the keyword
                            keyword_embedding = Retrieve.client.generate_embeddings(keyword)
                        # Add the keyword and its embedding to the retrieval dictionary
                        retrieval_kwds_with_embeddings[kw_type][keyword] = keyword_embedding

        ### 2. Gather keywords from the character's current goals ###

        try:
            # Attempt to retrieve the current goals for the current round as a string
            current_goals = character.goals.get_goals(
                round=game.round,
                priority="all",
                include_node_ids=True,
                include_description=False,
                include_priority_levels=False,
                include_scores=False,
                progress_as_percentage=False,
                to_str=False,
                list_prefix=""
            )
        except (AttributeError, KeyError):
            try:
                # Attempt to retrieve the current goals for the previous round as a string
                current_goals = character.goals.get_goals(
                    round=max(game.round - 1, 0),
                    priority="all",
                    include_node_ids=True,
                    include_description=False,
                    include_priority_levels=False,
                    include_scores=False,
                    progress_as_percentage=False,
                    to_str=False,
                    list_prefix="",
                )
            except (AttributeError, KeyError):
                # If the goals cannot be accessed, set current_goals to None
                current_goals = None

        # If current goals are available, extract keywords from them and combine with existing keywords
        if current_goals:
            for node_id in current_goals:
                if goal_description := character.memory.get_observation_description(node_id):
                    if goal_kwds := game.parser.extract_keywords(goal_description):
                        for kw_type, keywords in goal_kwds.items():
                            if kw_type not in retrieval_kwds_with_embeddings:
                                retrieval_kwds_with_embeddings[kw_type] = {}
                            for keyword in keywords:
                                # Add the keyword and its embedding to the retrieval dictionary
                                retrieval_kwds_with_embeddings[kw_type][keyword] = (
                                    character.memory.get_keyword_embedding(keyword)
                                )

        ### 3. Gather keywords from the provided query, if any ###

        if query:
            if isinstance(query, str):
                query = [query]
            elif isinstance(query, (list, tuple, set)):
                for q in query:
                    if query_kwds := game.parser.extract_keywords(q):
                        for kw_type, keywords in query_kwds.items():
                            if kw_type not in retrieval_kwds_with_embeddings:
                                retrieval_kwds_with_embeddings[kw_type] = {}
                            for keyword in keywords:
                                # Generate embeddings for the query keywords
                                retrieval_kwds_with_embeddings[kw_type][keyword] = Retrieve.client.generate_embeddings(keyword)
            elif isinstance(query, dict):
                # query -> ['embeddings', 'keywords'] -> {'kw_type': {'keyword': embedding}}
                for embedding_keyword_dicts in query.values():
                    for kw_embedding_dict in embedding_keyword_dicts['keywords']:
                        for kw_type, keywords in kw_embedding_dict.items():
                            if kw_type not in retrieval_kwds_with_embeddings:
                                retrieval_kwds_with_embeddings[kw_type] = {}
                            for keyword, embedding in keywords.items():
                                retrieval_kwds_with_embeddings[kw_type][keyword] = embedding

        # Return the dictionary of gathered keywords and their embeddings for retrieval
        return retrieval_kwds_with_embeddings

    @classmethod
    def _minmax_normalize(cls, lst, target_min: int, target_max: int):
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
                fixed_list = [
                    x if x else 0 for x in lst
                ]  # Replace None or invalid values with 0
                min_val = np.nanmin(fixed_list)  # Get the minimum value, ignoring NaNs
                max_val = np.nanmax(fixed_list)  # Get the maximum value, ignoring NaNs

        # Calculate the range of the values
        range_val = max_val - min_val

        # If there is no variance in the values (all values are the same), return a list of 0.5 for each element
        if range_val == 0:
            return [0.5] * len(lst)

        # Attempt to normalize the values to the specified range
        try:
            out = [
                ((x - min_val) * (target_max - target_min) / range_val + target_min)
                for x in lst
            ]
        except TypeError:
            # If there are None values in the list, replace them with the midpoint value of the range
            mid_val = (max_val + min_val) / 2  # Calculate the midpoint
            tmp = [x or mid_val for x in lst]  # Replace None values with the midpoint
            out = [
                ((x - min_val) * (target_max - target_min) / range_val + target_min)
                for x in tmp
            ]  # Normalize the adjusted list

        # Return the normalized values as a numpy array
        return np.array(out)

    @classmethod
    def _score_cos_sim(
        cls,
        relevances: np.ndarray, percentile: float = 0.9, method: str = "mean"
    ) -> np.ndarray:
        """
        Calculate the mean or median of the cosine similarity scores above a specified percentile.

        Args:
            relevances (np.ndarray): Array of shape (m, q) containing cosine similarity scores.
            percentile (float): The percentile threshold (e.g., 0.9) to filter the scores.
            method (str): The method to use for aggregation ('mean' or 'median').

        Returns:
            np.ndarray: An array of shape (m) with the aggregated scores.
        """
        if method not in ["mean", "median"]:
            raise ValueError("Method must be either 'mean' or 'median'")

        # Calculate the threshold value for the specified percentile
        threshold = np.percentile(relevances, percentile * 100, axis=1)

        # Filter the relevances to only include scores above the threshold
        filtered_scores = np.array(
            [
                relevances[i][relevances[i] >= threshold[i]]
                for i in range(relevances.shape[0])
            ]
        )

        # Calculate the mean or median of the filtered scores
        if method == "mean":
            result = np.array([np.mean(scores) for scores in filtered_scores])
        else:
            result = np.array([np.median(scores) for scores in filtered_scores])

        return result

    @classmethod
    def get_query_keywords_and_embeddings(
        cls,
        game: "Game",
        query: Union[list[str], str]
    ) -> dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]]:
        """
        Convert a list of query strings into a dictionary of 'keywords' and 'embeddings' keys.

        The 'keywords' key maps to a dictionary of keyword_type (str) mapping to dictionaries of keywords (str) mapping
        to their embeddings (np.ndarray). The 'embeddings' key maps to a dictionary of queries (str) mapping to
        embeddings (np.ndarray).

        Args:
            game (Game): The game context that provides parsing capabilities for extracting keywords.
            query (Union[list[str], str]): A list of query strings or a single query string to be converted.

        Returns:
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]]: A dictionary containing the
            keywords and their embeddings, and the query strings and their embeddings.
        """

        # Initialize the result dictionary with 'keywords' and 'embeddings' keys
        result = {
            'keywords': {},
            'embeddings': {}
        }

        # If a query is a string, convert it to a list for processing
        if isinstance(query, str):
            query = [query]

        # Iterate over each query string in the list
        for q in query:
            # Extract keywords from the query string using the extract_keywords method
            keywords_dict = game.parser.extract_keywords(q)

            # Generate embeddings for the query string using the generate_embeddings method
            query_embedding = Retrieve.client.generate_embeddings(q)

            # Add the query string and its embedding to the 'embeddings' key in the result dictionary
            result['embeddings'][q] = query_embedding

            # Iterate over the keyword types and their corresponding keywords
            for kw_type, keywords in keywords_dict.items():
                if kw_type not in result['keywords']:
                    result['keywords'][kw_type] = {}
                for keyword in keywords:
                    # Generate embeddings for each keyword
                    keyword_embedding = Retrieve.client.generate_embeddings(keyword)
                    # Add the keyword and its embedding to the 'keywords' key in the result dictionary
                    result['keywords'][kw_type][keyword] = keyword_embedding

        return result
