"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent_cognition/act.py
Description: defines how agents select an action given their perceptions and memory
"""

# TODO: Check where retrieve is called, trying to get a dict from the queries to their embeddings as input to reduce
# the number of calls to GPT.

from __future__ import annotations  # Enables postponed evaluation of type annotations

circular_import_prints = False

from collections import OrderedDict

if circular_import_prints:
    print("Importing Retrieve")

from typing import (
    TYPE_CHECKING,
    Union,
    List,
    Tuple,
    Literal,
    Optional,
)  # Allows conditional imports for type hints
import numpy as np  # Imports NumPy for numerical operations
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # Imports cosine similarity function for measuring similarity between vectors

# local imports
if (
    TYPE_CHECKING
):  # Ensures that the following imports are only evaluated during type checking
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game  # Imports Game class for type hinting

    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Character")
    from text_adventure_games.things.characters import (
        Character,
    )

if circular_import_prints:
    print(f"\t{__name__} calling Type Checking imports for MemoryStream")
from text_adventure_games.agent.memory_stream import MemoryStream

if circular_import_prints:
    print(f"\t{__name__} calling Type Checking imports for MemoryType")
from text_adventure_games.agent.memory_stream import MemoryType

if circular_import_prints:
    print(f"\t{__name__} calling imports for General")
from text_adventure_games.utils.general import (  # Imports utility functions for general use
    combine_dicts_helper,  # Function to combine dictionaries
    get_text_embedding,  # Function to obtain text embeddings
)

if circular_import_prints:
    print(f"\t{__name__} calling imports for Consts")
from text_adventure_games.utils.consts import get_models_config

# Importing GptCallHandler to interface with OpenAI client
if circular_import_prints:
    print(f"\t{__name__} calling imports for GptHelpers")
from ...gpt.gpt_helpers import GptCallHandler, limit_context_length


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

    DEBUG_MODE = False

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """

        if circular_import_prints:
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
        query: Union[
            dict[
                str,
                Union[
                    dict[
                        str,
                        Union[
                            Tuple[Tuple[float, int], np.ndarray], dict[str, np.ndarray]
                        ],
                    ],
                    dict[str, dict[str, np.ndarray]],
                ],
            ],
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ] = None,
        sort_nodes: Literal[
            "chronological", "reverse chronological", "importance", "reverse importance"
        ] = "importance",
        n: int = -1,
        threshold: float = 0.45,
        percentile: float = 0.75,
        memory_lookback: int | None = None,
        round: int | None = None,
        method: str = "mean",
        include_descriptions: bool = True,
        include_idx: bool = False,
        include_scores: bool = False,
        minmax_scale: bool = True,
        weighted: bool = False,
        max_tokens: int = None,
        prepend: str = "",
        memory_types: list[MemoryType] = [
            MemoryType.ACTION,
            MemoryType.DIALOGUE,
            MemoryType.REFLECTION,
            MemoryType.PERCEPT,
            # MemoryType.GOAL,
        ],
    ) -> list[str]:
        """
        Retrieve relevant memory nodes for a given character based on a query.
        This function gathers keywords, retrieves all memories associated with them, ranks these memory nodes, and
        returns a list of descriptions or indexed descriptions.

        Using character goals, current perceptions, and possibly additional inputs, this method parses these for
        keywords, retrieves a list of memory nodes based on the keywords, calculates the retrieval score for each, and
        returns a ranked list of memories.
        
        Note: Either descriptions or indices are required in the output. If descriptions are not being returned, max
        tokens and prepend are ignored. Max tokens is intended to be used when you want to limit memories displayed in a
        specific format. If you want node IDs without descriptions (for instance to get the MemoryType objects from
        them), these will be returned as integers.

        Args:
            game (Game): The game context in which the character exists.
            character (Character): The character whose memory is being queried.
            query (Union[dict[str, Union[dict[str, Union[Tuple[Tuple[float, int], np.ndarray],
            dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]]],
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str]): An optional search query to determine relevance. If not provided, default queries are used. Defaults
            to None.
            sort_nodes (Literal["chronological", "reverse chronological", "importance", "reverse importance"],
                          optional): Whether to sort the nodes by recency (chronological) or importance. Defaults to
                          "importance".
            n (int, optional): The maximum number of memory nodes to return. Defaults to -1, which returns all.
            threshold (float, optional): The cosine similarity threshold to consider a keyword as a match (default is
                                         0.45).
            percentile (float, optional): The percentile threshold for filtering memory scores. Defaults to 0.75.
            memory_lookback (int | None, optional): The number of memories to look back across. Defaults to None, which
                                                    utilizes the character's memory lookback value. Setting this
                                                    value to -1 will look back across all memories.
            round (int | None, optional): If specified, will only retrieve memories that occurred during or after the
                                          specified round.
            method (str, optional): The method to use for aggregation ('mean' or 'median'). Defaults to 'mean'.
            include_descriptions (bool, optional): If True, includes the descriptions of the memory nodes in the output.
                                                   Defaults to True.
            include_idx (bool, optional): If True, includes the index of each memory node in the output. Defaults to
                                          False.
            include_scores (bool, optional): If True, includes the scores in the output. Defaults to False.
            minmax_scale (bool, optional): Whether to normalize the scores between 0 and 1. Defaults to True.
            weighted (bool, optional): Whether to use weighted retrieval. Defaults to False.
            max_tokens (int, optional): The maximum number of tokens to use for the query. Defaults to None.
            prepend (str, optional): A string to prepend to each memory description. Defaults to "".
            memory_types (list[MemoryType], optional): A list of memory types to consider during retrieval. Defaults to
                                                      ["ACTION", "DIALOGUE", "REFLECTION", "PERCEPT"].

        Returns:
            list[str]: A list of memory node descriptions or indexed descriptions, or an empty list if no relevant memory
                          nodes are found.
        """

        if not (include_descriptions or include_idx):
            raise ValueError("Must include either descriptions or indices in the output")

        if not include_descriptions:
            max_tokens = None
            prepend = ""

        # Check if the query is weighted; it must correctly include weights for each embedding
        if weighted and not (
            isinstance(query, dict)
            and "embeddings" in query
            and isinstance(query["embeddings"], dict)
            and all(
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], tuple)
                and len(value[0]) == 2
                and isinstance(value[0][0], float)
                and isinstance(value[0][1], int)
                and isinstance(value[1], np.ndarray)
                for value in query["embeddings"].values()
            )
        ):
            raise ValueError(
                f"Query must be a dictionary with 'embeddings' mapping to individual query strings that map to tuples, "
                f"where the first index is a tuple of (float, int) and the second index is an np.ndarray. Instead, got {query}"
            )

        if circular_import_prints:
            print("-\tInitializing Retrieve")

        # Initialize the GPT handler if it hasn't been set up yet
        cls.initialize_gpt_handler()

        if memory_lookback != -1:

            # Gather keywords for searching relevant memory nodes based on the game and character context.
            # This gets the keywords associated with recent memories and goals, along with the optional query.
            keywords_to_embeddings_dict = Retrieve._gather_keywords_for_search(
                game=game,
                character=character,
                query=query,
                memory_lookback=memory_lookback,
            )

            # Retrieve memory node IDs that are relevant to the gathered search keys
            memory_node_ids = Retrieve._get_relevant_memory_ids(
                search_keys=keywords_to_embeddings_dict,
                character=character,
                memory_types=memory_types,
                threshold=threshold,
            )

        else:
            memory_node_ids = character.memory.get_observations_by_type(
                obs_type=memory_types
            )
        
        # If a round is specified, only retrieve memories from that round
        if round:
            memory_node_ids = [id for id in memory_node_ids if character.memory.observations[id].node_round >= round]

        # If no relevant memory node IDs are found, return an empty list
        if len(memory_node_ids) == 0:
            return []

        # Rank the memory nodes based on recency, importance, and relevance
        ranked_memory_ids = Retrieve._rank_nodes(
            character=character,
            node_ids=memory_node_ids,
            query=query,
            percentile=percentile,
            method=method,
            minmax_scale=minmax_scale,
            weighted=weighted,
        )

        ranked_memory_scores = cls.format_node_scores(
            character=character,
            node_scores=ranked_memory_ids,
            include_descriptions=include_descriptions,
            include_idx=include_idx,
            prepend=prepend,
        )

        ranked_memory_scores = cls.trim_memory_scores(
            memory_scores=ranked_memory_scores,
            sort_nodes=sort_nodes,
            n=n,
            max_tokens=max_tokens,
        )

        # If scores are not included, extract only the memory descriptions
        if not include_scores:
            ranked_memory_scores = [m[1] for m in ranked_memory_scores]

        if cls.DEBUG_MODE:
            # FOR DEBUGGING
            print(f"Ranked Memory IDs: {ranked_memory_scores}")

        return ranked_memory_scores

    @classmethod
    def format_node_scores(
        cls,
        character: "Character",
        node_scores: list[tuple],
        include_descriptions: bool = True,
        include_idx: bool = False,
        prepend: str = "",
    ) -> list[tuple]:
        """
        Format the node scores for output.

        This method formats the memory node scores into a list of tuples, where each tuple contains
        the score and the corresponding memory node description. The format of the description can
        include the index if specified.

        Args:
            character (Character): The character whose memory is being queried.
            node_scores (list[tuple]): A list of tuples containing memory scores and their corresponding node indices.
            include_descriptions (bool): A flag indicating whether to include the descriptions of the memory nodes in
                                          the output. Defaults to True.
            include_idx (bool): A flag indicating whether to include the index in the output. Defaults to False.
            prepend (str): A string to prepend to each memory description. Defaults to "".

        Returns:
            list[tuple]: A list of formatted memory node scores and descriptions, with or without indices.
        """
        # Check if the index should be included in the output
        if not include_idx:
            # Return a list of memory node descriptions as strings without indices
            return [
                (
                    t[0],
                    f"{prepend}{character.memory.observations[t[1]].node_description}",
                )
                for t in node_scores
            ]
        else:
            if include_descriptions:
                # Return a list of indexed memory node descriptions
                return [
                    (
                        t[0],
                        f"{prepend}{t[1]}. {character.memory.observations[t[1]].node_description}",
                    )
                    for t in node_scores
                ]
            else:
                # Return a list of memory node indices
                return [
                    (t[0], int(t[1])) for t in node_scores
                ]

    @classmethod
    def trim_memory_scores(
        cls,
        memory_scores: list[tuple],
        sort_nodes: Literal["chronological", "reverse chronological", "importance", "reverse importance"] = "importance",
        n: int = -1,
        max_tokens: Union[int, None] = None,
    ) -> list[tuple]:
        """
        Trims and sorts memory scores based on specified criteria.

        This method removes the lowest scoring memories to fit within a maximum token limit,
        sorts the remaining memories based on the specified sorting method, and limits the
        number of returned memory nodes if specified.

        Args:
            cls: The class reference.
            memory_scores (list[tuple]): A list of tuples containing memory scores and descriptions.
            sort_nodes (Literal): The sorting method to apply to the memory scores.
            n (int): The maximum number of memory nodes to return. Defaults to -1 (all).
            max_tokens (Union[int, None]): The maximum number of tokens allowed. If None, no limit is applied.

        Returns:
            list[tuple]: A list of trimmed and sorted memory scores and descriptions.
        """

        ### REMOVE LOWEST SCORING MEMORIES TO FIT MAX TOKENS ###

        if max_tokens:
            # Step 1: Sort by score in ascending order (first element of tuple)
            sorted_data = sorted(memory_scores, key=lambda x: x[0], reverse=False)

            # Extract memory descriptions from sorted data
            memories = [m[1] for m in sorted_data]

            # Step 2: Remove the lowest scoring values based on token limit
            trimmed_memories = limit_context_length(
                history=memories,
                max_tokens=max_tokens,
                keep_most_recent=True,
            )

            # Step 3: Sort back to original order based on index in the original list
            original_order = sorted(
                (
                    (i, score, description)
                    for i, (score, description) in enumerate(memory_scores)
                    if description in trimmed_memories
                ),
                key=lambda x: x[0],
            )

            # Extract the final (score, description) tuples in original order
            memory_scores = [
                (score, description) for _, score, description in original_order
            ]

        # If sorting is enabled, sort the memory scores based on the specified method
        if sort_nodes == "importance":  # Ascending importance
            memory_scores = sorted(memory_scores, key=lambda x: x[0], reverse=False)
            # If a positive integer is specified and sorting is enabled, limit the number of returned memory nodes
            if n > 0:
                memory_scores = memory_scores[-n:]
        elif sort_nodes == "reverse importance":  # Descending importance
            memory_scores = sorted(memory_scores, key=lambda x: x[0], reverse=True)
            # If a positive integer is specified and sorting is enabled, limit the number of returned memory nodes
            if n > 0:
                memory_scores = memory_scores[:n]
        elif sort_nodes == "chronological":  # Ascending chronologically
            # No sorting needed for chronological (currently in chronological order)
            # If a positive integer is specified and sorting is enabled, limit the number of returned memory nodes
            if n > 0:
                memory_scores = memory_scores[-n:]
        elif sort_nodes == "reverse chronological":  # Descending chronologically
            memory_scores = list(reversed(memory_scores))
            # If a positive integer is specified and sorting is enabled, limit the number of returned memory nodes
            if n > 0:
                memory_scores = memory_scores[:n]

        return memory_scores

    @classmethod
    def _rank_nodes(
        cls,
        character: "Character",
        node_ids: List[int],
        query: Union[
            dict[
                str,
                Union[
                    dict[
                        str,
                        Union[
                            Tuple[Tuple[int], np.ndarray],
                            Tuple[Tuple[float, int], np.ndarray],
                            dict[str, np.ndarray],
                        ],
                    ],
                    dict[str, dict[str, np.ndarray]],
                ],
            ],
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ],
        percentile: float = 0.75,
        method: str = "mean",
        standardize: bool = True,
        minmax_scale: bool = True,
        weighted: bool = False,
        separate_scores: bool = False,
    ) -> List[tuple[float, int]]:
        """
        Rank memory nodes based on recency, importance, and relevance to a given query.
        This function calculates scores for each node and returns them as tuples containing the total score and node ID.

        Args:
            character (Character): The character whose memory is being evaluated.
            node_ids (List[int]): A list of memory node IDs to be ranked.
            query (Union[dict[str, Union[dict[str, Tuple[Union[int, Tuple[float, int]]], np.ndarray]],
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]], list[str], str]): An optional
            search query to determine relevance.
            percentile (float): The percentile threshold to filter relevance scores (default is 0.75).
            method (str): The method to use for aggregation of relevance scores ('mean' or 'median', default is 'mean').
            standardize (bool): Whether to standardize the scores. Defaults to True.
            minmax_scale (bool): Whether to normalize the scores between 0 and 1. Defaults to True.
            weighted (bool): Whether to use weighted retrieval. Defaults to False.
            separate_scores (bool): Whether to return the recency, importance, and relevance scores separately.
                                    Defaults to False.

        Returns:
            List[tuple[float, int]]: A list of tuples containing total scores and their corresponding node IDs.
        """

        # Calculate the recency score for each memory node based on the character's memory
        recency = Retrieve._calculate_node_recency(
            character=character,
            memory_ids=node_ids,
            minmax_scale=minmax_scale,
            standardize=standardize,
        )

        # Calculate the importance score for each memory node based on the character's memory
        importance = Retrieve._calculate_node_importance(
            character=character,
            memory_ids=node_ids,
            minmax_scale=minmax_scale,
            standardize=standardize,
        )

        # Calculate the relevance score for each memory node in relation to the provided query
        relevance = Retrieve._calculate_node_relevance(
            character=character,
            memory_ids=node_ids,
            query=query,
            percentile=percentile,
            method=method,
            standardize=standardize,
            minmax_scale=minmax_scale,
            weighted=weighted,
        )

        # Scale the raw scores by the character's memory weights for recency, importance, and relevance
        recency = MemoryStream.recency_alpha * recency
        importance = MemoryStream.importance_alpha * importance
        relevance = MemoryStream.relevance_alpha * relevance

        # If separate scores are requested, return the recency, importance, and relevance scores separately
        if separate_scores:
            return {
                "recency": recency,
                "importance": importance,
                "relevance": relevance,
            }

        # Check if relevance is a 2D numpy array
        if isinstance(relevance, np.ndarray) and relevance.ndim == 2 and all(
            # Ensure each row in the relevance array is a 1D numpy array
            isinstance(row, np.ndarray) and (row.ndim == 1 or row.ndim == (1,))
            for row in relevance
        ):
            # Sum the relevance scores across the rows to get a single score for each memory node
            relevance = np.sum(relevance, axis=1)

        # Calculate the total score by summing the scaled scores
        total_score = recency + importance + relevance

        # Combine total scores with their corresponding node IDs into tuples and return
        return list(zip(total_score, node_ids))

    @classmethod
    def _calculate_node_recency(
        cls,
        character,
        memory_ids,
        standardize: bool = True,
        minmax_scale: bool = True,
    ) -> np.ndarray:
        """
        Calculate the recency scores for a list of memory nodes based on their age using an exponential decay assumption.
        This function determines how recent each memory node is relative to the most recent observation.

        Args:
            character: The character whose memory is being evaluated.
            memory_ids: A list of memory node IDs for which recency scores are calculated.
            standardize (bool): Whether to standardize the scores. Defaults to True.
            minmax_scale (bool): Whether to normalize the recency scores between 0 and 1. Defaults to True.

        Returns:
            ndarray: An array of normalized recency scores for the specified memory nodes, scaled between 0 and 1.
        """

        # The most recent memory node is represented by the last index, which corresponds to the total number of
        # observations made
        latest_node = character.memory.num_observations

        # Calculate the "age" of each memory node by taking the difference between the latest observation index and each
        # relevant node ID
        # This results in a list of recency scores, where more recent nodes have higher scores
        recency = [character.memory.gamma ** (latest_node - i) for i in memory_ids]

        # Standardize the recency scores
        if standardize:
            std_dev = np.std(recency, axis=0)
            # Avoid division by zero by checking if std_dev is not zero
            recency = (
                (recency - np.mean(recency, axis=0)) / std_dev
                if std_dev.all() != 0
                else recency
            )

        # Normalize the recency scores to a range between 0 and 1 for consistent scaling
        if minmax_scale:
            recency = Retrieve._minmax_normalize(recency, 0, 1)

        # Return the recency scores as a numpy array
        return np.array(recency)

    @classmethod
    def _calculate_node_importance(
        cls,
        character,
        memory_ids,
        standardize: bool = True,
        minmax_scale: bool = True,
    ) -> np.ndarray:
        """
        Calculate the importance scores for a list of memory nodes.
        This function retrieves the importance values of specified memory nodes and normalizes them for consistent scaling.

        Args:
            character: The character whose memory is being evaluated.
            memory_ids: A list of memory node IDs for which importance scores are calculated.
            standardize (bool): Whether to standardize the scores. Defaults to True.
            minmax_scale (bool): Whether to normalize the importance scores between 0 and 1. Defaults to True.
        Returns:
            ndarray: An array of normalized importance scores for the specified memory nodes, scaled between 0 and 1.
        """

        # Retrieve the importance scores for each memory node specified by the memory IDs
        importances = [
            character.memory.observations[i].node_importance for i in memory_ids
        ]

        # Standardize the importance scores, handling potential division by zero
        if standardize:
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            if std_importance > 0:  # Check to avoid division by zero
                importances = (importances - mean_importance) / std_importance
            else:
                importances = np.zeros_like(importances)  # Set to zero if std is zero

        # Normalize the importance scores to a range between 0 and 1 for consistent scaling
        if minmax_scale:
            importances = Retrieve._minmax_normalize(importances, 0, 1)

        # Return the importance scores as a numpy array
        return np.array(importances)

    @classmethod
    def _calculate_node_relevance(
        cls,
        character: "Character",
        memory_ids: list[int],
        query: Union[
            dict[
                str,
                Union[
                    dict[str, Tuple[Union[int, Tuple[float, int]], np.ndarray]],
                    dict[str, dict[str, np.ndarray]],
                ],
            ],
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ],
        percentile: float = 0.75,
        method: str = "mean",
        standardize: bool = True,
        minmax_scale: bool = True,
        weighted: bool = False,
    ) -> np.ndarray:
        """
        Calculate the relevance scores of memory nodes based on their similarity to a given query.
        This function uses embeddings to assess how closely related each memory node is to the query or to default
        queries if none is provided.

        Args:
            character (Character): The character whose memory is being evaluated.
            memory_ids (list[int]): A list of memory node IDs for which relevance scores are calculated.
            query (Union[dict[str, Union[dict[str, Tuple[Union[int, Tuple[float, int]], np.ndarray]],
            dict[str, dict[str, np.ndarray]]]], dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str], str]): An optional search query to determine relevance. If not provided, default queries are used.
            percentile (float): The percentile threshold to filter relevance scores (default is 0.75).
            method (str): The method to use for aggregation of relevance scores ('mean' or 'median', default is 'mean').
            standardize (bool): Whether to standardize the scores. Defaults to True.
            minmax_scale (bool): Whether to normalize the relevance scores between 0 and 1. Defaults to True.
            weighted (bool): Whether to use weighted retrieval. Defaults to False.

        Returns:
            ndarray: An array of normalized relevance scores for the specified memory nodes, scaled between 0 and 1.
        """

        # {'embeddings': {'query': embedding}, 'keywords': {'kw_type': {'keyword': embedding}}}

        # Retrieve the embeddings for each memory node specified by the memory IDs
        memory_embeddings = [
            character.memory.get_observation_embedding(i) for i in memory_ids
        ]

        # Check if a query is provided to determine relevance
        if query:
            if isinstance(query, str):
                # Compute the embeddings for the query
                query_embeddings = Retrieve.gpt_handler.generate_embeddings([query])
            elif isinstance(query, list):
                # Compute the embeddings for the query
                query_embeddings = Retrieve.gpt_handler.generate_embeddings(query)
            elif isinstance(query, dict):
                # If a query is a dictionary with embeddings, extract the embeddings
                if isinstance(query, dict) and "embeddings" in query:
                    query_embeddings = [
                        embedding if isinstance(embedding, np.ndarray) else embedding[1]
                        for embedding in query["embeddings"].values()
                    ]
            else:
                # raise an error if the query is not in the correct format
                raise ValueError(f"Query is not in the correct format: {query}")
            # Compute the cosine similarity between the memory embeddings and the query embeddings
            relevances = cosine_similarity(memory_embeddings, query_embeddings)
        else:
            # If no query is passed, use default queries (e.g., persona, goals) to assess relevance
            default_embeddings = character.memory.get_query_embeddings()
            relevances = cosine_similarity(memory_embeddings, default_embeddings)

        if weighted:
            # Get the weights for each query embedding
            weights = [
                (
                    embedding[0]
                    if isinstance(embedding[0], int | float)
                    else (
                        (embedding[0][0], embedding[0][1])
                        if isinstance(embedding[0], tuple)
                        else ValueError(f"Invalid weight type: {embedding[0]}")
                    )
                )
                for embedding in query["embeddings"].values()
            ]

            # Take a weighted average of the recency and importance weights for each query embedding
            recency_scaler = 0.5
            importance_scaler = 0.5

            weights = [
                (recency_scaler * weight[0], importance_scaler * weight[1])
                for weight in weights
            ]

        # Get aggregated relevance scores across all queries
        scored_relevances = Retrieve._score_cos_sim(
            relevances, percentile=percentile, method=method, weights=weights if weighted else None
        )

        # Standardize the scored_relevances
        if standardize:
            std_dev = np.std(scored_relevances, axis=0)
            # Avoid division by zero by checking if std_dev is not zero
            scored_relevances = (
                scored_relevances - np.mean(scored_relevances, axis=0)
            ) / std_dev if std_dev.all() != 0 else scored_relevances

        # # TODO: Modify _minmax_normalize to work with weighted scores
        # # Normalize the scored_relevances to a range between 0 and 1 for consistent scaling
        # if minmax_scale:
        #     scored_relevances = Retrieve._minmax_normalize(scored_relevances, 0, 1)

        # Return the scored relevance scores as a numpy array
        return np.array(scored_relevances)

    @classmethod
    def _get_relevant_memory_ids(
        cls,
        search_keys: dict[str, np.ndarray],
        character: "Character",
        memory_types: list[MemoryType],
        threshold: float = 0.45,
        get_all: bool = False,
    ) -> list[int]:
        """
        Retrieve a list of memory node IDs that are relevant to the provided search keywords. This function aggregates
        node IDs based on keywords and their associated search words from the character's memory.

        Args:
            search_keys (dict[str, np.ndarray]): A dictionary where keys are keywords and values are their embeddings.
            character (Character): The character whose memory is being queried for relevant node IDs.
            memory_types (list[MemoryType]): A list of memory types to consider during retrieval.
            threshold (float): The cosine similarity threshold to consider a keyword as a match (default is 0.45).
            get_all (bool): If True, returns all node IDs. Defaults to False.

        Returns:
            list[int]: A list of unique memory node IDs that match the search keywords.
        """

        # If get_all is True, return all node IDs of the specified memory type
        if get_all:
            memory_ids = []
            for memory_type in memory_types:
                memory_ids.extend(
                    character.memory.get_observations_by_type(memory_type)
                )
            return memory_ids

        if cls.DEBUG_MODE:
            # FOR DEBUGGING
            print(f"\n\n{'*'*35} Search Keys {'*'*35}\n")
            if isinstance(search_keys, dict):
                print(search_keys.items())

        if not search_keys:
            return []

        # Initialize an empty list to store relevant memory node IDs
        memory_ids = []

        # Collect all keywords and their associated node IDs from all keyword types
        all_keywords = []
        all_node_ids = []
        for kw_type_dict in character.memory.keyword_nodes.values():
            for keyword, node_ids in kw_type_dict.items():
                all_keywords.append(keyword)
                all_node_ids.append(node_ids)

        if cls.DEBUG_MODE:
            # FOR DEBUGGING
            print(f"All Node IDs: {all_node_ids}")

        # Get the embeddings of all search words
        search_word_embeddings = np.array(
            [embedding for embedding in search_keys.values()]
        )

        # Reshape search_word_embeddings if it's a 1D array
        if search_word_embeddings.ndim == 1:
            search_word_embeddings = search_word_embeddings.reshape(-1, 1)

        # Get the embeddings of all keywords
        keyword_embeddings = np.array(MemoryStream.get_keyword_embeddings(all_keywords))

        # Reshape keyword_embeddings if it's a 1D array
        if keyword_embeddings.ndim == 1:
            keyword_embeddings = keyword_embeddings.reshape(-1, 1)

        # Compute cosine similarity between all search words and all keywords
        similarities = cosine_similarity(search_word_embeddings, keyword_embeddings)

        ### THIS BLOCK IS FOR DEBUGGING ###
        if cls.DEBUG_MODE:
            # Iterate over each search word and its corresponding similarity scores (printing scores)
            for search_word, search_word_embedding in search_keys.items():
                search_word_similarities = cosine_similarity(
                    [search_word_embedding], keyword_embeddings
                )[0]
                for i, similarity in enumerate(search_word_similarities):
                    keyword = all_keywords[i]
                    print(
                        f"Cosine similarity between search word '{search_word}' and keyword '{keyword}': {similarity}"
                    )
                    if similarity >= threshold:
                        print(f"Similarity >= threshold: {similarity} >= {threshold}")
                        # Filter node IDs to include only those with specific valid memory types
                        valid_node_ids = [
                            node_id
                            for node_id in all_node_ids[i]
                            if character.memory.get_observation(node_id)
                            and character.memory.get_observation_type(node_id)
                            in (
                                memory_types
                                if isinstance(memory_types, list)
                                else [memory_types]
                            )
                        ]
                        print(
                            f"Memory IDs: {[character.memory.get_observation(node_id) for node_id in valid_node_ids]}"
                        )
                    else:
                        print(f"Similarity < threshold: {similarity} < {threshold}")

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
                        in (
                            memory_types
                            if isinstance(memory_types, list)
                            else [memory_types]
                        )
                    )

        # Return a sorted list of unique memory node IDs by converting the list to a set and back to a list
        return sorted(list(set(memory_ids)))

    @classmethod
    def _gather_keywords_for_search(
        cls,
        game: Game,
        character: Character,
        query: Union[
            dict[
                str,
                Union[
                    dict[str, Tuple[int, np.ndarray]], dict[str, dict[str, np.ndarray]]
                ],
            ],
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ] = None,
        memory_lookback: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Collect keywords for searching based on the character's recent memories, goals, and an optional query.
        This function extracts relevant keywords from the character's observations, current goals, and the provided
        query to facilitate memory retrieval.

        Args:
            game (Game): The game context that provides parsing capabilities for extracting keywords.
            character (Character): The character whose memories and goals are being analyzed for keyword extraction.
            query (Union[dict[str, Union[dict[str, Tuple[Union[int, Tuple[float, int]], np.ndarray]],
            dict[str, dict[str, np.ndarray]]]], dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str], str]): An optional search query to determine relevance. If not provided, default queries are used.
            memory_lookback (int | None): The number of memories to look back across. Defaults to None, which utilizes
                                          the character's memory lookback value.

        Returns:
            dict[str, np.ndarray]: A dictionary of keywords mapping to their embeddings.
        """

        if cls.DEBUG_MODE:
            # FOR DEBUGGING
            print(f"Query: {query}\n")

        # Initialize an empty dictionary to store keywords and their embeddings for retrieval
        retrieval_kwds_with_embeddings = {}

        # Initialize a list to store missing keywords for batch generation
        missing_keywords = []

        ### 1. Gather keywords from the last 'n' memories, simulating "short term memory" ###

        # Iterate over the last 'n' memories
        lookback = memory_lookback if memory_lookback else character.memory.lookback
        for node in character.memory.observations[-lookback :]:
            if cls.DEBUG_MODE:
                # FOR DEBUGGING
                print(f"Node: {node.node_description}")
                print(f"Node Keywords: {node.node_keywords}")

            # Get the node's keywords
            if keywords := node.node_keywords:
                # Check embeddings for all keywords
                keyword_embeddings = MemoryStream.get_keyword_embeddings(keywords)

                # Handle missing embeddings
                for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
                    # If the embedding is missing, add it to the list for batch generation
                    if keyword_embedding is None or (
                        isinstance(keyword_embedding, np.ndarray)
                        and not keyword_embedding.size
                    ):
                        missing_keywords.append(keyword)
                    else:
                        retrieval_kwds_with_embeddings[keyword] = keyword_embedding

        ### 2. Gather keywords from the character's current goals ###

        try:
            # Attempt to retrieve the current goals for the current round
            current_goals = character.goals.get_goals(
                round=game.round,
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
            try:
                # Attempt to retrieve the current goals for the previous round
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
            if keywords := {
                kw
                for node_id in current_goals
                for kw in character.memory.get_observation_keywords(node_id)
            }:
                # Check embeddings for all keywords
                keyword_embeddings = MemoryStream.get_keyword_embeddings(keywords)

                # Handle missing embeddings
                for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
                    # If the embedding is missing, add it to the list for batch generation
                    if keyword_embedding is None or (
                        isinstance(keyword_embedding, np.ndarray)
                        and not keyword_embedding.size
                    ):
                        missing_keywords.append(keyword)
                    else:
                        retrieval_kwds_with_embeddings[keyword] = keyword_embedding

        ### 3. Gather keywords from the provided query, if any ###

        if query:
            if isinstance(query, str):
                query = [query]
            elif isinstance(query, (list, tuple, set)):
                for q in query:
                    if query_kwds := game.parser.extract_keywords(q):
                        keywords = [
                            keyword
                            for keywords in query_kwds.values()
                            for keyword in keywords
                        ]
                        keyword_embeddings = MemoryStream.get_keyword_embeddings(
                            keywords
                        )
                        for keyword, keyword_embedding in zip(
                            keywords, keyword_embeddings
                        ):
                            if keyword_embedding is None or (
                                isinstance(keyword_embedding, np.ndarray)
                                and not keyword_embedding.size
                            ):
                                missing_keywords.append(keyword)
                            else:
                                retrieval_kwds_with_embeddings[keyword] = (
                                    keyword_embedding
                                )

            elif isinstance(query, dict):
                # {'embeddings': {'query': embedding}, 'keywords': {'kw_type': {'keyword': embedding}}}
                if "keywords" in query:
                    for kw_type, keywords in query["keywords"].items():
                        if isinstance(keywords, (dict, OrderedDict)) and all(
                            isinstance(kw, np.ndarray) for kw in keywords.values()
                        ):
                            retrieval_kwds_with_embeddings.update(keywords)
                        else:
                            keyword_embeddings = MemoryStream.get_keyword_embeddings(
                                keywords
                            )
                            for keyword, keyword_embedding in zip(
                                keywords, keyword_embeddings
                            ):
                                if keyword_embedding is None or (
                                    isinstance(keyword_embedding, np.ndarray)
                                    and not keyword_embedding.size
                                ):
                                    missing_keywords.append(keyword)
                                else:
                                    retrieval_kwds_with_embeddings[keyword] = (
                                        keyword_embedding
                                    )

        # Generate embeddings for missing keywords in batches
        if missing_keywords:
            # Batch generate embeddings for all missing keywords
            embeddings = Retrieve.gpt_handler.generate_embeddings(missing_keywords)
            generated_embeddings = {
                keyword: embedding
                for keyword, embedding in zip(missing_keywords, embeddings)
            }

            # Store the generated embeddings and add them to the retrieval dictionary in a single operation
            MemoryStream.keyword_embeddings.update(generated_embeddings)
            retrieval_kwds_with_embeddings.update(generated_embeddings)

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
        relevances: np.ndarray,
        percentile: float = 0.9,
        method: Literal["mean", "median"] = "mean",
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate the mean or median of the cosine similarity scores above a specified percentile.

        Args:
            relevances (np.ndarray): Array of shape (m, q) containing cosine similarity scores. m is the number of
                                     memory nodes and q is the number of queries.
            percentile (float): The percentile threshold (e.g., 0.9) to filter the scores.
            method (Literal["mean", "median"]): The method to use for aggregation ('mean' or 'median').
            weights (np.ndarray | None): An array of shape (m) containing weights for each memory node.

        Returns:
            np.ndarray: An array of shape (m) with the aggregated scores.
        """

        # Calculate the threshold value for the specified percentile
        threshold = np.percentile(relevances, percentile * 100, axis=1)

        # Filter the relevances to only include scores above the threshold
        filtered_scores = np.array(
            [
                relevances[i][relevances[i] >= threshold[i]]
                for i in range(relevances.shape[0])
            ]
        )

        # If weights are provided, multiply the filtered scores by their respective weights
        if weights is not None:
            # Get the indices of the scores that are above the threshold
            filtered_indices = np.array(
                [np.where(relevances[i] >= threshold[i])[0] for i in range(relevances.shape[0])]
            )

            # Multiply the filtered scores by their respective weights
            weighted_scores = []
            for i, indices in enumerate(filtered_indices):
                # Extract the weights corresponding to the filtered indices
                row_weights = [weights[idx] for idx in indices]
                row_scores = filtered_scores[i]
                # Check if weights are tuples or single values and multiply accordingly
                if isinstance(row_weights[0], tuple):
                    # Multiply each score by the corresponding tuple of weights
                    weighted_row = [
                        tuple(score * w for w in weight) for score, weight in zip(row_scores, row_weights)
                    ]
                else:
                    # Multiply each score by the corresponding single weight
                    weighted_row = [score * weight for score, weight in zip(row_scores, row_weights)]
                weighted_scores.append(weighted_row)
        else:
            weighted_scores = filtered_scores

        # Calculate the mean or median of the weighted scores
        if method == "mean":
            # Compute the mean of the weighted scores
            result = np.array([np.mean(scores, axis=0) for scores in weighted_scores])
        else:
            # Compute the median of the weighted scores
            result = np.array([np.median(scores, axis=0) for scores in weighted_scores])

        return result

    @classmethod
    def priority_scores(
        cls,
        game: "Game",
        character: "Character",
        query: Union[
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            dict[str, Union[dict[str, Tuple[Union[int, Tuple[float, int]], np.ndarray]],
            dict[str, dict[str, np.ndarray]]]],
            list[str],
            str,
        ] = None,
        percentile: float = 0.8,
        method: str = "mean",
        threshold: float = 0.45,
        standardize: bool = False,
        minmax_scale: bool = False,
        weighted: bool = False
    ) -> dict:
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
            query (Union[dict[str, Union[dict[str, Tuple[Union[int, Tuple[float, int]], np.ndarray]],
            dict[str, dict[str, np.ndarray]]]], dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
            list[str], str]): An optional search query to determine relevance. If not provided, default queries are
            used. Defaults to None.
            n (int, optional): The maximum number of memory nodes to return. Defaults to -1, which returns all.
            include_idx (bool, optional): If True, includes the index of each memory node in the output. Defaults to
                                          False.
            percentile (float, optional): The percentile threshold for filtering memory scores. Defaults to 0.75.
            method (str, optional): The method to use for aggregation ('mean' or 'median'). Defaults to 'mean'.
            threshold (float, optional): The cosine similarity threshold to consider a keyword as a match (default is
                                         0.9).
            standardize (bool, optional): Whether to standardize the scores. Defaults to False.
            minmax_scale (bool, optional): Whether to normalize the scores between 0 and 1. Defaults to False.
            weighted (bool, optional): Whether to weight the scores based on the weights list. Defaults to False.

        Returns:
            list or None: A list of memory node descriptions or indexed descriptions, or None if no relevant memory
                          nodes are found.
        """

        if circular_import_prints:
            print("-\tInitializing Retrieve")

        # Initialize the GPT handler if it hasn't been set up yet
        cls.initialize_gpt_handler()

        # Gather keywords for searching relevant memory nodes based on the game and character context.
        # This gets the keywords associated with recent memories and goals, along with the optional query.
        keywords_to_embeddings_dict = Retrieve._gather_keywords_for_search(
            game, character, query
        )

        memory_types = [
            MemoryType.PERSONA,
            MemoryType.RESPONSE,
            MemoryType.REFLECTION,
            MemoryType.IMPRESSION,
            MemoryType.GOAL,
        ]

        memory_type_scores = dict()

        for memory_type in memory_types:
            # Retrieve memory node IDs that are relevant to the gathered search keys
            memory_node_ids = Retrieve._get_relevant_memory_ids(
                search_keys=keywords_to_embeddings_dict,
                character=character,
                memory_types=[memory_type],
                threshold=threshold,
                get_all=True,  # if memory_type == MemoryType.PERSONA else False,
            )

            # If no relevant memory node IDs are found, set the score to 0
            if len(memory_node_ids) == 0:
                pass

            else:
                # Rank the memory nodes based on recency, importance, and relevance and their dictionary of scores to
                # the memory types scores dictionary
                memory_type_scores[memory_type] = Retrieve._rank_nodes(
                    character=character,
                    node_ids=memory_node_ids,
                    query=query,
                    percentile=percentile,
                    method=method,
                    standardize=standardize,
                    minmax_scale=minmax_scale,
                    separate_scores=True,
                    weighted=weighted,
                )

        return memory_type_scores

    # TODO: Consider removing the keyword types from the output dictionary
    @classmethod
    def get_query_keywords_and_embeddings(
        cls,
        game: "Game",
        query: Union[list[str], str],
        scores: list[Union[int, Tuple[int], Tuple[float, int]]] | None = None,
    ) -> dict[
        str,
        Union[
            OrderedDict[str, np.ndarray],
            dict[str, dict[str, Union[Tuple[int], Tuple[int, int], np.ndarray]]],
        ],
    ]:
        """
        Convert a list of query strings into a dictionary of 'keywords' and 'embeddings' keys.

        The 'keywords' key maps to a dictionary of keyword_type (str) mapping to dictionaries of keywords (str) mapping
        to their embeddings (np.ndarray). The 'embeddings' key maps to an ordered dictionary of queries (str) mapping to
        embeddings (np.ndarray).

        Args:
            game (Game): The game context that provides parsing capabilities for extracting keywords.
            query (Union[list[str], str]): A list of query strings or a single query string to be converted.
            scores (list[Union[int, Tuple[int], Tuple[float, int]]] | None): A list of importance scores for the queries.

        Returns:
            dict[str, Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]]: A dictionary containing the
            keyword types and their associated keywords and embeddings, and the query strings and their embeddings.
        """

        # Initialize the GPT handler if it hasn't been set up yet
        cls.initialize_gpt_handler()

        # Initialize the result dictionary with 'keywords' and 'embeddings' keys
        result = {"keywords": {}, "embeddings": OrderedDict()}

        # Initialize a dictionary to store missing keywords
        missing_keywords = {}

        # If the query is a single string, convert it to a list
        if isinstance(query, str):
            query = [query]

        # Compute the embeddings for the query, ensuring a list is returned
        query_embeddings = Retrieve.gpt_handler.generate_embeddings(query)

        # Iterate over each query string in the list
        for idx, (q, embedding) in enumerate(zip(query, query_embeddings)):
            # Extract keywords from the query string using the extract_keywords method
            keywords_dict = game.parser.extract_keywords(q)

            if cls.DEBUG_MODE:
                print(f"\n\nQuery: {q}\nKeywords: {keywords_dict}")

            # Add the query string and its embedding to the 'embeddings' key in the result dictionary
            result["embeddings"][q] = (
                embedding if scores is None else (scores[idx], embedding)
            )

            for kw_type, keywords in keywords_dict.items():
                if kw_type not in result["keywords"]:
                    result["keywords"][kw_type] = {}
                keyword_embeddings = MemoryStream.get_keyword_embeddings(keywords)
                for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
                    if keyword_embedding is None or (
                        isinstance(keyword_embedding, np.ndarray)
                        and not keyword_embedding.size
                    ):
                        if kw_type not in missing_keywords:
                            missing_keywords[kw_type] = []
                        missing_keywords[kw_type].append(keyword)
                    else:
                        result["keywords"][kw_type][keyword] = keyword_embedding

        # Generate embeddings for missing keywords in batches
        for kw_type, keywords in missing_keywords.items():
            if keywords:
                # Batch generate embeddings for all missing keywords of this type
                embeddings = Retrieve.gpt_handler.generate_embeddings(keywords)
                generated_embeddings = {
                    keyword: embedding
                    for keyword, embedding in zip(keywords, embeddings)
                }

                # Store the generated embeddings and add them to the retrieval dictionary in a single operation
                MemoryStream.keyword_embeddings.update(generated_embeddings)
                result["keywords"][kw_type].update(generated_embeddings)

        return result