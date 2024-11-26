"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent/memory_structures/memory_stream.py
Description: Defines Agent memory classes
"""

from __future__ import annotations  # Enables postponed evaluation of type annotations

circular_import_prints = False

from collections import defaultdict

if circular_import_prints:
    print("Importing MemoryStream")

from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Tuple,
    Union,
)  # Importing type hints for better type checking
from enum import Enum  # Importing Enum for defining enumerated constants
from collections import (
    defaultdict,
)  # Importing defaultdict for easier dictionary handling
from dataclasses import (
    dataclass,
    field,
)  # Importing dataclass for creating classes with minimal boilerplate
import re  # Importing re for regular expression operations
import numpy as np  # Importing NumPy for numerical operations
from spacy import load as spacyload  # Importing spaCy for natural language processing

# from uuid import uuid4  # Uncomment to use UUIDs for unique identifiers

# Local imports
# Importing utility functions for OpenAI client setup and text embedding
if circular_import_prints:
    print(f"\t{__name__} calling imports for General")
from ..utils.general import get_text_embedding

# Importing logging for logging messages
if circular_import_prints:
    print(f"\t{__name__} calling imports for Logging")
import logging

if circular_import_prints:
    print(f"\t{__name__} calling Type Checking imports for GptHelpers")
from ..gpt.gpt_helpers import GptCallHandler

if TYPE_CHECKING:
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking imports for Character")
    from ..things.characters import Character

    # from ..games import Game

if circular_import_prints:
    print(f"\tFinished calling imports for MemoryStream")


class MemoryType(Enum):
    """
    Enumeration for different types of memory.

    This class defines various memory types that can be used in the context of an agent's memory system.
    Each memory type represents a distinct category of information that the agent can store and recall.

    Attributes:
        ACTION (int): Represents an action memory type.
        DIALOGUE (int): Represents a dialogue memory type.
        REFLECTION (int): Represents a reflection memory type.
        PERCEPT (int): Represents a perception memory type.
        GOAL (int): Represents a goal memory type.
        IMPRESSIONS (int): Represents a impressions memory type.
        RESPONSES (int): Represents a responses memory type.
        PERSONA (int): Represents a persona memory type.
    """

    ACTION = 1  # Represents an action memory type
    DIALOGUE = 2  # Represents a dialogue memory type (summaries of all characters' dialogue replies)
    REFLECTION = 3  # Represents a reflection memory type
    PERCEPT = 4  # Represents a perception memory type
    GOAL = 5  # Represents a goal memory type
    IMPRESSION = 6  # Represents a impressions memory type
    RESPONSE = 7  # Represents a responses memory type (your verbatim dialogue replies)
    PERSONA = 8  # Represents a persona memory type


@dataclass
class ObservationNode:
    """
    Represents an observation made by an agent during a specific round and tick.

    This class encapsulates all relevant information about an observation, including its unique identifier,
    the round and tick at which it occurred, the level of the observation, and additional metadata such as location,
    description, success status, and importance. It is designed to facilitate the storage and retrieval of observations
    within the agent's memory system.

    Attributes:
        node_id (int): Unique identifier for the observation, representing its index in the agent's memory.
        node_round (int): The round in which the observation occurred.
        node_tick (int): The tick within the round when the observation was made.
        node_level (int): The level of the observation (1 for novel, 2 for reflections, 3 for ????).
        node_loc (str): The location where the observation took place.
        node_description (str): A description of the observation.
        node_success (bool): Indicates whether the observation was successful.
        embedding_key (int): Key for retrieving the embedding associated with the observation.
        node_importance (int): The importance of the observation, which could also be a float.
        node_is_self (int): Indicates whether the action was performed by the agent itself (1 for true, 0 for false)
        node_type (str, optional): The type of observation.
        node_keywords (set): A set of keywords associated with the observation.
    """

    node_id: int  # TODO: unique to this agent; represents the index in their memory
    node_round: int  # The round in which this occurred
    node_tick: int  # The round tick on which this observation occurred
    node_level: int  # The observation level: 1 for novel, 2 for reflections, 3 for ????
    node_loc: str  # The name of the location in which the observation occurred
    node_description: str
    node_success: bool
    embedding_key: (
        int  # Immediately get and store the embedding for faster retrieval later?
    )
    node_importance: int  # or could be float
    node_is_self: int  # ID of the agent making the observation, if relevant
    node_type: str = None  # the type of Observation
    node_keywords: set = field(
        default_factory=set
    )  # Keywords that were discovered in this node
    # associated_nodes: Optional[list[int]] = field(default_factory=list)


class MemoryStream:
    """
    Represents the memory stream for an agent, managing observations and memory types.

    This class provides functionality for storing and retrieving memories associated with an agent, including
    stopwords for natural language processing and methods for managing various types of memories.

    Attributes:
        _stopwords (set): A set of stopwords for natural language processing.
    """

    gpt_handler = None  # Class-level attribute to store the shared GPT handler

    model_params = {
        # "max_output_tokens": 256,
        # "temperature": 1,
        # "top_p": 1,
        # "max_retries": 5,
    }
    
    # Attributes for calculating relevancy scores
    importance_alpha = 0.4  # Weight for importance in relevancy scoring
    recency_alpha = 0.2  # Weight for recency in relevancy scoring
    relevance_alpha = 0.4  # Weight for relevance in relevancy scoring

    # store stopwords as a class variable for efficient access when adding new memories
    _stopwords = None

    # Dictionary to store embeddings of keywords, indexed by keyword
    keyword_embeddings = defaultdict(dict)

    @classmethod
    def _generate_stopwords(cls):
        """
        Generates and retrieves a set of stopwords for natural language processing.

        This class method initializes the stopwords from the spaCy library if they have not been generated yet.
        It returns the set of stopwords, which can be used to filter out common words in text processing.

        Returns:
            set: A set of stopwords used in natural language processing.
        """

        # Check if the stopwords have not been initialized
        if cls._stopwords is None:
            # Load the spaCy English model with specific components disabled
            nlp = spacyload(
                "en_core_web_sm", disable=["ner", "tagger", "parser", "textcat"]
            )
            # Assign the default stopwords from the spaCy model to the class variable
            cls._stopwords = nlp.Defaults.stop_words
            # Return the set of stopwords
        return cls._stopwords

    logger = None  # Class-level attribute to store the shared logger

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """

        if circular_import_prints:
            print(f"-\tGoals Module is initializing GptCallHandler")

        # Initialize the GPT handler if it hasn't been set up yet
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(
                model_config_type="miscellaneous", **cls.model_params
            )

    def __init__(self, character: "Character"):
        """
        Initializes an agent with identifying information and memory features.

        This constructor sets up the agent's identity based on the provided character and initializes various attributes
        related to memory management, observations, and relevancy scoring. It also establishes a connection to an OpenAI
        client and prepares the agent's stopwords for natural language processing.

        Args:
            character (Character): The character object containing identifying information for the agent.

        Attributes:
            character (Character): The character object containing identifying information for the agent.
            num_observations (int): The count of observations made by the agent.
            observations (list): A list to store observations.
            memory_embeddings (dict): A dictionary to store embeddings of observations.
            keyword_nodes (defaultdict): A nested dictionary to categorize keywords (dictionaries mapping keyword types
                                         to dictionaries mapping keywords to lists of observation IDs).
            memory_type_nodes (defaultdict): A dictionary to categorize memory types.
            this_round_nodes (defaultdict): A dictionary to store observations by round number.
            query_embeddings (dict): Cached embeddings for querying statements about the agent.
            lookback (int): The number of observations available without retrieval.
            gamma (float): The decay factor for memory importance.
            reflection_capacity (int): The number of reflections after each round.
            reflection_distance (int): The number of observations for reflection.
            reflection_rounds (int): The number of rounds for looking back.
            importance_alpha (float): Weight for importance in relevancy scoring.
        """

        # Initialize the GPT handler if it hasn't been set up yet
        MemoryStream.initialize_gpt_handler()

        if circular_import_prints:
            print(f"-\tInitializing MemoryStream")

        # Keep track of this agent's identifying information
        # Store the character
        self.character = character
        # Initialize the count of observations made by the agent
        self.num_observations = 0
        # List to hold the agent's observations
        self.observations = []

        # Dictionary to store embeddings of observations, indexed by observation index
        self.memory_embeddings = {}
        # Nested dictionary for categorizing keywords
        self.keyword_nodes = defaultdict(lambda: defaultdict(list))
        # Dictionary to categorize memory types using MemoryType enum values
        self.memory_type_nodes = defaultdict(list)
        # Dictionary to store observations by the current round number
        self.this_round_nodes = defaultdict(list)

        # Attributes defining the memory features of the agent
        # Number of observations available without retrieval; used for gathering keys
        self.lookback = 5
        # Decay factor for determining memory importance
        self.gamma = 0.95
        # Number of reflections to perform after each round
        self.reflection_capacity = 2
        # Number of observations the agent can look back for reflection
        self.reflection_distance = 200
        # Number of rounds the agent can look back
        self.reflection_rounds = 2

        # Cache for current querying statements about this agent
        # Cached embeddings include: persona summary, goals, and personal relationships
        # These will assist in the memory retrieval process
        self.query_embeddings = (
            self.set_query_embeddings()
        )  # Initialize query embeddings based on the character

        # Initialize stopwords for natural language processing
        self.stopwords = self._generate_stopwords()  # Generate and store stopwords

        if MemoryStream.logger is None:
            MemoryStream.logger = logging.getLogger("agent_cognition")

    # ----------- MEMORY CREATION -----------
    def add_memory(
        self,
        round: int,
        tick: int,
        description: str,
        keywords: dict[str, Union[list[str], dict[str, np.ndarray]]],
        location: str,
        success_status: bool,
        memory_importance: float,
        memory_type: Union[int, MemoryType],
        actor_id: str,
    ) -> int:
        """
        Adds a memory entry to the agent's memory system.

        This method validates the memory type, generates a unique node ID, and embeds the provided description.
        It then creates a new memory entry based on the specified memory type and caches it for future retrieval.

        Args:
            round (int): The current round in the game.
            tick (int): The current tick or time step in the game.
            description (str): A description of the memory to be added.
            keywords (dict[str, Union[list[str], dict[str, np.ndarray]]]): A dictionary mapping keyword types to
                either lists of keywords or dictionaries mapping keywords to their embeddings.
            location (str): The location where the memory was created.
            success_status (bool): Indicates whether the action associated with the memory was successful.
            memory_importance (float): A value representing the importance of the memory.
            memory_type (Union[int, MemoryType]): The type of memory being added, represented as an integer or
                MemoryType enum.
            actor_id (str): The ID of the actor associated with the memory.

        Raises:
            ValueError: If the provided memory type is not valid.

        Notes:
            The method supports different types of memories, including actions, reflections, and perceptions.
            It also caches the memory under relevant keywords and memory types for efficient retrieval.

        Returns:
            int: The unique identifier for the newly added memory.
        """

        # Check if keywords need to be modified before continuing
        if isinstance(keywords, dict):
            # Check if the dictionary has the correct format
            if all(isinstance(v, dict) for v in keywords.values()):
                # Convert the dictionary to a list of keywords
                keywords = self.convert_keyword_embeddings_to_keywords(keywords)

        # Import Character class from the text_adventure_games.things module (to avoid circular imports)
        from text_adventure_games.things import Character

        # Validate the memory type
        if not self.is_valid_memory_type(memory_type):
            valid_types = [type.name for type in MemoryType]
            raise ValueError(
                f"Memories must be created with valid type; one of {valid_types}"
            )

        # Get the next node id
        node_id = (
            self.num_observations
        )  # This works out to be the number of observations due to zero-indexing

        # Get a flattened list of keywords found in this memory
        node_kwds = [w for kw_type in keywords.values() for w in kw_type]

        # Embed the description
        memory_embedding = self.get_text_embedding(description)
        self.memory_embeddings[node_id] = memory_embedding

        # Check if this action was done by this agent
        self_is_actor = int(actor_id == self.character.id)

        original_memory_type = memory_type

        # Determine the memory type using the MemoryType enum
        memory_type = (
            MemoryType.ACTION
            if memory_type == MemoryType.ACTION.value
            else (
                MemoryType.REFLECTION
                if memory_type == MemoryType.REFLECTION.value
                else (
                    MemoryType.PERCEPT
                    if memory_type == MemoryType.PERCEPT.value
                    else (
                        MemoryType.GOAL
                        if memory_type == MemoryType.GOAL.value
                        else (
                            MemoryType.RESPONSE
                            if memory_type == MemoryType.RESPONSE.value
                            else (
                                MemoryType.DIALOGUE
                                if memory_type == MemoryType.DIALOGUE.value
                                else (
                                    MemoryType.PERSONA
                                    if memory_type == MemoryType.PERSONA.value
                                    else (
                                        MemoryType.IMPRESSION
                                        if memory_type == MemoryType.IMPRESSION.value
                                        else None
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # Determine the level of the memory based on its type
        node_level = (
            0
            if memory_type in [MemoryType.PERSONA]
            else (
                1
                if memory_type in [MemoryType.ACTION, MemoryType.PERCEPT]
                else (
                    2
                    if memory_type
                    in [MemoryType.REFLECTION, MemoryType.GOAL, MemoryType.RESPONSE]
                    else None
                )
            )
        )

        if memory_type is None:
            raise ValueError(
                f"Memories must be created with valid type; one of {valid_types}, not {original_memory_type}."
            )

        # Create a new memory entry
        new_memory = ObservationNode(
            node_id,
            node_round=round,
            node_tick=tick,
            node_level=node_level,
            node_loc=location,
            node_description=description,
            node_success=success_status,
            embedding_key=node_id,
            node_importance=memory_importance,
            node_type=memory_type,
            node_keywords=set(node_kwds),
            node_is_self=self_is_actor,
        )

        # Add the new memory to the sequential memory list
        self.observations.append(new_memory)

        # Initialize a list to store missing keywords
        missing_keywords = []

        # NODE CACHING
        # For each keyword type, cache the node under its keywords
        for kw_type, kw_list in keywords.items():
            # For each keyword in the type, cache the node
            for keyword in kw_list:
                # Append the node ID to the list of nodes for this keyword
                self.keyword_nodes[kw_type][keyword].append(node_id)

            # Retrieve embeddings for the current keywords
            if isinstance(kw_list, dict):
                print("KW_LIST TYPE:", type(kw_list))
                print("KW_LIST:", kw_list)
                print("KEYWORDS:", keywords)

            keyword_embeddings = MemoryStream.get_keyword_embeddings(kw_list)

            # Check each keyword's embedding and categorize missing ones
            for keyword, keyword_embedding in zip(kw_list, keyword_embeddings):
                # Check if the keyword embedding is missing
                if keyword_embedding is None or (
                    isinstance(keyword_embedding, np.ndarray)
                    and not keyword_embedding.size
                ):
                    # Collect missing keywords for batch processing
                    missing_keywords.append(keyword)
                else:
                    # Store the valid keyword embedding
                    MemoryStream.keyword_embeddings[keyword] = keyword_embedding

        # Check if there are any missing keywords of this type
        if missing_keywords:
            # Batch generate embeddings for all missing keywords of this type
            embeddings = MemoryStream.gpt_handler.generate_embeddings(missing_keywords)
            generated_embeddings = dict(zip(missing_keywords, embeddings))

            # Update the keyword embeddings and nodes with the newly generated embeddings
            MemoryStream.keyword_embeddings.update(generated_embeddings)

        # Cache the node under the value of its MemoryType and its round ID
        self.memory_type_nodes[memory_type.value].append(node_id)
        self.this_round_nodes[round].append(node_id)

        # Increment the internal count of nodes
        self.num_observations += 1

        return node_id  # Return the unique identifier for the newly added memory

    # ----------- GETTER METHODS -----------
    def get_observation(self, node_id):
        """
        Retrieves an observation node by its unique identifier.

        This method attempts to fetch an observation node from the agent's memory using the provided node ID.
        If the node ID is invalid or out of range, it returns None, indicating that the observation does not exist.

        Args:
            node_id (int): The unique identifier for the observation node to retrieve.

        Returns:
            ObservationNode or None: The observation node associated with the given ID, or None if not found.
        """

        # Attempt to retrieve the observation node using the provided node ID
        try:
            return self.observations[node_id]
        # Return None if the node ID is out of range or invalid
        except IndexError:
            return None

    def get_observation_description(self, node_id):
        """
        Retrieves the description of an observation node by its unique identifier.

        This method fetches the observation node using the provided node ID and returns its description.
        If the node does not exist, it may raise an error depending on the implementation of the `get_observation`
        method.

        Args:
            node_id (int): The unique identifier for the observation node whose description is to be retrieved.

        Returns:
            str: The description of the observation node, or raises an error if the node does not exist.
        """

        # Retrieve the observation node using the provided node ID
        node = self.get_observation(node_id)
        # Return the description of the retrieved observation node
        return node.node_description

    def get_observation_type(self, node_id):
        """
        Retrieves the type of an observation node by its unique identifier.

        This method fetches the observation node using the provided node ID and returns its type.
        If the node does not exist, it returns None, indicating that the type cannot be retrieved.

        Args:
            node_id (int): The unique identifier for the observation node whose type is to be retrieved.

        Returns:
            MemoryType or None: The type of the observation node, or None if the node does not exist.
        """

        # Retrieve the observation node using the provided node ID and assign it to 'node'.
        return node.node_type if (node := self.get_observation(node_id)) else None

    def get_observation_embedding(self, index):
        """
        Retrieves the embedding for a specified node index.

        This method checks if a node exists at the given index and, if so, returns its corresponding embedding from
        memory. It is useful for accessing the vector representation of a node's description for further processing or
        analysis.

        Args:
            index (int): The index of the node for which to retrieve the embedding.

        Returns:
            Union[np.array, None]: The embedding of the node description if the node exists; otherwise, it returns None.
        """

        # Check if a node exists at the specified index
        if self.node_exists(index):
            # Return the embedding associated with the node at the given index
            return self.memory_embeddings[index]
        else:
            return None

    def get_observation_keywords(self, index: int) -> set:
        """
        Retrieves the keywords associated with a specific observation node.

        This method fetches the observation node using the provided index and returns the set of keywords
        associated with that observation. If the node does not exist, it may raise an error.

        Args:
            index (int): The index of the observation node whose keywords are to be retrieved.

        Returns:
            set[str]: A set of keywords associated with the specified observation node.
        """
        return self.get_observation(index).node_keywords

    def get_enumerated_description_list(
        self, node_id_list, as_type: Literal["str", "tuple"] = True
    ) -> Union[List[Tuple], List[str]]:
        """
        Generates a list of observation descriptions paired with their IDs.

        This method takes a list of node IDs and retrieves their corresponding descriptions,
        returning them either as a list of tuples or a formatted string list based on the specified type.
        This allows for flexible output depending on the needs of the caller.

        Args:
            node_id_list (list): A list of unique identifiers for the observation nodes.
            as_type (Literal["str", "tuple"], optional): The format of the output; either "tuple" for a list of tuples
                or "str" for a list of formatted strings. Defaults to "tuple".

        Returns:
            Union[List[Tuple[int, str]], List[str]]: A list of tuples containing node IDs and descriptions,
                or a list of formatted strings based on the specified type.
        """

        enum_nodes = list(
            zip(
                node_id_list,
                [self.get_observation_description(i) for i in node_id_list],
            )
        )  # Create a list of tuples by pairing each node ID with its corresponding description

        # Check if the requested output type is a tuple
        if as_type == "tuple":
            # Return the list of tuples (node ID, description)
            return enum_nodes
        # If the requested output type is not a tuple
        else:
            # Return a list of formatted strings, each containing the node ID and description, followed by a newline
            # character
            return [f"{mem_id}. {mem_desc}\n" for mem_id, mem_desc in enum_nodes]

    def get_observations_by_round(self, round):
        """
        Retrieves all observations associated with a specific round.

        This method returns a list of observation nodes that were recorded during the specified round.
        It allows for easy access to all observations made in a particular round of the game.

        Args:
            round (int): The round number for which to retrieve the observations.

        Returns:
            list[int]: A list of node IDs associated with the specified round.
        """

        return self.this_round_nodes[round]

    def get_observations_after_round(self, round, inclusive=False):
        """
        Retrieves all observations recorded after a specified round.

        This method collects and returns observation nodes from rounds that occur after the given round number.
        The inclusion of the specified round can be controlled by the `inclusive` parameter, allowing for flexibility in
        the retrieval.

        Args:
            round (int): The round number after which to retrieve observations.
            inclusive (bool, optional): If True, includes observations from the specified round;
                if False, excludes it. Defaults to False.

        Returns:
            list[int]: A list of node IDs recorded after the specified round.
        """

        nodes = []  # Initialize an empty list to store the observation nodes

        # Determine the rounds to request based on the inclusive flag
        if inclusive:
            # Include the specified round
            requested_rounds = [r for r in self.this_round_nodes if r >= round]
        else:
            # Exclude the specified round
            requested_rounds = [r for r in self.this_round_nodes if r > round]

        # Iterate through the requested rounds to gather observation nodes
        for r in requested_rounds:
            # Retrieve observation nodes for the current round
            r_nodes = self.get_observations_by_round(r)
            # Add the retrieved nodes to the list
            nodes.extend(r_nodes)

        return nodes  # Return the collected observation nodes

    def get_observations_by_type(self, obs_type):
        """
        Retrieves all observations of a specified memory type or types.

        This method checks if the provided observation type is valid and retrieves the corresponding observation nodes
        associated with that memory type. If a list or set of types is provided, it retrieves nodes for all specified
        types without duplicating IDs. It raises an error if any type is unsupported, ensuring that only valid types
        are processed.

        Args:
            obs_type (Union[int, MemoryType, list[MemoryType], set[MemoryType]]): The type(s) of memory observations to 
            retrieve, which can be specified as an integer, a MemoryType enum, or a list/set of MemoryType enums.

        Raises:
            ValueError: If any provided observation type is not a supported MemoryType.

        Returns:
            list[int]: A list of node IDs associated with the specified memory type(s).
        """

        # If obs_type is a list or set, process each type
        if isinstance(obs_type, (list, set)):
            node_ids = set()  # Use a set to avoid duplicate IDs
            for type_item in obs_type:
                if not self.is_valid_memory_type(type_item):
                    raise ValueError(
                        f"{type_item} is not a supported MemoryType({list(MemoryType)})."
                    )
                if isinstance(type_item, MemoryType):
                    type_item = type_item.value
                node_ids.update(self.memory_type_nodes[type_item])
            return list(node_ids)

        # Single type processing
        if not self.is_valid_memory_type(obs_type):
            # Raise an error if the type is unsupported
            raise ValueError(
                f"{obs_type} is not a supported MemoryType({list(MemoryType)})."
            )

        # Check if obs_type is an instance of the MemoryType enum
        if isinstance(obs_type, MemoryType):
            # Convert Enum to its value
            obs_type = (
                obs_type.value
            )  # Assign the integer value of the enum to obs_type

        # Return the list of observation nodes associated with the specified memory type
        return self.memory_type_nodes[obs_type]

    def get_most_recent_summary(self):
        """
        Generates a summary of the most recent observations made by the agent.

        This method retrieves the last few observations based on the lookback attribute and formats them into a summary
        string. The summary provides a chronological list of the most recent observations, making it easy to review past
        actions.

        Returns:
            str: A formatted summary of the last `lookback` observations made by the agent.
        """

        # Retrieve the last 'lookback' number of observations from the agent's memory
        nodes = self.observations[-self.lookback :]
        # Initialize the summary string
        summary = f"The last {self.lookback} observations in chronological order you have made are:"

        # Iterate through the retrieved nodes to build the summary
        for i, node in enumerate(nodes):
            # Append each observation's description to the summary
            summary += "\n{idx}. {desc}".format(idx=i, desc=node.node_description)

        # Return the complete summary of the most recent observations
        return summary

    def get_query_embeddings(self):
        """
        Retrieves the query embeddings stored in the agent's memory.

        This method collects all non-empty query embeddings and returns them as a NumPy array.
        It is useful for accessing the embeddings that represent the current state of the agent's queries.

        Returns:
            np.array: An array containing the non-empty query embeddings.
        """

        # Create and return a NumPy array containing all non-empty query embeddings from the query_embeddings dictionary
        # The condition q.all() ensures that only embeddings with all elements non-zero are included
        # TODO: I'm switching this as it seems to ignore embeddings with 0's in the vector
        # return np.array([q for q in self.query_embeddings.values() if q.all()])

        # Non-nested embeddings version:
        # Create a NumPy array from the values of query_embeddings that aren't None.
        # return np.array([q for q in self.query_embeddings.values() if q is not None])

        # Create a NumPy array from the values of query_embeddings that aren't None.
        # Flatten and filter out None values, ensuring each slot is a list of floats
        return np.array(
            [
                embedding
                for value in self.query_embeddings.values()
                if value is not None
                for embedding in (
                    value
                    if isinstance(value, list)
                    and isinstance(value[0], (list, np.ndarray))
                    else [value]
                )
            ]
        )

    def get_text_embedding(self, text):
        """
        Generates an embedded vector representation of the given text.

        This method takes a text input and retrieves its corresponding embedded vector using a text embedding function.
        The resulting vector can be used for various applications, such as similarity comparisons or machine learning
        tasks.

        Args:
            text (str): The text to be converted into an embedded vector.

        Returns:
            numpy.ndarray: The embedded vector representation of the input text.
        """

        return MemoryStream.gpt_handler.generate_embeddings(text)

    def get_relationships_summary(self):
        """
        Generates a summary of the agent's relationships.

        This method is intended to provide an overview of the relationships associated with the agent.
        However, it is not yet implemented and will raise a NotImplementedError if called.

        Raises:
            NotImplementedError: Indicates that the method has not been implemented yet.
        """

        raise NotImplementedError

    # ----------- SETTER METHODS -----------
    def set_embedding(self, node_id, new_embedding):
        """
        Sets a new embedding for a specified node ID.

        This method updates the embedding of a node if it exists in the memory.
        It returns a boolean indicating whether the update was successful, allowing for easy error handling.

        Args:
            node_id (int): The unique identifier of the node for which to set the new embedding.
            new_embedding (np.array): The new embedding to be assigned to the specified node.

        Returns:
            bool: True if the embedding was successfully updated; False if the node does not exist.
        """

        # Check if the node with the specified ID exists
        if not self.node_exists(node_id):
            # Return False if the node does not exist, indicating the update was unsuccessful
            return False

        # Update the memory_embeddings dictionary with the new embedding for the node
        self.memory_embeddings.update({node_id: new_embedding})
        # Return True to indicate that the embedding was successfully updated
        return True

    def set_query_embeddings(self, round: int = 0):
        """
        Sets the default query embeddings for the character based on their persona and goals.

        This method retrieves the embeddings for the character's persona summary and current goals,
        caching them in a dictionary for later use. It handles cases where the character may not have goals defined,
        ensuring that the function can operate smoothly without raising errors.

        Args:
            round (int, optional): The round in the game for which to retrieve the goal embedding. Defaults to 0.

        Returns:
            dict: A dictionary containing the cached query embeddings for the persona and goals.
        """

        # # Import Character class from the text_adventure_games.things module (to avoid circular imports)
        # from text_adventure_games.things import Character

        # Initialize an empty dictionary to store cached query embeddings
        cached_queries = {}

        # Attempt to retrieve the goal embedding for the specified round
        try:
            current_goal_embed = self.character.goals.get_goal_embeddings(round=round)

        # If the character has no goals, set current_goal_embed to None
        except AttributeError:
            current_goal_embed = None

        # print("Old set_query_embeddings", self.character)
        # TODO: This is the old way that didn't use gpt_helpers.py
        # Get the embedding for the character's persona summary (old way didn't use gpt_helpers.py)
        # persona_embed = get_text_embedding(self.character.persona.summary)
        # print("New set_query_embeddings", self.character)

        # Get the persona embedding using the GPT call handler
        # TODO: Get embeddings for each component of the persona summary
        persona_embed = MemoryStream.gpt_handler.generate_embeddings(
            self.character.persona.summary
        )

        # Check if the persona embedding was successfully retrieved
        if persona_embed is not None:
            # Store the persona embedding in the cached_queries dictionary
            cached_queries["persona"] = persona_embed

        # Check if the goal embedding was successfully retrieved
        if current_goal_embed is not None:
            # Store the goal embedding in the cached_queries dictionary
            cached_queries["goals"] = current_goal_embed

        # Return the dictionary containing the cached query embeddings
        return cached_queries

    def set_goal_query(self, goal_node_ids):
        """
        Updates the query embeddings with the specified goal embeddings.

        This method sets the goal query embeddings in the agent's query embeddings dictionary using the provided
        node IDs. If the update encounters a KeyError, the exception is caught, and an error is logged, allowing
        the program to continue execution without interruption.

        Args:
            goal_node_ids (list): A list of node IDs corresponding to the goals for which embeddings are to be set.

        Returns:
            None
        """

        try:
            # Retrieve embeddings for each goal_node_id
            goal_embeddings = [
                self.memory_embeddings[node_id] for node_id in goal_node_ids
            ]

        # Catch a KeyError if the update fails due to a missing key
        except KeyError as e:
            # Raise a KeyError
            raise KeyError("Goal query embeddings update failed. %s", e)

        # Attempt to update the query_embeddings dictionary with the new goal embeddings
        try:
            self.query_embeddings.update({"goals": goal_embeddings})

        # Catch a KeyError if the update fails due to a missing key
        except KeyError as e:
            # Log the error
            MemoryStream.logger.error(
                "Goal query embeddings update failed. Skipping. Caught: %s", e
            )

    # ----------- UPDATE METHODS -----------
    def update_node(self, node_id, **kwargs) -> bool:
        """
        Updates the attributes of a specified node with new values.

        This method retrieves a node by its ID and updates its attributes based on the provided keyword arguments.
        If the node does not exist, it returns False; otherwise, it applies the updates and returns True.

        Args:
            node_id (int): The unique identifier of the node to be updated.
            **kwargs: Arbitrary keyword arguments representing the attributes to update and their new values.

        Returns:
            bool: True if the node was successfully updated; False if the node does not exist.
        """

        # Attempt to retrieve the observation node using the provided node ID
        try:
            node = self.get_observation(node_id)
        # Catch an IndexError if the node ID is invalid or does not exist
        except IndexError:
            # Return False to indicate that the update failed due to a non-existent node
            return False

        else:
            # Iterate over the keyword arguments provided for updating the node
            for k, v in kwargs.items():
                # Check if the node has the attribute specified by the keyword argument
                if hasattr(node, k):
                    # Update the attribute of the node with the new value
                    setattr(node, k, v)
            # Return True to indicate that the node was successfully updated
            return True

    def update_node_embedding(self, node_id, new_description) -> bool:
        """
        Updates the embedding of a specified node with a new description.

        This method checks if the node exists and, if so, retrieves the new embedding based on the provided description.
        It then updates the node's embedding and returns a boolean indicating whether the update was successful.

        Args:
            node_id (int): The unique identifier of the node whose embedding is to be updated.
            new_description (str): The new description used to generate the updated embedding.

        Returns:
            bool: True if the embedding was successfully updated; False if the node does not exist.
        """

        # Check if the node with the specified ID exists
        if not self.node_exists(node_id):
            # Return False if the node does not exist, indicating the update cannot proceed
            return False

        # Retrieve the updated embedding based on the new description
        updated_embedding = self.get_text_embedding(new_description)

        # Update the node's embedding and return the result of the update operation
        return self.set_embedding(node_id, updated_embedding)

    # ----------- MISC HELPER/VALIDATION METHODS -----------
    def replace_character(self, text, character_name, agent_descriptor):
        """
        Replaces occurrences of a character's name and descriptor in the given text with 'you'.

        This method uses regular expressions to identify and replace mentions of a specified character's name
        and their descriptor in the provided text, allowing for a more personalized narrative.
        It handles variations in how the character may be referenced, including optional articles and descriptors.

        Args:
            text (str): The text in which to replace the character's name and descriptor.
            character_name (str): The name of the character to be replaced.
            agent_descriptor (str): A descriptor associated with the character, which may also be replaced.

        Returns:
            str: The modified text with the character's name and descriptor replaced by 'you'.
        """

        # Escape any special regex characters in the descriptor and character_name
        escaped_name = re.escape(character_name)
        # Escape the descriptor, excluding any stopwords, to safely use it in a regex pattern
        escaped_descriptor = re.escape(
            " ".join([w for w in agent_descriptor.split() if w not in self.stopwords])
        )

        # Pattern to match 'the' optionally, then the descriptor optionally, followed by the character name
        # The descriptor and the character name can occur together or separately
        pattern = r"\b(?:the\s+)?(?:(?:{d}\s+)?{c}|{d})\b".format(
            d=escaped_descriptor, c=escaped_name
        )

        # Construct the regex pattern to match the character's descriptor and name, allowing for optional articles
        replacement = "you"  # Define the replacement string

        # Replace occurrences of the pattern in the text with 'you', ignoring case
        return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def node_exists(self, node_id):
        """
        Checks if a node exists based on its unique identifier.

        This method determines whether a node with the specified ID exists by comparing it to the total number of
        observations. It returns a boolean value indicating the existence of the node.

        Args:
            node_id (int): The unique identifier of the node to check.

        Returns:
            bool: True if the node exists; False otherwise.
        """

        # Return True if the node ID is less than the total number of observations, indicating the node exists
        return node_id < self.num_observations

    def is_valid_memory_type(self, memory_type):
        """
        Checks if the provided memory type is valid.

        This method verifies whether the given memory type is an instance of the MemoryType enum or can be converted to
        it. It returns True if the memory type is valid and False otherwise, ensuring that only supported types are
        processed.

        Args:
            memory_type (Union[int, MemoryType]): The memory type to validate, which can be an integer or a MemoryType
            enum.

        Returns:
            bool: True if the memory type is valid; False otherwise.
        """

        if isinstance(
            memory_type, MemoryType
        ):  # Check if the provided memory_type is already an instance of MemoryType
            return True  # Return True if it is a valid MemoryType

        try:
            # Attempt to convert the input value to a MemoryType
            _ = MemoryType(
                memory_type
            )  # This will succeed if memory_type is a valid integer representation of a MemoryType
        except ValueError:
            return False  # Return False if the conversion fails, indicating an invalid memory type
        else:
            return True  # Return True if the conversion is successful, confirming the memory type is valid

    @classmethod
    def get_keyword_embeddings(cls, keyword):
        """
        Retrieves the embedding associated with a specified keyword or a list of keywords.

        This method checks the keyword embeddings dictionary to find the embedding for the given keyword.
        If the keyword is found, the corresponding embedding is returned. If the keyword does not exist,
        it generates a new embedding for the keyword. When a list of keywords is provided, it returns
        a list of embeddings or None values for each keyword.

        Args:
            keyword (Union[str, list[str]]): A single keyword or a list of keywords for which the embedding is to be
                                             retrieved.

        Returns:
            Union[np.array, list[np.array]]: The embedding associated with the specified keyword, or a list of embeddings
            for the provided list of keywords.
        """

        if isinstance(keyword, dict):
            print("KEYWORDS in get_keyword_embeddings:", keyword)

        if isinstance(keyword, (list, set, tuple)):
            # Use map to retrieve embeddings for the list of keywords
            return list(map(lambda kw: cls.keyword_embeddings.get(kw, None), keyword))
        else:
            return cls.keyword_embeddings.get(keyword, None)

    # def reflect(self, game: Game):
    #     """
    #     Reflects on the current state of the game, allowing the agent to process and evaluate its experiences.

    #     This method invokes the reflection process, which may involve analyzing past actions, updating memory, and
    #     adjusting goals based on the game's current context.

    #     Args:
    #         game (Game): The current game object, which contains information about the game state and context.

    #     Returns:
    #         None
    #     """

    #     # Initialize the GPT handler for reflection
    #     Reflect.initialize_gpt_handler()

    #     # Perform the reflection process
    #     Reflect.reflect(game, self.character)

    def convert_keyword_embeddings_to_keywords(self, keyword_embeddings: dict) -> dict:
        """
        Converts keyword embeddings to keywords.

        This function takes a dictionary mapping keyword types to dictionaries that map keywords to their embeddings.
        It adds each keyword and its corresponding embedding to the MemoryStream.keyword_embeddings dictionary,
        which maps keywords to embeddings. The function returns a modified dictionary that maps keyword types to
        lists of keywords.

        Args:
            keyword_embeddings (dict): A dictionary in the following format:
                {
                    "keyword_type": {
                        "keyword": embedding
                    }
                }

        Returns:
            dict: A dictionary in the following format:
                {
                    "keyword_type": [keyword, ...]
                }
        """
        modified_keywords = {}

        for keyword_type, keywords_dict in keyword_embeddings.items():
            # Initialize a list to hold the keywords for this type
            keywords_list = []

            for keyword, embedding in keywords_dict.items():
                # Add the keyword and its embedding to the MemoryStream
                MemoryStream.keyword_embeddings[keyword] = embedding
                # Append the keyword to the list
                keywords_list.append(keyword)

            # Update the modified dictionary with the keyword type and its list of keywords
            modified_keywords[keyword_type] = keywords_list

        return modified_keywords
