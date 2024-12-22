circular_import_prints = False

if circular_import_prints:
    print("Importing Dialogue")

from typing import (
    TYPE_CHECKING,
    Union,
    List,
    Set,
    Literal,
    Tuple,
)  # Allows conditional imports for type hints

import numpy as np

import heapq

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

    DEBUG_MODE = False

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
        self.reflection_weight = 0.1
        self.impression_weight = 0.1
        self.goal_weight = 0.175

        self.percentile_threshold = 0.5  # TODO: Pick this hyperparameter

    def get_next_speaker(
        self,
        speaker: "Character",
        last_response: Union[dict[str, Union[str, int]], list[str], str, None],
        response_splits_token_counts: Union[list[int], None] = None,
        response_name_history: list[str] = None,
    ) -> Tuple["Character", bool]:
        """
        Get the next speaker in the dialogue priority queue.

        Args:
            speaker (Character): The current speaker.
            last_response (Union[dict[str, Union[str, int]], list[str], str, None]): The last response from the GPT model.

        Returns:
            Tuple[Character, bool]: The next speaker (Character) and a boolean indicating if the speaker must speak
            (everyone else declined and we are back to the original top scorer).
        """
        if last_response is None or last_response == []:
            # If the last response is None or an empty list, check if the heap is empty
            if not self.priority_queue.heap:
                print(
                    "NOBODY ELSE WANTS TO SPEAK, SO WE'RE GOING BACK TO THE ORIGINAL TOP SCORER"
                )
                return self.original_top_scoring_listener, True
            print("CHECKING IF SOMEONE ELSE WANTS TO SPEAK")
            return self.get_adj_next_speaker(response_name_history), False

        if isinstance(last_response, list) and all(
            isinstance(item, dict) for item in last_response
        ):
            query_keywords_embeddings = Retrieve.get_query_keywords_and_embeddings(
                game=self.game,
                query=[split["component"] for split in last_response],
                scores=[(response_splits_token_counts[i], split["importance_score"]) for i, split in enumerate(last_response)],
            )
        else:
            # Retrieve query keywords and embeddings based on the last response
            query_keywords_embeddings = Retrieve.get_query_keywords_and_embeddings(
                game=self.game, query=last_response
            )

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

        # Normalize the priority scores across all listeners for each memory type and score type (in-place)
        self._normalize_listener_scores(listener_scores)

        if self.DEBUG_MODE:
            print("\nNORMALIZED LISTENER SCORES")
            for listener, scores in listener_scores.items():
                print(listener.name + ":", scores)

        # Collapse the normalized scores into weighted averages for each listener and memory type (in-place)
        listener_scores = self._collapse_listener_scores(listener_scores)

        if self.DEBUG_MODE:
            print("\nCOLLAPSED LISTENER SCORES")
            for listener, scores in listener_scores.items():
                print(listener.name + ":", scores)

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

        if self.DEBUG_MODE:
            print("\nMEAN SCORES ABOVE PERCENTILE THRESHOLD")
            for memory_type, means in memory_type_means.items():
                print(str(memory_type) + ":", means)

        # Normalize the memory type means across listeners
        memory_type_means = self.normalize_means_across_listeners(
            listener_scores, memory_type_means
        )

        if self.DEBUG_MODE:
            print("\nNORMALIZED MEAN SCORES ACROSS LISTENERS")
            for memory_type, means in memory_type_means.items():
                print(str(memory_type) + ":", means)

        # Calculate the final score for each listener
        for listener_name, listener_priority_scores in listener_scores.items():
            listener_scores[listener_name] = self.calculate_final_score(
                listener_priority_scores
            )

        if self.DEBUG_MODE:
            print("\nNEW LISTENER SCORES")
            for listener, score in listener_scores.items():
                print(listener.name + ":", round(score, 2))

        # Update the priority queue with the new listener scores
        self.priority_queue.update_listeners(listener_scores)

        next_speaker = self.get_adj_next_speaker(response_name_history)

        self.original_top_scoring_listener = next_speaker

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

    def get_adj_next_speaker(self, response_name_history):
        # Temporarily adjust the scores to favor characters who have spoken less
        adjusted_scores = {}
        for listener, score in self.priority_queue.listener_map.items():
            # Calculate adjustment for listeners who have spoken
            if response_name_history and listener.name in response_name_history:
                # Calculate weighted response count
                weighted_response_count = sum(
                    (1 / (index + 1))
                    for index, name in enumerate(reversed(response_name_history))
                    if name == listener.name
                )
                # Calculate adjustment based on weighted response count
                adjustment = 0.5 / (1 + weighted_response_count)
                adjusted_scores[listener] = score + adjustment
            else:
                # Give a boost to listeners who haven't spoken at all
                adjusted_scores[listener] = (
                    score + 0.5
                )  # Boost for listeners who haven't spoken

        # Temporarily replace the scores in the priority queue
        original_scores = self.priority_queue.listener_map.copy()
        self.priority_queue.listener_map = adjusted_scores
        self.priority_queue._rebuild_heap()

        # Print the player names and scores in the dialogue queue
        # if self.DEBUG_MODE:
        print("\nDIALOGUE QUEUE:")
        for listener, score in self.priority_queue.listener_map.items():
            adjustment = adjusted_scores[listener] - original_scores[listener]
            print(f"{listener.name}: {round(score, 2)} (+{round(adjustment, 2)})")

        # Get the next speaker with the adjusted scores
        next_speaker = self.priority_queue.get_next_speaker()

        # Revert the priority queue to the original scores
        self.priority_queue.listener_map = original_scores
        self.priority_queue._rebuild_heap()

        # Remove the original top scoring listener from the priority queue
        if next_speaker in self.priority_queue.listener_map:
            del self.priority_queue.listener_map[next_speaker]

        return next_speaker

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

                                    # If the score is an array of shape (n, 2), average the numbers in the array
                                    if (
                                        score_value.ndim == 2
                                        and score_value.shape[1] == 2
                                        or score_value.shape == (2,)
                                    ):
                                        score_value = np.mean(score_value)

                                    # Add the weighted score to the weighted sum
                                    weighted_sum += alpha * score_value
                        # Append the weighted sum to the weighted scores list
                        weighted_scores.append(weighted_sum)
                    # Store the weighted scores list in the collapsed scores dictionary
                    collapsed_scores[listener][memory_type] = weighted_scores
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
