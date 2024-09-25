"""
Author: Rut Vyas

File: agent_cognition/goals.py
Description: defines how agents reflect upon their past experiences
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections import defaultdict

import numpy as np
from text_adventure_games.utils.general import get_logger_extras, get_text_embedding
from text_adventure_games.assets.prompts import goal_prompt as gp
from text_adventure_games.gpt.gpt_helpers import (
    GptCallHandler,
    limit_context_length,
    get_prompt_token_count,
    get_token_remainder,
    context_list_to_string,
)

if TYPE_CHECKING:
    from text_adventure_games.things.characters import Character
    from text_adventure_games.games import Game

GOALS_MAX_OUTPUT = 256

# 1. Get character's goals
# 2. Obtain a list of memories
# 3. ask

# TODO: max output length ? - TBD
# TODO: summarize impressions for goals ? - TBD
# TODO: pass previous round plan - try passing in system prompt - try on playground first


class Goals:

    def __init__(self, character: "Character"):
        """
        The goal is stored in the form of a dictionary based on the priority with the round number as the key in the
        following format:
            {Round #:
                {"Low Priority": _description_,
                 "Medium Priority: _description_,
                 "High Priority": _description_}
        """
        self.character = character
        self.goals = defaultdict(dict)
        self.goal_scores = defaultdict(dict)
        self.recent_reflection = None
        self.goal_embeddings = defaultdict(np.ndarray)

        # GPT Call handler attrs
        self.gpt_handler = self._set_up_gpt()
        self.token_offset = (
            50  # Taking into account a few variable tokens in the user prompt
        )
        self.offset_pad = 5

    def _set_up_gpt(self):
        """
        Configures and initializes the GPT handler with the specified model parameters.
        This method sets up the necessary configurations for the GPT model to be used in generating responses.

        Returns:
            GptCallHandler: An instance of the GptCallHandler configured with the specified parameters.
        """

        model_params = {
            "api_key_org": "Helicone",
            "model": "gpt-4",
            "max_tokens": GOALS_MAX_OUTPUT,
            "temperature": 1,
            "top_p": 1,
            "max_retries": 5,
        }

        return GptCallHandler(**model_params)

    def _log_goals(self, game, message):
        """
        Logs the specified message related to goals in the game's logger.
        This method adds additional context to the log entry, including character-specific information.

        Args:
            game: The current game instance used for logging.
            message (str): The message to be logged regarding the goals.

        Returns:
            None
        """

        # Retrieve additional logging context specific to the game and character.
        extras = get_logger_extras(game, self.character)

        # Set the type of log entry to "Goal" for categorization.
        extras["type"] = "Goal"

        # Log the debug message along with the additional context.
        game.logger.debug(msg=message, extra=extras)

    def gpt_generate_goals(self, game: "Game") -> str:
        """
        Calls GPT to create a goal for the character
        System prompt uses: world info, agent personal summary, and the target's name
        User prompt uses: impressions created by this character of other players, reflection of previous round

        Args:
            game (Game): the game

        Returns:
            str: a new goal for this round
        """

        # Build the system and user prompts required for goal generation.
        system, user = self.build_goal_prompts(game)

        # Generate a goal using the GPT handler based on the constructed prompts.
        goal = self.gpt_handler.generate(system=system, user=user)

        # Check if the generated goal is a tuple, indicating a potential error related to token limits.
        if isinstance(goal, tuple):
            # Unpack the tuple to get the success status and the token difference.
            success, token_difference = goal

            # Update the token offset to account for the exceeded limit and add padding for future calculations.
            self.token_offset = token_difference + self.offset_pad
            self.offset_pad += 2 * self.offset_pad

            # Recursively call the goal generation method to attempt generating a goal again with updated limits.
            return self.gpt_generate_goals(self.game)

        # Log the generated goal for tracking and debugging purposes.
        self._log_goals(game, goal)

        # Create an embedding for the generated goal to facilitate further processing.
        goal_embed = self._create_goal_embedding(goal)

        # Update the goal with the new embedding for experimentation purposes.
        self.goal_update(goal, goal_embed, game)

        # Return the generated goal.
        return goal

    def build_goal_prompts(self, game):
        """
        Constructs the system and user prompts necessary for goal generation.
        This method builds the prompts based on the current game state and the number of tokens consumed.

        Args:
            game: The current game instance used to generate the prompts.

        Returns:
            tuple: A tuple containing the system prompt and the user prompt for goal generation.
        """

        # Build the system prompt and retrieve the associated token count from the game.
        system_prompt, sys_tkn_count = self.build_system_prompt(game)

        # Calculate the total number of tokens consumed by adding the system token count to the current token offset.
        consumed_tokens = sys_tkn_count + self.token_offset

        # Generate the user prompt based on the game state and the total consumed tokens.
        user_prompt = self.build_user_prompt(game, consumed_tokens=consumed_tokens)

        # Return both the system prompt and the user prompt as a tuple for further processing.
        return system_prompt, user_prompt

    def build_system_prompt(self, game):
        """
        Constructs the system prompt used for generating goals in the game.
        This method combines the character's standard information with a predefined goals prompt and calculates the
        token count.

        Args:
            game: The current game instance used to gather character information.

        Returns:
            tuple: A tuple containing the constructed system prompt and its token count.
        """

        # Retrieve the standard information about the character from the game, excluding perceptions.
        system_prompt = self.character.get_standard_info(
            game, include_perceptions=False
        )

        # Append the predefined goals prompt to the character's standard information to form the complete system prompt.
        system_prompt += gp.gpt_goals_prompt

        # Calculate the token count for the constructed system prompt to manage token limits.
        system_tkn_count = get_prompt_token_count(
            system_prompt,
            role="system",
            pad_reply=False,
            tokenizer=game.parser.tokenizer,
        )

        # Return the complete system prompt along with its token count.
        return system_prompt, system_tkn_count

    def build_user_prompt(self, game, consumed_tokens=0):
        # sourcery skip: avoid-builtin-shadow
        """
        Constructs the user prompt for goal generation, incorporating relevant context and previous goals.
        This method gathers reflections, previous goals, and scores from the last two rounds to provide a comprehensive
        context for the user.

        Args:
            game: The current game instance used to gather information.
            consumed_tokens (int, optional): The number of tokens already consumed by previous prompts. Defaults to 0.

        Returns:
            str: The constructed user prompt that includes context for creating or updating goals.
        """

        # Define a list of strings that are always included in the user prompt
        always_included = [
            "Additional context for creating your goal:\n",
            "You can keep the previous goal, update the previous goal or create a new one based on your strategy.",
        ]

        # Calculate the token count for the always included strings
        always_included_count = get_prompt_token_count(
            always_included,
            role="user",
            pad_reply=True,
            tokenizer=game.parser.tokenizer,
        )

        # Determine the number of available tokens for reflections and goals
        available_tokens = get_token_remainder(
            self.gpt_handler.model_context_limit,
            self.gpt_handler.max_tokens,
            consumed_tokens,
            always_included_count,
        )

        # Set limits for reflections and goals based on available tokens
        reflections_limit, goals_limit = int(available_tokens * 0.6), int(
            available_tokens * 0.3
        )

        # Retrieve goals and scores for the previous round and two rounds prior
        round = game.round
        goal_prev = None
        goal_prev_2 = None

        # Get the previous round's goal and score if it exists
        if round > 0:
            goal_prev = self.get_goals(round=round - 1, as_str=True)
            score = self.get_goal_scores(round=round - 1, as_str=True)

        # Get the goal and score from two rounds prior if it exists
        if round > 1:
            goal_prev_2 = self.get_goals(round=round - 2, as_str=True)
            score_2 = self.get_goal_scores(round=round - 2, as_str=True)

        # Retrieve reflection nodes for two rounds prior
        reflection_raw_2 = []
        node_ids = self.character.memory.get_observations_after_round(
            round - 2, inclusive=True
        )

        # Collect reflection descriptions from the memory
        for node_id in node_ids:
            node = self.character.memory.get_observation(node_id)
            if node.node_type.value == 3:  # Check if the node type is a reflection
                reflection_raw_2.append(node.node_description)

        # Initialize the user prompt with the always included context
        user_prompt = always_included[0]

        # If there are reflections from two rounds prior, add them to the user prompt
        if reflection_raw_2:
            user_prompt += "Reflections on last two rounds:"
            context_list = limit_context_length(
                history=reflection_raw_2,
                max_tokens=reflections_limit,
                tokenizer=game.parser.tokenizer,
            )
            reflection_2 = context_list_to_string(context_list, sep="\n")
            user_prompt += f"{reflection_2}\n"

        # If there is a goal from the previous round, add it to the user prompt
        if goal_prev:
            context_list = [
                "Goals of prior round:",
                goal_prev,
                "Goal Completion Score of prior round:",
                score,
            ]
            context_list = limit_context_length(
                history=context_list,
                max_tokens=goals_limit // 2,
                tokenizer=game.parser.tokenizer,
            )
            goal_prev_str = context_list_to_string(context_list, sep="\n")
            user_prompt += f"{goal_prev_str}\n\n"

        # If there is a goal from two rounds prior, add it to the user prompt
        if goal_prev_2:
            context_list = [
                "Goals of two rounds prior:",
                goal_prev_2,
                "Goal Completion Score of two rounds prior:",
                score_2,
            ]
            context_list = limit_context_length(
                history=context_list,
                max_tokens=goals_limit // 2,
                tokenizer=game.parser.tokenizer,
            )
            goal_prev_2_str = context_list_to_string(context_list, sep="\n")
            user_prompt += f"{goal_prev_2_str}\n"

        # Append the final always included context to the user prompt
        user_prompt += always_included[1]

        # Return the constructed user prompt
        return user_prompt

    def goal_update(self, goal: str, goal_embedding: np.ndarray, game: "Game"):
        # sourcery skip: avoid-builtin-shadow
        """
        Updates the goals for the current round based on the provided goal string and its embedding.
        This method categorizes the goals into priority levels and stores them along with their embeddings for future
        reference.

        Args:
            goal (str): A string containing the goals, categorized by priority.
            goal_embedding (np.ndarray): The embedding representation of the goal for further processing.
            game (Game): The current game instance used to access the round information.

        Returns:
            None
        """

        # Get the current round from the game
        round = game.round

        # Initialize a dictionary for goals in the current round
        self.goals[round] = {}

        # Split the goal string into lines and process each line
        for line in goal.split("\n"):
            # Check for 'Low Priority' in the line and store the corresponding goal
            if "Low Priority" in line:
                self.goals[round]["Low Priority"] = line.replace("Low Priority: ", "")
            # Check for 'Medium Priority' in the line and store the corresponding goal
            elif "Medium Priority" in line:
                self.goals[round]["Medium Priority"] = line.replace(
                    "Medium Priority: ", ""
                )
            # Check for 'High Priority' in the line and store the corresponding goal
            elif "High Priority" in line:
                self.goals[round]["High Priority"] = line.replace("High Priority: ", "")

        # Update the goal embeddings for the current round
        self.goal_embeddings.update({round: goal_embedding})

        # Persist the updated goals in memory for the current round
        self.update_goals_in_memory(round)

    def get_goals(self, round=-1, priority="all", as_str=False):
        """Retrieves goals based on the specified round and priority.

        This method allows for flexible retrieval of goals, either for a specific round and priority or all goals if no
        specific criteria are provided. It can also return the goals as a string representation if requested.

        Args:
            round (int, optional): The round number for which to retrieve goals. Defaults to -1, which retrieves all
            goals.
            priority (str, optional): The priority level of the goals to retrieve. Defaults to "all", which retrieves
            goals of all priorities.
            as_str (bool, optional): If True, returns the goals as a string. Defaults to False.

        Returns:
            Union[dict, str]: The goals for the specified round and priority, or a string representation of the goals if
            as_str is True.
        """

        """
        Getter function for goal
            Args:
                round: round number (default is all rounds)
                priority: priority of goal needed (default is all priority goals)

            Returns:
                The goal
        """

        if (
            round != -1 and priority != "all"
        ):  # Check if a specific round and priority are provided
            goal = self.goals[round][
                priority
            ]  # Retrieve the goal for the specified round and priority
        elif round != -1:  # Check if only a specific round is provided
            goal = self.goals[round]  # Retrieve all goals for the specified round
        else:  # If no specific round is provided
            goal = self.goals  # Retrieve all goals

        # Return the goals as a string if as_str is True; otherwise, return the goals as is
        return self.stringify_goal(goal) if as_str else goal

    def stringify_goal(self, goal):
        """Converts a goal or a collection of goals into a string representation.

        This method checks the type of the provided goal and converts it to a string.
        If the goal is a collection, it concatenates the values into a single string, handling potential type errors
        gracefully.

        Args:
            goal (Union[str, dict]): The goal to be converted, which can be a string or a dictionary of goals.

        Returns:
            str: The string representation of the goal or an empty string if the conversion fails.
        """

        if isinstance(goal, str):  # Check if the provided goal is already a string
            return goal  # Return the goal as is if it is a string

        goal_str = (
            ""  # Initialize an empty string to accumulate the goal representation
        )

        try:
            # TODO: shouldn't this be "if len(goal) > 1:"?
            # Check if the goal is a collection with more than one item
            if len(goal) > 1:
                # Convert the entire collection to a string
                goal_str = str(goal)
            # If the goal is a collection with one item
            else:
                # Iterate through the values of the goal dictionary
                for g in goal.values():
                    # Concatenate each value to the goal_str with a space
                    goal_str += f"{g} "
            # Return the constructed string representation of the goal
            return goal_str
        # Catch a TypeError if the goal is not iterable
        except TypeError:
            # Return the accumulated goal_str, which may be empty
            return goal_str

    def _create_goal_embedding(self, goal: str) -> np.ndarray:
        """Generates an embedding for a specified goal.

        This method takes a goal represented as a string and converts it into a numerical embedding using a text
        embedding function. The resulting embedding can be used for various applications, such as similarity comparisons
        or machine learning tasks.

        Args:
            goal (str): The goal to be converted into an embedding.

        Returns:
            np.ndarray: The numerical embedding representation of the specified goal.
        """

        return get_text_embedding(goal)

    def get_goal_embedding(self, round: int):
        """Retrieves the goal embedding for a specified round.

        This method looks up the goal embedding associated with the given round number.
        If no embedding exists for that round, it returns None, allowing for easy handling of missing data.

        Args:
            round (int): The round number for which to retrieve the goal embedding.

        Returns:
            np.ndarray or None: The goal embedding for the specified round, or None if no embedding exists.
        """

        return self.goal_embeddings.get(round, None)

    def update_goals_in_memory(self, round):
        """Updates the character's memory with the current goal embedding for a specified round.

        This method retrieves the goal embedding associated with the given round and, if it exists,
        updates the character's memory with this embedding. This ensures that the character's memory reflects the
        current goals.

        Args:
            round (int): The round number for which to update the goals in memory.

        Returns:
            None
        """

        # Retrieve the current goal embedding for the specified round
        curr_embedding = self.get_goal_embedding(round)

        # Check if the retrieved embedding is not None
        if curr_embedding is not None:
            # Update the character's memory with the current goal embedding
            self.character.memory.set_goal_query(curr_embedding)

    # ----------- EVALUATION -----------
    def evaluate_goals(self, game: "Game"):
        """Evaluates the goals of the agent within the context of the game.

        This method constructs prompts for a language model to assess the agent's goals based on the current game state.
        It retrieves scores from the model and updates the agent's score accordingly, providing feedback on goal
        performance. User prompt uses: reflection, actions, goals of previous round

        Args:
            game (Game): The current game instance used to evaluate the agent's goals.

        Returns:
            list: The scores generated by the evaluation of the agent's goals.
        """

        # Retrieve the system prompt for evaluating goals from the gp module
        system_prompt = gp.evaluate_goals_prompt
        # Count the tokens in the system prompt for token management
        system_prompt_tokens = get_prompt_token_count(system_prompt, role="system")

        # Build the user prompt based on the game state and consumed tokens
        user_prompt = self.build_eval_user_prompt(
            game, consumed_tokens=system_prompt_tokens
        )

        # Generate scores by passing the system and user prompts to the GPT handler
        scores = self.gpt_handler.generate(system=system_prompt, user=user_prompt)
        # Update the agent's score based on the generated scores and the current game
        self.score_update(scores, game)

        # Return the scores obtained from the evaluation
        return scores

    def build_eval_user_prompt(self, game, consumed_tokens=0):
        # sourcery skip: avoid-builtin-shadow
        """
        Constructs a user prompt for evaluating progress towards a goal based on reflections and actions from the
        current round. This prompt includes the current goal, reflections, and actions, formatted for user input.

        Args:
            self: The instance of the class.
            game: The current game instance containing game state information.
            consumed_tokens (int, optional): The number of tokens already consumed. Defaults to 0.

        Returns:
            str: A formatted string containing the user prompt with the goal, reflections, and actions.
        """


        # Retrieve the current goal for the ongoing round and format it for the prompt
        goal = self.get_goals(round=game.round, as_str=True)
        goal_prompt = f"Goal:{goal}\n\n"

        # Define a list of strings that will always be included in the user prompt
        always_included = [
            "Score the progress toward the goal that is suggested by the reflections provided below:\n",
            goal_prompt,
            "Your reflections and actions from this round:",
        ]

        # Calculate the token count for the always included strings
        always_included_tokens = get_prompt_token_count(
            always_included, role="user", pad_reply=True
        )

        # Determine the number of available tokens for reflections and actions
        available_tokens = get_token_remainder(
            self.gpt_handler.model_context_limit,
            consumed_tokens,
            always_included_tokens,
        )

        # Initialize lists to store reflections and actions for the current round
        reflections_raw = []
        actions_raw = []
        round = game.round

        # Retrieve observation node IDs for the current round
        node_ids = self.character.memory.get_observations_by_round(round)

        # Collect reflections and actions made by this agent in the current round
        for node_id in node_ids:
            node = self.character.memory.get_observation(node_id)
            # Check if the node is a reflection made by this agent
            if node.node_type.value == 3 and node.node_is_self == 1:
                reflections_raw.append(node.node_description)
            # Check if the node is an action made by this agent
            if node.node_type.value == 1 and node.node_is_self == 1:
                actions_raw.append(node.node_description)

        # Limit the length of the reflections based on available tokens
        reflections_list = limit_context_length(
            history=reflections_raw,
            max_tokens=available_tokens // 2,
            tokenizer=game.parser.tokenizer,
        )
        # Convert the limited reflections list to a formatted string
        reflections_str = context_list_to_string(reflections_list, sep="\n")

        # Limit the length of the actions based on available tokens
        actions_list = limit_context_length(
            history=actions_raw,
            max_tokens=available_tokens // 2,
            tokenizer=game.parser.tokenizer,
        )


    def score_update(self, score: str, game: "Game"):
        # sourcery skip: avoid-builtin-shadow
        """
        Maintains the dictionary of goal completion scores for the character by round
        """
        
        # Get the current round from the game
        round = game.round

        # Initialize a dictionary to store goal scores for the current round
        self.goal_scores[round] = {}

        # Split the score string into lines and process each line
        for line in score.split("\n"):
            # Check for 'Low Priority' in the line and attempt to convert the score to an integer
            if "Low Priority" in line:
                try:
                    self.goal_scores[round]["Low Priority"] = int(
                        line.replace("Low Priority: ", "")
                    )
                except ValueError:
                    # Handle the case where conversion to integer fails
                    print("Error: Unable to convert 'Low Priority' to an integer.")
            
            # Check for 'Medium Priority' in the line and attempt to convert the score to an integer
            elif "Medium Priority" in line:
                try:
                    self.goal_scores[round]["Medium Priority"] = int(
                        line.replace("Medium Priority: ", "")
                    )
                except ValueError:
                    # Handle the case where conversion to integer fails
                    print("Error: Unable to convert 'Medium Priority' to an integer.")
            
            # Check for 'High Priority' in the line and attempt to convert the score to an integer
            elif "High Priority" in line:
                try:
                    self.goal_scores[round]["High Priority"] = int(
                        line.replace("High Priority: ", "")
                    )
                except ValueError:
                    # Handle the case where conversion to integer fails
                    print("Error: Unable to convert 'High Priority' to an integer.")


    def get_goal_scores(self, round=-1, priority="all", as_str=False):
        """
        Getter function for goal completion scores
            Args:
                round: round number (default is all rounds)
                priority: priority of goal score needed (default is all priority goal scores)

            Returns:
                The goal score
        """
        
        # Check if the current round is valid and a specific priority is requested
        if round != -1 and priority != "all":
            # Retrieve the score for the specified priority in the current round
            score = self.goal_scores[round][priority]
        # If the round is valid but no specific priority is requested, retrieve all scores for the current round
        elif round != -1:
            score = self.goal_scores[round]
        # If the round is invalid, retrieve all goal scores
        else:
            score = self.goal_scores

        # Return the score as a string if as_str is True, otherwise return the score as is
        return self.stringify_goal(score) if as_str else score

