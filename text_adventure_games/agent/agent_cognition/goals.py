"""
Author: Rut Vyas

File: agent_cognition/goals.py
Description: defines how agents reflect upon their past experiences
"""

# Import future annotations for forward reference type hints.
from __future__ import annotations

# Import TYPE_CHECKING to allow for type hints without circular imports.
from typing import TYPE_CHECKING

# Import defaultdict from collections for creating default dictionaries.
from collections import defaultdict

# Import numpy for numerical operations.
import numpy as np

# Import the get_models_config function from the consts module in the utils package.
# This function is used to retrieve the configuration for different models used in the game.
from text_adventure_games.utils.consts import get_models_config

# Import utility functions for logging and text embedding from the general module.
from text_adventure_games.utils.general import get_logger_extras, get_text_embedding

# Import the goal prompt from the prompts module.
from text_adventure_games.assets.prompts import goal_prompt as gp

# Import the logging module to enable logging functionality within this script
import logging

# Importing the inspect module, which provides useful functions to get information about live objects
import inspect

# Import helper functions for GPT calls and context management from the gpt_helpers module.
from text_adventure_games.gpt.gpt_helpers import (
    GptCallHandler,
    limit_context_length,
    get_prompt_token_count,
    get_token_remainder,
    context_list_to_string,
)

# Conditional import for type checking to avoid circular dependencies.
if TYPE_CHECKING:
    from text_adventure_games.things.characters import Character
    from text_adventure_games.games import Game

# 1. Get character's goals
# 2. Obtain a list of memories
# 3. ask

# TODO: max output length ? - TBD
# TODO: summarize impressions for goals ? - TBD
# TODO: pass previous round plan - try passing in system prompt - try on playground first

class Goals:
    """
    Manages the goals for a character within the game, allowing for the creation, updating, and evaluation of goals
    based on the character's actions and reflections. This class utilizes a shared GPT handler to generate goals and
    maintains a structured format for storing and retrieving goals by priority and round.

    Args:
        character (Character): The character for whom the goals are being managed.

    Attributes:
        character (Character): The character associated with this goals manager.
        goals (defaultdict): A dictionary storing goals categorized by round and priority.
        goal_scores (defaultdict): A dictionary storing goal completion scores categorized by round and priority.
        recent_reflection: Stores the most recent reflection made by the character.
        goal_embeddings (defaultdict): A dictionary storing embeddings for each goal by round.
        token_offset (int): The offset for managing token limits in GPT calls.
        offset_pad (int): Additional padding for token management.

    Class Attributes:
        gpt_handler (GptCallHandler): A class-level shared GPT handler for all instances.
        model_params (dict): Class-level model parameters for configuring the GPT handler.

    Methods:
        _log_goals(game, message):
            Logs the specified message related to goals in the game's logger.

        gpt_generate_goals(game):
            Calls GPT to create a new goal for the character based on the current game state.

        build_goal_prompts(game):
            Constructs the system and user prompts necessary for goal generation.

        build_system_prompt(game):
            Constructs the system prompt used for generating goals in the game.

        build_user_prompt(game, consumed_tokens=0):
            Constructs the user prompt for goal generation, incorporating relevant context and previous goals.

        goal_update(goal, goal_embeddings, game):
            Updates the goals for the current round based on the provided goal string and its embedding.

        get_goals(round=-1, priority="all", as_str=False):
            Retrieves goals based on the specified round and priority.

        stringify_goal(goal):
            Converts a goal or a collection of goals into a string representation.

        _create_goal_embedding(goal):
            Generates an embedding for a specified goal.

        get_goal_embeddings(round):
            Retrieves the goal embeddings for a specified round.

        update_goals_in_memory(round):
            Updates the character's memory with the current goal embedding for a specified round.

        evaluate_goals(game):
            Evaluates the goals of the agent within the context of the game.

        build_eval_user_prompt(game, consumed_tokens=0):
            Constructs a user prompt for evaluating progress towards a goal based on reflections and actions.

        score_update(score, game):
            Maintains the dictionary of goal completion scores for the character by round.

        get_goal_scores(round=-1, priority="all", as_str=False):
            Retrieves goal completion scores based on the specified round and priority.
    """

    gpt_handler = None  # Class-level attribute to store the shared GPT handler
    model_params = {
        "api_key_org": "Helicone",
        "model": get_models_config()["goals"]["model"],
        "max_tokens": 256,
        "temperature": 1,
        "top_p": 1,
        "max_retries": 5,
    }

    logger = None  # Class-level attribute to store the shared logger

    def __init__(self, character: "Character"):
        """
        Initializes the Goals manager for a character, setting up the necessary data structures to track goals, scores,
        and embeddings. This constructor also ensures the shared GPT handler is configured for generating goals and initializes parameters
        for managing token limits.

        The goal is stored in the form of a dictionary based on the priority with the round number as the key in the
        following format:
            {Round #:
                {"Low Priority": _description_,
                 "Medium Priority": _description_,
                 "High Priority": _description_}

        Args:
            character (Character): The character for whom the goals are being managed.

        Attributes:
            character (Character): The character associated with this goals manager.
            goals (defaultdict): A dictionary storing goals categorized by round and priority.
            goal_scores (defaultdict): A dictionary storing goal completion scores categorized by round and priority.
            recent_reflection: Stores the most recent reflection made by the character.
            goal_embeddings (defaultdict): A dictionary storing embeddings for each goal by round.
            token_offset (int): The offset for managing token limits in GPT calls.
            offset_pad (int): Additional padding for token management.
        """

        # Assign the character associated with this goals manager to an instance variable.
        self.character = character

        # Initialize a defaultdict to store goals categorized by round and priority.
        self.goals = defaultdict(dict)

        # Initialize a defaultdict to store goal completion scores categorized by round and priority.
        self.goal_scores = defaultdict(dict)

        # Initialize a variable to store the most recent reflection made by the character.
        self.recent_reflection = None

        # Ensure the shared GPT call handler is set up
        if Goals.gpt_handler is None:
            Goals.gpt_handler = GptCallHandler(**Goals.model_params)

        if Goals.logger is None:
            Goals.logger = logging.getLogger("agent_cognition")

        # Initialize a defaultdict to store embeddings for each goal by round.
        self.goal_embeddings = defaultdict(
            lambda: None  # np.zeros((3, Goals.gpt_handler.embedding_dimensions))
        )

        # Set the token offset to account for a few variable tokens in the user prompt.
        self.token_offset = 50  # Taking into account a few variable tokens in the user prompt

        # Initialize an offset padding value for managing token limits.
        self.offset_pad = 5

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
        extras = get_logger_extras(game, self.character, include_gpt_call_id=True)

        extras["type"] = "Goals"

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

        # Generate goals using the GPT handler based on the constructed prompts.
        goals = Goals.gpt_handler.generate(system=system, user=user, character=self.character)

        # Check if the generated goal is a tuple, indicating a potential error related to token limits.
        if isinstance(goals, tuple):
            # Unpack the tuple to get the success status and the token difference.
            _, token_difference = goals

            # Update the token offset to account for the exceeded limit and add padding for future calculations.
            self.token_offset = token_difference + self.offset_pad
            self.offset_pad += 2 * self.offset_pad

            # Recursively call the goal generation method to attempt generating a goal again with updated limits.
            return self.gpt_generate_goals(self.game)

        # Log the generated goal for tracking and debugging purposes.
        self._log_goals(game, goals)

        # Create an embedding for the generated goal to facilitate further processing.
        goal_embed = self._create_goal_embedding(goals, game)

        # Update the goal with the new embedding for experimentation purposes.
        self.goal_update(goals, goal_embed, game)

        # Return the generated goal.
        return goals

    def build_goal_prompts(self, game):
        """
        Constructs the system and user prompts necessary for goal generation.

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
        Constructs the system prompt used for generating goals in the game, utilizing the character's standard
        information.

        Args:
            game: The current game instance used to gather character information.

        Returns:
            tuple: A tuple containing the constructed system prompt and its token count.
        """

        # Retrieve the standard information about the character from the game, excluding goals and perceptions.
        system_prompt = self.character.get_standard_info(
            game, include_goals=False, include_perceptions=False
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
        """
        Constructs the user prompt for goal generation, incorporating relevant context including reflections, previous
        goals, and scores from the last two rounds to provide a comprehensive context for the user.

        Args:
            game: The current game instance used to gather information.
            consumed_tokens (int, optional): The number of tokens already consumed by previous prompts. Defaults to 0.

        Returns:
            str: The constructed user prompt that includes context for creating or updating goals.
        """

        # Define a list of strings that are always included in the user prompt
        always_included = [
            "Additional context for creating your goal:\n\n",
            "Reflections on last few rounds:\n\t",
            "Goals of prior round with current progress scores:\n\t",
            "Goals of two rounds prior with progress scores from prior round:\n\t",
            "Goals of three rounds prior with progress scores from two rounds prior:\n\t",
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
            Goals.gpt_handler.model_context_limit,
            Goals.gpt_handler.max_output_tokens,
            consumed_tokens,
            always_included_count,
        )

        # Set limits for reflections and goals based on available tokens
        reflections_limit, goals_limit = int(available_tokens * 0.6), int(
            available_tokens * 0.3
        ) # TODO: keep 0.3 + 0.6 < 1?

        # Retrieve goals and scores for the previous three rounds
        round = game.round
        goals_prev_1 = None
        goals_prev_2 = None
        goals_prev_3 = None

        # Get the previous round's goal and score if it exists
        if round > 0:
            goals_prev_1 = self.get_goals(round=round - 1, as_str=False)
            scores_prev_1 = self.get_goal_scores(round=round - 1, as_str=False)
            goals_prev_1 = Goals.format_goals_with_scores(goals_prev_1, scores_prev_1)

        # Get the goal and score from two rounds prior if it exists
        if round > 1:
            goals_prev_2 = self.get_goals(round=round - 2, as_str=False)
            scores_prev_2 = self.get_goal_scores(round=round - 2, as_str=False)
            goals_prev_2 = Goals.format_goals_with_scores(goals_prev_2, scores_prev_2)

        # Get the goal and score from three rounds prior if it exists
        if round > 2:
            goals_prev_3 = self.get_goals(round=round - 3, as_str=False)
            scores_prev_3 = self.get_goal_scores(round=round - 3, as_str=False)
            goals_prev_3 = Goals.format_goals_with_scores(goals_prev_3, scores_prev_3)

        # Retrieve reflection nodes for three rounds prior
        reflections_raw = []

        node_ids = self.character.memory.get_observations_after_round(
            round - 3, inclusive=True
        )

        # Collect reflection descriptions from the memory
        for node_id in node_ids:
            node = self.character.memory.get_observation(node_id)
            if node.node_type.value == 3:  # Check if the node type is a reflection
                reflections_raw.append(node.node_description)

        # Initialize the user prompt with the always included context
        user_prompt = always_included[
            0
        ]  # "Additional context for creating your goal:\n\n"

        # If there are reflections from two rounds prior, add them to the user prompt
        if reflections_raw:
            user_prompt += always_included[1]  # "Reflections on last few rounds:\n\t"
            context_list = limit_context_length(
                history=reflections_raw,
                max_tokens=reflections_limit,
                tokenizer=game.parser.tokenizer,
                keep_most_recent=True
            )
            reflection_str = context_list_to_string(context_list, sep="\n\t")
            user_prompt += f"{reflection_str}\n\n"

        # If there is a goal from the previous round, add it to the user prompt
        if goals_prev_1:
            context_list = [
                always_included[2],  # "Goals of prior round with current progress scores:\n\t"
                goals_prev_1,
            ]
            context_list = limit_context_length(
                history=context_list,
                max_tokens=goals_limit // 3,
                tokenizer=game.parser.tokenizer,
                keep_most_recent=True
            )
            goal_prev_str = context_list_to_string(context_list, sep="\n\t")
            user_prompt += f"{goal_prev_str}\n\n"

        # If there is a goal from two rounds prior, add it to the user prompt
        if goals_prev_2:
            context_list = [
                always_included[3],  # "Goals of two rounds prior with previous round's progress scores:\n\t"
                goals_prev_2,
            ]
            context_list = limit_context_length(
                history=context_list,
                max_tokens=goals_limit // 3,
                tokenizer=game.parser.tokenizer,
                keep_most_recent=True
            )
            goal_prev_2_str = context_list_to_string(context_list, sep="\n\t")
            user_prompt += f"{goal_prev_2_str}\n\n"

        # If there is a goal from three rounds prior, add it to the user prompt
        if goals_prev_3:
            context_list = [
                always_included[4],  # "Goals of three rounds prior with progress scores from two rounds prior:\n\t"
                goals_prev_3,
            ]
            context_list = limit_context_length(
                history=context_list,
                max_tokens=goals_limit // 3,
                tokenizer=game.parser.tokenizer,
                keep_most_recent=True
            )
            goal_prev_3_str = context_list_to_string(context_list, sep="\n\t")
            user_prompt += f"{goal_prev_3_str}\n\n"

        # Append the final always included context to the user prompt
        user_prompt += always_included[
            6
        ]  # "You can keep the previous goal, update the previous goal or create a new one based on your strategy."

        # Return the constructed user prompt
        return user_prompt

    def goal_update(self, goal: str, goal_embeddings: np.ndarray, game: "Game"):
        """
        Updates the goals for the current round based on the provided goal string and its embeddings.
        This method categorizes the goals into priority levels and stores them along with their embeddings for future
        reference.

        Args:
            goal (str): A string containing the goals, categorized by priority.
            goal_embeddings (np.ndarray): The embeddings representation of the goal for further processing.
            game (Game): The current game instance used to access the round information.

        Returns:
            None
        """

        # Initialize a dictionary for goals in the current round
        self.goals[game.round] = {}

        # Split the goal string into lines and process each line
        for line in goal.split("\n"):
            # Check for 'Low Priority' in the line and store the corresponding goal
            if "Low Priority" in line:
                self.goals[game.round]["Low Priority"] = line.replace(
                    "Low Priority: ", ""
                )
            # Check for 'Medium Priority' in the line and store the corresponding goal
            elif "Medium Priority" in line:
                self.goals[game.round]["Medium Priority"] = line.replace(
                    "Medium Priority: ", ""
                )
            # Check for 'High Priority' in the line and store the corresponding goal
            elif "High Priority" in line:
                self.goals[game.round]["High Priority"] = line.replace(
                    "High Priority: ", ""
                )

        # Update the goal embeddings for the current round
        self.goal_embeddings.update({game.round: goal_embeddings})

        # Persist the updated goals in memory for the current round
        self.update_goals_in_memory(game.round)

    def get_goals(self, round=-1, priority="all", as_str=False):
        """
        Retrieves goals based on the specified round and priority.

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
        """
        Converts a goal or a collection of goals into a string representation.

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

    def _create_goal_embedding(self, goal: str, game: "Game") -> np.ndarray:
        """
        Generates an embedding for a specified goal.

        This method takes a goal represented as a string and converts it into a numerical embedding using a text
        embedding function. The resulting embedding can be used for various applications, such as similarity comparisons
        or machine learning tasks.

        Args:
            goal (str): The goal to be converted into an embedding.

        Returns:
            np.ndarray: The numerical embedding representation of the specified goal.
        """

        # Initialize a dictionary for goals in the current round
        goals_dict = {}

        # Split the goal string into lines and process each line
        for line in goal.split("\n"):
            # Check for 'Low Priority' in the line and store the corresponding goal
            if "Low Priority" in line:
                goals_dict["Low Priority"] = line.replace("Low Priority: ", "")
            # Check for 'Medium Priority' in the line and store the corresponding goal
            elif "Medium Priority" in line:
                goals_dict["Medium Priority"] = line.replace("Medium Priority: ", "")
            # Check for 'High Priority' in the line and store the corresponding goal
            elif "High Priority" in line:
                goals_dict["High Priority"] = line.replace("High Priority: ", "")
            else:
                goals_dict["Unknown Priority"] = line
                Goals.logger.warning(
                    f"Unknown priority level: {line}",
                    extra=get_logger_extras(game, self.character, include_gpt_call_id=True),
                )

        return Goals.client.generate_embeddings(list(goals_dict.values()))

    def get_goal_embeddings(self, round: int):
        """
        Retrieves the goal embeddings for a specified round.

        This method looks up the goal embeddings associated with the given round number.
        If no embedding exists for that round, it returns None, allowing for easy handling of missing data.

        Args:
            round (int): The round number for which to retrieve the goal embeddings.

        Returns:
            np.ndarray or None: The goal embeddings for the specified round, or None if no embeddings exists.
        """

        return self.goal_embeddings.get(round, None)

    def update_goals_in_memory(self, round):
        """
        Updates the character's memory with the current goal embeddings for a specified round.

        This method retrieves the goal embeddings associated with the given round and, if they exist,
        updates the character's memory with these embeddings. This ensures that the character's memory reflects the
        current goals.

        Args:
            round (int): The round number for which to update the goals in memory.

        Returns:
            None
        """

        # Retrieve the current goal embeddings for the specified round
        curr_embeddings = self.get_goal_embeddings(round)

        # Check if the retrieved embeddings is not None
        if curr_embeddings is not None:
            # Update the character's memory with the current goal embeddings
            self.character.memory.set_goal_query(curr_embeddings)

    # ----------- EVALUATION -----------
    def evaluate_goals(self, game: "Game", impartial: bool = True):
        """
        Evaluates the goals of the agent within the context of the game.

        This method constructs prompts for a language model to assess the agent's goals based on the current game state.
        It retrieves scores from the model and updates the agent's score accordingly, providing feedback on goal
        performance. The evaluation can be done impartially or from the character's perspective.

        Args:
            game (Game): The current game instance used to evaluate the agent's goals.
            impartial (bool, optional): If True, evaluates goals impartially. If False, evaluates from the character's
            perspective. Defaults to True.

        Returns:
            list: The scores generated by the evaluation of the agent's goals.
        """

        if impartial:
            system_prompt = gp.impartial_evaluate_goals_prompt
        else:
            # Retrieve the standard information about the character from the game, excluding goals and perceptions.
            system_prompt = self.character.get_standard_info(
                game, include_goals=False, include_perceptions=False
            )

            # Retrieve the system prompt for evaluating goals from the gp module
            system_prompt += "\n" + gp.persona_evaluate_goals_prompt

        # Count the tokens in the system prompt for token management
        system_prompt_tokens = get_prompt_token_count(system_prompt, role="system")

        # Build the user prompt based on the game state and consumed tokens
        user_prompt = self.build_eval_user_prompt(
            game, consumed_tokens=system_prompt_tokens
        )

        # Generate scores by passing the system and user prompts to the GPT handler
        scores = Goals.gpt_handler.generate(
            system=system_prompt, user=user_prompt, character=self.character
        )
        # Update the agent's score based on the generated scores and the current game
        self.score_update(scores, game)

        # Return the scores obtained from the evaluation
        return scores

    def build_eval_user_prompt(self, game, consumed_tokens=0):
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

        # Retrieve the current goals for the ongoing round and format it for the prompt
        goals = self.get_goals(round=game.round, as_str=True)
        goal_prompt = f"Goals:\n{goals}\n\n"

        # Define a list of strings that will always be included in the user prompt
        always_included = [
            "Score the progress toward the goal that is suggested by the memories provided below:\n",
            goal_prompt,
        ]

        # Calculate the token count for the always included strings
        always_included_tokens = get_prompt_token_count(
            always_included, role="user", pad_reply=True
        )

        # Determine the number of available tokens for reflections and actions
        available_tokens = get_token_remainder(
            Goals.gpt_handler.model_context_limit,
            consumed_tokens,
            always_included_tokens,
        )

        # Initialize lists to store reflections and actions for the current round
        reflections_raw = []
        actions_raw = []
        dialogue_raw = []
        round = game.round

        # Retrieve observation node IDs for the current round
        node_ids = self.character.memory.get_observations_by_round(round)

        # Collect reflections and actions made by this agent in the current round
        for node_id in node_ids:
            node = self.character.memory.get_observation(node_id)
            # Check if the node is an action made by this agent
            if node.node_type.value == 1 and node.node_is_self == 1:
                actions_raw.append(node.node_description)
            # Check if the node is a dialogue made by this agent
            elif node.node_type.value == 2 and node.node_is_self == 1:
                dialogue_raw.append(node.node_description)
            # Check if the node is a reflection made by this agent
            elif node.node_type.value == 3 and node.node_is_self == 1:
                reflections_raw.append(node.node_description)

        # Set limits for actions, dialogue, and reflections based on available tokens
        actions_limit, dialogue_limit, reflections_limit = (
            int(available_tokens * 0.4),
            int(available_tokens * 0.3),
            int(available_tokens * 0.3),
        )

        # Limit the length of the actions based on available tokens
        actions_list = limit_context_length(
            history=actions_raw,
            max_tokens=actions_limit,
            tokenizer=game.parser.tokenizer,
        )

        # Limit the length of the actions based on available tokens
        dialogue_list = limit_context_length(
            history=dialogue_raw,
            max_tokens=dialogue_limit,
            tokenizer=game.parser.tokenizer,
        )

        # Limit the length of the reflections based on available tokens
        reflections_list = limit_context_length(
            history=reflections_raw,
            max_tokens=reflections_limit,
            tokenizer=game.parser.tokenizer,
        )

        # Convert the limited actions, dialogue, and reflections actions lists to formatted strings
        actions_str = context_list_to_string(actions_list, sep="\n")
        dialogue_str = context_list_to_string(dialogue_list, sep="\n")
        reflections_str = context_list_to_string(reflections_list, sep="\n")

        user_prompt = always_included[0]
        user_prompt += goal_prompt
        user_prompt += f"Actions:\n{actions_str}\n\n"
        user_prompt += f"Dialogues:\n{dialogue_str}\n\n"
        user_prompt += f"Reflections:\n{reflections_str}"
        return user_prompt

    def score_update(self, score: str, game: "Game"):
        """
        Updates the dictionary of goal completion scores for the character by round.

        This method processes a score string that contains scores for different priority levels 
        and updates the corresponding entries in the goal_scores dictionary for the current round.

        Args:
            score (str): A string containing the scores for different priority levels, formatted as 
                         'Low Priority: <score>', 'Medium Priority: <score>', and 'High Priority: <score>'.
            game (Game): The current game instance, used to retrieve the current round and for logging errors.

        Returns:
            None
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
                    Goals.logger.error(
                        "Unable to convert 'Low Priority' to an integer.",
                        extra=get_logger_extras(game, self.character, include_gpt_call_id=True),
                    )

            # Check for 'Medium Priority' in the line and attempt to convert the score to an integer
            elif "Medium Priority" in line:
                try:
                    self.goal_scores[round]["Medium Priority"] = int(
                        line.replace("Medium Priority: ", "")
                    )
                except ValueError:
                    # Handle the case where conversion to integer fails
                    Goals.logger.error(
                        "Unable to convert 'Medium Priority' to an integer.",
                        extra=get_logger_extras(game, self.character, include_gpt_call_id=True),
                    )

            # Check for 'High Priority' in the line and attempt to convert the score to an integer
            elif "High Priority" in line:
                try:
                    self.goal_scores[round]["High Priority"] = int(
                        line.replace("High Priority: ", "")
                    )
                except ValueError:
                    # Handle the case where conversion to integer fails
                    Goals.logger.error(
                        "Unable to convert 'High Priority' to an integer.",
                        extra=get_logger_extras(game, self.character, include_gpt_call_id=True),
                    )

    def get_goal_scores(self, round=-1, priority="all", as_str=False):
        """
        Retrieves goal completion scores based on the specified round and priority.

        Args:
            round (int, optional): The round number for which to retrieve scores. Defaults to -1, which retrieves scores
            for all rounds.
            priority (str, optional): The priority level of the scores to retrieve. Defaults to "all", which retrieves
            scores of all priorities.
            as_str (bool, optional): If True, returns the scores as a string. Defaults to False.

        Returns:
            Union[dict, str]: The goal score for the specified round and priority, or a string representation of the
            scores if as_str is True.
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

    @classmethod
    def format_goals_with_scores(cls, goals, scores):

        list_goals = list(goals.items())
        list_scores = scores.values()
        combined = zip(list_goals, list_scores)
        formatted_goals = [f"{item[0].capitalize()}: {item[1]} (progress score: {score})" for ((item, score)) in combined]
        return "\n\t".join(formatted_goals)
