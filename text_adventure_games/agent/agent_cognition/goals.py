# TODO: Read through this again. Make sure the memory_stream is compatible with the changes. Run the game with a few
# characters and a small number of ticks per round to make sure it works.
# TODO: Add the new agent personas. Afterward, update the memory_stream to include the agent persona info.
# TODO: Work on making a basic priority queue using the new memory_stream (includes goals and persona now)

"""
Author: Rut Vyas

File: agent_cognition/goals.py
Description: defines how agents reflect upon their past experiences
"""

# Import future annotations for forward reference type hints.
from __future__ import annotations

circular_import_prints = False

if circular_import_prints:
    print("Importing Goals")

# Import TYPE_CHECKING to allow for type hints without circular imports.
from typing import TYPE_CHECKING, Union, List, Set, Dict

# Import defaultdict from collections for creating default dictionaries.
from collections import defaultdict

# Import numpy for numerical operations.
import numpy as np

if circular_import_prints:
    print(f"\t{__name__} calling imports for Consts")
from text_adventure_games.utils.consts import get_models_config

if circular_import_prints:
    print(f"\t{__name__} calling imports for General")
# Import utility functions for logging and text embedding from the general module.
from text_adventure_games.utils.general import (
    get_logger_extras,
    get_text_embedding,
)

if circular_import_prints:
    print(f"\t{__name__} calling imports for GoalPrompt")
# Import the goal prompt from the prompts module.
from text_adventure_games.assets.prompts import goal_prompt as gp

# Import the logging module to enable logging functionality within this script
import logging

# Importing the inspect module, which provides useful functions to get information about live objects
import inspect

# Import the copy module to allow for deep copying of data structures
import copy

if circular_import_prints:
    print(f"\t{__name__} calling imports for Prompt Classes")
from ...assets.prompts import prompt_classes

if circular_import_prints:
    print(f"\t{__name__} calling imports for GptHelpers")
# Import helper functions for GPT calls and context management from the gpt_helpers module.
from text_adventure_games.gpt.gpt_helpers import (
    limit_context_length,
    get_prompt_token_count,
    get_token_remainder,
    context_list_to_string,
    GptCallHandler,
)

if circular_import_prints:
    print(f"\t{__name__} calling Type Checking import for MemoryStream")
from text_adventure_games.agent.memory_stream import MemoryType

# Conditional import for type checking to avoid circular dependencies.
if TYPE_CHECKING:
    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking import for Character")
    from text_adventure_games.things.characters import Character

    if circular_import_prints:
        print(f"\t{__name__} calling Type Checking import for Game")
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
        goal_embeddings (defaultdict): A dictionary storing embeddings for each goal by round.
        token_offset (int): The offset for managing token limits in GPT calls.
        offset_pad (int): Additional padding for token management.

    Class Attributes:
        max_score (int): The maximum progress score for a goal.
        gpt_handler (GptCallHandler): A class-level shared GPT handler for all instances.
        model_params (dict): Class-level model parameters for configuring the GPT handler.

    Methods:
        _log_goals(game, message):
            Logs the specified message related to goals in the game's logger.

        gpt_generate_goals(game):
            Calls GPT to create new goals for the character based on the current game state.

        build_goal_prompts(game):
            Constructs the system and user prompts necessary for goal generation.

        build_system_prompt(game):
            Constructs the system prompt used for generating goals in the game.

        build_user_prompt(game, consumed_tokens=0):
            Constructs the user prompt for goal generation, incorporating relevant context and previous goals.

        update_goals(game, goals_dict):
            Updates the goals dictionary and the character's memory with their goals for the current round.

        get_goals(round=-1, priority="all", as_str=False):
            Retrieves goals based on the specified round and priority.

        evaluate_goals(game):
            Evaluates the goals of the agent within the context of the game.

        build_eval_user_prompt(game, consumed_tokens=0):
            Constructs a user prompt for evaluating progress towards a goal based on reflections and actions.

    Class Methods:
        reconfigure_goals(game):
            Formats the provided goals optionally with priority levels and progress scores into a string representation
            or a list.
    """

    max_progress_score = 5

    gpt_handler = None  # Class-level attribute to store the shared GPT handler
    model_params = {
        "max_output_tokens": 800,
        "temperature": 1,
        "top_p": 1,
        "max_retries": 5,
    }

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
                model_config_type="goals", **cls.model_params
            )

    def __init__(self, character: "Character"):
        """
        Initializes the Goals manager for a character, setting up the necessary data structures to track goals, scores,
        and embeddings. This constructor also ensures the shared GPT handler is configured for generating goals and
        initializes parameters for managing token limits.

        The goals are stored in the form of a dictionary based on the priority with the round number as the key in the
        following format:
            {Round Number:
                {"Low Priority": Set(node_ids),
                 "Medium Priority": Set(node_ids),
                 "High Priority": Set(node_ids)}

        Args:
            character (Character): The character for whom the goals are being managed.

        Attributes:
            character (Character): The character associated with this goals manager.
            goals (defaultdict): A dictionary storing goals categorized by round and priority.
            goal_scores (dict): A dictionary mapping node ids to goal completion scores.
            goal_embeddings (defaultdict): A dictionary storing embeddings for each goal by round.
            token_offset (int): The offset for managing token limits in GPT calls.
            offset_pad (int): Additional padding for token management.
        """

        if circular_import_prints:
            print(f"-\tInitializing Goals")

        # Initialize the GPT handler if it hasn't been set up yet
        Goals.initialize_gpt_handler()

        # Assign the character associated with this goals manager to an instance variable.
        self.character = character

        # Initialize a defaultdict to store goals categorized by round and priority, where each priority is a set.
        self.goals = defaultdict(lambda: defaultdict(set))

        # Initialize a dict to store goal completion scores stored by node id.
        self.goal_scores = dict()

        # Initialize the logger if it hasn't been set up yet
        if Goals.logger is None:
            Goals.logger = logging.getLogger("agent_cognition")

        # Set the token offset to account for a few variable tokens in the user prompt.
        self.token_offset = (
            50  # Taking into account a few variable tokens in the user prompt
        )

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

    def gpt_generate_goals(self, game: "Game"):
        """
        Calls GPT to create a goal for the character
        System prompt uses: world info and the agent's personal summary. The user prompt uses: impressions created by this
        character of other players, reflection of previous round, goals of prior round with current progress scores,
        goals of two rounds prior with progress scores from prior round, and goals of three rounds prior with progress
        scores from two rounds prior.

        Args:
            game (Game): the game
        """

        # print("GPT GENERATING GOALS (goals) FOR:", self.character.name)

        # Build the system and user prompts required for goal generation.
        system, user = self.build_goal_prompts(game)

        # Generate goals using the GPT handler based on the constructed prompts.
        goals = Goals.gpt_handler.generate(
            system=system,
            user=user,
            character=self.character,
            response_format=prompt_classes.Goals,
            game=game,
        )

        # Check if the generated goal is a tuple, indicating a potential error related to token limits.
        if isinstance(goals, tuple):
            # Unpack the tuple to get the success status and the token difference.
            _, token_difference = goals

            # Update the token offset to account for the exceeded limit and add padding for future calculations.
            self.token_offset = token_difference + self.offset_pad
            self.offset_pad += 2 * self.offset_pad

            # Recursively call the goal generation method to attempt generating a goal again with updated limits.
            self.gpt_generate_goals(game)
            return  # Exit the function after the recursive call

        # Convert the parsed goals to a dictionary
        goals_dict = {}
        goals_dict["Low Priority"] = [goal.description for goal in goals.low_priority]
        goals_dict["Medium Priority"] = [
            goal.description for goal in goals.medium_priority
        ]
        goals_dict["High Priority"] = [goal.description for goal in goals.high_priority]

        # Log the generated goals for tracking and debugging purposes.
        self._log_goals(game, goals)

        # Update the goal with the new embedding for experimentation purposes.
        self.update_goals(game, goals_dict)

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
        system_prompt += "\n\nCURRENT TASK:\n" + gp.gpt_goals_prompt.format(
            max_progress_score=Goals.max_progress_score
        )

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
        goals, and scores from the last three rounds to provide a comprehensive context for the user.

        Args:
            game: The current game instance used to gather information.
            consumed_tokens (int, optional): The number of tokens already consumed by previous prompts. Defaults to 0.

        Returns:
            str: The constructed user prompt that includes context for creating or updating goals.
        """

        # Define a list of strings that are always included in the user prompt
        always_included = [
            "Additional context for creating your goal:",
            "\n\nReflections on last few rounds:",
            "\n\nGoals of prior round with current progress scores:",
            "\n\nGoals of two rounds prior with progress scores from prior round:",
            "\n\nGoals of three rounds prior with progress scores from two rounds prior:",
            "\n\nYou can keep the previous goal, update the previous goal or create a new one based on your strategy.",
            "You have no previous goals for reference. Brainstorm some.",
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
            available_tokens * 0.4
        )

        # Retrieve goals and scores for the previous three rounds
        round = game.round
        goals_prev_1 = None
        goals_prev_2 = None
        goals_prev_3 = None

        # Get the previous round's goal and score if it exists
        if round > 0:
            # print("GETTING GOALS FOR ROUND:", round - 1)
            # Get a dict mapping priority strings to dicts mapping node ids to goal descriptions
            goals_prev_1 = self.get_goals(
                round=round - 1,
                priority="all",
                include_node_ids=False,
                include_description=True,
                include_priority_levels=True,
                include_scores=True,
                progress_as_percentage=False,
                to_str=False,
                list_prefix="\n- ",
            )

        # Get the goal and score from two rounds prior if it exists
        if round > 1:
            # print("GETTING GOALS FOR ROUND:", round - 2)
            # Get a dict mapping priority strings to dicts mapping node ids to goal descriptions
            goals_prev_2 = self.get_goals(
                round=round - 2,
                priority="all",
                include_node_ids=False,
                include_description=True,
                include_priority_levels=True,
                include_scores=True,
                progress_as_percentage=False,
                to_str=False,
                list_prefix="\n- ",
            )

        # Get the goal and score from three rounds prior if it exists
        if round > 2:
            # print("GETTING GOALS FOR ROUND:", round - 3)
            # Get a dict mapping priority strings to dicts mapping node ids to goal descriptions
            goals_prev_3 = self.get_goals(
                round=round - 3,
                priority="all",
                include_node_ids=False,
                include_description=True,
                include_priority_levels=True,
                include_scores=True,
                progress_as_percentage=False,
                to_str=False,
                list_prefix="\n- ",
            )

        # Initialize a list to store previous reflections
        reflections_raw = []

        # Retrieve reflection nodes for three rounds prior
        node_ids = self.character.memory.get_observations_after_round(
            round - 3, inclusive=True
        )

        # Collect reflection descriptions from memory
        for node_id in node_ids:
            node = self.character.memory.get_observation(node_id)
            if node.node_type.value == 3:  # Check if the node type is a reflection
                reflections_raw.append("\n- " + node.node_description)

        # Initialize the user prompt
        user_prompt = ""

        # If there are reflections from two rounds prior, add them to the user prompt
        if reflections_raw:
            reflections_limited = limit_context_length(
                history=reflections_raw,
                max_tokens=reflections_limit,
                tokenizer=game.parser.tokenizer,
            )

            reflections_str = context_list_to_string(reflections_limited, sep="")

            user_prompt += always_included[1]  # "\n\nReflections on last few rounds:"
            user_prompt += reflections_str

        goals_raw = []
        # If there are goals from the previous rounds, add them to the user prompt
        for i, goal_round in enumerate([goals_prev_1, goals_prev_2, goals_prev_3]):
            if goal_round:
                goals_raw.append(always_included[2 + i])
                goals_raw.extend(goal_round)

        goals_limited = limit_context_length(
            history=goals_raw, max_tokens=goals_limit, tokenizer=game.parser.tokenizer
        )
        goals_str = context_list_to_string(goals_limited, sep="")

        user_prompt += goals_str

        # Return the constructed user prompt (if it's not empty, prepend "Additional context for creating your goal:"
        # and append "You can keep the previous goal, update the previous goal or create a new one based on your strategy.")
        return (
            always_included[0] + user_prompt + always_included[5]
            if user_prompt
            else always_included[6]
        )

    def get_goals(
        self,
        round: int = -1,
        priority: str = "all",
        include_node_ids: bool = False,
        include_description: bool = True,
        include_priority_levels: bool = True,
        include_scores: bool = True,
        progress_as_percentage: bool = False,
        to_str: bool = False,
        list_prefix: str = "",
        join_str: str = "\n",
        sep_initial: bool = False,
    ) -> Union[str, List[str], List[int], None]:
        """
        Retrieves goals based on the specified round and priority.

        This method allows for flexible retrieval of goals, either for a specific round and priority or all goals if no
        specific criteria are provided.

        Args:
            round (int, optional): The round number for which to retrieve goals. Defaults to -1, which retrieves all
                                   goals.
            priority (str, optional): The priority level of the goals to retrieve. Defaults to "all", which retrieves
                                      goals of all priorities.
            include_node_ids (bool, optional): If True, returns a dict mapping node_ids to goal descriptions instead of
                                               just descriptions.
            include_description (bool, optional): If True, includes goal descriptions in the returned goals.
            include_priority_levels (bool, optional): If True, includes priority levels in the returned goals.
            include_scores (bool, optional): If True, includes progress scores in the returned goals.
            progress_as_percentage (bool, optional): If True, formats progress scores as percentages.
            to_str (bool, optional): If True, returns the formatted goals as a string; otherwise, returns a list.
            list_prefix (str, optional): The string used to prefix the formatted goals. Defaults to "".
            join_str (str, optional): The string used to join the formatted goals. Defaults to "\n".
            sep_initial (bool, optional): If True, prepends the join_str at the beginning of the formatted string.

        Returns:
            Union[str, List[str], List[int], None]: The goals for the specified round and priority as a string, a list
            of goal descriptions (optionally including priority levels and progress scores), or a list of node ids.
        """
        # print("GETTING GOALS FOR:", self.character.name)
        # print(self.goals.items())

        if round != -1:
            if round not in self.goals:  # Check if the round exists in the dictionary
                return None  # Return None if the round does not exist
            if priority != "all":  # Check if a specific priority is provided
                goals = self.goals[round][
                    priority
                ]  # Retrieve the goal for the specified round and priority
            else:  # If no specific priority is provided
                goals = self.goals[round]  # Retrieve all goals for the specified round

        else:  # If no specific round is provided
            goals = self.goals  # Retrieve all goals

        if goals is None:
            return None

        # Make a deep copy of the goals to avoid modifying the original data structure
        goals_copy = copy.deepcopy(goals)

        # print("GOALS COPY FOR:", self.character.name, type(goals_copy))
        # for goal in goals_copy:
        #     print("-", goal)

        # Reconfigure the goals to replace node_ids with descriptions
        reconfigured_goals = self.reconfigure_goals(goals_copy)

        # print("RECONFIGURED GOALS FOR:", self.character.name, type(reconfigured_goals))
        # for goal in reconfigured_goals:
        #     print("-", goal)

        # print("GOALS PRE-FORMATTING:", reconfigured_goals)

        # print("CHARACTER's GOAL MEMORIES")
        # for obs in self.character.memory.observations:
        #     if obs.node_type.value == 5:
        #         print(
        #             "-",
        #             obs.node_id,
        #             obs.node_round,
        #             obs.node_tick,
        #             obs.node_type,
        #             obs.node_description,
        #         )

        # print("DONE GETTING (returning) GOALS")

        return self.format_goals(
            goals=reconfigured_goals,
            include_node_ids=include_node_ids,
            include_description=include_description,
            include_priority_levels=include_priority_levels,
            include_scores=include_scores,
            progress_as_percentage=progress_as_percentage,
            to_str=to_str,
            list_prefix=list_prefix,
            join_str=join_str,
            sep_initial=sep_initial,
        )

        # goals: dict,
        # include_node_ids: bool = False,
        # include_description: bool = True,
        # include_priority_levels: bool = False,
        # include_scores: bool = False,
        # progress_as_percentage: bool = False,
        # to_str: bool = False,
        # list_prefix: str = "",
        # join_str: str = "\n",
        # prefix_initial: bool = True,

    def update_goals(self, game: "Game", goals_dict: dict):
        """
        Updates the goals dictionary and the character's memory with their goals for the current round.

        This method processes the character's goals categorized by priority level (Low, Medium, High) and stores them
        in the goals dictionary and the character's memory. It also assigns an importance level to each goal based on
        its priority.

        Args:
            game (Game): The current game instance used to access round information and manage game state.
            goals_dict (dict): A dictionary containing goals categorized by priority levels.

        Returns:
            None
        """

        # Initialize a list to store the node IDs of the goals
        node_ids = []

        # Iterate through the goals for the current round
        for priority, goals in goals_dict.items():
            # Set the importance of the goal based on its priority
            if priority == "Low Priority":
                ref_importance = 8
            elif priority == "Medium Priority":
                ref_importance = 9
            elif priority == "High Priority":
                ref_importance = 10
            else:
                # If the priority level is unknown, set the importance to 9 and log an error
                ref_importance = 9
                Goals.logger.error(
                    f"Unknown priority level: {priority}",
                    extra=get_logger_extras(
                        game, self.character, include_gpt_call_id=True
                    ),
                )

            for goal in goals:

                # Summarize and score the action described in the statement, obtaining keywords and importance
                _, _, ref_kwds = game.parser.summarize_and_score_action(
                    description=goal,
                    thing=self.character,
                    needs_summary=False,
                    needs_score=False,
                )

                # Add the summarized memory to the character's memory with relevant details
                node_id = self.character.memory.add_memory(
                    game.round,
                    game.tick,
                    goal,
                    ref_kwds,
                    self.character.location.name,
                    success_status=True,
                    memory_importance=ref_importance,
                    memory_type=MemoryType.GOAL.value,
                    actor_id=self.character.id,
                )

                node_ids.append(node_id)
                self.goals[game.round][priority].add(node_id)

        # Update the character's memory with the current goal embeddings
        self.character.memory.set_goal_query(node_ids)

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
            system_prompt = gp.impartial_evaluate_goals_prompt.format(
                max_progress_score=Goals.max_progress_score
            )
        else:
            # Retrieve the standard information about the character from the game, excluding goals and perceptions.
            system_prompt = self.character.get_standard_info(
                game, include_goals=False, include_perceptions=False
            )

            # Retrieve the system prompt for evaluating goals from the gp module
            system_prompt += "\n" + gp.persona_evaluate_goals_prompt.format(
                max_progress_score=Goals.max_progress_score
            )

        # Count the tokens in the system prompt for token management
        system_prompt_tokens = get_prompt_token_count(system_prompt, role="system")

        # Build the user prompt based on the game state and consumed tokens
        user_prompt = self.build_eval_user_prompt(
            game, consumed_tokens=system_prompt_tokens
        )

        # Generate scores by passing the system and user prompts to the GPT handler
        scores = Goals.gpt_handler.generate(
            system=system_prompt,
            user=user_prompt,
            character=self.character,
            response_format=prompt_classes.Scores,
            game=game,
        )

        # Check if the generated goal is a tuple, indicating a potential error related to token limits.
        if isinstance(scores, tuple):
            # Unpack the tuple to get the success status and the token difference.
            _, token_difference = scores

            # Update the token offset to account for the exceeded limit and add padding for future calculations.
            self.token_offset = token_difference + self.offset_pad
            self.offset_pad += 2 * self.offset_pad

            # Recursively call the goal generation method to attempt generating a goal again with updated limits.
            self.evaluate_goals(game, impartial)
            return  # Exit the function after the recursive call

        # Convert the parsed scores to a dictionary
        scores_dict = {score.node_id: score.progress_score for score in scores.scores}

        # Update the goals dictionary with the new scores
        self.goal_scores.update(scores_dict)

    def build_eval_user_prompt(self, game, consumed_tokens=0):
        """
        Constructs a user prompt for evaluating progress towards goals based on reflections and actions from the
        current round. This prompt includes the current goals, reflections, and actions, formatted for user input.

        Args:
            game (Game): The current game instance containing game state information.
            consumed_tokens (int, optional): The number of tokens already consumed. Defaults to 0.

        Returns:
            str: A formatted string containing the user prompt with the goal, reflections, and actions.
        """

        # print("-" * 100)
        # print("BUILDING EVALUATION USER PROMPT FOR:", self.character.name)

        # print("GETTING GOALS FOR BUILDING EVALUATION USER PROMPT:", game.round)

        # Retrieve the goals for the current round
        goals = self.get_goals(
            round=game.round,
            priority="all",
            include_node_ids=True,
            include_description=True,
            include_priority_levels=False,
            include_scores=False,
            progress_as_percentage=False,
            to_str=True,
            join_str="\n- ",
            sep_initial=True,
        )

        # print("ROUND", game.round)
        # print("GOALS (build_eval_user_prompt)", self.goals)
        # for obs in self.character.memory.observations:
        #     if obs.node_type.value == 5:
        #         print(
        #             "-",
        #             obs.node_id,
        #             obs.node_type,
        #             obs.node_round,
        #             obs.node_tick,
        #             obs.node_description,
        #         )

        # Check if no goals have been set for the current round
        if goals is None:
            raise ValueError("No goals have been set for this round.")

        # Format the goals as a string
        goal_prompt = f"Goals:{goals}"

        # Define a list of strings that will always be included in the user prompt
        always_included = [
            "Score your progress toward each goal based on provided action, dialogue, and reflection memories.\n\n",
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

        # Initialize lists to store reflections, actions, and dialogues for the current round
        reflections_raw = []
        actions_raw = []
        dialogue_raw = []

        # Retrieve observation node IDs for the current round
        node_ids = self.character.memory.get_observations_by_round(game.round)

        # Collect reflections and actions made by this agent in the current round
        for node_id in node_ids:
            node = self.character.memory.get_observation(node_id)
            # Check if the node is an action
            if node.node_type.value == 1:
                actions_raw.append(node.node_description)
            # Check if the node is a dialogue
            elif node.node_type.value == 2:  # and node.node_is_self == 1:
                dialogue_raw.append(node.node_description)
            # Check if the node is a reflection
            elif node.node_type.value == 3:
                reflections_raw.append(node.node_description)

        # Set limits for actions, dialogue, and reflections based on available tokens
        actions_limit, dialogue_limit, reflections_limit = (
            int(available_tokens * 0.4),
            int(available_tokens * 0.3),
            int(available_tokens * 0.3),
        )

        # Limit the length of the actions based on available tokens
        actions_list = (
            limit_context_length(
                history=["\n\nActions:"] + ["\n- " + a for a in actions_raw],
                max_tokens=actions_limit,
                tokenizer=game.parser.tokenizer,
            )
            if actions_raw
            else []
        )

        # Limit the length of the dialogues based on available tokens
        dialogue_list = (
            limit_context_length(
                history=["\n\nDialogues:"] + ["\n- " + d for d in dialogue_raw],
                max_tokens=dialogue_limit,
                tokenizer=game.parser.tokenizer,
            )
            if dialogue_raw
            else []
        )

        # Limit the length of the reflections based on available tokens
        reflections_list = (
            limit_context_length(
                history=["\n\nReflections:"] + ["\n- " + r for r in reflections_raw],
                max_tokens=reflections_limit,
                tokenizer=game.parser.tokenizer,
            )
            if reflections_raw
            else []
        )

        # Convert the limited actions, dialogue, and reflections lists to formatted strings
        actions_str = context_list_to_string(actions_list, sep="")
        dialogue_str = context_list_to_string(dialogue_list, sep="")
        reflections_str = context_list_to_string(reflections_list, sep="")

        user_prompt = (
            always_included[0]
            + goal_prompt
            + actions_str
            + dialogue_str
            + reflections_str
        )

        return user_prompt

    def reconfigure_goals(
        self,
        goals_data: Union[Set[str], Dict[str, Union[Set[str], Dict[str, Set[str]]]]],
    ) -> Union[List[str], Dict[str, Union[List[str], Dict[str, List[str]]]]]:
        """
        Reconfigures the input data by replacing node_ids with a mapping to their corresponding goal descriptions.

        Args:
            goals_data: Can be a set of node_ids, a dict of priority levels to node_ids,
                        or a dict of rounds to priority levels to node_ids.

        Returns:
            Union[List[str], Dict[str, Union[List[str], Dict[str, List[str]]]]]: The reconfigured data with node_ids
                                                                                 replaced by goal descriptions.
        """

        if isinstance(goals_data, set):
            return self._node_ids_to_descriptions(goals_data)
        elif isinstance(goals_data, dict):
            return {
                key: self.reconfigure_goals(value) for key, value in goals_data.items()
            }
        else:
            return goals_data

    def _node_ids_to_descriptions(self, node_ids: Set[str]) -> Dict[str, str]:
        """
        Converts a set of node IDs into a mapping to their corresponding goal descriptions.

        Args:
            node_ids (Set[str]): A set of node IDs to be converted.

        Returns:
            Dict[str, str]: A dictionary mapping node IDs to descriptions.
        """

        return {
            node_id: self.character.memory.get_observation_description(node_id)
            for node_id in node_ids
        }

    def format_goals(
        self,
        goals: dict,
        include_node_ids: bool = False,
        include_description: bool = True,
        include_priority_levels: bool = False,
        include_scores: bool = False,
        progress_as_percentage: bool = False,
        to_str: bool = False,
        list_prefix: str = "",
        join_str: str = "\n",
        sep_initial: bool = False,
    ) -> Union[str, list[str], list[int]]:
        """
        Formats the provided goals optionally with priority levels and progress scores into a string representation or
        a list.

        Args:
            goals (dict): A dictionary where keys are either round numbers mapping to priority levels mapping to node IDs
                          and their corresponding goal descriptions, or priority levels mapping to node IDs and their
                          corresponding goal descriptions, or node IDs mapping to goal descriptions.
            include_node_ids (bool, optional): Indicates whether to include node IDs in the formatted output. Defaults
                                                to False.
            include_priority_levels (bool, optional): Indicates whether to include priority levels in the
                                                      formatted output. Defaults to False.
            include_scores (bool, optional): Indicates whether to include progress scores in the formatted
                                              output. Defaults to False.
            progress_as_percentage (bool, optional): If True, progress scores are formatted as percentages. Defaults to
                                                      False.
            to_str (bool, optional): Indicates whether to return the formatted goals as a string. Defaults to False.
            join_str (str, optional): The string used to join the formatted goals. Defaults to "\n- ".
            sep_initial (bool, optional): Indicates whether to prepend the join_str at the beginning of the
                                          formatted string. Defaults to True.
            include_description (bool, optional): Indicates whether to include goal descriptions in the formatted output.
                                                  Defaults to True.

        Returns:
            Union[str, list[str], list[int]]: If to_str is True, returns a formatted string containing goal descriptions
                                              and their progress scores. If to_str is False, returns a list of formatted
                                              goals. If include_node_ids is True, include_priority_levels is False,
                                              include_scores is False, and to_str is False, returns a list of ints.
        """

        if not (include_node_ids or include_description):
            raise ValueError(
                "At least one of include_node_ids or include_description must be True."
            )

        # Initialize a list to store the formatted goals
        formatted_goals = []

        def format_goal_dict(goal_dict):
            for priority_level, node_desc_dict in goal_dict.items():
                for node_id, node_desc in node_desc_dict.items():
                    if include_description:
                        goal_str = f"{str(node_id) + '. ' if include_node_ids else ''}{node_desc}"
                    else:
                        goal_str = f"{str(node_id) if include_node_ids else ''}"
                    details = []
                    if include_priority_levels:
                        details.append(
                            f"priority: {priority_level.split(' ')[0].lower()}"
                        )
                    if include_scores:
                        score = (
                            f"{round(100 * self.goal_scores.get(node_id, 'Not Scored') / Goals.max_progress_score, 1)}%"
                            if progress_as_percentage
                            else self.goal_scores.get(node_id, "Not Scored")
                        )
                        details.append(f"progress score: {score}")
                    if details:
                        goal_str += f" ({', '.join(details)})"
                    formatted_goals.append(goal_str)

        # Check if the top-level keys are round numbers
        if all(isinstance(key, int) for key in goals.keys()):
            for round_num, priority_dict in goals.items():
                format_goal_dict(priority_dict)
        # Check if the top-level keys are priority levels
        elif all(isinstance(key, str) for key in goals.keys()):
            format_goal_dict(goals)
        # Assume the dictionary is node IDs mapping to descriptions
        else:
            for node_id, node_desc in goals.items():
                if include_description:
                    goal_str = (
                        f"{str(node_id) + '. ' if include_node_ids else ''}{node_desc}"
                    )
                else:
                    goal_str = f"{str(node_id) if include_node_ids else ''}"
                formatted_goals.append(goal_str)

        # Special case: return list of ints
        if (
            include_node_ids
            and not include_priority_levels
            and not include_scores
            and not to_str
            and not include_description
        ):
            return [
                int(node_id)
                for priority_level, node_desc_dict in goals.items()
                for node_id in node_desc_dict.keys()
            ]

        # Format the goals as a string
        if to_str:
            if sep_initial:
                return join_str + join_str.join(formatted_goals)
            else:
                return join_str.join(formatted_goals)
        else:
            return [list_prefix + goal for goal in formatted_goals]
