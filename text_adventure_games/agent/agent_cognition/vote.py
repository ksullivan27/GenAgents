"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent_cognition/vote.py
Description: defines how agents vote for one another. This is specific to Survivor or other similar competitive games.
"""

print("Imported Vote")

from collections import (
    Counter,
    defaultdict,
)  # Importing Counter and defaultdict for counting and default dictionary functionality
import json  # Importing json for handling JSON data
import logging  # Importing logging for logging messages
from random import choice  # Importing choice for random selection from a list
from typing import (
    List,
    TYPE_CHECKING,
    Union,
)  # Importing type hints for better code clarity
import openai  # Importing OpenAI library for interacting with GPT models

# Local imports for specific functionalities within the project
# Importing Retrieve class for data retrieval functions
print(f"\t{__name__} calling imports for Retrieve")
from .retrieve import Retrieve

# Importing vote_prompt for voting-related prompts
from text_adventure_games.assets.prompts import vote_prompt as vp

# Importing utility for logging extras
print(f"\t{__name__} calling imports for General")
from text_adventure_games.utils.general import get_logger_extras

print(f"\t{__name__} calling imports for Consts")
from text_adventure_games.utils.consts import get_models_config

print(f"\t{__name__} calling imports for GptHelpers")
from text_adventure_games.gpt.gpt_helpers import (  # Importing helper functions for GPT interactions
    limit_context_length,  # Function to limit the context length for GPT
    get_prompt_token_count,  # Function to count tokens in a prompt
    get_token_remainder,  # Function to get remaining tokens
    context_list_to_string,  # Function to convert context list to string
)

print(f"\t{__name__} calling imports for MemoryType")
from text_adventure_games.agent.memory_stream import (
    MemoryType,
)  # Importing MemoryType for memory management

print(f"\t{__name__} calling Type Checking imports for GptCallHandler")
from text_adventure_games.gpt.gpt_helpers import GptCallHandler

if TYPE_CHECKING:  # Conditional import for type checking
    print(f"\t{__name__} calling Type Checking imports for Character")
    from text_adventure_games.things import (
        Character,
    )  # Importing Character class for type hints

    print(f"\t{__name__} calling Type Checking imports for Game")
    from text_adventure_games.games import Game  # Importing Game class for type hints

VOTING_MAX_OUTPUT = 100  # Constant defining the maximum output for voting


class VotingSession:
    """
    Manages the voting process within a game session, allowing participants to cast votes and tally results.
    It handles the setup of participants, the voting mechanism, and the recording of votes.

    Args:
        game (Game): The game instance in which the voting session takes place.
        participants (List[Character]): A list of characters participating in the voting session.

    Attributes:
        game (Game): The game instance.
        participants (List[Character]): The list of participants in the voting session.
        tally (Counter): A counter to tally the votes received by each participant.
        voter_record (defaultdict): A record of the votes cast by each participant.

    Class-level Attributes:
        gpt_handler (GptCallHandler): Class-level attribute for making calls to the GPT model.
    """

    gpt_handler = None  # Class-level attribute to store the shared GPT handler
    # Define the parameters for configuring the GPT model
    model_params = {
        "max_output_tokens": 250,  # Maximum number of tokens to generate in a response
        "temperature": 1,  # Controls the randomness of the output (higher is more random)
        "top_p": 1,  # Nucleus sampling parameter for controlling diversity
        "max_retries": 5,  # Maximum number of retries for API calls in case of failure
    }

    logger = None  # Class-level attribute to store the shared logger

    @classmethod
    def initialize_gpt_handler(cls):
        """
        Initialize the shared GptCallHandler if it hasn't been created yet.
        """

        print(f"-\tVoting Session Module is initializing GptCallHandler")

        # Initialize the GPT handler if it hasn't been set up yet
        if cls.gpt_handler is None:
            cls.gpt_handler = GptCallHandler(
                model_config_type="vote", **cls.model_params
            )

    def __init__(self, game: "Game", participants: List["Character"]):
        """
        Initializes a VotingSession instance with the specified game and participants.
        This constructor sets up the game context, participant list, and initializes attributes for vote tallying and
        GPT interactions.

        Args:
            game (Game): The game instance in which the voting session takes place.
            participants (List[Character]): A list of characters participating in the voting session.
        """

        # Assigning the game instance to the session for context and management
        self.game = game
        # Setting up the participants for the voting session
        self.participants = self._set_participants(participants)
        # Initializing a counter to keep track of votes received by each participant
        self.tally = Counter()
        # Creating a default dictionary to record the votes cast by each participant
        self.voter_record = defaultdict(str)

        # Initialize the GPT handler if it hasn't been set up yet
        VotingSession.initialize_gpt_handler()

        # Initial token offset for managing token limits during GPT calls
        self.token_offset = 10
        # Padding for additional token management during GPT interactions
        self.offset_pad = 5

        if VotingSession.logger is None:
            VotingSession.logger = logging.getLogger("agent_cognition")

    def _set_participants(self, participants):
        """
        Sets the participants for the voting session and identifies any that are immune to voting.
        This method checks each participant for immunity and updates the session's memory accordingly.

        Args:
            participants (List[Character]): A list of characters participating in the voting session.

        Returns:
            List[Character]: A unique list of participants after filtering out duplicates.
        """

        # Creating a list of participants who have immunity
        immune = [p for p in participants if p.get_property("immune")]

        # Checking if there are any immune participants
        if immune:
            # Printing the name of the first immune participant
            print(f"{immune[0].name} is safe from the vote")
            # Adding the immunity information to memory for the session
            self._add_idol_possession_to_memory(immune, participants)

        # Storing the list of immune participants in the session's attribute
        self.immune = immune
        # Returning a unique list of participants, removing any duplicates
        return list(set(participants))

    def _add_idol_possession_to_memory(self, immune_players, participants):
        """
        Add idol possession information to the memory of participants.

        This function updates the memory of each participant in the game with details about immune players. It formats a
        description of the immune players and extracts relevant keywords to enhance the memory entry.

        Args:
            immune_players (list): A list of players who are immune.
            participants (list): A list of participants whose memory will be updated.

        Returns:
            None
        """

        # Format a description of immune players using a predefined prompt
        immune_desc = vp.immunity_memory_prompt.format(
            immune_players=", ".join([ip.name for ip in immune_players])
        )

        # Extract keywords from the immune description for memory tagging
        immunity_kwds = self.game.parser.extract_keywords(immune_desc)

        # Iterate over each participant to update their memory
        for p in participants:
            # Add a memory entry for the participant with relevant details
            p.memory.add_memory(
                self.game.round,  # Current game round
                self.game.tick,  # Current game tick
                immune_desc,  # Description of immune players
                keywords=immunity_kwds,  # Extracted keywords for the memory
                location=None,  # No specific location associated
                success_status=True,  # Indicate that the memory addition was successful
                memory_importance=10,  # Set the importance level of the memory
                memory_type=MemoryType.ACTION.value,  # Type of memory being added
                actor_id=p.id,  # ID of the participant adding the memory
            )

    def get_vote_options(self, current_voter: "Character", names_only=False):
        """
        Retrieve the voting options available to the current voter.

        This function filters the list of participants to exclude the current voter and any immune players. Depending on
        the `names_only` flag, it can return either the names of the eligible participants or the participant objects
        themselves.

        Args:
            current_voter (Character): The character who is currently voting.
            names_only (bool): If True, return only the names of the participants; otherwise, return the participant
            objects.

        Returns:
            list: A list of eligible voting options, either as names or participant objects based on the `names_only`
            flag.
        """

        # Define a predicate function to filter participants based on voting eligibility
        predicate = lambda p: (p != current_voter) and (p not in self.immune)

        # Check if only names of participants should be returned
        if names_only:
            # Return a list of names of participants who are eligible to vote
            return [p.name for p in self.participants if predicate]
        else:
            # Return a list of participant objects who are eligible to vote
            return [p for p in self.participants if predicate]

    def run(self):
        """
        Execute the voting process for all participants.

        This function iterates through each participant in the voting process, gathering necessary context and allowing
        each voter to cast their vote. It ensures that all participants have the opportunity to contribute to the voting
        outcome.

        Returns:
            None
        """

        # Iterate through each participant in the voting process
        for voter in self.participants:
            # Uncomment the line below to print the name of the current voter
            # print(f"Voter: {voter.name}")

            # Gather necessary context for the current voter (implementation details omitted)
            # 1. Gather context for voter
            # 2. Allow the voter to cast their vote
            self._run_character_vote(voter)

    def _run_character_vote(self, voter):
        """
        Facilitate the voting process for a given voter.

        This function gathers the necessary context for the voter and attempts to cast a vote using the GPT model. It
        retries the voting process up to five times if the initial attempts fail, and resorts to a random vote if all
        attempts are unsuccessful.

        Args:
            voter: The character who is casting the vote.

        Returns:
            None
        """

        # Gather the context needed for the voter, including system and user prompts
        system_prompt, user_prompt = self._gather_voter_context(voter)

        # Initialize success flag for tracking the voting process
        success = False

        # Attempt to cast a vote up to five times
        for i in range(5):
            # Cast a vote using the gathered prompts and the voter
            vote = self.gpt_cast_vote(system_prompt, user_prompt, voter)

            # Validate the cast vote and retrieve its details
            vote_name, vote_confessional, success = self._validate_vote(vote, voter)

            # Check if the vote was successful
            if success:
                # Record the successful vote
                self._record_vote(voter, vote_name, vote_confessional)
                break
            elif i == 4:
                # If the vote has failed too many times, make a random choice
                print(
                    f"{voter.name} has failed to vote properly. Making a random choice."
                )

                # Retrieve valid voting options for the voter
                valid_options = self.get_vote_options(voter, names_only=True)

                # Select a random vote from the valid options
                random_vote = choice(valid_options)

                # Record the randomized vote with a note
                self._record_vote(
                    voter,
                    random_vote,
                    "This was a randomized vote because GPT failed to vote properly.",
                )
                break
            else:
                VotingSession.logger.error(
                    f"Voter {voter.name} failed to vote properly.",
                    extra=get_logger_extras(
                        self.game, self.voter, include_gpt_call_id=True
                    ),
                )

        # Clean up any idols used during this round of voting
        self._cleanup()

    def _record_vote(self, voter, vote_name, vote_confessional):
        """
        Record the vote cast by a voter and update relevant records.

        This function increments the tally for the specified vote name, adds the vote to the voter's memory, and updates
        the voter's record with their chosen vote. Additionally, it logs any confessional remarks made by the voter
        during the voting process.

        Args:
            voter: The character who is casting the vote.
            vote_name (str): The name of the vote being recorded.
            vote_confessional (str): Any confessional remarks made by the voter.

        Returns:
            None
        """

        # Increment the tally for the specified vote name by 1
        self.tally[vote_name] += 1

        # Add the vote to the voter's memory for future reference
        self._add_vote_to_memory(voter, vote_name)

        # Update the voter's record with their chosen vote name
        self.voter_record[voter.name] = vote_name

        # Log any confessional remarks made by the voter during the voting process
        self._log_confessional(voter, vote_confessional)

    def _validate_vote(self, vote_text, voter):
        """
        Validate the vote cast by a voter.

        This function checks the format and content of the provided vote text to ensure it is valid. It verifies that
        the vote target exists and is not the voter themselves, returning the target and reason if valid, or indicating
        failure otherwise.

        Args:
            vote_text (str): The text representation of the vote to be validated.
            voter: The character who is casting the vote.

        Returns:
            tuple: A tuple containing the vote target, the reason for the vote, and a boolean indicating the validity of
            the vote.
                Returns (None, None, False) if the vote is invalid.
        """

        # Attempt to parse the vote text into a dictionary
        try:
            vote_dict = json.loads(vote_text)  # Convert JSON string to dictionary
            vote_target = vote_dict["target"]  # Extract the target of the vote
            vote_reason = vote_dict["reason"]  # Extract the reason for the vote
        except (json.JSONDecodeError, KeyError, TypeError):
            # Return None values and False if there is an error in parsing or missing keys
            return None, None, False
        else:
            # Check if the vote target is valid (exists in characters or jury)
            if vote_target not in list(
                self.game.characters.keys()
            ) and vote_target not in list(self.game.jury.keys()):
                return None, None, False  # Invalid target, return None values and False
            elif vote_target == voter.name:
                return (
                    None,
                    None,
                    False,
                )  # Voter cannot vote for themselves, return None values and False
            else:
                # Retrieve valid voting options for the voter
                valid_names = self.get_vote_options(voter, names_only=True)
                if vote_target not in valid_names:
                    return (
                        None,
                        None,
                        False,
                    )  # Target not in valid options, return None values and False
                else:
                    # Return the valid vote target, reason, and indicate success
                    return vote_target, vote_reason, True

    def _add_vote_to_memory(self, voter: "Character", vote_target: str) -> None:
        """
        Record the voter's action in their memory.

        This function creates a description of the voter's action during the voting process and adds it to their memory.
        It also extracts relevant keywords from the description to enhance the memory entry.

        Args:
            voter (Character): The character who is casting the vote.
            vote_target (str): The target of the vote.

        Returns:
            None
        """

        # Create a description of the voter's action during the voting process
        vote_desc = f"During the end of round session, {voter.name} voted for {vote_target} in secret."

        # Print the vote description for logging or debugging purposes
        print(vote_desc)

        # Extract keywords from the vote description for memory tagging
        vote_kwds = self.game.parser.extract_keywords(vote_desc)

        # Add the vote description to the voter's memory with relevant details
        voter.memory.add_memory(
            self.game.round,  # Current game round
            self.game.tick,  # Current game tick
            vote_desc,  # Description of the vote
            keywords=vote_kwds,  # Extracted keywords for the memory
            location=None,  # No specific location associated
            success_status=True,  # Indicate that the memory addition was successful
            memory_importance=10,  # Set the importance level of the memory
            memory_type=MemoryType.ACTION.value,  # Type of memory being added
            actor_id=voter.id,  # ID of the voter adding the memory
        )

    def read_votes(self):
        """
        Determine the participant to be exiled based on the voting tally.

        This function analyzes the voting results to identify the participant with the highest vote count. If there is a
        tie, it randomly selects one of the participants with the highest votes to be exiled.

        Returns:
            Participant: The participant who is exiled based on the voting results, or None if no participant is found.
        """

        # Initialize top_count to None for tracking the highest vote count
        top_count = None

        # Attempt to retrieve the highest vote count from the tally
        try:
            top_count = self.tally.most_common(1)[0][
                1
            ]  # Get the count of the most common vote
        except (KeyError, IndexError):
            # If there is an error, fallback to getting the key of the most common vote
            exiled_key, _ = self.tally.most_common(1)[0]

        # If a top count was successfully retrieved
        if top_count:
            # Create a list of choices that have the highest vote count
            choices = [(c, v) for c, v in self.tally.items() if v == top_count]
            # Randomly select one of the choices to determine the exiled participant
            exiled_key = choice(choices)[0]

        # Find the participant corresponding to the exiled key
        exiled_participant = next(
            (p for p in self.participants if p.name == exiled_key), None
        )

        # Store the name of the exiled participant in the exiled list
        self.exiled = [exiled_participant.name]

        # Return the exiled participant object
        return exiled_participant

    def record_vote(self, voter):
        """
        Record and retrieve the voting details for a specific voter.

        This function compiles information about the voter's received votes, their voting target, and whether they are
        considered safe from exile. It returns this information in a structured format.

        Args:
            voter: The character who cast the vote.

        Returns:
            list: A list of dictionaries containing the voter's received votes, their voting target, and their safety
            status.
        """

        return [
            {"votes_received": self.tally.get(voter.name, 0)},
            {"target": self.voter_record.get(voter.name, "None")},
            {"is_safe": voter.name not in self.exiled},
        ]

    def _gather_voter_context(self, voter: "Character"):
        """
        Collect and prepare the context needed for a voter before casting a vote.

        This function retrieves the voter's standard information, valid voting options, and their impressions of other
        participants. It constructs prompts for both the system and user, incorporating relevant memories and token
        counts to facilitate the voting process.

        Args:
            voter (Character): The character who is casting the vote.

        Returns:
            tuple: A tuple containing the system prompt and user prompt for the voter.
        """

        # Retrieve the standard information of the voter without including their perceptions
        voter_std_info = voter.get_standard_info(self.game, include_perceptions=False)

        # Get the valid voting options available to the voter
        valid_options = self.get_vote_options(voter)

        # Attempt to retrieve the voter's impressions of the valid options
        try:
            impressions = voter.impressions.get_multiple_impressions(valid_options)
        except AttributeError:
            # If the impressions attribute is not available, set impressions to None
            impressions = None

        # Construct a query to recall relevant actions taken by the voter before casting their vote
        query = "".join(
            [
                f"Before the vote, I need to remember what {' '.join(self.get_vote_options(voter, names_only=True))} ",
                "have done to influence my position in the game.",
            ]
        )

        # Retrieve hyper-relevant memories based on the constructed query
        hyper_relevant_memories = Retrieve.retrieve(
            game=self.game,
            character=voter,
            query=query,
            n=50
        )

        # Build the system prompt using the voter's standard information and a predefined ending
        system = self._build_system_prompt(
            voter_std_info, prompt_ending=vp.vote_system_ending
        )

        # Calculate the token count for the system prompt
        system_token_count = get_prompt_token_count(
            content=system, role="system", tokenizer=self.game.parser.tokenizer
        )

        # Calculate the total tokens consumed by adding the system token count to the token offset
        tokens_consumed = system_token_count + self.token_offset

        # Build the user prompt using the voter's information, impressions, memories, and token count
        user = self._build_user_prompt(
            voter=voter,
            impressions=impressions,
            memories=hyper_relevant_memories,
            prompt_ending=vp.vote_user_prompt,
            consumed_tokens=tokens_consumed,
        )

        # Return both the system and user prompts for further processing
        return system, user

    def _build_system_prompt(self, standard_info, prompt_ending):
        """
        Construct the system prompt for the voting process.

        This function builds a system prompt by combining the provided standard information with a specified ending. It
        ensures that the prompt is formatted correctly for use in the voting context.

        Args:
            standard_info (str): The standard information to include in the prompt.
            prompt_ending (str): The ending text to append to the prompt.

        Returns:
            str: The constructed system prompt.
        """

        # Initialize an empty string for the system prompt
        system_prompt = ""

        # If standard information is provided, append it to the system prompt
        if standard_info:
            system_prompt += standard_info

        # Append the specified prompt ending to the system prompt
        system_prompt += prompt_ending

        # Return the constructed system prompt
        return system_prompt

    def _build_user_prompt(
        self,
        voter,
        impressions: list,
        memories: list,
        prompt_ending: str,
        consumed_tokens: int = 0,
    ):
        """
        Construct the user prompt for the voting process.

        This function generates a user prompt that includes the voter's reflections on other participants and relevant
        memories related to the vote. It formats the prompt based on available tokens and ensures that the content is
        concise and relevant.

        Args:
            voter: The character who is casting the vote.
            impressions (list): A list of the voter's impressions of other participants.
            memories (list): A list of relevant memories related to the voting context.
            prompt_ending (str): The ending text to append to the user prompt.
            consumed_tokens (int, optional): The number of tokens already consumed. Defaults to 0.

        Returns:
            str: The constructed user prompt for the voter.
        """

        # Retrieve the valid voting options for the voter, formatted as names only
        choices = self.get_vote_options(voter, names_only=True)

        # Format the ending of the user prompt with the available vote options
        user_prompt_end = prompt_ending.format(vote_options=choices)

        # Calculate the token count that must always be included in the user prompt
        always_included_count = get_prompt_token_count(
            content=user_prompt_end, role="user", pad_reply=True
        )

        # Determine the number of tokens available for the user prompt
        user_available_tokens = get_token_remainder(
            self.gpt_handler.model_context_limit,
            self.gpt_handler.max_output_tokens,
            consumed_tokens,
            always_included_count,
        )

        # Initially allocate half of the remaining tokens for the impressions section
        context_limit = user_available_tokens // 2

        # Initialize an empty string for the user prompt
        user_prompt = ""

        # If there are impressions available, process them
        if impressions:
            # Limit the context length of impressions to fit within the context limit
            impressions, imp_token_count = limit_context_length(
                impressions, context_limit, return_count=True
            )

            # Add the reflections on other participants to the user prompt
            user_prompt += (
                f"Your REFLECTIONS on other:\n{context_list_to_string(impressions)}\n\n"
            )

            # Update the context limit based on the token count of the impressions
            context_limit = get_token_remainder(user_available_tokens, imp_token_count)

        # If there are memories available, process them
        if memories:
            # Limit the context length of memories to fit within the updated context limit
            memories = limit_context_length(memories, context_limit)

            # Add the relevant memories to the user prompt
            user_prompt += f"SELECT RELEVANT MEMORIES to the vote:\n{context_list_to_string(memories)}\n\n"

        # Append the formatted ending of the user prompt
        user_prompt += user_prompt_end

        # Return the constructed user prompt
        return user_prompt

    def gpt_cast_vote(self, system_prompt, user_prompt, voter):
        """
        Generate a vote using the GPT model based on provided prompts.

        This function constructs a call to the GPT model, passing in the system and user prompts that contain relevant
        context for the voting decision. It handles potential errors related to token limits and adjusts the token
        offset accordingly before reattempting the vote if necessary.

        Args:
            system_prompt (str): The system prompt providing context for the GPT model.
            user_prompt (str): The user prompt containing recent memories and voting options.
            voter: The character who is casting the vote.

        Returns:
            str or None: The generated vote from the GPT model, or None if a reattempt is needed.
        """

        # Generate a vote by calling the GPT model with the provided system and user prompts
        # The system prompt includes context, while the user prompt contains recent memories and valid voting options
        vote = self.gpt_handler.generate(system_prompt, user_prompt, character=voter)

        # Check if the vote returned is a tuple, indicating a potential error
        if isinstance(vote, tuple):
            # This occurs when there was a Bad Request Error due to exceeding the token limit
            success, token_difference = vote

            # Update the token offset to account for the token difference and padding
            self.token_offset = token_difference + self.offset_pad + self.token_offset

            # Double the offset padding for the next calculation
            self.offset_pad += 2 * self.offset_pad

            # Reattempt the character's vote if there was an error
            return self._run_character_vote(voter)

        # Return the generated vote if no errors occurred
        return vote

    def _log_confessional(self, voter: "Character", message: str):
        """
        Log a confessional message from a voter.

        This function records a confessional message along with the target of the voter's action. It enriches the log
        entry with additional context about the voter and the game state.

        Args:
            voter (Character): The character who is providing the confessional message.
            message (str): The confessional message to be logged.

        Returns:
            None
        """

        # Retrieve additional logging context specific to the game and the voter
        extras = get_logger_extras(self.game, voter, include_gpt_call_id=True)

        # Set the type of log entry to "Confessional"
        extras["type"] = "Confessional"

        # Format the message to include the target of the vote and the confessional message
        message = f"Target: {self.voter_record.get(voter.name, 'None')}; {message}"

        # Log the confessional message at the debug level with the additional context
        self.game.logger.debug(msg=message, extra=extras)

    def log_vote(self, exiled: "Character", message: str):
        """
        Log a voting action for an exiled character.

        This function records a message related to the voting action of an exiled character, enriching the log entry
        with additional context about the character and the game state. It ensures that the log entry is categorized as
        a "Vote" type for better organization.

        Args:
            exiled (Character): The character who has been exiled and whose vote is being logged.
            message (str): The message detailing the voting action to be logged.

        Returns:
            None
        """

        # Retrieve additional logging context specific to the game and the exiled character
        extras = get_logger_extras(self.game, exiled)

        # Set the type of log entry to "Vote"
        extras["type"] = "Vote"

        # Log the voting message at the debug level with the additional context
        self.game.logger.debug(msg=message, extra=extras)

    def _cleanup(self):
        """
        Reset the immune status of characters and remove associated items.

        This function checks if there are any characters with immunity and, if so, removes their "idol" items from
        inventory and resets their immune status. It ensures that the game state is properly updated by clearing any
        immunity effects.

        Returns:
            bool: Always returns True after performing the cleanup operations.
        """

        # If there are no characters with immunity, exit the function early
        if not self.immune:
            return True

        # Iterate through each character that is currently immune
        for character in self.immune:
            # Attempt to retrieve the "idol" item from the character's inventory
            if idol := character.get_item_by_name("idol"):
                # Remove the idol from the character's inventory if it exists
                character.remove_from_inventory(idol)

            # Reset the character's immune status to False
            character.set_property("immune", False)

        # Return True to indicate that the cleanup process has completed
        return True


class JuryVotingSession(VotingSession):
    """
    Manage the voting session for a jury of characters.

    This class extends the VotingSession to specifically handle jury members and finalists in a voting scenario. It
    provides methods to gather voter context, determine voting options, and identify the winner based on the jury's
    votes.

    Args:
        game (Game): The game instance in which the voting session takes place.
        jury_members (List[Character]): The characters serving as jury members.
        finalists (List[Character]): The characters who are finalists in the voting process.

    Methods:
        get_vote_options(current_voter: Character, names_only=False):
            Retrieve the voting options available to the current voter, focusing on the finalists.

        _gather_voter_context(voter):
            Collect and prepare the context needed for a jury member before casting their vote.

        determine_winner():
            Identify the winner based on the votes cast by the jury members and return the winner character.
    """

    def __init__(
        self,
        game: "Game",
        jury_members: List["Character"],
        finalists: List["Character"],
    ):
        """
        Initialize a JuryVotingSession with game, jury members, and finalists.

        This constructor sets up the jury voting session by initializing the parent VotingSession with the provided
        jury members and storing the list of finalists. It ensures that the session is ready for managing the voting
        process among the jury.

        Args:
            game (Game): The game instance in which the jury voting session occurs.
            jury_members (List[Character]): The characters serving as jury members in the voting session.
            finalists (List[Character]): The characters who are finalists in the voting process.

        Returns:
            None
        """

        # Call the constructor of the parent VotingSession class, passing the game instance and jury members as
        # participants
        super().__init__(game=game, participants=jury_members)

        # Store the list of finalists for the voting session
        self.finalists = finalists

    def get_vote_options(self, current_voter: "Character", names_only=False):
        """
        Retrieve the voting options available to the current voter.

        This function returns a list of finalists that the current voter can choose from. Depending on the `names_only`
        flag, it can return either the names of the finalists or the full character objects.

        Args:
            current_voter (Character): The character who is currently voting.
            names_only (bool): If True, return only the names of the finalists; otherwise, return the full finalist
            objects.

        Returns:
            list: A list of voting options, either as names or character objects based on the `names_only` flag.
        """

        return [f.name for f in self.finalists] if names_only else self.finalists

    def _gather_voter_context(self, voter):
        """
        Collect and prepare the context needed for a voter before casting their final vote.

        This function retrieves the voter's standard information and their impressions of the finalists. It constructs a
        query to recall relevant actions taken by the voter and gathers hyper-relevant memories, ultimately building
        prompts for both the system and user to facilitate the voting process.

        Args:
            voter: The character who is casting the vote.

        Returns:
            tuple: A tuple containing the system prompt and user prompt for the voter.
        """

        # Retrieve the standard information of the voter, excluding goals and perceptions
        voter_std_info = voter.get_standard_info(
            self.game, include_goals=False, include_perceptions=False
        )

        # Attempt to get the voter's impressions of the finalists
        try:
            impressions = voter.impressions.get_multiple_impressions(self.finalists)
        except AttributeError:
            # If the impressions attribute is not available, set impressions to None
            impressions = None

        # Construct a query to recall relevant actions taken by the voter before the final vote
        query = "".join(
            [
                "Before the final vote, I need to remember what ",
                f"{' '.join(self.get_vote_options(voter, names_only=True))} ",
                "have done over the course of their game, focusing on their strategy, ",
                "critical moves made, and strength as a player.",
            ]
        )

        # Retrieve hyper-relevant memories based on the constructed query
        hyper_relevant_memories = Retrieve.retrieve(
            game=self.game,
            character=voter,
            query=query,
            n=50
        )

        # Build the system prompt using the voter's standard information and a predefined ending
        system = self._build_system_prompt(
            voter_std_info, prompt_ending=vp.jury_system_ending
        )

        # Calculate the token count for the system prompt
        system_token_count = get_prompt_token_count(
            content=system, role="system", tokenizer=self.game.parser.tokenizer
        )

        # Calculate the total tokens consumed by adding the system token count to the token offset
        tokens_consumed = system_token_count + self.token_offset

        # Build the user prompt using the voter's information, impressions, memories, and token count
        user = self._build_user_prompt(
            voter=voter,
            impressions=impressions,
            memories=hyper_relevant_memories,
            prompt_ending=vp.jury_user_prompt,
            consumed_tokens=tokens_consumed,
        )

        # Return both the system and user prompts for further processing
        return system, user

    def determine_winner(self):
        """
        Identify the winner of the voting session among the finalists.

        This function analyzes the voting tally to determine which finalist received the most votes. It also compiles a
        list of finalists who were not selected as the winner and marks them as exiled.

        Returns:
            Character: The finalist who is determined to be the winner of the voting session.
        """

        # Retrieve the key of the finalist with the most votes from the tally
        winner_key, _ = self.tally.most_common(1)[0]

        # Find the finalist object corresponding to the winner's key
        winner = next((f for f in self.finalists if f.name == winner_key), None)

        # Create a list of names for finalists who are not the winner, marking them as exiled
        exiled = [f.name for f in self.finalists if f.name != winner.name]

        # Store the list of exiled finalists in the instance variable
        self.exiled = exiled

        # Return the winner of the voting session
        return winner
