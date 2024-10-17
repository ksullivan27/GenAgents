import tiktoken  # Importing the tiktoken library for tokenization tasks
from text_adventure_games.gpt.gpt_helpers import (  # Importing helper functions for GPT interactions
    limit_context_length,  # Function to limit the context length for GPT
    get_prompt_token_count,  # Function to count tokens in a prompt
    GptCallHandler,  # Class to handle GPT calls
)
from text_adventure_games.assets.prompts import (
    dialogue_prompt as dp,
)  # Importing the dialogue prompt for use in the game
from ..utils.general import (
    set_up_openai_client,
)  # Importing a utility function to set up the OpenAI client
from ..agent.agent_cognition.retrieve import (
    retrieve,
)  # Importing a function to retrieve information for agent cognition

ACTION_MAX_OUTPUT = 100  # Constant defining the maximum output length for actions


class Dialogue:
    """
    Manages the dialogue between two characters in a text adventure game.
    It facilitates interactions, updates character instructions, and handles GPT responses.

    Args:
        game (Game): The game instance managing the overall game state.
        participants (List[Character]): Sorted list of characters by initiative.
        command (str): The initial command or intent of the dialogue.

    Attributes:
        game (Game): The game instance.
        gpt_handler (GptCallHandler): Handler for GPT interactions.
        token_offset (int): Offset for token management.
        offset_pad (int): Padding for token limits.
        model_context_limit (int): Maximum token limit for the model.
        participants (List[Character]): List of dialogue participants.
        characters_system (dict): System instructions for each character.
        characters_user (dict): User instructions for each character.
        participants_number (int): Number of participants in the dialogue.
        command (str): The command initiating the dialogue.
        dialogue_history (List[str]): History of dialogue exchanges.
        dialogue_history_token_count (int): Token count of the dialogue history.
        characters_mentioned (List[str]): List of characters mentioned in the dialogue.
    """

    def __init__(self, game, participants, command):
        """
        Initializes the Dialogue instance, setting up the game state and participants.
        It prepares the dialogue history and character instructions for the interaction.

        Args:
            game (Game): The game instance managing the overall game state.
            participants (List[Character]): Sorted list of characters by initiative.
            command (str): The initial command or intent of the dialogue.

        Returns:
            None
        """

        # Assign the game instance to the instance variable
        self.game = game

        # Set up the GPT handler for generating responses
        self.gpt_handler = self._set_up_gpt()

        # Initialize token offset for managing token limits
        self.token_offset = 0

        # Set padding for token limits
        self.offset_pad = 5

        # Define the maximum token limit for the model based on the GPT handler
        self.model_context_limit = self.gpt_handler.model_context_limit

        # Store the participants in the dialogue
        self.participants = participants

        # Initialize dictionaries to hold system and user instructions for characters
        self.characters_system = {}
        self.characters_user = {}

        # Count the number of participants in the dialogue
        self.participants_number = len(participants)

        # Store the initial command for the dialogue
        self.command = command

        # Initialize the dialogue history with a starting message
        self.dialogue_history = [
            f"{self.participants[0].name} wants to {self.command}. The dialogue just started."
        ]

        # Calculate the token count for the initial dialogue history
        self.dialogue_history_token_count = get_prompt_token_count(
            content=self.dialogue_history, role=None, pad_reply=False
        )

        # Create a list of all characters mentioned in the conversation
        self.characters_mentioned = [
            character.name for character in self.participants
        ]  # Characters mentioned so far in the dialogue

        # Iterate over each participant to set up their instructions
        for participant in self.participants:
            # Initialize a dictionary for the character's system instructions using their name as the key
            self.characters_system[participant.name] = dict()

            # Initialize a dictionary for the character's user instructions using their name as the key
            self.characters_user[participant.name] = dict()

            # Update the character's system instructions
            self.update_system_instruction(participant)

            # Update the character's user instructions, including impressions and memories
            self.update_user_instruction(
                participant,
                update_impressions=True,  # Indicate that impressions should be updated
                update_memories=True,  # Indicate that memories should be updated
                system_instruction_token_count=self.get_system_instruction(participant)[
                    0  # Get the token count of the system instruction for the participant
                ],
            )

    def _set_up_gpt(self):
        """
        Initializes the GPT handler with the specified model parameters.
        This method configures the settings required for generating responses from the GPT model.

        Returns:
            GptCallHandler: An instance of the GptCallHandler configured with the model parameters.
        """

        # Define the parameters for the GPT model configuration
        model_params = {
            "api_key_org": "Helicone",  # Organization API key for accessing the model
            "model": "gpt-4",  # Specify the model version to use
            "max_tokens": 250,  # Maximum number of tokens to generate in a response
            "temperature": 1,  # Sampling temperature for randomness in responses
            "top_p": 1,  # Nucleus sampling parameter for controlling diversity
            "max_retries": 5,  # Maximum number of retries for generating a response
        }

        # Create and return an instance of GptCallHandler with the specified parameters
        return GptCallHandler(**model_params)

    def get_user_instruction(self, character):
        """
        Retrieves the user instructions for a specified character.
        This includes the token count and string representation of the character's impressions, memories, and dialogue
        history.

        Args:
            character (Character): The character whose user instructions are to be retrieved.

        Returns:
            tuple: A tuple containing the total token count and the combined string representation of the character's
                impressions, memories, and dialogue history.
        """

        # Retrieve the dictionary of user instruction components for the specified character
        char_inst_comp = self.characters_user[character.name]

        # Return a tuple containing:
        # 1. The total token count, which is the sum of the token counts for impressions, memories, and dialogue history.
        # 2. The combined string representation of impressions, memories, and dialogue history.
        return (
            char_inst_comp["impressions"][0]  # Token count for impressions
            + char_inst_comp["memories"][0]  # Token count for memories
            + char_inst_comp["dialogue_history"][0],  # Token count for dialogue history
            char_inst_comp["impressions"][1]  # String representation for impressions
            + char_inst_comp["memories"][1]  # String representation for memories
            + char_inst_comp["dialogue_history"][
                1
            ],  # String representation for dialogue history
        )

    def get_system_instruction(self, character):
        """Retrieve the system instruction for a specified character.

        This function accesses the character's system prompt components and returns
        a tuple containing the token count and string representation of the system
        instructions.

        Args:
            character: The character for which to retrieve the system instruction.

        Returns:
            tuple: A tuple where the first element is the token count of the system
            instructions and the second element is the string representation of the
            instructions.
        """

        # get this character's dictionary of system prompt components
        char_inst_comp = self.characters_system[character.name]

        # return a tuple containing the system instructions token count and string representation
        return (char_inst_comp["intro"][0], char_inst_comp["intro"][1])

    def update_user_instruction(
        self,
        character,
        update_impressions=False,
        update_memories=False,
        system_instruction_token_count=0,
    ):
        """Update the user's instructions based on character impressions and memories.

        This function modifies the user's character data by updating impressions of other
        characters and relevant memories. It also manages the dialogue history to ensure it
        fits within the model's context limits.

        This method constructs and updates the user instructions which include the impressions, the memory and the
        dialogue history. Currently, the impressions are also passed in without being shortened. The memories are
        reduced if necessary to fit into GPT's context. Note that these aren't returned, but rather are stored in the
        characters system dictionary as a dictionary of lists. Each component serves as a dictionary key, and its value
        is a list where the first index is the component's token count and the second is its string representation.

        Args:
            character: The character whose instructions are being updated.
            update_impressions (bool, optional): Flag to indicate if impressions should be updated. Defaults to False.
            update_memories (bool, optional): Flag to indicate if memories should be updated. Defaults to False.
            system_instruction_token_count (int, optional): The token count of the system instruction. Defaults to 0.

        Returns:
            None: This function updates the character's data in place and does not return a value.
        """

        ### IMPRESSIONS OF OTHER CHARACTERS###
        if update_impressions:
            # Attempt to retrieve and update impressions of other game characters
            try:
                # Get multiple impressions from the character's impressions
                impressions = character.impressions.get_multiple_impressions(
                    self.game.characters.values()
                )
                # Format the impressions into a string with a header
                impressions = (
                    "YOUR IMPRESSIONS OF OTHERS:\n" + "\n".join(impressions) + "\n\n"
                )

                # Calculate the token count for the formatted impressions
                impressions_token_count = get_prompt_token_count(
                    content=impressions, role=None, pad_reply=False
                )

                # Store the impressions token count and string in the user's character data
                self.characters_user[character.name]["impressions"] = (
                    impressions_token_count,
                    impressions,
                )
            except AttributeError:
                # If an error occurs, set impressions to a default value
                self.characters_user[character.name]["impressions"] = (0, "")

        ### MEMORIES OF CHARACTERS IN DIALOGUE/MENTIONED ###

        # Check if memories should be updated
        if update_memories:
            # Construct a query based on the current command and mentioned characters
            query = self.command
            query += ", ".join(self.characters_mentioned)

            # Retrieve the most relevant memories for the character
            if context_list := retrieve(self.game, character, query, n=25):
                # Prepend a primer message to the list of memories
                context_list = [
                    "These are select MEMORIES in ORDER from MOST to LEAST RELEVANT:\n"
                ] + [m + "\n" for m in list(context_list)]

                # Get the impressions token count for the character
                impressions_token_count = self.characters_user[character.name][
                    "impressions"
                ][0]

                # Limit the memories to fit within the model's context by trimming
                memories_limited = limit_context_length(
                    history=context_list,
                    max_tokens=self.model_context_limit
                    - self.characters_user[character.name]["impressions"][0]
                    - system_instruction_token_count
                    - self.dialogue_history_token_count
                    - 5
                    - self.gpt_handler.max_output_tokens,
                    keep_most_recent=False,
                )

                # Convert the list of limited memories into a single formatted string
                memories_limited_str = "".join([f"{m}\n" for m in memories_limited])

                # Calculate the token count for the limited memories string
                memories_limited_token_count = get_prompt_token_count(
                    content=memories_limited_str, role=None, pad_reply=False
                )

                # Update the character's memories in the user's character data
                self.characters_user[character.name]["memories"] = (
                    memories_limited_token_count,
                    memories_limited_str,
                )

            else:
                # If no memories are found, set a default message
                self.characters_user[character.name]["memories"] = (2, "No memories")

        # Update the dialogue history
        # Limit the number of dialogue messages to fit within the model's context
        limited_dialog = limit_context_length(
            history=self.get_dialogue_history_list(),
            max_tokens=self.model_context_limit
            - system_instruction_token_count
            - self.characters_user[character.name]["impressions"][0]
            - self.characters_user[character.name]["memories"][0]
            - self.gpt_handler.max_output_tokens,
        )

        # Join the limited dialogue messages into a single string
        dialog_str = "\n".join(limited_dialog)

        # Calculate the token count for the current dialogue history
        dialogue_history_token_count = get_prompt_token_count(
            content=dialog_str, role=None, pad_reply=False
        )

        # Format the dialogue history prompt for the GPT model
        dialogue_history_prompt = dp.gpt_dialogue_user_prompt.format(
            character=character.name, dialogue_history=dialog_str
        )

        # Update the character's dialogue history in the user's character data
        self.characters_user[character.name]["dialogue_history"] = (
            dialogue_history_token_count,
            dialogue_history_prompt,
        )

    def update_system_instruction(self, character):
        """Update the system instruction for a specified character.

        This function constructs the system prompt for a character by combining their
        standard information with dialogue instructions based on other participants in the
        game. It also calculates the token count for the system prompt and updates the
        character's information in the system dictionary.

        This method constructs and updates the system instructions which now only includes the intro (updated). The
        intro must be included without trimming. Each component serves as a dictionary key, and its value is a list
        where the first index is the component's token count and the second is its string representation.

        Args:
            character: The character for whom the system instruction is being updated.

        Returns:
            None: This function updates the character's system instruction in place and does not return a value.
        """

        ### REQUIRED START TO SYSTEM PROMPT (CAN'T TRIM) ###
        # Retrieve the standard information for the specified character
        intro = character.get_standard_info(self.game)

        # Construct a string of dialogue instructions by joining the names of other participants
        other_character = ", ".join(
            [x.name for x in self.participants if x.name != character.name]
        )
        # Append the formatted dialogue system prompt to the intro
        intro += dp.gpt_dialogue_system_prompt.format(other_character=other_character)

        # Calculate the token count for the constructed system prompt intro
        intro_token_count = get_prompt_token_count(
            content=intro, role="system", pad_reply=False
        )

        # Add the token count for the user role, including padding for GPT's reply
        intro_token_count += get_prompt_token_count(
            content=None, role="user", pad_reply=True
        )

        # Update the character's intro information in the system dictionary with the token count and intro string
        self.characters_system[character.name]["intro"] = (intro_token_count, intro)

    def get_dialogue_history_list(self):
        """Retrieve the list of dialogue history.

        This function returns the stored dialogue history, which contains the
        interactions that have occurred in the conversation. It provides access
        to the dialogue data for further processing or analysis.

        Args:
            None

        Returns:
            list: The list of dialogue history entries.
        """

        return self.dialogue_history

    def get_dialogue_history(self):
        """Retrieve the dialogue history as a formatted string.

        This function concatenates the entries in the dialogue history into a single
        string, with each entry separated by a newline. It provides a readable format
        of the conversation for display or logging purposes.

        Args:
            None

        Returns:
            str: A string representation of the dialogue history, with entries separated by newlines.
        """

        return "\n".join(self.dialogue_history)

    def add_to_dialogue_history(self, message):
        """Add a message to the dialogue history.

        This function appends a new message to the existing dialogue history,
        allowing for the accumulation of conversation entries over time. It
        ensures that all interactions are recorded for future reference.

        Args:
            message (str): The message to be added to the dialogue history.

        Returns:
            None: This function updates the dialogue history in place and does not return a value.
        """

        self.dialogue_history.append(message)

    def get_gpt_response(self, character):
        """Generate a response from the GPT model for a specified character.

        This function retrieves the necessary system and user instructions, checks if the
        combined token count exceeds the model's context limit, and updates the user
        instructions if necessary. It then generates a response from the GPT model and
        handles any potential errors related to token limits.

        Args:
            character: The character for whom the GPT response is being generated.

        Returns:
            str or None: The generated response from the GPT model, or None if an error occurs.
        """

        # Get the system instruction token count and string representation.
        # To change this, pass in True to any system prompt components that we want to update
        system_instruction_token_count, system_instruction_str = (
            self.get_system_instruction(character=character)
        )
        user_instruction_token_count, user_instruction_str = self.get_user_instruction(
            character=character
        )

        # if the sum of the system prompt and dialogue history token counts exceeds the max tokens
        if (
            system_instruction_token_count + user_instruction_token_count
            >= self.model_context_limit
        ):

            # reduce the max token count by the dialogue count, and reduce the number of memories included in the prompt
            self.model_context_limit = (
                self.model_context_limit - user_instruction_token_count
            )
            self.update_user_instruction(
                character,
                update_impressions=False,
                update_memories=True,
                system_instruction_token_count=system_instruction_token_count,
            )
            system_instruction_token_count, system_instruction_str = (
                self.get_system_instruction(character=character, memories=True)
            )

        # get GPT's response
        response = self.gpt_handler.generate(
            system=system_instruction_str, user=user_instruction_str
        )

        if isinstance(response, tuple):
            print("Bad Request Error")
            # This occurs when there was a Bad Request Error cause for exceeding token limit
            success, token_difference = response
            # Add this offset to the calculations of token limits and pad it
            self.token_offset = token_difference + self.offset_pad
            self.offset_pad += 2 * self.offset_pad
            return self.get_gpt_response(character)

        return response

    def is_dialogue_over(self):
        """Determine if the dialogue has concluded.

        This function checks the number of participants in the dialogue. If there is one
        or no participant, it indicates that the dialogue is over.

        Args:
            None

        Returns:
            bool: True if the dialogue is over, otherwise False.
        """

        return len(self.participants) <= 1

    def dialogue_loop(self):
        """Manage the interactive dialogue loop among participants.

        This function initiates a dialogue session, allowing each participant to respond
        in turn while monitoring for new characters mentioned and potential conversation
        termination conditions. It continues the dialogue until a specified number of
        iterations is reached or if the conversation ends prematurely.

        Args:
            None

        Returns:
            list: The complete dialogue history recorded during the session.
        """

        # Initialize a counter to limit the duration of the dialogue loop
        i = 10  # Counter to avoid dialogue dragging on for too long
        print("Dialogue started successfully")

        # Begin the dialogue loop, allowing for a maximum of 10 iterations
        while i > 0:
            # Iterate over each character participating in the dialogue
            for character in self.participants:
                # Retrieve the last line of dialogue to analyze for new character mentions
                last_line = self.dialogue_history[-1]
                # Extract keywords from the last line, specifically looking for character names
                keywords = self.game.parser.extract_keywords(last_line).get(
                    "characters", None
                )
                update_memories = (
                    False  # Flag to determine if memories need to be updated
                )

                # If keywords (character names) are found in the last line
                if keywords:
                    for k in keywords:
                        # Check if the character is not already mentioned
                        if k not in self.characters_mentioned:
                            update_memories = True  # Set flag to update memories
                            self.characters_mentioned.append(
                                k
                            )  # Add new character to the list

                # Update the user's instructions based on the current character and any new mentions
                self.update_user_instruction(
                    character,
                    update_impressions=False,
                    update_memories=update_memories,
                    system_instruction_token_count=self.get_system_instruction(
                        character
                    )[0],
                )

                # Get the response from the GPT model for the current character
                response = self.get_gpt_response(character)
                # Format the response to include the character's name
                response = f"{character.name} said: {response}"
                print(response)  # Print the character's response
                self.add_to_dialogue_history(
                    response
                )  # Add the response to the dialogue history

                # The following lines are commented out but could be used to update the token count
                # response_token_count = get_prompt_token_count(content=response, role=None, pad_reply=False)
                # self.dialogue_history_token_count += response_token_count

                # Check if the response indicates that a character is leaving the conversation
                if "I leave the conversation" in response:
                    self.participants.remove(
                        character
                    )  # Remove the character from the participants
                    print(
                        "The conversation is over"
                    )  # Notify that the conversation has ended
                    break  # Exit the for loop if a character leaves

            # Check if the dialogue is over based on the number of participants
            if self.is_dialogue_over():
                break  # Exit the while loop if the dialogue is over

            i -= 1  # Decrement the counter to eventually end the loop

        # Return the complete dialogue history recorded during the session
        return self.dialogue_history
