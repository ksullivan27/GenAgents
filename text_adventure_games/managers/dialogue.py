import tiktoken
from text_adventure_games.gpt.gpt_helpers import (
    limit_context_length,
    get_prompt_token_count,
    GptCallHandler,
)
from text_adventure_games.assets.prompts import dialogue_prompt as dp
from ..utils.general import set_up_openai_client
from ..agent.agent_cognition.retrieve import retrieve

ACTION_MAX_OUTPUT = 100


class Dialogue:
    """
    Represents a dialogue system for managing interactions between two game characters. This class handles the
    initialization of participants, manages dialogue history, and interfaces with a GPT model to generate character
    responses based on their instructions and memories.

    Args:
        game: The game context in which the dialogue takes place.
        participants (List[Character]): A sorted list of characters participating in the dialogue, ordered by
        initiative.
        command: The initial command or intent that starts the dialogue.

    Attributes:
        game: The game context.
        gpt_handler: An instance of the GPT handler for generating responses.
        token_offset: An offset for managing token limits.
        offset_pad: A padding value for token calculations.
        model_context_limit: The maximum token limit for the GPT model.
        participants: The list of dialogue participants.
        characters_system: A dictionary storing system instructions for each character.
        characters_user: A dictionary storing user instructions for each character.
        participants_number: The number of participants in the dialogue.
        command: The command that initiated the dialogue.
        dialogue_history: A list tracking the history of dialogue exchanges.
        dialogue_history_token_count: The token count of the dialogue history.
        characters_mentioned: A list of characters mentioned during the dialogue.
    """

    def __init__(self, game, participants, command):
        """
        Initializes a Dialogue instance for managing interactions between game characters. This constructor sets up the
        necessary attributes, including the game context, participants, and command, while also preparing the dialogue
        history and character instructions.

        Args:
            game: The game context in which the dialogue occurs.
            participants (List[Character]): A sorted list of characters participating in the dialogue, ordered by
            initiative.
            command: The initial command or intent that starts the dialogue.

        Attributes:
            game: The game context.
            gpt_handler: An instance of the GPT handler for generating responses.
            token_offset: An offset for managing token limits.
            offset_pad: A padding value for token calculations.
            model_context_limit: The maximum token limit for the GPT model.
            participants: The list of dialogue participants.
            characters_system: A dictionary storing system instructions for each character.
            characters_user: A dictionary storing user instructions for each character.
            participants_number: The number of participants in the dialogue.
            command: The command that initiated the dialogue.
            dialogue_history: A list tracking the history of dialogue exchanges.
            dialogue_history_token_count: The token count of the dialogue history.
            characters_mentioned: A list of characters mentioned during the dialogue.
        """

        # Store the game context in the instance variable
        self.game = game

        # Set up the GPT handler for generating responses
        self.gpt_handler = self._set_up_gpt()

        # Initialize token offset and padding for managing token limits
        self.token_offset = 0
        self.offset_pad = 5

        # Set the maximum token limit for the GPT model
        self.model_context_limit = self.gpt_handler.model_context_limit

        # Store the participants in the dialogue
        self.participants = participants

        # Initialize dictionaries to hold system and user instructions for each character
        self.characters_system = {}
        self.characters_user = {}

        # Count the number of participants in the dialogue
        self.participants_number = len(participants)

        # Store the initial command that starts the dialogue
        self.command = command

        # Initialize the dialogue history with the starting message
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
            # Add each participant to the characters system dictionary using their name as the key
            self.characters_system[participant.name] = dict()
            self.characters_user[participant.name] = dict()

            # Update the character's system instructions, including intro, impressions, and memories
            self.update_system_instruction(participant)
            self.update_user_instruction(
                participant,
                update_impressions=True,  # Indicate that impressions should be updated
                update_memories=True,      # Indicate that memories should be updated
                system_instruction_token_count=self.get_system_instruction(participant)[0],  # Get the token count for the system instructions
            )


    def _set_up_gpt(self):
        """
        Sets up the GPT handler with the necessary parameters for generating responses. This method initializes the GPT
        model with specific configurations, including the API key, model type, token limits, and other settings.

        Returns:
            GptCallHandler: An instance of the GPT call handler configured with the specified parameters.
        """

        model_params = {
            "api_key_org": "Helicone",
            "model": "gpt-4",
            "max_tokens": 250,
            "temperature": 1,
            "top_p": 1,
            "max_retries": 5,
        }

        return GptCallHandler(**model_params)

    def get_user_instruction(self, character):
        """
        Retrieves the user instructions for a specified character, including their impressions, memories, and dialogue
        history. This method returns both the total token count and the string representation of the user's
        instructions.

        Args:
            character: The character whose user instructions are to be retrieved.

        Returns:
            tuple: A tuple containing the total token count and the string representation of the user instructions.
        """

        # get this character's dictionary of system prompt components
        char_inst_comp = self.characters_user[character.name]

        # return a tuple containing the system instructions token count and string representation
        return (
            char_inst_comp["impressions"][0]
            + char_inst_comp["memories"][0]
            + char_inst_comp["dialogue_history"][0],
            char_inst_comp["impressions"][1]
            + char_inst_comp["memories"][1]
            + char_inst_comp["dialogue_history"][1],
        )

    def get_system_instruction(self, character):
        """
        Retrieves the system instructions for a specified character, focusing on their introductory information. This
        method returns both the token count and the string representation of the character's system instructions.

        Args:
            character: The character whose system instructions are to be retrieved.

        Returns:
            tuple: A tuple containing the token count and the string representation of the system instructions.
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
        """
        Updates the user instructions for a specified character, including their impressions, memories, and dialogue
        history. This method allows for the optional updating of impressions and memories, ensuring that the character's
        instructions are current and relevant for the dialogue context.

        This method constructs and updates the user instructions which include the impressions, the memory and the
        dialog history. Currently, the impressions are also passed in without being shortened. The memories are
        reduced if necessary to fit into GPT's context. Note that these aren't returned, but rather are stored in the
        characters system dictionary as a dictionary of lists. Each component serves as a dictionary key, and its value
        is a list where the first index is the component's token count and the second is its string representation.

        Args:
            character: The character whose user instructions are to be updated.
            update_impressions (bool, optional): Indicates whether to update the character's impressions. Defaults to
            False.
            update_memories (bool, optional): Indicates whether to update the character's memories. Defaults to False.
            system_instruction_token_count (int, optional): The token count of the system instructions for the
            character. Defaults to 0.

        Returns:
            None
        """

        ### IMPRESSIONS OF OTHER CHARACTERS###
        # Check if impressions need to be updated
        if update_impressions:
            # Retrieve impressions of other game characters
            try:
                impressions = character.impressions.get_multiple_impressions(
                    self.game.characters.values()
                )
                # Format the impressions into a readable string
                impressions = (
                    "YOUR IMPRESSIONS OF OTHERS:\n" + "\n".join(impressions) + "\n\n"
                )

                # Calculate the token count for the impressions string
                impressions_token_count = get_prompt_token_count(
                    content=impressions, role=None, pad_reply=False
                )

                # Update the character's impressions in the characters system dictionary
                self.characters_user[character.name]["impressions"] = (
                    impressions_token_count,
                    impressions,
                )
            except AttributeError:
                # If the character has no impressions, set default values
                self.characters_user[character.name]["impressions"] = (0, "")

        ### MEMORIES OF CHARACTERS IN DIALOGUE/MENTIONED ###

        # Check if memories need to be updated
        if update_memories:
            # Create a query string based on the current command and mentioned characters
            query = self.command
            query += ", ".join(self.characters_mentioned)

            # Retrieve the most relevant memories for the character
            if context_list := retrieve(self.game, character, query, n=25):
                # Prepend a primer message to the list of memories
                context_list = [
                    "These are select MEMORIES in ORDER from MOST to LEAST RELEVANT:\n"
                ] + [m + "\n" for m in list(context_list)]

                # Get the token count for the character's impressions
                impressions_token_count = self.characters_user[character.name][
                    "impressions"
                ][0]

                # Limit the memories to fit within the GPT's context by trimming less relevant memories
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

                # Convert the list of limited memories into a single string
                memories_limited_str = "".join([f"{m}\n" for m in memories_limited])

                # Calculate the token count for the limited memories string
                memories_limited_token_count = get_prompt_token_count(
                    content=memories_limited_str, role=None, pad_reply=False
                )

                # Update the character's memories in the characters system dictionary
                self.characters_user[character.name]["memories"] = (
                    memories_limited_token_count,
                    memories_limited_str,
                )

            else:
                # If no memories are found, set default values
                self.characters_user[character.name]["memories"] = (2, "No memories")

        # Update the dialogue history
        # Limit the number of dialogue messages to fit within GPT's context
        limited_dialog = limit_context_length(
            history=self.get_dialogue_history_list(),
            max_tokens=self.model_context_limit
            - system_instruction_token_count
            - self.characters_user[character.name]["impressions"][0]
            - self.characters_user[character.name]["memories"][0]
            - self.gpt_handler.max_output_tokens,
        )

        # Convert the limited dialogue to a single string
        dialog_str = "\n".join(limited_dialog)

        # Calculate the token count for the current dialogue history
        dialogue_history_token_count = get_prompt_token_count(
            content=dialog_str, role=None, pad_reply=False
        )

        # Format the dialogue history for the GPT prompt
        dialogue_history_prompt = dp.gpt_dialogue_user_prompt.format(
            character=character.name, dialogue_history=dialog_str
        )

        # Update the character's dialogue history in the characters system dictionary
        self.characters_user[character.name]["dialogue_history"] = (
            dialogue_history_token_count,
            dialogue_history_prompt,
        )


    def update_system_instruction(self, character):
        """
        Updates the system instructions for a specified character, focusing on their introductory information and
        dialogue context. This method constructs the system prompt by incorporating the character's standard information
        and the names of other participants in the dialogue.

        Args:
            character: The character whose system instructions are to be updated.

        Returns:
            None
        """

        ### REQUIRED START TO SYSTEM PROMPT (CAN'T TRIM) ###
        intro = character.get_standard_info(self.game)

        # add dialogue instructions
        other_character = ", ".join(
            [x.name for x in self.participants if x.name != character.name]
        )
        intro += dp.gpt_dialogue_system_prompt.format(other_character=other_character)

        # get the system prompt intro token count
        intro_token_count = get_prompt_token_count(
            content=intro, role="system", pad_reply=False
        )

        # account for the number of tokens in the resulting role (just the word 'user'),
        # including a padding for GPT's reply containing <|start|>assistant<|message|>
        intro_token_count += get_prompt_token_count(
            content=None, role="user", pad_reply=True
        )

        # update the character's intro in the characters system dictionary
        self.characters_system[character.name]["intro"] = (intro_token_count, intro)

    def get_dialogue_history_list(self):
        """
        Retrieves the list of dialogue history for the current conversation. This method provides access to the recorded
        exchanges that have occurred during the dialogue.

        Returns:
            list: A list containing the history of dialogue exchanges.
        """

        return self.dialogue_history

    def get_dialogue_history(self):
        """
        Retrieves the dialogue history as a single formatted string. This method concatenates all recorded exchanges in
        the dialogue, separating them with newline characters for readability.

        Returns:
            str: A string representation of the dialogue history, with each exchange on a new line.
        """

        return "\n".join(self.dialogue_history)

    def add_to_dialogue_history(self, message):
        """
        Adds a new message to the dialogue history. This method appends the provided message to the list of recorded
        exchanges, allowing for the tracking of the conversation's progression.

        Args:
            message (str): The message to be added to the dialogue history.

        Returns:
            None
        """

        self.dialogue_history.append(message)

    def get_gpt_response(self, character):
        """
        Generates a response from the GPT model for a specified character based on their system and user instructions.
        This method handles token limits and updates the character's instructions as necessary to ensure a valid
        response is generated.

        Args:
            character: The character for whom the GPT response is to be generated.

        Returns:
            str: The generated response from the GPT model.

        Raises:
            Exception: If there is a bad request error due to exceeding token limits.
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
        """
        Determines whether the dialogue has concluded based on the number of participants. This method checks if there
        is one or no participants left in the dialogue, indicating that the conversation is over.

        Returns:
            bool: True if the dialogue is over (one or no participants), otherwise False.
        """

        return len(self.participants) <= 1

    def dialogue_loop(self):
        """
        Manages the dialogue loop for the participants in the conversation. This method facilitates the interaction
        between characters, updates their instructions, retrieves responses from the GPT model, and handles the flow of
        dialogue until the conversation ends or a specified limit is reached.

        Returns:
            list: The updated dialogue history after the loop concludes.
        """

        i = 10  # Counter to avoid dialogue dragging on for too long
        print("Dialogue started successfully")
        while i > 0:
            for character in self.participants:
                # Get last line of dialogue and if any new characters are mentioned update system prompts
                last_line = self.dialogue_history[-1]
                keywords = self.game.parser.extract_keywords(last_line).get(
                    "characters", None
                )
                update_memories = False
                if keywords:
                    for k in keywords:
                        if k not in self.characters_mentioned:
                            update_memories = True
                            self.characters_mentioned.append(k)

                self.update_user_instruction(
                    character,
                    update_impressions=False,
                    update_memories=update_memories,
                    system_instruction_token_count=self.get_system_instruction(
                        character
                    )[0],
                )
                # Get GPT response
                response = self.get_gpt_response(character)
                response = f"{character.name} said: {response}"
                print(response)
                self.add_to_dialogue_history(response)

                # update the dialog history token count with the latest reply
                # response_token_count = get_prompt_token_count(content=response, role=None, pad_reply=False)
                # self.dialogue_history_token_count += response_token_count

                # End conversation if a character leaves
                if "I leave the conversation" in response:
                    self.participants.remove(character)
                    print("The conversation is over")
                    break
            if self.is_dialogue_over():
                break
            i -= 1
        return self.dialogue_history
