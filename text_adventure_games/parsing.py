"""The Parser

The parser is the module that handles the natural language understanding in
the game. The players enter commands in text, and the parser interprets them
and performs the actions that the player intends.  This is the module with
the most potential for improvement using modern natural language processing.
The implementation that I have given below only uses simple keyword matching.
"""

import inspect
import textwrap
import re
import json
import tiktoken
import numpy as np
from openai import OpenAI

from .things import Character, Item, Location
from . import actions, blocks
from .utils.general import set_up_openai_client


class Parser:
    """
    The Parser is the class that handles the player's input.  The player
    writes commands, and the parser performs natural language understanding
    in order to interpret what the player intended, and how that intent
    is reflected in the simulated world.
    """

    def __init__(self, game, echo_commands=False):
        # A list of the commands that the player has issued,
        # and the respones given to the player.
        self.command_history = []

        # Build default scope of actions
        self.actions = game.default_actions()

        # Build default scope of blocks
        self.blocks = game.default_blocks()

        # A pointer to the game.
        self.game = game
        self.game.parser = self

        # Print the user's commands
        self.echo_commands = echo_commands

    def ok(self, description: str):
        """
        In the next homework, we'll replace this with a call to the OpenAI API
        in order to create more evocative descriptions.
        """
        print(Parser.wrap_text(description))
        self.add_description_to_history(description)

    def fail(self, description: str):
        """
        In the next homework, we'll replace this with a call to the OpenAI API
        in order to create more evocative descriptions.
        """
        print(Parser.wrap_text(description))

    @staticmethod
    def wrap_text(text: str, width: int = 80) -> str:
        """
        Keeps text output narrow enough to easily be read
        """
        lines = text.split("\n")
        wrapped_lines = [textwrap.fill(line, width) for line in lines]
        return "\n".join(wrapped_lines)

    def add_command_to_history(self, command: str):
        message = {"role": "user", "content": command}
        self.command_history.append(message)
        # CCB - todo - manage command_history size

    def add_description_to_history(self, description: str):
        message = {"role": "assistant", "content": description}
        self.command_history.append(message)
        # CCB - todo - manage command_history size

    def add_action(self, action: actions.Action):
        """
        Add an Action class to the list of actions a parser can use
        """
        self.actions[action.action_name()] = action

    def add_block(self, block):
        """
        Adds a block class to the list of blocks a parser can use. This is
        primarily useful for loading game states from a save.
        """
        self.blocks[block.__class__.__name__] = block

    def init_actions(self):
        self.actions = {}
        for member in dir(actions):
            attr = getattr(actions, member)
            if inspect.isclass(attr) and issubclass(attr, actions.Action):
                # dont include base class
                if not attr == actions.Action:
                    self.add_action(attr)

    def determine_intent(self, command: str, character):
        """
        This function determines what command the player wants to do.
        Here we have implemented it with a simple keyword match. Later
        we will use AI to do more flexible matching.
        """
        # check which character is acting (defaults to the player)
        # character = self.get_character(command, character)  <-- don't need this if passing in the current character
        command = command.lower()
        if "," in command:
            # Let the player type in a comma separted sequence of commands
            return "sequence"
        elif self.get_direction(command, character.location):
            # Check for the direction intent
            return "direction"
        elif command == "look" or command == "l":
            # when the user issues a "look" command, re-describe what they see
            return "describe"
        elif "examine " in command or command.startswith("x "):
            return "examine"
        elif "take " in command or "get " in command:
            return "take"
        elif "light" in command:
            return "light"
        elif "drop " in command:
            return "drop"
        elif (
            "eat " in command
            or "eats " in command
            or "ate " in command
            or "eating " in command
        ):
            return "eat"
        elif "drink" in command:
            return "drink"
        elif "give" in command:
            return "give"
        elif "attack" in command or "hit " in command or "hits " in command:
            return "attack"
        elif "inventory" in command or command == "i":
            return "inventory"
        elif "quit" in command:
            return "quit"
        else:
            for _, action in self.actions.items():
                special_command = action.action_name()
                if special_command in command:
                    return action.action_name()
        return None

    def parse_action(self, command: str, character) -> actions.Action:
        """
        Routes an action described in a command to the right action class for
        performing the action.
        """
        command = command.lower().strip()
        if command == "":
            return None
        intent = self.determine_intent(command, character)
        if intent in self.actions:
            action = self.actions[intent]
            return action(self.game, command, character)
        elif intent == "direction":
            return actions.Go(self.game, command, character)
        elif intent == "take":
            return actions.Get(self.game, command, character)
        self.fail(f"No action found for {command}")
        return None

    def parse_command(self, command: str, character: Character):
        # print("\n>", command, "\n", flush=True)
        # add this command to the history
        self.add_command_to_history(command)
        action = self.parse_action(command, character)
        if not action:
            self.fail("I'm not sure what you want to do.")
            return False
        else:
            action()
            return True

    @staticmethod
    def split_command(command: str, keyword: str) -> tuple[str, str]:
        """
        Splits the command string into two parts based on the keyword.

        Args:
        command (str): The command string to be split.
        keyword (str): The keyword to split the command string around.

        Returns:
        tuple: A tuple containing the part of the command before the keyword and the part after.
        """
        command = command.lower()
        keyword = keyword.lower()
        # Find the position of the keyword in the command
        keyword_pos = command.find(keyword)

        # If the keyword is not found, return the entire command and an empty string
        if keyword_pos == -1:
            return (command, "")

        # Split the command into two parts
        before_keyword = command[:keyword_pos]
        after_keyword = command[keyword_pos + len(keyword):]

        return (before_keyword, after_keyword)

    def get_character(self, command: str, character: Character) -> Character:
        # ST 3/10 - add character arg for sake of override in GptParser3
        """
        This method tries to match a character's name in the command.
        If no names are matched, it returns the default value.
        """
        command = command.lower()
        # matched_character_name = ""  # JD logical change
        for name in self.game.characters.keys():
            if name.lower() in command:
                return self.game.characters[name]
        return self.game.player

    def get_character_location(self, character: Character) -> Location:
        return character.location

    def match_item(self, command: str, item_dict: dict[str, Item]) -> Item:
        """
        Check whether the name any of the items in this dictionary match the
        command. If so, return Item, else return None.
        """
        for item_name in item_dict:
            if item_name in command:
                item = item_dict[item_name]
                return item
        return None

    def get_items_in_scope(self, character=None) -> dict[str, Item]:
        """
        Returns a list of items in character's location and in their inventory
        """
        if character is None:
            character = self.game.player
        items_in_scope = {}
        for item_name in character.location.items:
            items_in_scope[item_name] = character.location.items[item_name]
        for item_name in character.inventory:
            items_in_scope[item_name] = character.inventory[item_name]
        return items_in_scope

    def get_direction(self, command: str, location: Location = None) -> str:
        """
        Converts aliases for directions into its primary direction name.
        """
        command = command.lower()
        if command == "n" or "north" in command:
            return "north"
        if command == "s" or "south" in command:
            return "south"
        if command == "e" or "east" in command:
            return "east"
        if command == "w" or "west" in command:
            return "west"
        if command.endswith("go up"):
            return "up"
        if command.endswith("go down"):
            return "down"
        if command.endswith("go out"):
            return "out"
        if command.endswith("go in"):
            return "in"
        if location:
            for exit in location.connections.keys():
                if exit.lower() in command:
                    return exit
        return None


class GptParser(Parser):
    def __init__(self, game, echo_commands=True, verbose=False):
        super().__init__(game, echo_commands=echo_commands)
        self.verbose = verbose
        self.client = set_up_openai_client()
        self.gpt_model = "gpt-4"  # REMEMBER TO SWITCH BACK TO GPT-4
        self.max_output_tokens = 256  # You get to pick this
        self.max_tokens = 8192 - self.max_output_tokens  # GPT-4's max total tokens
        # self.max_tokens = 4096
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def gpt_describe(self, system_instructions, command_history):
        """
        Generate a description with GPT.  This takes two arguments:
        * The system instructions, which is the prompt that describes 
          how you'd like GPT to behave.
        * The command history - this is a list of previous user input 
          commands and game descriptions. It's given as context to GPT.
        The Parser class manages the command_history via the 
        `add_command_to_history` and `add_description_to_history` functions
        which use the ChatGPT format with commands being assigned role: user,
        and descriptions being assigned role: assistant.
        """
        try:
            messages = [{
                "role": "system",
                "content": system_instructions
            }]
            context = self.limit_context_length(command_history, self.max_tokens)
            messages.extend(context)
            if self.verbose:
                print(json.dumps(messages, indent=2))
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
                temperature=1,
                max_tokens=self.max_output_tokens,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            return f"Something went wrong with GPT: {e}"
        
    def limit_context_length(self, command_history, max_tokens, max_turns=1000):
        """
        This method limits the length of the command_history 
        to be less than the max_tokens and less than max_turns. 
        The least recent messages are disregarded from the context. 
        This function is non-destructive and doesn't modify command_history.
        """
        encoding = tiktoken.get_encoding("cl100k_base")

        if len(command_history) > max_turns:
            adj_command_history = command_history[-max_turns:]
        else:
            # print(f"history not greater than {max_turns}")
            adj_command_history = command_history[:]

        commands = [i['content'] for i in adj_command_history]
        command_lengths = [len(encoding.encode(i)) for i in commands]
        command_lengths.reverse()
        cum_list_lengths = np.cumsum(command_lengths)
        for index, i in enumerate(cum_list_lengths):
            # print("Total token length so far:", i)
            if i > max_tokens:
                # print("returning truncated history of length:")
                # print(len(command_history[-index + 1:]))
                return command_history[-index + 1:]
        return command_history  

    def ok(self, description):
        """
        In this homework, we'll replace this with a call to the OpenAI API
        in order to create more evocative descriptions.  For this part, 
        all you need to do is create your own system instructions.   
        """
        self.command_history.append({"role": "assistant", "content": description})
        
        system_instructions = """You are the narrator for a text adventure game. 
        You create short, evocative descriptions of the scenes in the game.
        Include descriptions of the items and exits available to the Player."""
        
        response = self.gpt_describe(system_instructions, self.command_history)
        self.add_description_to_history(response)
        print(self.wrap_text(response) + '\n')

    def fail(self, description):
        """
        In this homework, we'll replace this with a call to the OpenAI API
        in order to create more useful error messages descriptions.  You should
        do the same steps as for the ok() command, but optionally customize 
        the system_instructions to tell the user that their command failed.
        """
        self.command_history.append({"role": "assistant", "content": description})
        print("COMMAND FAILED")
        
        system_instructions = "You are the narrator for a text adventure game. \
The player entered a command that failed in the game. \
Try to help the player understand why the command failed."

        response = self.gpt_describe(system_instructions, self.command_history)
        if self.verbose:
            print("GPT's Error Description:")
        # self.add_description_to_history(response)
        print(self.wrap_text(response) + '\n')

    
class GptParser2(GptParser):
    def __init__(self, game, echo_commands=True, verbose=False):
        super().__init__(game, echo_commands, verbose)
        self.refresh_command_list()

    def refresh_command_list(self):
        # Command descriptions is a dictionary that maps
        # action descriptions and aliases onto action names 
        command_descriptions = {}
        for _, action in self.actions.items():
            description = action.ACTION_DESCRIPTION
            if action.ACTION_ALIASES:
                description += " (can also be invoked with '{aliases}')".format(
                    aliases="', '".join(action.ACTION_ALIASES)
                )
            action_name = action.ACTION_NAME
            if action_name:
                command_descriptions[description] = action_name
        
        self.command_descriptions = command_descriptions
        return self

    def determine_intent(self, command, character: Character):
        """
        Given a user's command, this method returns the ACTION_NAME of
        the intended action.  
        """
        lst = [str(idx) + ': ' + str(i) + '\n' for idx, i in enumerate(self.command_descriptions.keys())]
        choices = ''.join(lst)
        prompt_beg = "You are matching an input text to the most similar action phrase. If you are unsure of the correct input, the action is most likely \"go\".\n\
Focus on matching the action to the verb in the command. Here are some examples of matching verbs:\n\
\"grab\", \"take\", \"pick up\" ==> \"get\"\n\
\"leave behind\", \"discard\", \"leave\" ==> \"drop\"\n\
\"leave for\", \"head to\", \"head towards\" ==> \"go\"\n\
\"strike\", \"fight\", \"hit\" ==> \"attack\"\n\
\"inspect\", \"study\", \"look at\" ==> \"examine\"\n\
\"gift\", \"hand over\", \"pass\" ==> \"give\"\
\"consume\", \"taste\" ==> \"eat.\"\
\"consume\", \"taste\", \"imbibe\", \"sip\" ==> \"drink\"\n\
\"sniff\"  ==> \"smell\"\n\
\"go fishing\" ==> \"catch fish\"\n\
You must only return the single number whose corresponding text best matches the given command:\n"
        prompt_end = "The best match is number: "
        total_prompt = prompt_beg + choices + prompt_end
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": total_prompt
                },
                {
                    "role": "user",
                    "content": command
                },
            ],
            temperature=0,
            max_tokens=10,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
        )

        idx = response.choices[0].message.content

        # print(total_prompt)
        print("Chosen Command:", list(self.command_descriptions.values())[int(idx)])
        
        return list(self.command_descriptions.values())[int(idx)]


class GptParser3(GptParser2):
    def __init__(self, game, echo_commands=True, verbose=False, model='gpt-4'):
        super().__init__(game, echo_commands, verbose)
        self.model = model

    def extract_digit(self, text):
        # extracted_digit = list(filter(str.isdigit, text))
        # return extracted_digit[0]

        return re.findall(r"[-]?\d+", text)[0]
    
    def get_characters_and_find_current(self, character):
        current_idx = -999
        chars = {}
        for i, char in enumerate(list(self.game.characters)):
            chars[i] = char
            if char == character.name:
                current_idx = i
        return chars, current_idx
    
    def get_character(
        self, command: str, character: Character = None, hint: str = None, split_words=None, position=None
    ) -> Character:
        """
        This method tries to match a character's name in the command.
        If no names are matched, it defaults to the passed character. 
        Args:
            hint: A hint about the role of character we're looking for 
                  (e.g. "giver" or "recipent")
            split_words: not needed for our GptParser
            position: not needed for our GptParser
        """

        system_prompt = "Given a command, return the character who can be described as: \"{h}\". ".format(h=hint)
        # Create an enumerated dict of the characters in the game
        chars, curr_idx = self.get_characters_and_find_current(character)

        system_prompt += f"Unless specified, assume \"{curr_idx}: {character.name}\" performs all actions.\nChoose from the following characters:\n"
        # Format the characters into a list structure for the system prompt
        system_prompt += "{c}".format(c='\n'.join([str(i)+": "+str(c) for i, c in chars.items()]))

        system_prompt += "\nYou must only return the single number whose corresponding character is performing the action.\n\
If no command is given, return \"{curr_idx}: {character.name}\""
        # if hint:
        #     system_prompt += "As a hint, in the given command, the subject can be described as: \"{h}\". ".format(h=hint)
        #     system_prompt += "If there are no good matches, the action is performed by the game player, so you should return 0.\n"
        # else:
        #     system_prompt += "If there are no good matches, the action is performed by the game player, so you should return 0.\n"

        # create a new client
        # client = OpenAI()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": "Command: {c}\nThe best character match is number: ".format(c=command)
                },
            ],
            temperature=0,
            max_tokens=10,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Will probably need to do some parsing of the output here
        char_idx = response.choices[0].message.content
        try: 
            char_idx = self.extract_digit(char_idx)
            char_idx = int(char_idx)
        except Exception as e:
            print("Couldn't match the following response to a number:")
            print(char_idx)
            print(e)

        # print("Item system prompt: ", system_prompt)
        print(f"GPTParse selected character: {char_idx}")
        if char_idx not in chars:
            print(f"no player with id {char_idx} in {str(chars)}")
            return self.game.player
        else:
            name = chars[char_idx]
            return self.game.characters[name]

    def match_item(
        self, command: str, item_dict: dict[str, Item], hint: str = None
    ) -> Item:
        """
        Check whether the name any of the items in this dictionary match the
        command. If so, return Item, else return None.

        Args:
            item_dict: A map from item names to Items (could be a player's 
                       inventory or the items at a location)
            hint: what kind of item we're looking for
        """

        system_prompt = "Given a command, return the item that is the direct object of the action.\nChoose from the following items:\n"
        items = {i: it for i, it in enumerate(list(item_dict.keys()))}
        system_prompt += "{c}".format(c=''.join([str(i)+": "+str(item)+"\n" for i, item in items.items()]))
        system_prompt += """You must only return the single number whose corresponding item best matches the given command. \
If there are no good matches, return '-999'\n"""
        if hint:
            system_prompt += "As a hint, in the given command, the item can be described as:\"{h}\".\n".format(h=hint)
        else:
            system_prompt += "\n"
        
        # print("Item system prompt: ", system_prompt)
        # client = OpenAI()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": "Command: {c}\n  The best item match is number: ".format(c=command)
                },
            ],
            temperature=0,
            max_tokens=10,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
        )

        item_idx = response.choices[0].message.content
        try:
            item_idx = self.extract_digit(item_idx)
            item_idx = int(item_idx)
        except Exception as e:
            print(e)

        print(f"GPTParse selected item: {item_idx}")
        if item_idx == -999:
            return None
        elif item_idx in items:
            name = items[item_idx]
            return item_dict[name]
        else:
            print(f'Item index {item_idx} not found in {str(items)}')

    def get_direction(self, command: str, location: Location = None) -> str:
        """
        Return the direction from `location.connections` which the player
        wants to travel to.
        """
        dirs = list(location.connections.keys())
        names = [loc.name for loc in location.connections.values()]
        connections = {i: dl for i, dl in enumerate(zip(dirs, names))}
        print('Found connections: ', connections)

        system_prompt = """
        You must select the direction that best matches the description given in a command.
        The possible directions to choose are:\n
        """
        
        system_prompt += "\n" + "{c}".format(c=''.join([str(i)+": "+str(d)+" or "+str(l)+"\n" for i, (d, l) in connections.items()]))
        
        system_prompt += """\nYou must only return the single number whose corresponding direction best matches the given command.
            If there are no good matches, return '-999'\n"""

        # print("Direction system prompt: ", system_prompt)

        # client = OpenAI()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": "Command: {c}\n  The best direction match is number:  ".format(c=command)
                }
            ],
            temperature=0,
            max_tokens=100,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
        )

        dir_idx = response.choices[0].message.content
        try:
            dir_idx = self.extract_digit(dir_idx)
            dir_idx = int(dir_idx)
        except Exception as e:
            print(e)
        print(f"GPTParse selected direction: {dir_idx}")

        if dir_idx in connections:
            dir_name = connections[dir_idx][0]
            return dir_name
        else:
            print(f'direction id "{dir_idx}" not in location connections: {connections}')
            return None
