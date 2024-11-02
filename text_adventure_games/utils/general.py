"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: utils/general.py
Description: helper methods used throughout the project
"""

print("Importing General")

# Importing defaultdict from collections for creating dictionaries with default values.
from collections import defaultdict

# Importing os module for interacting with the operating system.
import os

# Importing re module for regular expression operations.
import re

# Importing json module for working with JSON data.
import json

# Importing string module for string manipulation and constants.
import string

# Importing the inspect module, which provides useful functions to get information about live objects
import inspect

# Importing numpy for numerical operations and array handling.
import numpy as np

# Importing Dict type for type hinting in function signatures.
from typing import Dict

# Importing OpenAI class from the openai module for API interactions.
from openai import OpenAI

# # Importing OpenAIEngine from kani.engines.openai for engine-specific operations.
# from kani.engines.openai import OpenAIEngine

# # Importing GptCallHandler from gpt.gpt_helpers for handling GPT call counts.
# from GenAgentsBoardroom.text_adventure_games.gpt.gpt_helpers import GptCallHandler

# Local imports from the current package, specifically constants used in the module.
print(f"\t{__name__} calling imports for Consts")
from . import consts


def set_up_openai_client(org="Penn", **kwargs):
    """
    Sets up and returns an OpenAI client configured with the specified organization and additional parameters. This
    function retrieves the API key for the organization, updates the parameters accordingly, and initializes the OpenAI
    client, optionally setting a custom base URL for specific organizations.

    Args:
        org (str, optional): The organization for which to set up the OpenAI client. Defaults to "Penn".
        **kwargs: Additional parameters to be passed to the OpenAI client.

    Returns:
        OpenAI: An instance of the OpenAI client configured with the provided parameters.

    Raises:
        ValueError: If the organization is not recognized or if the API key cannot be retrieved.
    """

    # Retrieve the OpenAI API key for the specified organization.
    key = consts.get_openai_api_key(org)

    # Create a copy of the provided keyword arguments to avoid modifying the original.
    params = kwargs.copy()

    # Update the parameters with the retrieved API key.
    params["api_key"] = key

    # TODO: See if this block below is needed
    # Set the model to the value in the models config file if it is not already set
    if params.get("model") is None:
        params["model"] = consts.get_models_config()["miscellaneous"]["model"]

    # If the organization is "Helicone", retrieve and set the base URL for the API.
    if org == "Helicone":
        base_url = consts.get_helicone_base_path()
        params["base_url"] = base_url

    # Initialize and return the OpenAI client with the specified parameters.
    return OpenAI(**params)


def set_up_kani_engine(org="Penn", model="gpt-4o-mini", **kwargs):
    """
    Sets up and returns an instance of the Kani engine configured with the specified organization and model. This
    function retrieves the API key for the organization and initializes the OpenAIEngine with the provided model and
    additional parameters.

    Args:
        org (str, optional): The organization for which to set up the Kani engine. Defaults to "Penn".
        model (str, optional): The model to be used by the Kani engine. Defaults to 'gpt-4'.
        **kwargs: Additional parameters to be passed to the OpenAIEngine.

    Returns:
        OpenAIEngine: An instance of the Kani engine configured with the provided parameters.

    Raises:
        ValueError: If the organization is not recognized or if the API key cannot be retrieved.
    """

    key = consts.get_openai_api_key(org)
    return OpenAIEngine(key, model=model, **kwargs)


def get_logger_extras(game, character=None, include_gpt_call_id=False, stack_level=1):
    """
    Retrieves additional logging information related to the current game state and character attributes. This function
    constructs a dictionary containing relevant details about the character and the game, which can be used for logging
    purposes.

    Args:
        game (Game): The current game instance containing game-related information.
        character (Character, optional): The character instance whose attributes are to be logged. Defaults to None .
        include_gpt_call_id (bool, optional): A flag indicating whether to include the GPT call ID in the logging
        information. Defaults to False.
        stack_level (int, optional): The level of the stack to inspect for the calling module. Defaults to 1.

    Returns:
        dict: A dictionary containing logging extras, including character details and game state information.

    Raises:
        AttributeError: If the character does not have the expected attributes.
    """

    # Import GptCallHandler here to avoid circular import issues
    print(f"\t{__name__} interior calling imports for GptCallHandler")
    from ..gpt.gpt_helpers import GptCallHandler

    # # Get the filename of the module that called this function
    # module_name = inspect.stack()[stack_level].filename
    # # Remove the .py extension
    # module_name = os.path.splitext(module_name)[0]
    # # Capitalize name
    # module_name = module_name.title()

    if include_gpt_call_id:
        gpt_call_id = GptCallHandler.get_calls_count()
    else:
        gpt_call_id = "N/A"

    return {
        "gpt_call_id": gpt_call_id,
        "character_name": character.name if character else "none",
        "character_id": character.id if character else "none",
        "character_group": character.group if character else "none",
        "action_location": character.location.name if character else "none",
        "round": game.round,
        "tick": game.tick,
        "total_ticks": game.total_ticks,
        "experiment_name": game.experiment_name,
        "experiment_id": game.experiment_id,
    }


def normalize_name(name):
    """
    Normalizes a given character name by removing common prefixes and suffixes, converting it to lowercase, and
    eliminating non-alphanumeric characters. This function processes the name to ensure a standardized format, making it
    suitable for consistent storage or display.

    Args:
        name (str): The name to be normalized.

    Returns:
        str: The normalized name without prefixes, suffixes, or non-alphanumeric characters.

    Raises:
        ValueError: If the input name is empty or only contains non-alphanumeric characters.
    """

    # Convert the input name to lowercase to ensure uniformity in processing.
    name = name.lower()

    # Remove non-alphanumeric characters from the name, allowing only letters, numbers, and spaces.
    name = "".join(e for e in name if e.isalnum() or e.isspace())

    # Split the cleaned name into individual parts for easier handling of prefixes and suffixes.
    name_parts = name.split()

    # Check if there are any name parts to process.
    if name_parts:
        # Define a list of common prefixes that may appear in names.
        common_prefixes = [
            "mr",
            "ms",
            "mrs",
            "dr",
            "sir",
            "lady",
            "captain",
            "prof",
            "professor",
        ]

        # If the first part of the name is a common prefix, remove it from the list of name parts.
        if name_parts[0] in common_prefixes:
            name_parts = name_parts[1:]  # Remove the first element (prefix)

    # Define a list of common suffixes that may appear in names.
    common_suffixes = ["jr", "sr", "ii", "iii", "iv", "phd", "md"]

    # If there are remaining name parts and the last part is a common suffix, remove it.
    if name_parts and name_parts[-1] in common_suffixes:
        name_parts = name_parts[:-1]  # Remove the last element (suffix)

    # Return the cleaned and rejoined name parts as a single string.
    return " ".join(name_parts)


def extract_target_word(response):
    """
    Extracts the target word from a given response string by splitting the response into words and returning the first
    word. This function also removes any surrounding punctuation from the extracted word, ensuring a clean output.

    Args:
        response (str): The response string from which to extract the target word.

    Returns:
        str: The first word from the response, stripped of surrounding punctuation.

    Raises:
        IndexError: If the response is an empty string, leading to an attempt to access an index that does not exist.
    """

    words = response.split()
    # For debugging purposes check when it fails to return only 1 word.
    if len(words) > 1:
        # print("target word list returned is: ", words)
        return words[0].strip(string.punctuation)
    else:
        return words[0].strip(string.punctuation)


def extract_enumerated_list(response):
    """
    Extracts a list of enumerated words from a response string formatted with numbered lines. This function identifies
    lines that match a specific enumeration pattern and returns the corresponding words as a list.

    Args:
        response (str): The response string containing enumerated lines.

    Returns:
        list: A list of words extracted from the enumerated lines.

    Raises:
        None
    """

    # Split the response string into individual lines using the newline character as the delimiter.
    lines = response.split("\n")

    # Initialize an empty list to store the extracted words from the enumerated lines.
    extracted_words = []

    # Compile a regular expression pattern to match lines that start with a number followed by a period
    # and a space, capturing the subsequent word.
    pattern = re.compile(r"^\d+\.\s*(\w+)")

    # Iterate through each line in the split response.
    for line in lines:
        # Use the walrus operator (:=) to match the line against the compiled pattern.
        if match := pattern.match(line):
            # If a match is found, extract the captured word and append it to the list of extracted words.
            extracted_words.append(match.group(1))

    # Return the list of extracted words.
    return extracted_words


def extract_json_from_string(s: str):
    """
    Extracts a JSON string from a given input string using a regular expression. This function searches for a valid JSON
    structure within the input and attempts to fix any issues before returning the JSON string along with a flag
    indicating whether a fix was applied.

    Args:
        s (str): The input string from which to extract the JSON.

    Returns:
        tuple: A tuple containing the extracted JSON string and a flag indicating if a fix was applied. Returns None if
        no JSON structure is found.

    Raises:
        None
    """

    # Uncomment the following line to print the string being processed for JSON extraction from GPT for debugging
    # purposes.
    # print(f"Pre-JSON extraction string from GPT: {s}")

    # Compile a regular expression pattern to match a JSON structure, looking for an opening curly brace,
    # followed by any characters (non-greedily), and then a closing curly brace. The DOTALL flag allows
    # the dot to match newline characters as well.
    pattern = re.compile(r"\{.*?\}", re.DOTALL)

    # Use the walrus operator (:=) to find all matches of the pattern in the input string 's'.
    if match := pattern.findall(s):
        # If a match is found, extract the first JSON string from the list of matches.
        json_str = match[0]

        # Attempt to fix any issues with the extracted JSON string and return the fixed string along with a flag.
        json_str, flag = try_fix_json(json_str)
        return json_str, flag

    # If no matches are found, return None to indicate that no valid JSON structure was extracted.
    return None


def try_fix_json(json_str):
    """
    Attempts to fix and parse a potentially malformed JSON string. This function tries to load the JSON string and, if a
    decoding error occurs, it attempts to correct common issues, such as missing quotation marks, before retrying the
    parsing.

    Args:
        json_str (str): The JSON string to be fixed and parsed.

    Returns:
        tuple: A tuple containing the parsed JSON object and a boolean indicating whether a fix was applied.

    Raises:
        None
    """

    try:
        # Attempt to parse the JSON string into a Python object and return it along with a flag indicating no fix was
        # needed.
        return json.loads(json_str), False
    except json.JSONDecodeError as e:
        # Capture the JSONDecodeError and convert the error message to a string for analysis.
        err_msg = str(e)

        # Check if the error message indicates a missing value in the JSON string.
        if "Expecting value" in err_msg:
            # Split the JSON string into two parts: before and after the error position.
            before_error = json_str[: e.pos]
            after_error = json_str[e.pos :]

            # Attempt to insert a missing quotation mark at the error position to fix the JSON string.
            fixed_json_str = f'{before_error}"{after_error}'

            # Recursively call the try_fix_json function to attempt to fix the modified JSON string.
            return try_fix_json(fixed_json_str)

    # If no fix was applied, return the original JSON string along with a flag indicating a fix was attempted.
    return json_str, True


def parse_json_garbage(s):
    """
    Parses a JSON string that may contain extraneous characters before the actual JSON structure. This function trims
    the input string to start from the first occurrence of a JSON opening character and attempts to load the JSON,
    returning the successfully parsed object or a partial object if a decoding error occurs.

    Args:
        s (str): The input string potentially containing JSON data.

    Returns:
        object: The parsed JSON object, or a partial object if an error occurs during parsing.

    Raises:
        None
    """

    # Slice the input string 's' to start from the first occurrence of a JSON opening character ('{' or '[').
    # The generator expression finds the index of the first character that matches the condition.
    s = s[next(idx for idx, c in enumerate(s) if c in "{[") :]

    # Attempt to parse the trimmed string as JSON and return the resulting object.
    try:
        return json.loads(s)
    # If a JSONDecodeError occurs, catch the exception and attempt to parse the string up to the position of the error.
    except json.JSONDecodeError as e:
        return json.loads(
            s[: e.pos]
        )  # Return the parsed object from the valid portion of the string.


def parse_location_description(text):
    """
    Parses a location description from a given text input and organizes the observations into a structured format. This
    function extracts the description type and associated locations, handling multiple lines of input to create a
    dictionary of observations.

    Args:
        text (str): The input text containing location descriptions, formatted with types and observations.

    Returns:
        dict: A dictionary where keys are description types and values are lists of corresponding observations.

    Raises:
        ValueError: If the input text cannot be split into the expected format.
    """

    # Split the input text into lines based on one or more newline characters.
    by_description_type = re.split("\n+", text)

    # Initialize an empty dictionary to store observations categorized by description type.
    new_observations = {}

    # Check if there is at least one line in the split text.
    if len(by_description_type) >= 1:
        # Extract the description type and location from the first line, stripping any whitespace.
        desc_type, loc = map(lambda x: x.strip(), by_description_type[0].split(":"))
        # strip leading/trailing whitespaces around each piece
        desc_type, loc = desc_type.strip(), loc.strip()
        # Store the location in the observations dictionary under the description type.
        new_observations[desc_type] = [loc]

        # Iterate over the remaining lines in the split text.
        for description in by_description_type[1:]:
            # Proceed only if the line is not empty.
            if description:
                try:
                    # Attempt to unpack the description into description type, player, and observed values.
                    desc_type, player, observed = map(
                        lambda x: x.strip(), description.split(":")
                    )
                    # strip leading/trailing whitespaces around each piece
                    desc_type, player, observed = (
                        desc_type.strip(),
                        player.strip(),
                        observed.strip(),
                    )
                except ValueError:
                    # If unpacking fails due to insufficient values, handle the error.
                    # Assume the first part is the description type and indicate no observations.
                    desc_type = description.split(":")[0]
                    # strip leading/trailing whitespaces
                    desc_type = desc_type.strip()
                    new_observations[desc_type] = [f"No {desc_type}"]
                    continue  # Skip to the next line.

                # Check if the observed value contains parentheses, indicating a specific format.
                if "(" in observed:
                    # Split the observed values by semicolon and format them with the player name.
                    new_observations[desc_type] = [
                        f"{player} {obs.strip()}" for obs in observed.split(";") if obs
                    ]
                else:
                    # Split the observed values by comma and format them with the player name.
                    new_observations[desc_type] = [
                        f"{player} {obs.strip()}" for obs in observed.split(",") if obs
                    ]

    # Return the dictionary containing the categorized observations.
    return new_observations


def find_difference_in_dict_lists(dict1, dict2):
    """
    Finds the differences between two dictionaries containing lists as values. This function compares the values
    associated with matching keys in both dictionaries and returns a new dictionary containing the elements from the
    second dictionary that are not present in the first.

    Args:
        dict1 (dict): The first dictionary to compare.
        dict2 (dict): The second dictionary to compare.

    Returns:
        dict: A dictionary containing the keys from dict2 and their corresponding values that differ from dict1.

    Raises:
        TypeError: If both dictionaries are None, indicating they are not comparable.
    """

    # Check if the first dictionary is None.
    if dict1 is None:
        # If the first dictionary is None, check if the second dictionary is also None.
        if dict2 is None:
            # Raise a TypeError if both dictionaries are None, indicating they cannot be compared.
            raise TypeError(f"{type(dict2)} is not comparable.")
        else:
            # If only the first dictionary is None, return the second dictionary as the difference.
            return dict2

    # Initialize an empty dictionary to store the differences between the two dictionaries.
    diff = {}

    # Iterate through each key and value in the second dictionary.
    for key, value2 in dict2.items():
        # Iterate through each description in the list of values for the current key in dict2.
        for description2 in value2:
            # Check if the current key exists in the first dictionary.
            if key in dict1:
                # If the key exists, check if the current description doesn't match against all in the list from dict1.
                if all(description2 != desc1 for desc1 in dict1[key]):
                    diff[key] = [description2]
            else:
                # If the key doesn't exist in the first dictionary, add the key and its value from dict2 to the
                # differences.
                diff[key] = [value2]

    # Return the dictionary containing the differences between the two dictionaries.
    return diff


def enumerate_dict_options(options, names_only=False, inverted=False):
    """
    Generates a formatted string of enumerated options from a dictionary, with the ability to customize the output based
    on the provided parameters. This function can return either just the names of the options or include both keys and
    values, and it can also handle inverted mappings. Used by GPT-pick an option. Expects keys are descriptions and
    values are the name of the corresponding option.

    Args:
        options (dict): A dictionary of options to enumerate: Dict[description of option, option_name].
        names_only (bool, optional): If True, only the names (keys) will be included in the output. Defaults to False.
        inverted (bool, optional): If True, the enumeration will use the keys as values and vice versa. Defaults to
        False.

    Returns:
        tuple: A tuple containing the formatted string of options and a list of the dictionary keys.

    Raises:
        None
    """

    # Create a list of the keys from the options dictionary.
    options_list = list(options.keys())

    # Initialize an empty string to accumulate the formatted choices.
    choices_str = ""

    # Check if only the names of the options should be included in the output.
    if names_only:
        # Handle cases where the dictionary is structured as description of option: option_name.
        if inverted:
            # Enumerate through the keys (option names) of the options dictionary and create a numbered list of names.
            for i, name in enumerate(options.keys()):
                choices_str += "{i}. {n}\n".format(i=i, n=name)
        else:
            # Enumerate through the values of the options dictionary and create a numbered list of names.
            for i, name in enumerate(options.values()):
                choices_str += "{i}. {n}\n".format(i=i, n=name)
        # Return the formatted choices string and None since no keys are needed.
        return choices_str, None
    else:
        # If names_only is False, create a numbered list that includes both values and keys.
        for i, (k, v) in enumerate(options.items()):
            choices_str += "{i}. {v}: {k}\n".format(i=i, v=v, k=k)
        # Return the formatted choices string along with the list of option keys.
        return choices_str, options_list


def combine_dicts_helper(existing, new):
    """
    Combines two dictionaries by merging values from the new dictionary into the existing one. If a key already exists
    in the existing dictionary, the function extends the list of values; otherwise, it adds the new key-value pair.

    Args:
        existing (dict): The dictionary to which new values will be added.
        new (dict): The dictionary containing new key-value pairs to be merged.

    Returns:
        dict: The updated existing dictionary after merging with the new dictionary.

    Raises:
        None
    """

    # Iterate through each key-value pair in the new dictionary.
    for k, v in new.items():
        # Check if the current key already exists in the existing dictionary.
        if k in existing:
            # If the key exists, extend the list of values in the existing dictionary with the new values.
            existing[k].extend(v)
        else:
            # If the key does not exist, add the key-value pair from the new dictionary to the existing dictionary.
            existing[k] = v

    # Return the updated existing dictionary after merging with the new dictionary.
    return existing


def get_text_embedding(text, model="text-embedding-3-small", *args):
    """
    Generates a text embedding (calls the OpenAI embeddings api) for the provided input text using a specified model.
    This function utilizes the OpenAI client to create an embedding vector, which can be used for various natural
    language processing tasks.

    Args:
        text (str): The input text for which to generate the embedding.
        model (str, optional): The model to be used for generating the embedding. Defaults to "text-embedding-3-small".
        *args: Additional arguments to be passed to the embedding creation method.

    Returns:
        np.ndarray: A (1, 1536) shape NumPy array representing the text embedding vector, or None if the input text is
        empty.

    Raises:
        None
    """

    """
    Calls the OpenAI embeddings api

    Args:
        text (str): text to embed
        model (str, optional): the embedding model to use. Defaults to "text-embedding-3-small".

    Returns:
        np.array (1, 1536): array of embeddings 
    """

    # Check if the input text is empty; if so, return None to indicate no embedding can be generated.
    if not text:
        return None

    # Set up the OpenAI client for the specified organization.
    client = set_up_openai_client(org="Penn")

    # Create an embedding for the input text using the specified model and additional arguments.
    # The embedding is extracted from the response data returned by the OpenAI API.
    text_vector = (
        client.embeddings.create(input=[text], model=model, *args).data[0].embedding
    )

    # Convert the embedding to a NumPy array for further processing or analysis.
    return np.array(text_vector)


def create_dirs(fp):
    """
    Creates all necessary directories for the specified file path. This function ensures that the directory structure
    exists, creating any missing directories as needed.

    Args:
        fp (str): The file path for which to create the necessary directories.

    Returns:
        None
    """

    # Create all necessary directories for the specified file path 'fp'.
    # The os.path.dirname(fp) function retrieves the directory name from the file path.
    # The exist_ok=True parameter allows the function to succeed without raising an error
    # if the target directory already exists.
    os.makedirs(os.path.dirname(fp), exist_ok=True)
