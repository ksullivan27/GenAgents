"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: gpt_agent.py
Description: Methods that access the OPENAI API and make a call to GPT
"""

import re  # Import the regular expressions module for pattern matching and string manipulation.
import openai  # Import the OpenAI library to interact with the OpenAI API.

# Relative imports for utility functions and the GptCallHandler class.
from ..utils import (
    general,
)  # Import general utility functions from the parent directory.
from .gpt_helpers import (
    GptCallHandler,
)  # Import the GptCallHandler class from the current package.

# Initialize the GPT handler with specific parameters for API interaction.
GPT_HANDLER = GptCallHandler(
    **{
        "api_key_org": "Helicone",  # Set the organization identifier for the OpenAI API.
        "model": "gpt-4",  # Specify the model to be used for API calls.
        "max_tokens": 100,  # Define the maximum number of tokens to generate in the response.
        "temperature": 1,  # Set the sampling temperature for randomness in responses.
        "top_p": 1,  # Define the nucleus sampling parameter for controlling diversity.
        "max_retries": 5,  # Set the maximum number of retries for API calls in case of failure.
    }
)

DEFAULT_MODEL = "gpt-4o-mini"


def get_new_character_from_gpt(description, model: str = DEFAULT_MODEL):
    """
    Generates a new character based on a provided description using the GPT model. This function constructs a system
    prompt to guide the character generation and formats the output as a JSON structure.

    Args:
        description (str): A short description of the character to be generated.
        model (str, optional): The model to be used for generating the character. Defaults to the default model
        (e.g., "gpt-4o-mini").

    Returns:
        tuple: A tuple containing the generated character information in JSON format and any error encountered during
        JSON extraction.

    Raises:
        ValueError: If the response cannot be parsed into valid JSON.
    """

    # client = general.set_up_openai_client(org="Penn")
    GPT_HANDLER.update_params(max_tokens=200, temperature=1.25)

    system_prompt = (
        """You are a character generator. You should fill in the following character information based on a short """
        """description that will be provided. Create realistic, diverse characters.\n\n"""
        """Example prompt: A college student from New Mexico\n"""
        """Example Output: {"""
        """"Name": "Emily Sanchez", """
        """"Age": "20", """
        """"Likes": ["studying, "cinema"], """
        """"Dislikes": ["procrastination", "fast food"], """
        """"Occupation": "College Student", """
        """"Home city": "Albuquerque, New Mexico\""""
        """}\n\n"""
        """Create a JSON structure from the output."""
    )

    user_prompt = f"Create a character who fits this description: {description}"
    response = GPT_HANDLER.generate(system_prompt, user_prompt)
    GPT_HANDLER.reset_defaults()

    facts_json, error_in_json = general.extract_json_from_string(response)
    return facts_json, error_in_json


def get_trait_continuum(low: str, high: str, mid: str = None, model=DEFAULT_MODEL):
    """
    Generates a list of adjectives that represent a semantic continuum between two specified extremes. This function
    uses the GPT model to create a smooth transition of adjectives based on the provided low, high, and optional mid
    values.

    Args:
        low (str): The adjective representing the low end of the continuum.
        high (str): The adjective representing the high end of the continuum.
        mid (str, optional): An optional adjective representing the midpoint of the continuum. Defaults to None.
        model (str, optional): The model to be used for generating the adjectives. Defaults to the default model
        (e.g., "gpt-4o-mini").

    Returns:
        list: A list of adjectives that smoothly transition from low to high, potentially including the midpoint.

    Raises:
        ValueError: If the response cannot be parsed into a valid list of adjectives.
    """

    system_prompt = (
        """You will be provided two anchor words that represent the extremes on a semantic continuum. Consider one """,
        """end to have a score of 0 and the other a score of 100. For example: Evil=0 and Good=100. You may also """,
        """receive a third word which represents the midpoint of the continuum (e.g. neutral=50). Your job is to """,
        """fill in the scale with adjectives.""",
    )

    user_prompt = ""
    if mid:
        user_prompt += f"Provide a list of 15 adjectives that range from\
        'Low: {low}' to 'Mid: {mid}' to 'High: {high}' with a smooth transition in between."
    else:
        user_prompt += f"Provide a list of 15 adjectives that range from\
        'Low: {low}' to 'High: {high}' with a smooth transition in between."

    GPT_HANDLER.update_params(top_p=0.5)

    continuum = GPT_HANDLER.generate(system=system_prompt, user=user_prompt)
    GPT_HANDLER.reset_defaults()

    return general.extract_enumerated_list(continuum)


def get_target_adjective(
    low: str,
    high: str,
    target: int,
    model=DEFAULT_MODEL,
    low_int: int = 0,
    high_int: int = 100,
):
    """
    Generates a target adjective based on a specified position within a semantic continuum defined by two extremes. This
    function uses the GPT model to determine the appropriate adjective that corresponds to a given target score between
    the low and high values.

    Args:
        low (str): The adjective representing the low end of the continuum.
        high (str): The adjective representing the high end of the continuum.
        target (int): The target score along the continuum for which to find the corresponding adjective.
        model (str, optional): The model to be used for generating the adjective. Defaults to the default model
        (e.g., "gpt-4o-mini").
        low_int (int, optional): The integer score assigned to the low adjective. Defaults to 0.
        high_int (int, optional): The integer score assigned to the high adjective. Defaults to 100.

    Returns:
        str: The adjective that best describes the position of the target on the continuum.

    Raises:
        ValueError: If the response cannot be parsed into a valid adjective.
    """

    system_prompt = (
        """You will be provided two anchor words that represent the extremes on a semantic continuum. Consider one """
        f"""end to have a score of {low_int} and the other a score of {high_int}. You will then receive a target """
        """number somewhere along the scale. You should provide a single adjective that describes the position of """
        f"""the target on the continuum. For example: Evil={low_int} and Good={high_int} and target is """
        f"""{int((low_int + high_int) / 2)} --> predict: \"Neutral\""""
    )

    user_prompt = (
        f"""On a smooth transition scale from {low_int} = {low} to {high_int} = {high}, a target score of {target} """
        """is represented by the adjective: """
    )

    GPT_HANDLER.update_params(max_tokens=10, top_p=0.5)

    response = GPT_HANDLER.generate(system=system_prompt, user=user_prompt)
    GPT_HANDLER.reset_defaults()

    return general.extract_target_word(response)


def summarize_agent_facts(facts: str, model="gpt-4") -> str:
    """
    Generates a concise summary of an agent's characteristics based on provided facts. This function uses the GPT model
    to create a two-sentence description that captures the essence of the individual without merely listing their likes
    and dislikes.

    Args:
        facts (str): A string containing facts about a person, which will be summarized. model (str, optional): The
        model to be used for generating the summary. Defaults to "gpt-4".

    Returns:
        str: A concise summary of the person's core characteristics.

    Raises:
        ValueError: If the response cannot be parsed into a valid summary.
    """

    dummy_facts = {
        "Name": "Jacob Harrison",
        "Age": 25,
        "Likes": ["coffee brewing", "indie music", "baking", "dogs", "reading"],
        "Dislikes": [
            "rude customers",
            "early mornings",
            "negativity",
            "instant coffee",
        ],
        "Occupation": "Barista",
        "Home city": "Philadelphia, Pennsylvania",
    }

    # TODO: This asks for a two-sentence summary, but gives an example using four sentences. Which do we want?
    system_prompt = (
        """You will get a dictionary of traits that tell you about a person. You should write a concise, """
        """two-sentence summary of the person and describe their core characteristics without just listing the """
        """person's likes and dislikes.\n\n"""
        """The facts:\n"""
        f"""{dummy_facts}\n"""
        """are summarized as:\n"""
        """Jacob Harrison is a 25-year-old barista from Philadelphia who has a passion for creating delicious """
        """coffee blends and treats. He finds solace in indie music and enjoys spending his free time baking and """
        """getting lost in the pages of a good book. Jacob is a compassionate individual who values positivity and """
        """dislikes rude behavior or early mornings. His love for dogs adds a playful and nurturing aspect to his """
        """personality, creating a warm and inviting presence in both his professional and personal life."""
    )

    # TODO consider removing period (".") stop word to allow multiple sentences
    GPT_HANDLER.update_params(stop=".", max_tokens=100, presence_penalty=0.2)
    response = GPT_HANDLER.generate(system=system_prompt, user=facts)
    GPT_HANDLER.reset_defaults()

    summary = response.lower()

    # Remove the word "summary" from the response, if present, to clean up the output.
    summary = re.sub("summary:?", "", summary)

    return summary
