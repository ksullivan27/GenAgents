"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: setup_agent.py
Description: helper methods for agent setup
"""

print("Importing Build Agent")

import os
import json
from importlib.resources import files, as_file
import random
from typing import Dict, List, Literal
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# relative imports
print(f"{__name__} calling imports for Persona")
from ..agent.persona import Persona
print(f"{__name__} calling imports for TraitScale")
from ..managers.scales import TraitScale
print(f"{__name__} calling imports for GPT Agent Setup")
from ..gpt import gpt_agent_setup, gpt_helpers
print(f"{__name__} calling imports for General")
from .general import set_up_openai_client
print(f"{__name__} calling imports for Consts")
from . import consts, general

# from . import consts

GPT_RETRIES = 5
DEFAULT_MODEL = "gpt-4o-mini"


def find_similar_character(query, characters, top_n=1):
    """
    Finds the index of the most similar character to a given query based on cosine similarity. This function compares
    the query vector against a set of character vectors and returns the index of the top N most similar characters.

    Args:
        query (array-like): The vector representation of the query character.
        characters (dict): A dictionary where keys are character identifiers and values are their vector
        representations.
        top_n (int, optional): The number of top similar characters to return. Defaults to 1.

    Returns:
        int: The index of the most similar character in the characters dictionary.

    Raises:
        None
    """

    # Calculate the cosine similarity between the query vector and the vectors of all characters.
    # The query is reshaped to ensure it has the correct dimensions for the similarity calculation.
    sim = cosine_similarity(
        np.array(query).reshape(1, -1),
        np.array([np.array(v) for v in characters.values()]),
    )

    # Sort the similarity scores in descending order and retrieve the index of the top N most similar characters.
    # The sorted function enumerates the similarity scores, allowing us to sort by the similarity value.
    # Note that sim has the shape (1, n), where n is the number of characters. Thus, sim[0] is a 1D array of size n,
    # where each value is the cosine similarity score between the query vector and a corresponding character vector.
    idx = [
        i[0]
        for i in sorted(enumerate(sim[0]), key=lambda x: x[1], reverse=True)[:top_n]
    ]

    # Return the index of the most similar character.
    # TODO: Check if this should always return the most similar vs. a list of top_n most similar
    return idx[0]


def get_or_create_base_facts(description: str, make_new=False):
    """
    Retrieves or creates base facts (JSON) for a character based on a given description. This function either generates
    a new character using the GPT model if requested or compares the description to existing characters to
    find a match.

    Args:
        description (str): The description of the character for which to retrieve or create facts.
        make_new (bool, optional): A flag indicating whether to create a new character. Defaults to False.

    Returns:
        object: The character object that matches the description or is newly created.

    Raises:
        ValueError: If GPT fails to create a character after the specified number of retries.
    """

    print("GET OR CREATE BASE FACTS PLAYER DESCRIPTION:", description, make_new)

    # Check if a new character is requested by the user.
    if make_new:
        # Attempt to create a new character using the GPT model, retrying up to GPT_RETRIES times.
        for i in range(GPT_RETRIES):
            # Call the function to get a new character from GPT, passing the description and model.
            char, error_flag = gpt_agent_setup.get_new_character_from_gpt(
                description
            )
            # If an error occurs during character creation, log the retry attempt and continue.
            if error_flag:
                print(f"retry {i} for character setup: {description}")
                continue
            else:
                # If character creation is successful, return the newly created character.
                return char

        # If all retries fail, raise a ValueError indicating that character creation was unsuccessful.
        raise ValueError(
            "GPT failed to create a character with your description. Try something different."
        )

    # If a new character is not requested, proceed to compare the description to existing characters.
    else:
        try:
            # Attempt to retrieve the existing character facts from storage.
            characters = get_character_facts()
        except FileNotFoundError:
            # If no character presets are found, log a message and default to creating a new character.
            print("No character presets found. Defaulting to new character creation.")
            return get_or_create_base_facts(description, make_new=True)

        # Initialize a dictionary to hold the embedded representations of existing characters.
        embedded_characters = {}
        # Iterate over the existing characters to create their vector representations.
        for i, c in enumerate(characters):
            # Convert the character to a string and get its vector representation.
            c_vec = general.get_text_embedding(c.__str__())
            # Store the vector in the embedded_characters dictionary with the index as the key.
            embedded_characters[i] = c_vec

        # Get the vector representation of the requested character description.
        requested_vector = general.get_text_embedding(description)

        # Find the index of the character that is most similar to the requested description vector.
        idx = find_similar_character(
            query=requested_vector, characters=embedded_characters
        )
        # Return the character that matches the found index.
        # TODO: Alter this to get the top_n > 1 characters
        return characters[idx]


def create_persona(
    facts: Dict,
    trait_scores: List = None,
    archetype=None,
    model=DEFAULT_MODEL,
    file_path=None,
):
    """
    Creates a persona based on provided facts, trait scores, or an archetype. This function can either load an existing
    persona from a file or generate a new one, assigning traits and strategies based on the specified parameters.

    Args:
        facts (Dict): A dictionary containing the foundational facts for the persona.
        trait_scores (List, optional): A list of scores representing the persona's traits. Defaults to None.
        archetype (optional): An optional archetype to define the persona's characteristics. Defaults to None.
        model (str, optional): The GPT model to use for generating traits and adjectives. Defaults to "gpt-4o-mini".
        file_path (str, optional): The path to a file from which to load an existing persona. Defaults to None.

    Returns:
        Persona: The created or loaded persona object.

    Raises:
        FileNotFoundError: If the specified file path does not exist when loading a persona.
        ValueError: If neither trait_scores nor archetype is provided.
    """

    # A mapping of archetypes to their corresponding game theory strategies.
    # This dictionary defines how different archetypes are expected to behave in terms of strategy.
    archetype_game_theory_mapping = {
        "Hubris": "Backstabbing",  # Given their self-centered and assertive traits.
        "Villain": "Backstabbing",  # Villains are typically manipulative and self-serving.
        "Hero": "Cooperation",  # Heroes are often altruistic and collaborative.
        "Student": "Tit-for-Tat",  # Students are learners and may adapt their strategy based on others.
        "Leader": "Cooperation",  # Leaders are usually cooperative, aiming to unite and guide.
        "Damsel in Distress": "Cooperation",  # Likely to seek help and cooperate in situations.
        "Mother": "Cooperation",  # Embodying nurturing and caring traits, inclined to help and cooperate.
        "Warrior": "Backstabbing",  # Focused and combative, might prioritize individual goals over cooperation.
        "Sage Advisor": "Tit-for-Tat",  # Wise and adaptive, responding strategically to the actions of others.
    }

    # Check if a file path is provided to load an existing persona from a file.
    if file_path is not None:
        # Verify that the specified file path exists; raise an error if it does not.
        if not os.path.isfile(file_path):  # check that filepath exists
            raise FileNotFoundError(f"No file found at {file_path}")

        # Import and return the persona from the specified file.
        return Persona.import_persona(file_path)

    # If no file path is provided, create a new Persona instance using the provided facts.
    p = Persona(facts)

    # TODO: Property of Character (Troll, etc.) - Placeholder for future implementation.

    # If trait scores are provided, validate and create TraitScale instances for each.
    if trait_scores:
        scores = validate_trait_scores(
            trait_scores
        )  # Validate the provided trait scores.
        monitored_traits = (
            TraitScale.get_monitored_traits()
        )  # Retrieve the monitored traits.

        # Iterate through the validated scores and create TraitScale instances.
        for named_trait, score in zip(monitored_traits.items(), scores):
            name, dichotomy = named_trait  # Extract the trait name and its dichotomy.
            trait = TraitScale(
                name, dichotomy, score=score
            )  # Create a new TraitScale instance.

            # TODO: would be more cost/time effective to ask this to GPT once - Placeholder for optimization.
            trait.set_adjective(
                model
            )  # Set the adjective for the trait based on the model. Note that this only adds new traits. It doesn't
            # perform any trait alterations.
            p.add_trait(trait)  # Add the trait to the persona.

    # If trait scores aren't provided...
    else:
        # If an archetype isn't provided, randomly select one.
        if not archetype:
            archetype = random.choice(list(archetype_game_theory_mapping.keys()))
        # Retrieve the archetype's profile and create corresponding traits.
        profile = get_archetype_profiles(archetype)  # Get the archetype profile.

        # Iterate through the traits defined in the archetype profile.
        for scale in profile["traits"]:
            low, high, target, name = (
                scale["lowAnchor"],
                scale["highAnchor"],
                scale["targetScore"],
                scale["name"],
            )
            dichotomy = (low, high)  # Define the dichotomy for the trait.

            # Add wiggle/variance to the target score (+/- 5%).
            # Personas are only *instantiations* of archetypes, so they can vary.
            random_wiggle = np.random.uniform(-5, 5)  # Generate a random variance.
            target = target + target * random_wiggle / 100  # Adjust the target score.

            trait = TraitScale(
                name, dichotomy, score=target
            )  # Create a new TraitScale instance.

            # TODO: would be more cost/time effective to ask this to GPT once - Placeholder for optimization.
            trait.set_adjective(
                model=model
            )  # Set the adjective for the trait based on the model.
            p.add_trait(trait)  # Add the trait to the persona.

        # Set the game theory strategy based on the selected archetype.
        p.set_game_theory_strategy(
            archetype_game_theory_mapping[archetype]
        )  # Sets default strategy based on archetype.

        # Assign the archetype to the persona.
        p.set_archetype(archetype)

    # Return the fully constructed persona object.
    return p


def build_agent(
    agent_description,
    facts_new,
    trait_scores: List = None,
    archetype=None,
    model=DEFAULT_MODEL,
):
    """
    Builds an agent based on the provided description and associated facts. This function generates or retrieves the
    necessary facts for the agent, validates them, and creates a persona using the specified trait scores and archetype.

    Args:
        agent_description (str): A description of the agent to be created.
        facts_new (bool): A flag indicating whether to create new facts for the agent.
        trait_scores (List, optional): A list of scores representing the agent's traits. Defaults to None.
        archetype (optional): An optional archetype to define the agent's characteristics. Defaults to None.
        model (str, optional): The model to use for generating facts and personas. Defaults to Default Model, e.g.,
        "gpt-4o-mini".

    Returns:
        object: The created persona representing the agent.

    Raises:
        None
    """

    # Retrieve or create the base facts for the agent using the provided description.
    # The make_new flag indicates whether to generate new facts if they do not already exist.
    facts = get_or_create_base_facts(agent_description, make_new=facts_new)

    # Print the generated facts for debugging or informational purposes.
    print(f"Generated facts: {facts}")

    # Validate the retrieved facts to check for any missing keys.
    missing_keys = validate_facts(facts)

    # If there are any missing keys, attempt to randomly fill them with corresponding backup facts (name, age, and
    # occupation values).
    if len(missing_keys) > 0:
        # Iterate over the missing keys to retrieve backup facts for each category.
        # The backup facts are limited to specific categories such as name, age, and occupation.
        for k in missing_keys:
            facts[k] = get_backup_fact(k)

    # Create and return a persona for the agent using the validated facts, trait scores, and optional archetype.
    return create_persona(facts, trait_scores, archetype=archetype, model=model)

    # TODO: How to add affinities? We need the game information to know how many
    # characters exist in the world. This may need to happen later
    # Maybe once characters are set, there is a start up sequence that sets
    # all "affinities" in each persona.


def validate_facts(facts_dict):
    """
    Validates the presence of required keys in a dictionary of facts. This function checks if the input is a dictionary
    and verifies that it contains the necessary keys for a character's attributes, returning a list of any missing keys.

    Args:
        facts_dict (dict): The dictionary containing character facts to validate.

    Returns:
        list: A list of missing keys that are required for the character's attributes.

    Raises:
        TypeError: If the input is not a dictionary, indicating that the facts must be provided in the correct format.
    """

    # Define a list of required keys that must be present in the facts dictionary.
    required_keys = ["Name", "Age", "Occupation"]

    # Check if the provided facts_dict is a dictionary.
    if not isinstance(facts_dict, dict):
        # Print the contents of facts_dict for debugging purposes.
        print(facts_dict)
        # Raise a TypeError if facts_dict is not a dictionary, indicating the expected type.
        raise TypeError(f"facts must be a dictionary. Got {type(facts_dict)}")

    # Initialize an empty list to keep track of any missing required keys.
    missing = []

    # Iterate over the list of required keys to check their presence in the facts_dict.
    for k in required_keys:
        if k not in facts_dict:
            # If a required key is missing, print a message indicating which key is absent.
            print(f"facts is missing {k}")
            # Add the missing key to the list of missing keys.
            missing.append(k)

    # Return the list of missing keys.
    return missing


def validate_goals(goals_dict):
    """
    Validates and ensures the presence of specific goals in a goals dictionary. This function checks for flexible,
    short-term, and long-term goals, adding default values for short-term and long-term goals if they are not already
    set.

    Args:
        goals_dict (dict): A dictionary containing the goals to validate.

    Returns:
        dict: The updated goals dictionary, including any newly added default goals.

    Raises:
        None
    """

    # Check if the "flex" goal is not present in the goals dictionary.
    if "flex" not in goals_dict.keys():
        # Print a message indicating that no flexible goal has been set.
        print("No flexible goal set.")

    # Check if the "short-term" goal is not present in the goals dictionary.
    if "short-term" not in goals_dict.keys():
        # TODO: Modify this goal wording as needed
        # Set a default short-term goal if it is missing.
        goals_dict["short-term"] = (
            "Gain the trust of others. Find allies to prevent yourself from being voted off the island."
        )

    # Check if the "long-term" goal is not present in the goals dictionary.
    if "long-term" not in goals_dict.keys():
        # TODO: Modify this goal wording as needed
        # Set a default long-term goal if it is missing.
        goals_dict["long-term"] = (
            "Develop strong alliances. Position yourself to win the game of Survivor."
        )

    # Return the updated goals dictionary, which now includes any newly added goals.
    return goals_dict


def validate_trait_scores(scores):
    """
    Validates and adjusts a list of trait scores to ensure it contains a fixed number of scores. If the provided list
    has fewer than the required number of scores, the function fills in the missing values with random scores, ensuring
    all scores are within the acceptable range.

    Args:
        scores (list): A list of trait scores to validate and adjust.

    Returns:
        list: A list of validated and adjusted trait scores, clipped to be within the range of 0 to 100.

    Raises:
        None
    """

    # Determine the number of scores provided in the input list.
    nscores = len(scores)

    # Define the expected number of traits being measured; currently set to 9.
    # This value can be modified to accommodate a dynamic number of traits in the future.
    # Here, we check if the number of provided scores matches the expected count.
    if nscores != 9:
        # If the number of scores is less than expected, print a message indicating how many were provided.
        print(
            f"Only {nscores} for the character were provided. Filling others randomly..."
        )

        # Generate random scores to fill in the missing values, ensuring they are between 0 and 100.
        rand_scores = np.random.randint(0, 100, size=(9 - nscores))

        # Extend the original scores list with the newly generated random scores.
        scores.extend(rand_scores.tolist())

    # Clip the scores to ensure all values are within the range of 0 to 100.
    scores = np.clip(scores, 0, 100).tolist()

    # Return the adjusted list of scores.
    return scores


def get_archetype_profiles(target: str) -> Dict:
    """
    Retrieves the archetype profile corresponding to a specified target name. This function loads archetype data from a
    JSON file and searches for a matching archetype, returning the profile if found.

    Args:
        target (str): The name of the archetype to retrieve.

    Returns:
        Dict: The archetype profile as a dictionary if found, or None if no matching archetype exists.

    Raises:
        None
    """

    # Retrieve the path to the assets directory using a function from the consts module.
    asset_path = consts.get_assets_path()

    # Construct the full path to the archetypes JSON file by joining the assets path with the filename.
    asset_path = os.path.join(asset_path, "archetypes.json")

    # Open the archetypes JSON file for reading.
    with open(asset_path, "r") as f:
        # Load the contents of the JSON file into a Python dictionary.
        profiles = json.load(f)

    # Return the first archetype from the profiles that matches the target name.
    # If no match is found, return None.
    return next(
        (atype for atype in profiles["archetypes"] if target == atype["name"]),
        None,
    )


def get_character_facts():
    """
    Retrieves character facts from a JSON file containing character data. This function loads the character facts from
    the specified file and returns them as a dictionary.

    Returns:
        dict: A dictionary containing the character facts loaded from the JSON file.

    Raises:
        FileNotFoundError: If the character facts file does not exist or cannot be opened.
    """

    # Retrieve the path to the assets directory using a function from the consts module.
    asset_path = consts.get_assets_path()

    # Construct the full path to the character facts JSON file by joining the assets path with the filename.
    # TODO: shouldn't this use "character_traits.json"? Originally, "character_facts.json".
    asset_path = os.path.join(asset_path, "character_traits.json")

    # Open the character facts JSON file for reading.
    with open(asset_path, "r") as f:
        # Load the contents of the JSON file into a Python dictionary.
        characters = json.load(f)

    # Return the dictionary containing the character facts.
    return characters


def get_backup_fact(key):
    """
    Retrieves a random backup fact associated with a specified key from a JSON file containing backup facts. This
    function loads the backup facts from the file and returns a randomly selected fact for the given key.

    Args:
        key (str): The key for which to retrieve a backup fact.

    Returns:
        str: A randomly selected backup fact corresponding to the specified key.

    Raises:
        KeyError: If the specified key does not exist in the backup facts.
        FileNotFoundError: If the backup facts file cannot be found or opened.
    """

    # Retrieve the path to the assets directory using a function from the consts module.
    asset_path = consts.get_assets_path()

    # Construct the full path to the backup facts JSON file by joining the assets path with the filename.
    asset_path = os.path.join(asset_path, "backup_fact_lists.json")

    # Open the backup facts JSON file for reading.
    with open(asset_path, "r") as f:
        # Load the contents of the JSON file into a Python dictionary.
        facts = json.load(f)

        # Print the loaded facts for debugging or informational purposes.
        print(facts)

    # Return a randomly selected backup fact corresponding to the specified key from the facts dictionary.
    return np.random.choice(facts[key])
