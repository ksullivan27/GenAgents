"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: consts.py
Description: get/set any necessary API keys, constant values, etc.
"""

# Importing the json module for working with JSON data, including parsing and serialization.
import json

# Importing the os module for interacting with the operating system, such as file and directory operations.
import os

# Importing PathLike from os to allow type hinting for objects that behave like file system paths.
from os import PathLike

# Importing Union from typing to enable type hinting for variables that can hold multiple types.
from typing import Union

# Import the logging module to enable logging functionality within this script
import logging

# Set up the logger at the module level
logger = logging.getLogger("utilities")


def get_root_dir(n=2) -> Union[str, PathLike]:
    """
    Retrieves the absolute path of the root directory by navigating up a specified number of parent directories from the
    current file's location. This function allows for flexible navigation of the directory structure based on the
    provided level of ascent. With respect to this file (consts.py), get root directories n levels above.

    Args:
        n (int, optional): The number of parent directories to traverse (climb). Defaults to 2.

    Returns:
        Union[str, PathLike]: The absolute path of the root directory.

    Raises:
        None
    """

    # Create a list of path components starting with the current file's path and adding 'n' instances of the parent
    # directory.
    path_components = [__file__] + [os.pardir] * n

    # Uncomment the following line to print the resolved root directory for debugging purposes.
    # print(f"ROOT DIR: {os.path.abspath(os.path.join(*path_components))}")

    # Join the provided path components into a single path and convert it to an absolute path.
    # This results in the absolute path of the root directory based on the specified number of parent directories to
    # traverse.
    return os.path.abspath(os.path.join(*path_components))


# TODO: helper method for finding any dir or file in this project
# def find_path(name: Union[str, PathLike]) -> Union[str, PathLike]:
#     """
#     Given a name or path of a directory or file, discover its path within this project.

#     Args:
#         name (Union[str, PathLike]): _description_

#     Returns:
#         Union[str, PathLike]: _description_
#     """


def get_config_file():
    """
    Retrieves configuration variables from a JSON configuration file. This function first attempts to locate a visible
    configuration file and, if not found, checks for a hidden configuration file, raising an error if neither is
    available.

    Returns:
        dict: A dictionary containing the configuration variables loaded from the JSON file.

    Raises:
        FileNotFoundError: If neither the visible nor hidden configuration file is found, indicating that the required
        configuration is missing.
    """

    # Construct the path to the visible configuration file by joining the root directory path with "config.json".
    config_path = os.path.join(get_root_dir(n=3), "config.json")

    # Check if the visible configuration file exists.
    if not os.path.exists(config_path):
        # If the visible config is not found, print a message and attempt to find the hidden config file.
        print("visible config not found, trying invisible option")

        # Construct the path to the hidden configuration file by joining the root directory path with ".config.json".
        config_path = os.path.join(get_root_dir(n=3), ".config.json")

        # Check if the hidden configuration file exists.
        if not os.path.exists(config_path):
            # Raise an error if neither configuration file is found, indicating that the required configuration is
            # missing.
            raise FileNotFoundError(
                'No config file found. Store your OpenAI key in a variable "OPENAI_API_KEY".'
            )

    # Open the configuration file for reading.
    with open(config_path, "r") as cfg:
        # Load the configuration variables from the JSON file into a dictionary.
        config_vars = json.load(cfg)

    # Return the loaded configuration variables.
    return config_vars


def get_openai_api_key(organization):
    # TODO: Specify Helicone vs Personal API key
    """
    Retrieves the OpenAI API key associated with a specified organization from the configuration file. This function
    checks the configuration for the organization and returns the corresponding API key, or indicates if the
    organization is not found.

    Args:
        organization (str): The name of the organization for which to retrieve the API key.

    Returns:
        str: The API key for the specified organization, or None if the organization is not found.

    Raises:
        None
    """

    # Retrieve the configuration variables from the configuration file.
    config_vars = get_config_file()

    # Use the walrus operator (:=) to assign the organization configuration to 'org_config'
    # while checking if it exists in the 'config_vars' dictionary.
    if org_config := config_vars.get("organizations", {}).get(organization, None):

        # Retrieve the API key from the organization's configuration.
        api_key = org_config.get("api_key", None)
        # Uncomment the following line to print a portion of the API key for debugging purposes.
        # print(f"{api_key[:5]}...")
        return api_key

    # If no matches are found for the specified organization, print a warning message.
    print(
        f"{organization} not found in list of valid orgs. You may not have a key set up for {organization}."
    )
    # Return None if the organization does not have an associated API key.
    return None


def get_models_config():
    """
    Retrieves the models configuration from the configuration file. This function
    checks the configuration for the 'models' section and returns it.

    Returns:
        dict: A dictionary containing the models configuration, or None if not found.

    Raises:
        None
    """

    # Retrieve the configuration variables from the configuration file.
    config_vars = get_config_file()

    # Get the 'models' section from the configuration.
    models_config = config_vars.get("models", None)

    if models_config is None:
        # Return None if the file doesn't include "models".
        print("No 'models' configuration found in the config file.")
        return None

    # Ensure that all required model types are present
    required_models = [
        "act",
        "goals",
        "impressions",
        "reflect",
        "retrieve",
        "dialogue",
        "vote",
    ]
    # Set up a default value if any are missing.
    default_model = "gpt-4o-mini"
    for model in required_models:
        if model not in models_config:
            logger.warning(
                f"'{model}' model configuration is missing. Defaulting to {default_model}."
            )
            models_config[model] = {"model": default_model}  # Set a default value

    return models_config


def get_helicone_base_path(organization="Helicone"):
    """
    Retrieves the base URL for the Helicone organization from the configuration file. This function checks if the
    specified organization is valid and returns the corresponding base URL, raising an error if the organization is not
    Helicone.

    Args:
        organization (str, optional): The name of the organization for which to retrieve the base URL. Defaults to
        "Helicone".

    Returns:
        str: The base URL for the specified organization, or None if the organization is not found in the configuration.

    Raises:
        ValueError: If the organization is not "Helicone", indicating that the method is only valid for that
        organization.
    """

    # TODO: Specify Helicone vs Personal API key

    # Check if the specified organization is "Helicone"; raise an error if it is not.
    if organization != "Helicone":
        raise ValueError("Method only valid for organization == 'Helicone'.")

    # Retrieve the configuration variables from the configuration file.
    config_vars = get_config_file()

    # If it exists, access the configuration for the specified organization from the loaded config variables.
    if org_config := config_vars.get("organizations", {}).get(organization, None):
        # Retrieve the base URL from the organization's configuration.
        return org_config.get("base_url", None)

    # If no matches are found for the specified organization, print a warning message.
    print(
        f"{organization} not found in list of valid orgs. You may not have a base url set up for {organization}."
    )
    # Return None if the organization does not have an associated base URL.
    return None


def get_assets_path() -> Union[str, PathLike]:
    """
    Retrieves the file path to the assets directory located two levels up from the current root directory. This function
    constructs the path by joining the root directory path with the "assets" folder.

    Returns:
        Union[str, PathLike]: The path to the assets directory as a string or a PathLike object.

    Raises:
        None
    """

    return os.path.join(get_root_dir(n=2), "assets")


def get_custom_logging_path():
    """
    Retrieves the file path to the custom logging directory located one level up from the current root directory. This
    function constructs the path by joining the root directory path with the "custom_logging" folder.

    Returns:
        str: The path to the custom logging directory as a string.

    Raises:
        None
    """

    return os.path.join(get_root_dir(n=1), "custom_logging")


def get_output_logs_path():
    """
    Retrieves the file path to the output logs directory located three levels up from the current root directory. This
    function calls another function to obtain the root directory and returns the path for the output logs.

    Returns:
        str: The path to the output logs directory as a string.

    Raises:
        None
    """

    return get_root_dir(n=3)


def validate_output_dir(fp, name, sim_id):
    """
    Validates the output directory for simulation logs, ensuring that it either does not exist or prompts the user for
    action if it does. This function checks if the specified file path exists and either increments the simulation ID
    for a new log path or allows the user to overwrite the existing log data.

    Args:
        fp (str): The file path to the output directory to validate.
        name (str): The name of the log file or directory.
        sim_id (int): The simulation identifier used in the log file name.

    Returns:
        tuple: A tuple containing a boolean indicating whether overwriting is allowed, the file path, and the simulation
        ID.

    Raises:
        None
    """

    # Check if the specified file path already exists.
    if os.path.exists(fp):
        # Print a blank line for better readability in the console output.
        print()

        # Prompt the user for input regarding whether to overwrite the existing log.
        decision = check_user_input(name, sim_id)

        # If the user decides not to overwrite the existing log.
        if not decision:
            print("Incrementing id...")
            # Construct a new log path by incrementing the simulation ID.
            new_log_path = os.path.join(
                get_output_logs_path(), f"logs/{name}-{sim_id+1}/"
            )
            # Recursively validate the new log path with the incremented simulation ID.
            return validate_output_dir(new_log_path, name, sim_id + 1)
        else:
            # Inform the user that the existing log file will be overwritten.
            print("Overwriting log file is data...")
            print(
                "The game data will be overwritten when you run `game.save_simulation_data()`"
            )

            # Return the overwrite flag, the original file path, and the current simulation ID.
            return True, fp, sim_id
    else:
        # If the file path does not exist, return the False overwrite flag, the file path, and the simulation ID.
        return False, fp, sim_id


def check_user_input(name, sim_id):
    """
    Prompts the user for confirmation on whether to overwrite existing data associated with a given name and simulation
    ID. This function handles user input and ensures that the decision to overwrite is made explicitly, with additional
    confirmation if the user chooses to proceed.

    Args:
        name (str): The name associated with the saved data.
        sim_id (int): The simulation identifier associated with the saved data.

    Returns:
        bool: True if the user confirms they want to overwrite the data, False otherwise.

    Raises:
        None
    """

    # Create a prompt message indicating that data has already been saved with the specified name and simulation ID.
    p1 = f"It appears you've already saved data using '{name}-{sim_id}. Do you want to overwrite the data?"

    # Create a secondary prompt message asking the user to type 'y' or 'n' for their decision.
    p2 = "Type 'y' or 'n'"

    # Prompt the user for input, displaying both messages.
    decision = input(f"{p1}\n{p2}\n")

    # Check if the user's input is neither 'y' nor 'n'.
    if decision not in ["y", "n"]:
        # If the input is invalid, recursively call the function to prompt the user again.
        return check_user_input()

    elif decision == "y":
        # Ask the user for a second confirmation to ensure they really want to overwrite.
        decision = input(
            "Are you REALLY sure you want to do this (it cannot be undone)??? y or n\n"
        )

        # If the user confirms again with 'y', return True to indicate overwriting is allowed. Otherwise return False
        # to indicate overwriting is not allowed.
        return decision == "y"
    else:
        # Return False to indicate that the overwrite action will not proceed.
        return False
