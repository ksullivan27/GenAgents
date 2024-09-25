"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: logging_setup.py
Description: create a logger that exists for the entire program.
             Any external library logging should be captured as well, though we don't care about this info.
"""

# Import Union from the typing module to allow type hinting for variables that can hold multiple types.
from typing import Union

# Import PathLike from the os module to enable type hinting for objects that behave like file system paths.
from os import PathLike

# Import the os module for interacting with the operating system, including file and directory operations.
import os

# Import the json module for working with JSON data, including parsing and serialization.
import json

# Import the logging.config module to configure logging settings in the application.
import logging.config

# Import the pathlib module for object-oriented file system path manipulation.
import pathlib

# Import the datetime class from the datetime module and alias it as 'dt' for easier access to date and time functions.
from datetime import datetime as dt

# Local imports from the parent package, specifically functions related to directory and logging path management.
from ..consts import (
    get_root_dir,  # Function to retrieve the root directory path.
    get_custom_logging_path,  # Function to get the path for custom logging.
    get_output_logs_path,  # Function to get the path for output logs.
    validate_output_dir,  # Function to validate the output directory.
)


def setup_logger(
    experiment_name: str,
    simulation_id: int,
    # other_exp_info: dict
):
    """
    Sets up a logger for tracking experiments and simulation runs. This function configures the logging system based on
    a specified configuration file and prepares the log file paths for storing simulation data.

    Args:
        experiment_name (str): The name of the experiment being logged.
        simulation_id (int): The identifier for the specific simulation run.

    Returns:
        tuple: A tuple containing the configured logger and the validated simulation ID.

    Raises:
        FileNotFoundError: If the logging configuration file does not exist.
        KeyError: If the logging configuration file is not structured properly.
        ValueError: If there are issues with the logging configuration or paths.
    """

    # TODO: Determine the strategy for tracking experiments and simulation runs.
    # NOTE: Consider whether all runs for a single experiment should be logged to the same file,
    # and what the maximum file size should be in that case.

    # Retrieve the path for custom logging (the location where log files should be stored).
    logging_path = get_custom_logging_path()

    # Construct the path to the logging configuration file.
    logger_config = pathlib.Path(os.path.join(logging_path, "logger_config.json"))

    # Check if the logging configuration file exists; raise an error if it does not.
    if not os.path.exists(logger_config):
        raise FileNotFoundError(
            (
                """You must set up a logging configuration file called logging_setup.py to extract data from the """
                """simulation. This must be saved in the \"custom_logging\" directory."""
            )
        )

    # Open the logging configuration file and load its contents as a JSON object.
    with open(logger_config, "r") as log_cfg:
        config = json.load(log_cfg)

    # Create a new log path for the current experiment and simulation ID.
    new_log_path = os.path.join(
        get_output_logs_path(), f"logs/{experiment_name}-{simulation_id}/"
    )

    # Validate the output directory and determine if it should be overwritten.
    overwrite, validated_path, validated_id = validate_output_dir(
        new_log_path, experiment_name, simulation_id
    )

    # TODO: Update this logic as needed to maintain either one log per run or one log per experiment.
    experiment_logfile_name = os.path.join(
        validated_path, f"sim_{experiment_name}-{validated_id}.jsonl"
    )

    # NOTE: Keeping this static path for global warnings log file (a shared warnings log for the project).
    global_warnings_logfile = os.path.join(
        get_output_logs_path(), "logs/warnings/project_warnings.jsonl"
    )

    # Set the paths for the logs in the logging configuration.
    set_logs_paths(config, experiment_logfile_name, global_warnings_logfile, overwrite)

    # Attempt to configure logging using the loaded configuration.
    try:
        logging.config.dictConfig(config)
    except ValueError:
        # If a ValueError occurs, it may indicate that a module couldn't be found; attempt to amend the paths.
        # The default is to look for text adventure games, so try to prepend "SurvivorWorld" to the formatter and
        # handler paths.
        try:
            config["formatters"]["json"]["()"] = (
                "SurvivorWorld." + config["formatters"]["json"]["()"]
            )
            config["handlers"]["queue_handler"]["()"] = (
                "SurvivorWorld." + config["handlers"]["queue_handler"]["()"]
            )
            # Re-attempt to configure logging with the amended paths.
            logging.config.dictConfig(config)
        except KeyError as e:
            # If a KeyError occurs, print an error message indicating the logging config file structure is likely
            # incorrect.
            print("Your logging config file is likely not structured properly.")
            raise e
        except ValueError as e:
            # If a ValueError occurs again, print a message indicating potential issues with the logger's components.
            print(
                (
                    """Flexible path could not find components of your logger. Are you running from a Jupyter """
                    """notebook or run_experiment.py?"""
                )
            )
            print(
                (
                    """You should run this package by following the quickstart guidelines on github: """
                    """www.github.com/sthudium25/SurvivorWorld"""
                )
            )
            raise e
    finally:
        # Retrieve the logger instance for global logging.
        logger = logging.getLogger("survivor_global_logger")

    # Return the configured logger and the validated simulation ID.
    return logger, validated_id


def set_logs_paths(
    logs_config,
    experiment_log_path: Union[str, PathLike],
    global_warnings_logfile: Union[str, PathLike],
    overwrite: bool = False,
) -> None:
    """
    Sets the file paths for logging configurations, including paths for experiment logs and global warning logs. This
    function ensures that the specified directories exist, handles potential overwriting of existing log files, and
    updates the logging configuration with the appropriate file paths. It assumes that path begins with the "logs"
    directory.

    Args:
        logs_config (dict): The logging configuration dictionary to be updated with file paths.
        experiment_log_path (Union[str, PathLike]): The path for the experiment log file.
        global_warnings_logfile (Union[str, PathLike]): The path for the global warnings log file.
        overwrite (bool, optional): A flag indicating whether to overwrite existing log files. Defaults to False.

    Returns:
        None

    Raises:
        Exception: If the logging configuration file structure is incorrect or if the specified paths cannot be created.
    """

    # Attempt to set the filename for the JSON log handlers in the logging configuration.
    try:
        logs_config["handlers"]["file_json"]["filename"] = experiment_log_path
        logs_config["handlers"]["warning_json"]["filename"] = global_warnings_logfile
    except KeyError:
        # If a KeyError occurs, print an error message indicating that the logging configuration structure is incorrect.
        print(
            (
                """Your log config file structure is incorrect. Expects to set filename with: """
                """config["handlers"][<handler_name>]["filename"]"""
            )
        )
        # Raise an exception to halt execution, prompting the user to fix the configuration.
        raise Exception("Fix your log config file before running a simulation.")

    # Retrieve the root directory path, moving up three levels from the current directory.
    root = get_root_dir(n=3)

    # Create a list of directory paths for the experiment and global warning logs by extracting the directory names.
    dir_paths = map(
        os.path.dirname,
        [
            os.path.join(
                root, experiment_log_path
            ),  # Full path for the experiment log file.
            os.path.join(
                root, global_warnings_logfile
            ),  # Full path for the global warnings log file.
        ],
    )

    # Iterate through each directory path to ensure it exists.
    for log_dir in dir_paths:
        if not os.path.exists(log_dir):
            # If the directory does not exist, create it.
            # print("couldn't find path: ", dir_path)  # Uncomment for debugging purposes.
            os.makedirs(log_dir)

    # Check if the user has opted to overwrite the existing experiment log file.
    if overwrite and os.path.exists(experiment_log_path):
        # If overwriting is allowed, clear the contents of the existing log file.
        open(experiment_log_path, "w").close()
    # If not overwriting, check if the experiment log file exists.
    elif not os.path.exists(experiment_log_path):
        # If the file does not exist, create it to prevent file not found errors.
        open(experiment_log_path, "a").close()

    # # Check if the user has opted to overwrite the existing experiment log file.
    # if (not os.path.exists(experiment_log_path)) or (overwrite and os.path.exists(experiment_log_path)):
    #     # If the file does not exist, or if overwriting is allowed, clear the contents of the existing log file.
    #     open(experiment_log_path, "w").close()
