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
    name: str,
    experiment_name: str,
    simulation_id: int,
    logfile_prefix: str = None,
    overwrite: bool = False,
    # other_exp_info: dict
):
    """
    Sets up a logger for tracking experiments and simulation runs. This function configures the logging system based on
    a specified configuration file and prepares the log file paths for storing simulation data.

    Args:
        experiment_name (str): The name of the experiment being logged.
        simulation_id (int): The identifier for the specific simulation run.
        logfile_prefix (str, optional): A prefix for the log file name. Defaults to None, which uses the experiment name.
        overwrite (bool, optional): Indicates whether to overwrite existing log files. Defaults to False.

    Returns:
        tuple: A tuple containing the configured logger and the validated simulation ID.

    Raises:
        FileNotFoundError: If the logging configuration file does not exist.
        KeyError: If the logging configuration file is not structured properly.
        ValueError: If there are issues with the logging configuration or paths.
    """

    # Check if logfile_prefix is not provided; if so, set it to the experiment name.
    if not logfile_prefix:
        logfile_prefix = experiment_name

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

    if not overwrite:
        # Validate the output directory and determine if it should be overwritten. If the directory exists, this asks
        # the user if they want to overwrite. If the directory does not exist, this creates the directory, and overwrite
        # is set to False.
        overwrite, validated_path, validated_id = validate_output_dir(
            new_log_path, experiment_name, simulation_id
        )
    # If overwrite is True, set the validated path to the new log path and the validated id to the simulation id.
    else:
        validated_path = new_log_path
        validated_id = simulation_id

    # TODO: Update this logic as needed to maintain either one log per run or one log per experiment.
    custom_logfile_name = os.path.join(
        validated_path, f"{logfile_prefix}_{experiment_name}-{validated_id}.jsonl"
    )

    # NOTE: Keeping this static path for global warnings log file (a shared warnings log for the project).
    global_warnings_logfile = os.path.join(
        get_output_logs_path(), "logs/warnings/project_warnings.jsonl"
    )

    # Set the paths for the logs in the logging configuration.
    set_logs_paths(
        config, name, custom_logfile_name, global_warnings_logfile, overwrite
    )

    # Attempt to configure logging using the loaded configuration.
    try:
        logging.config.dictConfig(config)
    except ValueError:
        # If a ValueError occurs, it may indicate that a module couldn't be found; attempt to amend the paths.
        # The default is to look for text adventure games, so try to prepend "GenAgents" to the formatter and
        # handler paths.
        try:
            # Iterate through all formatters and handlers in the config
            for section in ["formatters", "handlers"]:
                for key, value in config[section].items():
                    if isinstance(value, dict) and "()" in value:
                        # If the value is a dict and contains a '()' key, prepend 'GenAgents.'
                        config[section][key]["()"] = "GenAgents." + value["()"]

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
        custom_logger = logging.getLogger(name)

    # Return the configured logger and the validated simulation ID.
    return custom_logger, validated_id


def set_logs_paths(
    logger_name: str,
    logs_config: dict,
    custom_log_path: Union[str, PathLike],
    global_warnings_logfile: Union[str, PathLike],
    gpt_calls_logfile: Union[str, PathLike],
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
        gpt_calls_logfile (Union[str, PathLike]): The path for the GPT calls log file.
        overwrite (bool, optional): A flag indicating whether to overwrite existing log files. Defaults to False.

    Returns:
        None

    Raises:
        Exception: If the logging configuration file structure is incorrect or if the specified paths cannot be created.
    """

    # Attempt to set the filename for the JSON log handlers in the logging configuration.
    try:
        # Set to store the paths of the log files.
        paths_set = set()

        # Function to count the number of filename matches in the log handlers.
        def count_filename_matches(handler_name, log_path):
            handler = logs_config["handlers"].get(handler_name)
            match_count = 0
            if handler:
                if "filename" in handler:
                    match_count += 1
                if "handlers" in handler:
                    for sub_handler in handler["handlers"]:
                        match_count += count_filename_matches(sub_handler, log_path)
            return match_count

        # Function to update the filename of a log handler.
        def update_handler_filenames(handler_name, log_path, prepend):
            handler = logs_config["handlers"].get(handler_name)
            if handler:
                if "filename" in handler:
                    if prepend:
                        handler["filename"] = os.path.join(
                            os.path.dirname(log_path),
                            handler_name,
                            os.path.basename(log_path),
                        )
                        paths_set.add(handler["filename"])
                    else:
                        handler["filename"] = log_path
                        paths_set.add(handler["filename"])
                if "handlers" in handler:
                    for sub_handler in handler["handlers"]:
                        update_handler_filenames(sub_handler, log_path, prepend)

        # Update filenames for the specified logger.
        for name, logger_config in logs_config["loggers"].items():
            if name == logger_name:
                match_count = 0
                for handler_name in logger_config.get("handlers", []):
                    match_count += count_filename_matches(handler_name, custom_log_path)
                # To avoid overwriting the custom log file, prepend the handler name to the file name.
                prepend = match_count > 1
                for handler_name in logger_config.get("handlers", []):
                    update_handler_filenames(handler_name, custom_log_path, prepend)

        # Update filenames for the root logger.
        root_config = logs_config["loggers"].get("root", {})
        match_count = 0
        for handler_name in root_config.get("handlers", []):
            match_count += count_filename_matches(handler_name, global_warnings_logfile)
        # To avoid overwriting the global warnings log file, prepend the handler name to the file name.
        prepend = match_count > 1
        for handler_name in root_config.get("handlers", []):
            update_handler_filenames(handler_name, global_warnings_logfile, prepend)

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
                root, custom_log_path
            ),  # Full path for the experiment log file.
            os.path.join(
                root, global_warnings_logfile
            ),  # Full path for the global warnings log file.
        ]
        + [os.path.join(root, path) for path in paths_set],
    )

    # Iterate through each directory path to ensure it exists.
    for log_dir in dir_paths:
        if not os.path.exists(log_dir):
            # If the directory does not exist, create it.
            # print("couldn't find path: ", dir_path)  # Uncomment for debugging purposes.
            os.makedirs(log_dir)

    # Check if the user has opted to overwrite the existing experiment log file.
    if overwrite and os.path.exists(custom_log_path):
        # If overwriting is allowed, clear the contents of the existing log file.
        open(custom_log_path, "w").close()
    # If not overwriting, check if the experiment log file exists.
    elif not os.path.exists(custom_log_path):
        # If the file does not exist, create it to prevent file not found errors.
        open(custom_log_path, "a").close()

    # # Check if the user has opted to overwrite the existing experiment log file.
    # if (not os.path.exists(experiment_log_path)) or (overwrite and os.path.exists(experiment_log_path)):
    #     # If the file does not exist, or if overwriting is allowed, clear the contents of the existing log file.
    #     open(experiment_log_path, "w").close()
