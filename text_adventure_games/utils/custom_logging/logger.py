import logging

# local imports
from .logging_setup import setup_logger


class CustomLogger():
    """
    CustomLogger is a class that sets up a logger for tracking experiments and simulations. It initializes a logger
    with a specified experiment name and simulation ID, and provides methods to access the logger instance and the
    associated simulation ID.

    Args:
        name (str): The name of the logger.
        experiment_name (str): The name of the experiment for which the logger is being configured.
        simulation_id (str): The unique identifier for the simulation, used to differentiate between various simulations.

    Methods:
        get_logger(): Returns the logger instance for logging messages.
        get_simulation_id(): Returns the validated simulation ID associated with the logger.

    Returns:
        None
    """

    def __init__(self, name, experiment_name, simulation_id, logfile_prefix=None, overwrite=True):
        """
        Initializes a CustomLogger instance by setting up a logger for a specified experiment and simulation. This
        constructor validates the provided simulation ID and creates a logger instance for logging messages related to
        the experiment.

        Args:
            name (str): The name of the logger.
            experiment_name (str): The name of the experiment for which the logger is being configured.
            simulation_id (str): The unique identifier for the simulation.
            logfile_prefix (str, optional): A prefix for the log file name. Defaults to None, which uses the experiment
            name.
            overwrite (bool, optional): Indicates whether to overwrite existing log files. Defaults to True.

        Returns:
            None
        """

        self.logger, self.simulation_id = setup_logger(
            name=name,
            experiment_name=experiment_name,
            simulation_id=simulation_id,
            logfile_prefix=logfile_prefix,
            overwrite=overwrite,
        )
        self.experiment_name = experiment_name

    def get_logger(self):
        """
        Retrieves the logger instance associated with the CustomLogger. This method allows access to the logger for
        logging messages related to the experiment and simulation.

        Returns:
            logging.Logger: The logger instance for logging messages.
        """

        return self.logger

    def get_simulation_id(self):
        """
        Retrieves the validated simulation ID associated with the CustomLogger. This method provides access to the
        simulation ID for tracking and logging purposes.

        Returns:
            str: The validated simulation ID.
        """

        return self.simulation_id

    def get_experiment_name(self):
        """
        Retrieves the experiment name of the CustomLogger. This method provides access to the name of the experiment for
        logging messages related to the experiment.

        Returns:
            str: The name of the experiment.
        """

        return self.experiment_name
