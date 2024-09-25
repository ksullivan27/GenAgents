import logging

# local imports
from .logging_setup import setup_logger


class CustomLogger():
    """
    CustomLogger is a class that sets up a logger for tracking experiments and simulations. It initializes the logger
    with a specified experiment name and simulation ID, providing methods to retrieve the logger instance and the
    simulation ID.

    Args:
        experiment_name (str): The name of the experiment for which the logger is being set up.
        sim_id (str): The simulation identifier used to distinguish between different simulations.

    Methods:
        get_logger(): Returns the logger instance for logging messages.
        get_simulation_id(): Returns the validated simulation ID associated with the logger.

    Returns:
        None
    """

    def __init__(self, experiment_name, sim_id):
        """
        Initializes a CustomLogger instance by setting up a logger for a specified experiment and simulation. This
        constructor validates the simulation ID and creates a logger instance for logging messages related to the
        experiment.

        Args:
            experiment_name (str): The name of the experiment for which the logger is being set up.
            sim_id (str): The simulation identifier used to distinguish between different simulations.

        Returns:
            None
        """

        _, validated_id = setup_logger(experiment_name, sim_id)

        self.simulation_id = validated_id
        self.logger = logging.getLogger("survivor_global_logger")

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
