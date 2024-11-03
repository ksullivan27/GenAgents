"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: custom_logging_handlers.py
Description: define extensions of the base logging QueueHandler and QueueListener
             to allow for non-blocking writing of logging statements 
"""

# Import the atexit module to register cleanup functions that will be called upon normal program termination.
import atexit

# Import the logging module to provide a flexible framework for emitting log messages from Python programs.
import logging

# Import the logging.handlers module to access various logging handlers for different logging needs.
import logging.handlers

# Import the queue module to use queue data structures for thread-safe communication between threads.
import queue

# Import the threading module to create and manage threads for concurrent execution in the program.
import threading


class CustomQueueHandler(logging.handlers.QueueHandler):
    """
    CustomQueueHandler is a logging handler that manages log messages using a queue. It automatically initializes and
    starts a listener thread to process log messages asynchronously, ensuring efficient logging in multi-threaded
    applications. It's a subclass of QueueHandler to automatically manage the listener.

    Args:
        *args: Variable length argument list for initializing the parent QueueHandler.
        **kwargs: Arbitrary keyword arguments for initializing the parent QueueHandler.

    Methods:
        start_listener(): Sets up and starts the queue listener for processing log messages.
        stop_listener(): Stops the queue listener when the application exits.

    Returns:
        None
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a CustomQueueHandler instance by calling the parent class's initializer and automatically starting
        the listener thread for processing log messages. This setup ensures that log messages are handled asynchronously
        from the main application flow.

        Args:
            *args: Variable length argument list for initializing the parent QueueHandler.
            **kwargs: Arbitrary keyword arguments for initializing the parent QueueHandler.

        Returns:
            None
        """

        super().__init__(*args, **kwargs)
        # Initialize and start the listener thread automatically
        self.start_listener()

    def start_listener(self):
        """
        Sets up and starts the queue listener for processing log messages asynchronously. This method initializes the
        listener with the specified queue and handlers, and ensures that the listener is stopped when the application
        exits.

        Returns:
            None
        """

        # Set up the queue listener
        self.queue_listener = CustomQueueListener(self.queue, *self._handlers)
        self.queue_listener.start()

        # Stop the listener when the application exits
        atexit.register(self.stop_listener)

    def stop_listener(self):
        """
        Stops the queue listener that processes log messages. This method ensures that the listener is properly shut
        down, preventing any further log processing.

        Returns:
            None
        """

        self.queue_listener.stop()

    def _handlers(self):
        """
        Defines a method to retrieve logging handlers for the queue listener. This method currently serves as a
        placeholder and should be implemented to return actual logging handlers as needed.

        Returns:
            list: An empty list, indicating that no handlers are currently defined.
        """

        # Define how to get handlers; this is placeholder logic
        # Typically, you'd pass actual handlers to the listener
        return []


class CustomQueueListener(logging.handlers.QueueListener):
    """
    CustomQueueListener is a logging listener that processes log records from a queue in a separate thread. It provides
    methods to start and stop the listener, ensuring that log messages are handled asynchronously and efficiently. It is
    a subclass of QueueListener to add a stopping mechanism.

    Args:
        queue (queue.Queue): The queue from which log records will be retrieved.
        *handlers: Variable length argument list for initializing the parent QueueListener with specific handlers.

    Methods:
        start(): Starts the listener thread to monitor the queue for log records.
        stop(): Stops the listener thread and ensures it exits cleanly.
        enqueue_sentinel(): Enqueues a sentinel value to signal the thread to stop.

    Returns:
        None
    """

    def __init__(self, queue, *handlers):
        """
        Initializes a CustomQueueListener instance with a specified queue and optional logging handlers. This
        constructor sets up the listener to process log records from the queue and initializes a stopping condition for
        graceful termination.

        Args:
            queue (queue.Queue): The queue from which log records will be retrieved.
            *handlers: Variable length argument list for initializing the parent QueueListener with specific handlers.

        Returns:
            None
        """

        # Call the initializer of the parent class (a logging handler) to set up the queue listener with the provided
        # queue and any additional handlers.
        super().__init__(queue, *handlers)

        # Initialize an instance variable to track whether the listener is in the process of stopping.
        # This flag is set to False, indicating that the listener is currently active and not stopping.
        self._stopping = False

    def start(self):
        """
        Starts the listener thread for the CustomQueueListener, allowing it to process log records from the queue
        asynchronously. This method initializes the thread as a daemon, ensuring it will exit when the main program
        terminates.

        Returns:
            None
        """

        # Create a new thread that will run the _monitor method, which is responsible for monitoring the queue for log
        # records.
        self._thread = threading.Thread(target=self._monitor)

        # Set the thread as a daemon thread, meaning it will automatically close when the main program exits.
        self._thread.daemon = True  # Daemon thread exits with the program

        # Start the thread, allowing it to begin executing the _monitor method concurrently.
        self._thread.start()

    def stop(self):
        """
        Stops the listener thread for the CustomQueueListener, ensuring that it exits cleanly. This method sets a
        stopping condition, enqueues a sentinel value to signal the thread, and waits for the thread to terminate.

        Returns:
            None
        """

        # Set the stopping flag to True, indicating that the listener should stop processing log records.
        self._stopping = True

        # Enqueue a sentinel value in the queue to signal the monitoring thread that it should exit.
        # This allows the thread to finish its current operations and terminate gracefully. This allows _monitor() to
        # get past the thread's blocking get() operation, which would otherwise wait for something to get added to the
        # queue before getting back to the top of the loop to check the stopping flag.
        self.enqueue_sentinel()

        # Wait for the monitoring thread to terminate before proceeding.
        # The join() method blocks the calling thread until the thread whose join() method is called is terminated.
        # It ensures that the main program waits for the listener thread to finish processing any remaining log records
        # and exit cleanly. Without this, the program could terminate while the listener thread is still running,
        # potentially causing log records to be lost or unhandled.
        self._thread.join()

    def _monitor(self):
        """
        Monitors the queue for incoming log records and processes them as they arrive. This method runs in a loop until
        a stopping condition is met, handling each log record retrieved from the queue. Essentially, it overrides the
        monitor method to add a stopping condition.

        Returns:
            None
        """

        # Continuously monitor the queue for log records until the stopping flag is set to True.
        while not self._stopping:
            try:
                # Retrieve a log record from the queue, blocking if necessary and timing out after 1 second.
                # The timeout can be adjusted based on the application's needs.
                record = self.queue.get(
                    block=True, timeout=1
                )  # Adjust timeout as needed

                # Process the retrieved log record using the handle method.
                self.handle(record)
            except queue.Empty:
                # If the queue is empty and a timeout occurs, continue the loop to check for new records.
                continue  # Timeout occurred, loop again

    def enqueue_sentinel(self):
        """
        Enqueues a sentinel value into the queue to signal the listener thread to stop processing. This method is used
        to ensure that the thread exits gracefully when requested.

        Returns:
            None
        """

        # Enqueue a sentinel value (None) into the queue to signal the listener thread to stop processing.
        # This is a common technique used to gracefully terminate threads that are waiting for new items in a queue.
        self.queue.put(None)


# Import specific classes and functions from the logging.config module.
# ConvertingList is used for handling lists in logging configurations.
from logging.config import (
    ConvertingList,
    # ConvertingDict, valid_ident
)

# Import QueueHandler and QueueListener from the logging.handlers module to manage logging in a multi-threaded
# environment.
from logging.handlers import QueueHandler, QueueListener

# Import the Queue class from the queue module to create a thread-safe queue for log records.
from queue import Queue

# Import the register function from the atexit module to register cleanup functions that will be called upon normal
# program termination.
from atexit import register


class QueueListenerHandler(QueueHandler):
    """
    QueueListenerHandler is a logging handler that manages a queue listener for processing log records. It initializes
    with specified handlers and can automatically start the listener, allowing for asynchronous logging in applications.

    Args:
        handlers (list): A list of logging handlers to be used by the listener.
        respect_handler_level (bool, optional): If True, respects the logging level of each handler. Defaults to False.
        auto_run (bool, optional): If True, automatically starts the listener upon initialization. Defaults to True.
        queue (Queue, optional): The queue used for log record processing. Defaults to an unbounded queue.

    Methods:
        start(): Starts the queue listener to begin processing log records.
        stop(): Stops the queue listener and ensures it exits cleanly.
        emit(record): Emits a log record to the appropriate handlers.
        _resolve_handlers(handlers_list): Resolves and returns a list of handlers from a ConvertingList or a standard
        list.

    Returns:
        None
    """

    def __init__(
        self, handlers, respect_handler_level=False, auto_run=True, queue=None
    ):
        """
        Initializes a QueueListenerHandler instance with specified logging handlers and configuration options. This
        constructor sets up the queue listener to process log records asynchronously and can automatically start the
        listener if desired.

        Args:
            handlers (list): A list of logging handlers to be used by the listener.
            respect_handler_level (bool, optional): If True, respects the logging level of each handler. Defaults to
            False.
            auto_run (bool, optional): If True, automatically starts the listener upon initialization. Defaults to True.
            queue (Queue, optional): The queue used for log record processing. Defaults to an unbounded queue. Defaults
            to None.

        Returns:
            None
        """

        if queue is None:
            queue = Queue(-1)

        # Call the initializer of the parent class, passing the queue to set up the logging handler.
        super().__init__(queue)

        # Resolve and prepare the logging handlers using the _resolve_handlers method.
        handlers = self._resolve_handlers(handlers)

        # Initialize a QueueListener with the specified queue and handlers, allowing it to process log records.
        # The respect_handler_level parameter determines whether to respect the logging level of each handler.
        self._listener = QueueListener(
            self.queue, *handlers, respect_handler_level=respect_handler_level
        )

        # If auto_run is True, start the listener and register the stop method to be called upon program termination.
        if auto_run:
            self.start()  # Start the listener to begin processing log records.
            register(
                self.stop
            )  # Register the stop method to ensure it is called when the program exits.

    def start(self):
        """
        Starts the queue listener for processing log records. This method initiates the listener, allowing it to begin
        handling log messages from the queue asynchronously.

        Returns:
            None
        """

        self._listener.start()

    def stop(self):
        """
        Stops the queue listener that processes log records. This method ensures that the listener is properly shut
        down, preventing any further log processing.

        Returns:¡
            None
        """

        self._listener.stop()

    def emit(self, record):
        """
        Emits a log record to the appropriate handlers. This method overrides the emit function of the parent class to
        ensure that log records are processed correctly by the configured handlers.

        Args:
            record (logging.LogRecord): The log record to be emitted.

        Returns:
            None
        """

        return super().emit(record)

    def _resolve_handlers(self, handlers_list):
        """
        Resolves and returns a list of logging handlers from a given handlers list. This method checks if the provided
        list is an instance of ConvertingList and, if so, evaluates it to return the actual handlers.

        Args:
            handlers_list (list): A list of handlers to be resolved.

        Returns:
            list: A list of resolved logging handlers.

        Raises:
            TypeError: If handlers_list is not a list or ConvertingList.
        """

        # Check if the provided handlers_list is not an instance of ConvertingList.
        # If it is not, return the handlers_list as it is, indicating no conversion is needed.
        if not isinstance(handlers_list, ConvertingList):
            return handlers_list

        # If handlers_list is a ConvertingList, create a new list by indexing into it.
        # This indexing operation evaluates the list, ensuring that all elements are properly converted.
        # Return the evaluated list of handlers.
        return [handlers_list[i] for i in range(len(handlers_list))]

    def print_handlers(self):
        """
        Prints the details of the handlers managed by the QueueListenerHandler.
        This method iterates over the handlers and prints their type and file path if applicable.

        Returns:
            None
        """
        # Access the handlers from the listener
        print("Handlers managed by QueueListenerHandler:")
        for handler in self._listener.handlers:
            print(f"Handler: {handler}")
            print(f"Handler type: {type(handler)}")
            if isinstance(handler, logging.FileHandler):
                print(f"FileHandler is saving JSON file at: {handler.baseFilename}")
            else:
                print("This handler does not save to a file.")
