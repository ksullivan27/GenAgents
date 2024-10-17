"""
Author: Samuel Thudium (sam.thudium1@gmail.com)
"""

# Import necessary modules and classes from the standard library and third-party packages.
from dataclasses import asdict, dataclass, field  # For creating data classes.
import json  # For handling JSON data.
import logging  # For logging messages.
import os  # For interacting with the operating system.
import re  # For regular expression operations.
import time  # For time-related functions.
from typing import ClassVar  # For type hinting class variables.
import openai  # For interacting with the OpenAI API.
import tiktoken  # For tokenization of text.
import httpx  # For making HTTP requests.
import numpy as np  # Importing numpy for numerical operations and array handling.

# Local imports from the project's utility modules.
from ..utils.general import (
    enumerate_dict_options,
)  # Function to enumerate dictionary options.
from ..utils.consts import get_config_file, get_assets_path
from ..utils.general import get_logger_extras
from ..assets.prompts import (
    gpt_helper_prompts as hp,
)  # Importing predefined prompts for GPT helper.
from ..games import Game
from ..things.characters import Character

# Set up a logger for this module to log messages with the module's name.
logger = logging.getLogger(__name__)


class ClientInitializer:
    """
    ClientInitializer is responsible for managing the initialization and retrieval of OpenAI clients based on provided
    API keys and organization information. It ensures that valid parameters are used and maintains a count of how many
    times the API keys have been loaded.

    Attributes:
        VALID_CLIENT_PARAMS (set): A set of valid client parameters that can be used for client initialization.
        load_count (int): The number of times the API keys have been loaded.
        api_info (dict): The API key information loaded from the configuration.
        clients (dict): A dictionary storing the initialized clients for each organization.

    Methods:
        get_client(org):
            Retrieves the client for the specified organization, initializing it if necessary.

        set_client(org):
            Sets up the client for the specified organization using the provided API parameters.
    """

    # class attribute containing all valid client parameters
    VALID_CLIENT_PARAMS = set(
        [
            "api_key",
            "organization",
            "base_url",
            "timeout",
            "max_retries",
            "default_headers",
            "default_query",
            "http_client",
        ]
    )

    def __init__(self):
        """
        Initializes a new instance of the ClientInitializer class. This constructor sets up the initial state by loading
        API keys and initializing the client storage.

        Attributes:
            load_count (int): The number of times the API keys have been loaded.
            api_info (dict): The API key information loaded from the configuration.
            clients (dict): A dictionary to store initialized clients for different organizations.
        """

        # Initializes a counter to track how many times the API keys have been loaded.
        self.load_count = 0

        # Loads the API keys from the configuration file and stores the information in api_info.
        self.api_info = self._load_api_keys()

        # Initializes an empty dictionary to hold client instances for different organizations.
        self.clients = {}

    def _load_api_keys(self):
        """
        Loads the API keys from the configuration file and increments the load count. This method retrieves the
        organization information associated with the API keys, returning it as a JSON object.

        Returns:
            dict or None: The organizations value from the configuration if it exists, otherwise None.
        """

        # Increments the count of how many times the API keys have been loaded, tracking usage.
        self.load_count += 1

        # Retrieves the content of the configuration file, which contains the OpenAI API key. This configuration file
        # must be named "config_file" and is expected to be in JSON format.
        configs = get_config_file()

        # Returns the value associated with the "organizations" key from the configuration. If the key does not exist,
        # it returns None.
        return configs.get("organizations", None)

    def get_client(self, org):
        """
        Retrieves the client associated with the specified organization. If the client does not exist, it initializes
        the client for the organization before returning it.

        Args:
            org (str): The organization identifier for which to retrieve the client.

        Returns:
            Client: The client instance associated with the specified organization.
        """

        try:
            # Attempts to retrieve the client associated with the specified organization from the clients dictionary.
            return self.clients[org]  # Returns the found client instance.
        except KeyError:
            # If the organization client does not exist, initialize the client for the organization.
            self.set_client(org)
            # Recursively call get_client to retrieve the newly created client.
            return self.get_client(org)

    def set_client(self, org):
        """
        Sets up the client for the specified organization using the provided API parameters. This method validates the
        API parameters and initializes the OpenAI client if the organization is correctly configured.

        Args:
            org (str): The organization identifier for which to set up the client.

        Raises:
            AttributeError: If the API information has not been initialized correctly.
            ValueError: If no API key has been set up for the specified organization.
        """

        # Check if the API information has been initialized; if not, raise an error indicating improper initialization.
        if not self.api_info:
            raise AttributeError("api_info may not have been initialized correctly")

        try:
            # Attempt to retrieve the API parameters for the specified organization from the API information.
            org_api_params = self.api_info[org]
        except KeyError:
            # If the organization key does not exist, raise an error indicating that the API key has not been set up.
            raise ValueError(
                f"You have not set up an api key for {org}. Valid orgs are: {list(self.api_info.keys())}"
            )
        else:
            # Validate the retrieved API parameters to ensure they are correct and conform to expected values.
            valid_api_params = self._validate_client_params(org_api_params)
            # Initialize the OpenAI client for the organization using the validated parameters.
            self.clients[org] = openai.OpenAI(**valid_api_params)

    def _validate_client_params(self, params):
        """
        Validates the parameters provided for client initialization, ensuring that only valid parameters are included.
        This method checks for the presence of required parameters and removes any invalid ones before returning the
        validated set.

        Args:
            params (dict): A dictionary of parameters to validate for client initialization.

        Returns:
            dict: A dictionary containing only the valid parameters for client initialization.

        Raises:
            ValueError: If the 'api_key' is missing from the provided parameters.
        """

        # Initializes a dictionary to hold valid parameters after validation. Remove any invalid parameters that were
        # attempted to add.
        validated_params = {}

        # Check if the 'api_key' is present in the provided parameters; if not, raise an error.
        if "api_key" not in params:
            raise ValueError("'api_key' must be included in your config.")

        # Load the model limits from the JSON file
        asset_path = get_assets_path()
        model_limits_path = os.path.join(asset_path, "openai_model_limits.json")
        with open(model_limits_path, "r") as f:
            model_limits = json.load(f)

        # Check if 'model' is present in the provided parameters
        if "model" in params:
            # Check if the provided model is in the model_limits under the "LLM" key
            if params["model"] not in model_limits["LLM"]:
                raise ValueError(
                    f"The provided model '{params['model']}' is not a valid OpenAI LLM model."
                )

        # Check if 'embedding_model' is present in the provided parameters
        if "embedding_model" in params:
            # Check if the provided embedding model is in the model_limits under the "Embeddings" key
            if params["embedding_model"] not in model_limits["Embeddings"]:
                raise ValueError(
                    f"The provided embedding model '{params['embedding_model']}' is not a valid OpenAI Embedding model."
                )

        # If 'timeout' is not specified, set a default timeout for the connection and read/write operations.
        if "timeout" not in params:
            # Limit connection to 15 seconds and read/write to 60 seconds.
            params["timeout"] = httpx.Timeout(60, connect=15)

        # Iterate through the provided parameters to validate them.
        for k, v in params.items():
            # If the parameter is not in the set of valid client parameters, log a message and skip it.
            if k not in self.VALID_CLIENT_PARAMS:
                print(f"{k} is not a valid argument to openai.OpenAI(). Removing {k}.")
            else:
                # If the parameter is valid, add it to the validated parameters dictionary.
                validated_params[k] = v

        # Return the dictionary containing only the validated parameters.
        return validated_params


@dataclass
class GptCallHandler:
    """
    A class to facilitate uniform interactions with the OpenAI GPT API. Users can configure various parameters for the
    OpenAI client and model, and utilize the "generate" method to make API calls with built-in error handling and retry
    logic.

    Attributes:
        model_limits (ClassVar): Limits for the model's input/output.
        client_handler (ClassVar): An instance of ClientInitializer to manage API clients.
        calls_made (ClassVar[int]): A count of the total API calls made.
        input_tokens_processed (ClassVar[int]): A count of the total input tokens processed.
        output_tokens_processed (ClassVar[int]): A count of the total output tokens processed.
        embedding_tokens_processed (ClassVar[int]): A count of the total embedding tokens processed.

        api_key_org (str): The organization identifier for the OpenAI API.
        model (str): The model to be used for API calls.
        embedding_model (str): The model to be used for embedding API calls.
        model_context_limit (int): The maximum number of tokens the model can process as input.
        max_output_tokens (int): The maximum number of tokens to generate in the response.
        embedding_dimensions (int): The number of dimensions in the embedding output.
        temperature (float): Sampling temperature for randomness in responses.
        top_p (float): Nucleus sampling parameter.
        frequency_penalty (float): Penalty for repeated tokens.
        presence_penalty (float): Penalty for new tokens.
        max_retries (int): The maximum number of retries for API calls.
        stop: Optional stop sequence for the API response.
        openai_internal_errors (int): Count of internal errors encountered.
        openai_rate_limits_hit (int): Count of rate limit errors encountered.
    """

    # Class variables that are shared across all instances of the class.
    model_limits: ClassVar = field(
        init=False
    )  # Holds the limits for the model's input/output, initialized later.
    client_handler: ClassVar = (
        ClientInitializer()
    )  # An instance of ClientInitializer to manage API client setup.
    calls_made: ClassVar[int] = (
        0  # Tracks the total number of API calls made by all instances.
    )
    input_tokens_processed: ClassVar[int] = (
        0  # Tracks the total number of input tokens processed by all instances.
    )
    output_tokens_processed: ClassVar[int] = (
        0  # Tracks the total number of output tokens processed by all instances.
    )
    embedding_tokens_processed: ClassVar[int] = (
        0  # Tracks the total number of embedding tokens processed by all instances.
    )

    # Instance variables that are unique to each instance of the class.
    game: Game = None  # The game instance associated with this GPT call handler.
    api_key_org: str = "Helicone"  # The organization identifier for the OpenAI API.
    model: str = "gpt-4"  # The specific model to be used for API calls.
    embedding_model: str = (
        "text-embedding-3-small"  # The specific model to be used for embedding API calls.
    )
    model_context_limit: int = (
        8192  # The maximum number of tokens the model can process as input.
    )
    max_output_tokens: int = (
        256  # The maximum number of tokens to generate in the response.
    )
    embedding_dimensions: int = 768  # The number of dimensions in the embedding output.
    temperature: float = (
        1.0  # Controls the randomness of the output; higher values mean more randomness.
    )
    top_p: float = 0.75  # Nucleus sampling parameter that controls diversity.
    frequency_penalty: float = 0  # Penalty for using repeated tokens in the output.
    presence_penalty: float = 0  # Penalty for introducing new tokens in the output.
    max_retries: int = (
        5  # The maximum number of retries allowed for API calls in case of failure.
    )
    stop = None  # Optional stop sequence for the API response.
    openai_internal_errors: int = (
        0  # Counter for internal errors encountered during API calls.
    )
    openai_rate_limits_hit: int = (
        0  # Counter for how many times rate limits have been hit during API calls.
    )

    def __post_init__(self):
        """
        Post-initialization method that sets up the instance after the default initialization. This method saves the
        initial parameters, retrieves the API client for the specified organization, loads the model limits, and sets
        the requested model limits for the instance.
        """

        # Save the initial parameters of the instance by converting them to a dictionary.
        self.original_params = self._save_init_params()

        # Retrieve the OpenAI client associated with the specified organization using the client handler.
        self.client = self.client_handler.get_client(self.api_key_org)

        # Load the model limits from the configuration file to determine input/output constraints.
        self.model_limits = self._load_model_limits()

        # Set the requested model limits based on the loaded limits to ensure compliance with the model's capabilities.
        self._set_requested_model_limits()

        def _save_init_params(self):
            """
            Converts the instance's attributes into a dictionary format for easy access and manipulation. This method is
            useful for saving the initial state of the instance, allowing for later restoration or inspection of its
            parameters.

            Returns:
                dict: A dictionary representation of the instance's attributes.
            """

            # Converts the instance's attributes to a dictionary and returns it.
            return asdict(self)

        def _load_model_limits(self):
            """
            Loads the model limits from a JSON file located in the assets directory. This method constructs the file
            path, attempts to read the contents, and returns the parsed limits, handling any file-related errors that
            may occur during the process.

            Returns:
                dict or None: The model limits loaded from the JSON file if successful, otherwise None if the file
                cannot be found or read.
            """

            # Retrieves the path to the assets directory and constructs the full path to the model limits JSON file.
            assets = get_assets_path()
            full_path = os.path.join(assets, "openai_model_limits.json")
            try:
                # Attempts to open the model limits JSON file and load its contents.
                with open(full_path, "r") as f:
                    limits = json.load(f)
            except IOError:
                # If the file cannot be found, print an error message indicating the bad path.
                print(f"Bad path. Couldn't find asset at {full_path}")
                # TODO: Determine how to handle this error case appropriately.
            else:
                # Return the loaded limits if the file was successfully read.
                return limits

    def _set_requested_model_limits(self):
        """
        Sets the input/output limits for the requested model based on the available model limits. If the model limits
        are not found, it defaults to using the GPT-4 limits, ensuring that the maximum tokens do not exceed the context
        limit.
        """

        if self.model_limits:
            # If model limits are available, retrieve the limits for the specified model.
            limits = self.model_limits.get("LLM", {}).get(self.model, {})
            # Set the model context limit based on the retrieved limits, defaulting to 8192 if not specified.
            self.model_context_limit = limits.get("context", 8192)
            # Ensure that the maximum tokens do not exceed either the context or max output limits.
            self.max_output_tokens = min(
                self.max_output_tokens,
                self.model_context_limit,
                limits.get("max_out", self.model_context_limit),
            )
        else:
            # If no model limits are available, default the model context limit to 8192.
            self.model_context_limit = 8192
            # Ensure that the maximum tokens do not exceed the default context limit.
            self.max_output_tokens = min(self.max_output_tokens, 8192)

    def _set_requested_embeddings_dimensions(self):
        """
        Sets the embedding dimensions for the requested embedding model based on the available model limits.
        If the embedding model limits are not found, it defaults to using the dimensions for 'text-embedding-3-small'.
        """

        if self.model_limits:
            # If model limits are available, retrieve the dimensions for the specified embedding model.
            embedding_info = self.model_limits.get("Embeddings", {}).get(
                self.embedding_model, {}
            )
            # Set the embedding dimensions based on the retrieved info, defaulting to 768 if not specified.
            self.embedding_dimensions = embedding_info.get("dimensions", 768)

    def update_params(self, **kwargs):
        """
        Updates the instance parameters with new values provided as keyword arguments. This method checks if each
        parameter exists in the instance and updates its value accordingly.

        Args:
            **kwargs: Arbitrary keyword arguments representing the parameters to update and their new values.
        """

        # Iterate over the keyword arguments passed to the method, unpacking each parameter name and its new value.
        for param, new_val in kwargs.items():
            # Check if the instance has an attribute with the name of the current parameter.
            if hasattr(self, param):
                # Set the attribute to the new value if it exists.
                setattr(self, param, new_val)

    def reset_defaults(self):
        """
        Resets the instance parameters to their original values as stored during initialization. This method restores
        the state of the instance to ensure consistent behavior after modifications.
        """

        # Reset the parameters to the original values
        for param, value in self.original_params.items():
            setattr(self, param, value)

    @classmethod
    def get_calls_count(cls):
        """
        Returns the total number of API calls made across all instances of the class. This class method provides a way
        to access the call count without needing an instance of the class.

        Returns:
            int: The total number of API calls made.
        """

        return cls.calls_made

    @classmethod
    def increment_calls_count(cls):
        """
        Increments the total number of API calls made across all instances of the class. This class method updates the
        call count, allowing for tracking of how many times the API has been accessed.
        """

        cls.calls_made += 1

    @classmethod
    def get_input_tokens_processed(cls):
        """
        Returns the total number of input tokens processed across all instances of the class. This class method provides
        a way to access the token count without needing an instance of the class.

        Returns:
            int: The total number of input tokens processed.
        """

        return cls.input_tokens_processed

    @classmethod
    def get_output_tokens_processed(cls):
        """
        Returns the total number of output tokens processed across all instances of the class. This class method
        provides a way to access the token count without needing an instance of the class.

        Returns:
            int: The total number of output tokens processed.
        """

        return cls.output_tokens_processed

    @classmethod
    def get_all_tokens_processed(cls):
        """
        Returns the total number of all tokens (input + output) processed across all instances of the class. This
        class method provides a way to access the token count without needing an instance of the class.

        Returns:
            int: The total number of all (input + output) tokens processed.
        """

        return cls.get_input_tokens_processed + cls.get_output_tokens_processed

    @classmethod
    def get_embedding_tokens_processed(cls):
        """
        Returns the total number of embedding tokens processed across all instances of the class. This class method
        provides a way to access the token count without needing an instance of the class.

        Returns:
            int: The total number of embedding tokens processed.
        """

        return cls.embedding_tokens_processed

    @classmethod
    def update_input_token_count(cls, add_on: int):
        """
        Updates the total count of input tokens processed across all instances of the class by adding a specified
        number. This class method allows for tracking the cumulative number of input tokens processed during API
        interactions.

        Args:
            add_on (int): The number of tokens to add to the total input processed count.
        """

        cls.input_tokens_processed += add_on

    @classmethod
    def update_output_token_count(cls, add_on: int):
        """
        Updates the total count of output tokens processed across all instances of the class by adding a specified
        number. This class method allows for tracking the cumulative number of output tokens processed during API
        interactions.

        Args:
            add_on (int): The number of tokens to add to the total output processed count.
        """

        cls.output_tokens_processed += add_on

    @classmethod
    def update_embedding_token_count(cls, add_on: int):
        """
        Updates the total count of embedding tokens processed across all instances of the class by adding a specified
        number. This class method allows for tracking the cumulative number of embedding tokens processed during API
        interactions.

        Args:
            add_on (int): The number of tokens to add to the total embedding processed count.
        """

        cls.embedding_tokens_processed += add_on

    def generate(
        self, system: str = None, user: str = None, messages: list = None, character: Character = None
    ) -> str:
        """
        Makes a call to the OpenAI API to generate a response based on the provided messages. This method can accept
        system and user messages directly or a list of messages, and it includes error handling and retry logic for
        various API errors. Also, it optionally accepts a character argument for GPT logging purposes. This method is a
        wrapper for making a call to the OpenAI API. It expects a function as an argument that should produce the
        messages argument.

        Args:
            system (str, optional): The system message to guide the behavior of the model.
            user (str, optional): The user message to provide context for the model's response.
            messages (list, optional): A list of messages to send to the model, overriding system and user parameters if
            provided.
            character (Character, optional): The character to use for the conversation.
        Returns:
            str: The content of the model's response.

        Raises:
            ValueError: If neither system and user strings nor a valid list of messages is provided.
        """

        if system and user:
            # If both system and user messages are provided, create a list of messages for the API call.
            messages = [
                {
                    "role": "system",
                    "content": system,
                },  # Add the system message with its role.
                {
                    "role": "user",
                    "content": user,
                },  # Add the user message with its role.
            ]
        elif not messages or not isinstance(messages, list):
            # If messages are not provided or are not a list, raise an error indicating the required input.
            raise ValueError(
                "You must supply 'system' and 'user' strings or a list of ChatMessages in 'messages'."
            )

        i = 0
        # Attempt to make the API call with a maximum number of retries.
        while i < self.max_retries:
            try:
                # Make a call to the OpenAI API to generate a response based on the provided messages.
                response = self.client.chat.completions.create(
                    model=self.model,  # Specify the model to use for the API call.
                    messages=messages,  # Pass the messages to the API.
                    temperature=self.temperature,  # Set the randomness of the output.
                    max_tokens=self.max_output_tokens,  # Limit the number of tokens in the response.
                    top_p=self.top_p,  # Set the nucleus sampling parameter.
                    frequency_penalty=self.frequency_penalty,  # Apply penalty for repeated tokens.
                    presence_penalty=self.presence_penalty,  # Apply penalty for new tokens.
                    stop=self.stop,  # Specify any stop sequences for the response.
                )
                # return response.choices[0].message.content  # Uncomment to return the content of the response.
            except openai.APITimeoutError as e:
                # Handle timeout errors where the request took too long.
                self._log_gpt_error(e)  # Log the error for debugging.
                self._handle_TimeoutError(
                    e, attempt=i
                )  # Handle the timeout error with custom logic.
                continue  # Retry the API call.
            except openai.RateLimitError as e:
                # Handle rate limit errors when the API usage exceeds allowed limits.
                self._log_gpt_error(e)  # Log the error for debugging.
                wait_time = self._handle_RateLimitError(
                    e, attempt=i
                )  # Get the wait time before retrying.
                print(
                    f"Rate limit exceeded, waiting {wait_time} seconds."
                )  # Inform the user of the wait time.
                time.sleep(wait_time)  # Pause execution for the specified wait time.
                continue  # Retry the API call.
            except openai.BadRequestError as e:
                # Handle bad request errors due to invalid input.
                success, info = self._handle_BadRequestError(
                    e
                )  # Process the error and get relevant info.
                self._log_gpt_error(e)  # Log the error for debugging.
                return (
                    success,
                    info,
                )  # Return the success status and info for further handling.
            except openai.InternalServerError as e:
                # Handle internal server errors indicating issues with the OpenAI service.
                self._handle_InternalServerError(
                    e
                )  # Handle the internal server error with custom logic.
                self._log_gpt_error(e)  # Log the error for debugging.
                continue  # Retry the API call.
            except openai.APIConnectionError as e:
                # Handle connection errors that may occur during the API call.
                self._handle_APIConnectionError(
                    e
                )  # Handle the connection error with custom logic.
                self._log_gpt_error(e)  # Log the error for debugging.
            except openai.AuthenticationError as e:
                # Handle authentication errors due to invalid API credentials.
                print(
                    "Your api credentials caused an error. Check your config file."
                )  # Inform the user of the issue.
                raise e  # Raise the error to stop execution.
            else:
                # Log the GPT call details
                self._log_gpt_call(messages, response, character)

                # If the API call is successful, update the token counts and increment the call count.
                self._set_token_counts(
                    response=response, system=system, user=user, messages=messages
                )  # Update the token counts based on the messages.

                # print("incrementing the number of calls to GPT")  # Uncomment to log the increment action.
                GptCallHandler.increment_calls_count()  # Increment the total number of API calls made.
                return response.choices[
                    0
                ].message.content  # Return the content of the model's response.

    def generate_embeddings(self, text: str, *args) -> list:
        """
        Generates embeddings for the provided text using the OpenAI client.

        This method attempts to create an embedding vector for the specified text, handling various potential errors
        that may arise during the API call. It includes retry logic for timeouts and rate limits, ensuring robust
        interaction with the OpenAI service.

        Args:
            text (str): The input text for which to generate embeddings. If empty, the function returns None.
            *args: Additional parameters to be passed to the OpenAI client.

        Returns:
            list: A NumPy array containing the generated embedding vector, or None if the input text is empty.

        Raises:
            openai.AuthenticationError: If the API credentials are invalid.
        """

        # Check if the input text is empty; if so, return None to indicate no embedding can be generated.
        if not text:
            return None

        i = 0
        # Attempt to make the API call with a maximum number of retries.
        while i < self.max_retries:
            try:
                # Create an embedding vector for the provided text using the OpenAI client.
                response = self.client.embeddings.create(
                    input=[text], model=self.embedding_model, *args
                )
            except openai.APITimeoutError as e:
                # Handle timeout errors where the request took too long.
                self._log_gpt_error(e)  # Log the error for debugging.
                self._handle_TimeoutError(
                    e, attempt=i
                )  # Handle the timeout error with custom logic.
                continue  # Retry the API call.
            except openai.RateLimitError as e:
                # Handle rate limit errors when the API usage exceeds allowed limits.
                self._log_gpt_error(e)  # Log the error for debugging.
                wait_time = self._handle_RateLimitError(
                    e, attempt=i
                )  # Get the wait time before retrying.
                print(
                    f"Rate limit exceeded, waiting {wait_time} seconds."
                )  # Inform the user of the wait time.
                time.sleep(wait_time)  # Pause execution for the specified wait time.
                continue  # Retry the API call.
            except openai.BadRequestError as e:
                # Handle bad request errors due to invalid input.
                success, info = self._handle_BadRequestError(
                    e
                )  # Process the error and get relevant info.
                self._log_gpt_error(e)  # Log the error for debugging.
                return (
                    success,
                    info,
                )  # Return the success status and info for further handling.
            except openai.InternalServerError as e:
                # Handle internal server errors indicating issues with the OpenAI service.
                self._handle_InternalServerError(
                    e
                )  # Handle the internal server error with custom logic.
                self._log_gpt_error(e)  # Log the error for debugging.
                continue  # Retry the API call.
            except openai.APIConnectionError as e:
                # Handle connection errors that may occur during the API call.
                self._handle_APIConnectionError(
                    e
                )  # Handle the connection error with custom logic.
                self._log_gpt_error(e)  # Log the error for debugging.
            except openai.AuthenticationError as e:
                # Handle authentication errors due to invalid API credentials.
                print(
                    "Your api credentials caused an error. Check your config file."
                )  # Inform the user of the issue.
                raise e  # Raise the error to stop execution.
            else:
                # If the API call is successful, update the embedding token counts and increment the call count.
                self._set_embedding_token_counts(text)

                # print("incrementing the number of calls to GPT")  # Uncomment to log the increment action.
                GptCallHandler.increment_calls_count()  # Increment the total number of API calls made.

                # Get the response embeddings, and convert them to a NumPy array for further processing or analysis.
                return np.array(response.data[0].embedding)

    def _set_token_counts(self, response, system=None, user=None, messages=None):
        """
        Updates the total token counts based on the provided API response and optional system, user, or message inputs.
        This method extracts token usage information from the response and updates the input and output token counts,
        handling errors gracefully and calculating token counts from system and user messages if necessary.

        Args:
            response (dict): The API response containing token usage information, including 'usage' with
            'completion_tokens' and 'total_tokens'.
            system (str, optional): The system message to calculate token count for.
            user (str, optional): The user message to calculate token count for.
            messages (list, optional): A list of message dictionaries to calculate token count for.

        Raises:
            KeyError: If expected keys are not found in the response.
            TypeError: If the response is not a dictionary.
            ValueError: If the token calculations result in unexpected values.
            AttributeError: If the GptCallHandler methods are not found.
        """

        try:
            # Attempt to extract input and output token counts from the API response.
            input_tokens = response["usage"]["prompt_tokens"]
            output_tokens = response["usage"]["completion_tokens"]

            # Update the input and output token counts in the GptCallHandler.
            GptCallHandler.update_input_token_count(input_tokens)
            GptCallHandler.update_output_token_count(output_tokens)

        # Catch potential errors related to missing keys or incorrect types in the response.
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            # Log the error for debugging purposes.
            self._log_gpt_error(e)

            # If both system and user messages are provided, calculate their token counts.
            if system and user:
                system_tkn_count = get_prompt_token_count(
                    system, role="system", pad_reply=False
                )  # Get token count for the system message without padding.

                user_tkn_count = get_prompt_token_count(
                    user, role="user", pad_reply=True
                )  # Get token count for the user message with padding.

                # Update the total token count by adding the counts of system and user messages.
                GptCallHandler.update_input_token_count(
                    system_tkn_count + user_tkn_count
                )

            # If messages are provided, calculate the token count based on the list of messages.
            elif messages:
                pad = (
                    len(messages) * 3
                )  # Calculate padding based on the number of messages.

                # Extract the content from each message to calculate the total token count.
                prompt_contents = [
                    chat.get("content")
                    for chat in messages
                    if chat.get("content", None)
                ]

                # Get the token count for the message contents.
                prompt_tkn_count = get_prompt_token_count(content=prompt_contents)

                # Update the total token count by adding the calculated prompt token count and padding.
                GptCallHandler.update_input_token_count(prompt_tkn_count + pad)

    def _set_embedding_token_counts(self, text):
        """
        Sets the embedding token counts based on the provided text.

        This method calculates the number of embedding tokens for the given text and updates the total input and output
        token counts in the GptCallHandler. It ensures that the token usage is accurately tracked for the embedding
        process.

        Args:
            text (str): The text for which to calculate the embedding token count.

        Returns:
            None
        """

        # Calculate the number of embedding tokens for the provided text.
        if embedding_tokens := get_prompt_token_count(
            text,
            role=None,
            pad_reply=False,  # Call the function to get the token count without padding the role or reply.
        ):
            # If the token count is successfully retrieved, update the input and output token counts in the
            # GptCallHandler.
            GptCallHandler.update_embedding_token_count(embedding_tokens)

    def _log_gpt_call(self, messages, response, character=None):
        """
        Logs the details of a GPT call, including the messages sent, the response received, and the character associated
        with the call if provided.

        Args:
            messages (list): A list of message dictionaries representing the conversation history.
            response (dict): The response dictionary containing the model's response details.
            character (Character, optional): The character associated with the GPT call, used for logging.
        """

        if self.game:

            extras = get_logger_extras(
                self.game, character=character, include_gpt_call_id=True, stack_level=2
            )

            extras["type"] = "GPT Call"

            # Format the messages from the log record.
            messages_list = []
            for message in messages:
                messages_list.append(
                    f"{message['role'].title()}:\n{message['content']}"  # Format each message with its role.
                )
            messages = "\n\n".join(messages_list)  # Join formatted messages with double newlines.

            choices = response["choices"][0]  # Get the first choice from the response.

            # Add relevant response information to the extras dictionary.
            extras["id"] = response["id"]
            extras["model"] = response["model"]
            extras["messages"] = messages
            extras["response"] = choices["message"]["content"]
            extras["finish_reason"] = choices["finish_reason"]

            extras["max_output_tokens"] = self.max_output_tokens
            extras["temperature"] = self.temperature
            extras["top_p"] = self.top_p
            extras["frequency_penalty"] = self.frequency_penalty
            extras["presence_penalty"] = self.presence_penalty

            # Extract and add token usage information to the message dictionary.
            usage = response["usage"]
            extras["prompt_tokens"] = usage["prompt_tokens"]
            extras["completion_tokens"] = usage["completion_tokens"]
            extras["total_tokens"] = usage["total_tokens"]

            # Log the details of the GPT call, including the input and the response received.
            self.game.gpt_call_logger.debug(f"GPT Call Details", extra=extras)

    def _log_gpt_error(self, e):
        """
        Logs an error message related to GPT API interactions. This method captures the error details and logs them
        using the configured logger for debugging and monitoring purposes.

        Args:
            e (Exception): The exception object containing details about the error to be logged.
        """

        # Log an error message indicating a GPT-related error, including the details of the exception.
        logger.error(f"GPT Error: {e}")

    def _handle_TimeoutError(self, e, attempt):
        """
        Handles timeout errors encountered during API calls to the OpenAI service. This method logs the error,
        implements an exponential backoff strategy for retries, and pauses execution for a specified duration based on
        the number of attempts made.

        Args:
            e (Exception): The exception object representing the timeout error.
            attempt (int): The current attempt number for the API call, used to calculate the backoff duration.
        """

        # # Log the GPT error for debugging purposes. # This gets logged when the error occurs, just prior to this call.
        # self._log_gpt_error(e)

        # Implement exponential backoff for retrying the API call after a timeout. If the rate limits have been hit more
        # than once, set a longer wait duration. openai_rate_limits_hit doesn't get incremented, so this is dependent on
        # the initialization value.
        duration = 61 if self.openai_rate_limits_hit > 1 else min(0.1**attempt, 2)
        # Inform the user about the wait time before retrying the API call.
        print(f"rate limit reached, sleeping {duration} seconds")

        # Pause execution for the calculated duration to avoid hitting the rate limit again.
        time.sleep(duration)

    def _handle_RateLimitError(self, e, attempt):
        """
        Handles rate limit errors encountered during API calls to the OpenAI service. This method analyzes the error
        response to determine the appropriate wait time before retrying the request, implementing a strategy based on
        the error message and the number of attempts made.

        Args:
            e (Exception): The exception object representing the rate limit error.
            attempt (int): The current attempt number for the API call, used to calculate the backoff duration.

        Returns:
            float: The calculated wait time in seconds before retrying the API call.

        """

        # Calculate the default wait time based on the number of attempts, capped at 2 seconds.
        default_seconds = min(attempt * 1, 2)

        # Check if the exception object has a response attribute.
        if hasattr(e, "response"):
            # Parse the error response as JSON to extract error details.
            error_response = e.response.json()

            # If the response does not contain an error key, return the default wait time.
            if "error" not in error_response:
                return default_seconds

            # Extract the error details from the response.
            error = error_response.get("error")
            code = error.get("code", None)  # Get the error code.

            # If the error code is not related to rate limits, return the default wait time.
            if code != "rate_limit_exceeded":
                return default_seconds

            msg = error.get("message", None)  # Get the error message.
            if msg and isinstance(msg, str):
                return self._log_rate_limit_error_message(msg)
            return default_seconds  # Return the default wait time if no specific duration is found.

        # Return the default wait time if no response is available.
        return default_seconds

    def _log_rate_limit_error_message(self, msg):
        """
        Processes a rate limit error message to extract the suggested wait time before retrying an API call. This method
        uses regular expressions to identify the duration and unit of the wait time, converting it to seconds and adding
        a padding value before returning the final wait time.

        Args:
            msg (str): The error message containing the suggested wait time.

        Returns:
            float: The calculated wait time in seconds, including a padding of 0.5 seconds.

        Raises:
            ValueError: If the message does not contain a valid wait time format.
        """

        sleep_request = re.search(r"Please try again in \d+(\.\d+)?(ms|s)\b", msg)

        if sleep_request:
            sleep_request = sleep_request.group(
                0
            )  # Extract the matched wait time string.

        # Define patterns to extract the unit and duration from the sleep request.
        unit_pattern = r"(?:\d+)(ms|s)"
        duration_pattern = r"\d+(\.\d+)?(?=\s*(ms|s))"
        units = re.findall(unit_pattern, sleep_request)  # Find the unit (ms or s).
        unit = units[0] if units else None  # Get the first matched unit.
        duration = re.search(
            duration_pattern, sleep_request
        )  # Find the duration value.

        # Determine the wait time based on the extracted unit and duration.
        if not unit:
            unit = (
                "s" if "." in duration else "ms"
            )  # Default to seconds if no unit is found.
        if not duration:
            return 2  # Return a default wait time of 2 seconds if no duration is found.
        else:
            duration = duration.group(0)  # Extract the duration value as a string.

        # Convert the duration to seconds based on its unit.
        if unit == "ms":
            wait_time = float(duration) / 1000.0  # Convert milliseconds to seconds.

        elif unit == "s":
            wait_time = float(duration)  # Convert duration to float if in seconds.
        # Set a padding time to add to the calculated wait time.
        pad = 0.5
        # Return the calculated wait time with padding added.
        return wait_time + pad

    def _handle_BadRequestError(self, e):
        """
        Handles bad request errors encountered during API calls to the OpenAI service. This method checks for specific
        error codes in the response and extracts relevant information, particularly when the context length exceeds the
        model's limits.

        Args:
            e (Exception): The exception object representing the bad request error.

        Returns:
            tuple: A tuple containing False and the difference between the input token count and the model's maximum
            token limit, or None if not applicable.
        """

        # Check if the exception object has a response attribute.
        if hasattr(e, "response"):
            # Parse the error response as JSON to extract error details.
            error_response = e.response.json()

            # Check if the response contains an error key.
            if "error" in error_response:
                error = error_response.get("error")  # Extract the error details.

                # If the error code indicates that the context length has been exceeded.
                if error.get("code") == "context_length_exceeded":
                    msg = error.get("message")  # Get the error message.
                    # Use a regex pattern to find numeric values in the message.
                    matches = re.findall(r"\d+", msg)

                    # If there are at least two numeric matches in the message.
                    if matches and len(matches) > 1:
                        model_max = int(
                            matches[0]
                        )  # The maximum token limit for the model.
                        input_token_count = int(
                            matches[1]
                        )  # The token count of the input.
                        diff = (
                            input_token_count - model_max
                        )  # Calculate the difference.
                        return (
                            False,
                            diff,
                        )  # Return False and the difference in token counts.

        # If no relevant error was found, return False and None.
        return False, None

    def _handle_APIConnectionError(self, e):
        """
        Handles API connection errors encountered during interactions with the OpenAI service. This method provides
        feedback to the user about potential causes of the error and pauses execution for a specified duration to allow
        for recovery.

        Args:
            e (Exception): The exception object representing the API connection error.
        """

        # Inform the user that an API connection error has been encountered.
        print("APIConnectionError encountered:\n")

        # Print the details of the exception for debugging purposes.
        print(e)

        # Suggest checking the API key or organization base URL for potential misconfigurations.
        print("Did you set your API key or organization base URL incorrectly?")

        # Indicate that the error could also be due to a poor internet connection.
        print("This could also be raised by a poor internet connection.")

        # Pause execution for a specified duration to allow for recovery from the error.
        self._wait_an_interval(total_wait_time=120)

    def _handle_InternalServerError(self, e):
        """
        Handles internal server errors encountered during interactions with the OpenAI service. This method logs the
        error, increments the internal error count, and pauses execution for a calculated duration to allow for
        potential recovery.

        Args:
            e (Exception): The exception object representing the internal server error.

        """

        # Inform the user that an OpenAI service error has been encountered.
        print("OpenAI Service Error encountered:\n")

        # Print the details of the exception for debugging purposes.
        print(e)

        # Increment the count of internal errors encountered during API interactions.
        self.openai_internal_errors += 1

        # Calculate the total wait time based on the number of internal errors, with a base wait time of 15 seconds per
        # error.
        total_wait_time = 15 * self.openai_internal_errors

        # Notify the user about the wait time and suggest stopping the run if necessary.
        print(
            f"\nYou may want to stop the run and try later. Otherwise, waiting {total_wait_time} seconds..."
        )

        # Pause execution for the calculated wait time to allow for potential recovery from the error.
        self._wait_an_interval(total_wait_time)

    def _wait_an_interval(self, total_wait_time=15):
        """
        Pauses execution for a specified total wait time, decrementing by a one second interval. This method provides a
        countdown display to inform the user of the remaining wait time before resuming execution.

        Args:
            total_wait_time (int, optional): The total time to wait in seconds. Defaults to 15 seconds.

        """

        # Set the interval for the countdown display to 1 second.
        interval = 1

        # Continue the countdown while there is remaining wait time.
        while total_wait_time > 0:
            # Print the remaining wait time, updating the same line in the console.
            print(f"Resuming in {total_wait_time} seconds...", end="\r")

            # Pause execution for the specified interval to create the countdown effect.
            time.sleep(interval)

            # Decrease the total wait time by the interval amount.
            total_wait_time -= interval


def gpt_get_summary_description_of_action(
    statement, call_handler: GptCallHandler, **handler_kwargs
):
    """
    Generates a summary description of a specified action using the GPT model. This function constructs a message based
    on the provided statement and calls the GPT model to generate a summary, resetting parameters after each call to
    ensure consistent behavior.

    Args:
        statement (str): The action statement for which a summary description is to be generated.
        call_handler (GptCallHandler): An instance of GptCallHandler used to interact with the GPT model.
        **handler_kwargs: Additional keyword arguments to update the parameters of the call handler.

    Returns:
        str: The generated summary description of the action.

    Raises:
        TypeError: If the provided call_handler is not an instance of GptCallHandler.
    """

    # Check if the provided call_handler is an instance of GptCallHandler; raise an error if not.
    if not isinstance(call_handler, GptCallHandler):
        raise TypeError("'call_handler' must be a GptCallHandler.")

    # Update the parameters of the call handler with any additional keyword arguments provided.
    call_handler.update_params(**handler_kwargs)

    # Retrieve the system prompt (from gpt_helper_prompts.py) for generating the action summary.
    system = hp.action_summary_prompt

    # Construct the messages to be sent to the GPT model, including the system and user roles.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": statement},
    ]

    # Generate the summary statement by calling the GPT model with the constructed messages.
    summary_statement = call_handler.generate(messages=messages)

    # Reset the call handler parameters to their original values after the first generation.
    call_handler.reset_defaults()

    # Return the generated summary statement.
    return summary_statement


def gpt_get_action_importance(
    statement: str, call_handler: GptCallHandler, **handler_kwargs
):
    """
    Determines the importance of a specified action by generating a response from the GPT model. This function
    constructs a message based on the provided statement and retrieves the importance value, returning it as an integer.

    Args:
        statement (str): The action statement for which the importance is to be evaluated.
        call_handler (GptCallHandler): An instance of GptCallHandler used to interact with the GPT model.
        **handler_kwargs: Additional keyword arguments to update the parameters of the call handler.

    Returns:
        int or None: The importance value of the action as an integer, or None if no valid importance value is found.

    Raises:
        TypeError: If the provided call_handler is not an instance of GptCallHandler.
    """

    # Check if the provided call_handler is an instance of GptCallHandler; raise an error if not.
    if not isinstance(call_handler, GptCallHandler):
        raise TypeError("'call_handler' must be a GptCallHandler.")

    # Update the parameters of the call handler with any additional keyword arguments provided.
    call_handler.update_params(**handler_kwargs)

    # Retrieve the system prompt for evaluating action importance.
    system = hp.action_importance_prompt

    # Construct the messages to be sent to the GPT model, including the system and user roles.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": statement},
    ]

    # Generate the importance string by calling the GPT model with the constructed messages.
    importance_str = call_handler.generate(messages=messages)

    # Reset the call handler parameters to their original values after the first generation.
    call_handler.reset_defaults()

    # Define a regex pattern to find numeric values in the importance string.
    pattern = r"\d+"

    if matches := re.findall(pattern, importance_str):
        # Iterate through the matched numbers to find the first valid integer.
        for match in matches:
            try:
                return int(
                    match
                )  # Convert the matched string to an integer and return it.
            except ValueError:
                continue  # If conversion fails, continue to the next match.
        return None  # Return None if no valid integer was found.

    # Return None if no matches were found in the importance string.
    return None


def gpt_pick_an_option(
    instructions, options, input_str, call_handler: GptCallHandler, **handler_kwargs
):
    """
    Selects the best option from a set of choices based on user input and system instructions using the GPT model. This
    function generates a prompt for the model, retrieves a selection, and returns the corresponding option name. CREDIT
    of generalized option picking method: Dr. Chris Callison-Burch (UPenn). The function generates an enumerated list of
    option descriptions that are shown to GPT. It then returns a number (which is matched with a regex, in case it
    generates more text than is necessary), and then returns the option name.

    Args:
        instructions (str): The system instructions that guide the model's response.
        options (dict): A dictionary mapping option descriptions to option names.
        input_str (str): The user input that is used to match against the options.
        call_handler (GptCallHandler): An instance of GptCallHandler used to interact with the GPT model.
        **handler_kwargs: Additional keyword arguments to update the parameters of the call handler.

    Returns:
        str or None: The name of the selected option if a valid selection is made, or None if no valid selection is
        found.

    Raises:
        TypeError: If the provided call_handler is not an instance of GptCallHandler.
    """

    # Check if the provided call_handler is an instance of GptCallHandler; raise an error if not.
    if not isinstance(call_handler, GptCallHandler):
        raise TypeError("'call_handler' must be a GptCallHandler.")

    # Update the parameters of the call handler with any additional keyword arguments provided.
    call_handler.update_params(**handler_kwargs)

    # Enumerate the options to create a string of choices and a list of option names.
    choices_str, options_list = enumerate_dict_options(options)

    # Construct the messages to be sent to the GPT model, including the system instructions and user input.
    messages = [
        {
            "role": "system",
            "content": "{instructions}\n\n{choices_str}\nReturn just the number of the best match.".format(
                instructions=instructions, choices_str=choices_str
            ),
        },
        {"role": "user", "content": input_str},
    ]

    # Call the OpenAI API to generate a selection based on the constructed messages.
    selection = call_handler.generate(messages=messages)

    # Reset the call handler parameters to their original values after the first generation.
    call_handler.reset_defaults()

    # Define a regex pattern to find numeric values in the selection string.
    pattern = r"\d+"

    if not (matches := re.findall(pattern, selection)):
        return None  # Return None if no matches were found in the selection string.
    index = int(matches[0])  # Convert the first matched string to an integer index.

    # Check if the index is within the bounds of the options list.
    return None if index >= len(options_list) else options[options_list[index]]


def limit_context_length(
    history,
    max_tokens,
    max_turns=1000,
    tokenizer=None,
    keep_most_recent=True,
    return_count=False,
):
    """
    Limits the context length of the command history to ensure it does not exceed the specified maximum number of tokens
    or turns. The function can retain the most recent messages or discard them based on the provided parameters, and it
    operates non-destructively, leaving the original command history unchanged.

    Args:
        history (list): The command history to be limited, which can contain either strings or dictionaries representing
        messages.
        max_tokens (int): The maximum number of tokens allowed in the limited history.
        max_turns (int, optional): The maximum number of messages to retain in the limited history. Defaults to 1000.
        tokenizer (Tokenizer, optional): A tokenizer instance used to calculate token counts. Defaults to None, which
        uses a default tokenizer.
        keep_most_recent (bool, optional): If True, the function will trim the oldest messages first. Defaults to True.
        return_count (bool, optional): If True, the function will also return the total token count consumed. Defaults
        to False.

    Raises:
        TypeError: If the history is not a list or if the elements in history are not of type dict or str.

    Returns:
        list: The limited command history, which contains the most relevant messages based on the specified constraints.
        int (optional): The total number of tokens consumed by the limited context if return_count is True.
    """

    # Initialize the total token count and an empty list to hold the limited history.
    total_tokens = 0
    limited_history = []

    # If no tokenizer is provided, use the default tokenizer for encoding.
    if not tokenizer:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Check if the history is a list; raise an error if it is not.
    if not isinstance(history, list):
        raise TypeError("history must be a list, not ", type(history))

    # If the history contains any elements, determine the type of the first element.
    if len(history) > 0:
        if isinstance(history[0], dict):
            # If the first element is a dictionary, we are parsing ChatMessages. Define a function to extract the token
            # count from the "content" and "role" fields.
            extract = lambda x: get_prompt_token_count(
                content=x["content"],
                role=x["role"],
                pad_reply=False,
                tokenizer=tokenizer,
            )

            # Each reply carries an additional 3 tokens for formatting that need to be accounted for.
            total_tokens += 3
        elif isinstance(history[0], str):
            # If the first element is a string, define a function to get the token count based on the string length.
            extract = lambda x: len(tokenizer.encode(x))
        else:
            # Raise an error if the elements in history are neither dict nor str.
            raise TypeError("Elements in history must be either dict or str")

    # Create a reversed copy of the history if we are keeping only the most recent items. Otherwise, make a normal copy
    # of the list.
    copy_history = reversed(history) if keep_most_recent else history.copy()
    # Iterate through the messages in the copied history.
    for message in copy_history:
        msg_tokens = extract(message)  # Get the token count for the current message.

        # If adding the current message's tokens exceeds the maximum allowed, stop processing.
        if total_tokens + msg_tokens > max_tokens:
            break
        total_tokens += msg_tokens  # Update the total token count.
        limited_history.append(
            message
        )  # Add the current message to the limited history.

        # If the limited history has reached the maximum number of turns, stop processing.
        if len(limited_history) >= max_turns:
            break

    # Reverse the limited history back to the original order if we kept only the most recent items.
    if keep_most_recent:
        limited_history.reverse()

    # If requested, return the total number of tokens consumed along with the limited history.
    if return_count:
        return list(limited_history), total_tokens

    # Return the limited history as a list.
    return list(limited_history)


def get_prompt_token_count(content=None, role=None, pad_reply=False, tokenizer=None):
    """
    Calculates the token count for a given prompt based on its content and role, considering the structure used by the
    GPT API. This function can handle both string and list inputs for content and allows for optional padding to account
    for the GPT message structure.

    Args:
        content (str or list of str): The prompt content; if a list of strings is provided, it returns the total token
        count for the list without padding.
        role (str, optional): The role associated with the prompt; if None, it processes without the GPT message padding
        - essentially just a plain token counter if pad_reply is False. Padding is applied if pad_reply is True.
        pad_reply (bool, optional): If True, adds padding to account for GPT's reply primer tokens
        (<|start|>assistant<|message|>). GPT only adds one reply primer for the entire collective messages given as
        input. To avoid repeatedly accounting for the reply primer in each message in the larger passed messages. It
        should only be set to true in the final message given in GPT's prompt. Defaults to False.
        tokenizer (Tokenizer, optional): A tokenizer instance used to calculate token counts. Defaults to None, which
        uses a default tokenizer.

    Raises:
        TypeError: If content is not a string or list, if the role is not a string, or if the content list contains
        non-string elements.

    Returns:
        int: The total number of tokens calculated for the provided content and role.
    """

    # If no content is provided, return a token count of 0.
    if not content:
        return 0

    # Check if content is neither a string nor a list; raise a TypeError if so.
    if (
        content is not None
        and not isinstance(content, str)
        and not isinstance(content, list)
    ):
        raise TypeError("content must be a string or list, not ", type(content))

    # If content is a list, ensure it is not empty and contains only strings; raise a TypeError if not.
    if (
        content is not None
        and isinstance(content, list)
        and (len(content) != 0 and not isinstance(content[0], str))
    ):
        raise TypeError("content list must contain strings, not ", type(content[0]))

    # Check if the role is provided and ensure it is a string; raise a TypeError if not.
    if role is not None and not isinstance(role, str):
        raise TypeError("role must be a string, not ", type(role))

    # If no tokenizer is provided, use the default tokenizer for encoding.
    if not tokenizer:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Initialize the token count to 0.
    token_count = 0

    # If padding is required for GPT's reply structure, add 3 tokens to the count.
    if pad_reply:
        token_count += 3

    # If the content is a string, calculate its token count and add it to the total.
    if content and isinstance(content, str):
        token_count += len(tokenizer.encode(content))

    # If the content is a list of strings, iterate through each string and add its token count.
    elif content and isinstance(content, list):
        for c in content:
            token_count += len(tokenizer.encode(c))

    # If a role is provided, add padding for the role and its token count.
    if role:
        # Add 3 tokens to account for the role's structure in the GPT API.
        token_count += 3
        token_count += len(tokenizer.encode(role))

    # Return the total token count calculated for the provided content and role.
    return token_count


def get_token_remainder(max_tokens: int, *consumed_counts):
    """
    Calculates the number of remaining tokens available for use based on a maximum token limit and the counts of tokens
    that have already been consumed. This function takes the maximum token limit and any number of consumed token counts
    as arguments to determine the available token remainder.

    Args:
        max_tokens (int): The maximum number of tokens allowed for the model.
        *consumed_counts (int): A variable number of integers representing the counts of tokens that have been consumed.

    Returns:
        int: The number of remaining available tokens after subtracting the consumed counts from the maximum.
    """

    # Calculate and return the number of remaining tokens by subtracting the total consumed token counts from the
    # maximum allowed tokens.
    return max_tokens - sum(consumed_counts)


def context_list_to_string(context, sep: str = "", convert_to_string: bool = False):
    """
    Converts a list of context messages into a single string, optionally converting each message to a string format. The
    messages can be joined using a specified separator, allowing for flexible formatting of the output.

    Args:
        context (list): A list of messages to be converted into a string.
        sep (str, optional): The separator to use when joining the messages. Defaults to an empty string.
        convert_to_string (bool, optional): If True, each message will be explicitly converted to a string. Defaults to
        False.

    Returns:
        str: The concatenated string of messages, separated by the specified separator.
    """

    # If convert_to_string is True, convert each message in the context to a string and join them using the specified
    # separator.
    if convert_to_string:
        return sep.join([f"{str(msg)}" for msg in context])
    # If convert_to_string is False, join the messages directly without converting them to strings.
    else:
        return sep.join(list(context))
