# Import the datetime module and alias it as 'dt' for easier access to date and time functions.
import datetime as dt

# Import the json module for working with JSON data, including parsing and serialization.
import json

# Import the logging module to enable logging functionality in the application.
import logging

from typing import override

# Define a set of built-in attributes that are commonly found in log records.
# This set can be used to filter or process log records effectively.
LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class CustomJSONFormatter(logging.Formatter):
    """
    CustomJSONFormatter is a logging formatter that outputs log records in JSON format. It allows for customization of
    the keys in the JSON output while ensuring that essential log information, such as the message and timestamp, is
    always included.

    Args:
        fmt_keys (dict[str, str], optional): A dictionary mapping custom keys to log record attributes. If None,
        defaults to an empty dictionary.

    Methods:
        format(record): Formats the log record into a JSON string.
        _prepare_log_dict(record): Prepares a dictionary representation of the log record, including always fields and
        custom fields, which is then used by format(record) to make the record JSON string.

    Returns:
        str: A JSON string representation of the log record.

    Raises:
        TypeError: If fmt_keys is not a dictionary or None.
    """

    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        """
        Initializes a CustomJSONFormatter instance, allowing for the specification of custom keys for formatting log
        records. If no custom keys are provided, it defaults to an empty dictionary.

        Args:
            fmt_keys (dict[str, str], optional): A dictionary mapping custom keys to log record attributes for renaming
            purposes. Defaults to None, which initializes fmt_keys as an empty dictionary.
        """

        # Call the initializer of the parent class to ensure proper initialization of the inherited attributes and
        # methods.
        super().__init__()

        # Initialize the instance variable 'fmt_keys' with the provided 'fmt_keys' argument.
        # If 'fmt_keys' is None, assign an empty dictionary to 'fmt_keys' to avoid potential errors.
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record into a JSON string representation. This method prepares the log record as a dictionary and
        then converts it to a JSON string, ensuring that all relevant information is included.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: A JSON string representation of the log record.
        """

        # Prepare a dictionary representation of the log record by calling the _prepare_log_dict method.
        message = self._prepare_log_dict(record)

        # Convert the log message dictionary to a JSON string.
        # The 'default=str' argument ensures that any non-serializable objects are converted to strings.
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        """
        Prepares a dictionary representation of a log record, including essential fields and any additional custom
        fields specified. This method extracts relevant information from the log record and formats it into a structured
        dictionary for further processing.

        Args:
            record (logging.LogRecord): The log record to be processed.

        Returns:
            dict: A dictionary containing the formatted log information, including the message, timestamp, and any
            additional fields.

        Raises:
            KeyError: If a specified field in fmt_keys does not exist in the log record.
        """

        # Create a dictionary to hold fields that are always included in the log message.
        always_fields = {
            "message": record.getMessage(),  # Get the log message from the record.
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),  # Convert the timestamp from the record to an ISO 8601 formatted string in UTC.
        }

        # If the record contains exception information, format and add it to the always_fields dictionary.
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        # If the record contains stack information, format and add it to the always_fields dictionary.
        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        # Create a message dictionary by populating it with values from always_fields and the record that match with the
        # keys in fmt_keys.
        message = {
            key: (
                msg_val  # Use the value from always_fields if it exists.
                if (msg_val := always_fields.pop(val, None)) is not None
                else getattr(
                    record, val
                )  # Otherwise, get the value directly from the record.
            )
            for key, val in self.fmt_keys.items()  # Iterate over the keys and values defined in fmt_keys.
        }

        # Update the message dictionary with any remaining fields from always_fields. These are the fields that are not
        # in fmt_keys but still need to be included, like the message or timestamp if they weren’t popped earlier (if
        # they weren't in fmt_keys – they weren't being renamed).
        message |= always_fields

        # Iterate over all attributes in the record's __dict__ to add any additional fields not already included.
        for key, val in record.__dict__.items():
            if (
                key not in LOG_RECORD_BUILTIN_ATTRS
            ):  # Exclude built-in log record attributes.
                message[key] = val  # Add the custom attribute to the message.

        # Return the constructed message dictionary containing all relevant log information.
        return message


# class CustomGPTCallJSONFormatter(CustomJSONFormatter):
#     """
#     CustomGPTCallJSONFormatter is a specialized formatter designed to format log records related to GPT calls. It extends
#     the CustomJSONFormatter and adds specific fields to capture details related to GPT calls, such as the conversation
#     history and the response received.

#     Args:
#         fmt_keys (dict[str, str], optional): A dictionary mapping custom keys to log record attributes. If None,
#         defaults to an empty dictionary.

#     Methods:
#         format(record): Formats the log record into a JSON string.
#         _prepare_log_dict(record): Prepares a dictionary representation of the log record, including always fields and
#         custom fields, which is then used by format(record) to make the record JSON string.

#     Returns:
    
#         str: A JSON string representation of the log record.

#     Raises:
#         TypeError: If fmt_keys is not a dictionary or None.
#     """

#     def __init__(
#         self,
#         *,
#         fmt_keys: dict[str, str] | None = None,
#     ):
#         """
#         Initializes a CustomGPTCallJSONFormatter instance, allowing for the specification of custom keys for formatting
#         log records related to GPT calls. If no custom keys are provided, it defaults to an empty dictionary.

#         Args:
#             fmt_keys (dict[str, str], optional): A dictionary mapping custom keys to log record attributes for renaming
#             purposes. Defaults to None, which initializes fmt_keys as an empty dictionary.
#         """

#         # Call the initializer of the parent class to ensure proper initialization of the inherited attributes and
#         # methods.
#         super().__init__(fmt_keys=fmt_keys)

#     @override
#     def _prepare_log_dict(self, record: logging.LogRecord):
#         """
#         Prepares a dictionary representation of a log record related to GPT calls, including essential fields and any
#         additional custom fields specified. This method extracts relevant information from the log record and formats it
#         into a structured dictionary for further processing.

#         Args:
#             record (logging.LogRecord): The log record to be processed.

#         Returns:
#             dict: A dictionary containing the formatted log information related to GPT calls, including the conversation
#             history, the response received, and any additional fields.

#         Raises:
#             KeyError: If a specified field in fmt_keys does not exist in the log record.
#         """

#         # Retrieve the base log dictionary from the parent class.
#         message = super()._prepare_log_dict(record)

#         # Extract and format the messages from the log record.
#         messages = message.pop("messages")
#         messages_list = []
#         for message in messages:
#             messages_list.append(
#                 f"{message['role'].title()}:\n{message['content']}"  # Format each message with its role.
#             )
#         messages = "\n\n".join(messages_list)  # Join formatted messages with double newlines.

#         # Extract response details from the log record.
#         response = message.pop("response")
#         choices = response["choices"][0]  # Get the first choice from the response.

#         # Add relevant response information to the message dictionary.
#         message["id"] = response["id"]
#         message["model"] = response["model"]
#         message["messages"] = messages
#         message["response"] = choices["message"]["content"]
#         message["finish_reason"] = choices["finish_reason"]

#         # Extract and add token usage information to the message dictionary.
#         usage = response["usage"]
#         message["prompt_tokens"] = usage["prompt_tokens"]
#         message["completion_tokens"] = usage["completion_tokens"]
#         message["total_tokens"] = usage["total_tokens"]

#         # Add specific fields related to GPT calls to the message dictionary.
#         message["gpt_call_id"] = record.gpt_call_id  # Add the GPT call ID.
#         message["gpt_call_history"] = record.gpt_call_history  # Add the GPT call history.
#         message["gpt_call_response"] = record.gpt_call_response  # Add the GPT call response.

#         # Return the constructed message dictionary containing all relevant log information related to GPT calls.
#         return message
