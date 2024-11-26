circular_import_prints = False

if circular_import_prints:
    print("Importing Things Base")

from collections import defaultdict
import json
import itertools

class Thing:
    """
    Represents an item in a text adventure game with properties and commands. It's a supertype that will add shared
    functionality to Items, Locations and Characters.

    This class allows for the creation of game items that have a name, description, and various properties. It also
    supports serialization to and from JSON format.

    Attributes:
        id (int): Unique identifier for the thing.
        name (str): A short name for the thing.
        description (str): A description of the thing.
        properties (defaultdict): A dictionary of properties and their values.
        commands (set): A set of special commands associated with this item.

    Methods:
        to_primitive(): Converts the main fields of this class into a dictionary representation.
        from_primitive(data, instance=None): Creates an instance from a dictionary of values.
        to_json(): Converts the instance to a JSON string.
        from_json(data_json): Creates an instance from a JSON string.
        set_property(property_name, property): Sets a property for the item.
        get_property(property_name): Retrieves the value of a specified property.
        has_property(property_name): Checks if the item has a specified property.
        add_command_hint(command): Adds a special command to the item.
        get_command_hints(): Returns a list of special commands associated with the item.
        remove_command_hint(command): Removes a special command from the item.
    """

    # Create a counter starting from 1 to generate unique IDs for each instance of Thing.
    new_id = itertools.count(1)

    # Initialize a variable to keep track of the last assigned ID.
    _last_id = 0

    def __init__(self, name: str, description: str):
        """
        Initializes a new instance of the Thing class with a unique ID, name, and description.

        This constructor assigns a unique identifier to the instance, sets its name and description, and initializes
        properties and commands associated with the item. The properties are stored in a dictionary, while commands are
        stored in a set.

        Args:
            name (str): A short name for the thing.
            description (str): A description of the thing.

        Returns:
            None
        """

        if circular_import_prints:
            print(f"-\tInitializing Thing", name)

        self.id = next(Thing.new_id)  # Assign the next unique ID from the counter.
        Thing._last_id = (
            self.id
        )  # Update the last assigned ID to the current instance's ID.

        # Set the short name for the thing based on the provided argument.
        self.name = name

        # Set the description for the thing based on the provided argument.
        self.description = description

        # Initialize a dictionary to hold properties and their values.
        # Boolean properties may include: gettable, is_wearable, is_drink, is_food, is_weapon,
        # is_container, and is_surface.
        self.properties = defaultdict(bool)

        # Initialize a set to hold special commands associated with this item.
        # The commands are used to invoke specific actions related to the item.
        self.commands = set()

    def to_primitive(self):
        """
        Converts the main fields of the Thing instance into a dictionary representation.

        This method creates a dictionary that includes the name, description, commands, and properties of the thing,
        making it suitable for serialization or further processing.

        Returns:
            dict: A dictionary representation of the Thing instance containing its main attributes.
        """

        # Create and return a dictionary to hold the main attributes of the Thing instance.
        return {
            "name": self.name,  # Include the name of the thing.
            "description": self.description,  # Include the description of the thing.
            "commands": list(
                self.commands
            ),  # Convert the set of commands to a list for serialization.
            "properties": self.properties,  # Include the properties dictionary.
        }

    @classmethod
    def from_primitive(cls, data, instance=None):
        """
        Creates an instance of the Thing class from a dictionary of values.

        This method populates a Thing instance with attributes such as name, description, commands, and properties based
        on the provided dictionary. If an existing instance is not provided, a new instance is created.

        Args:
            data (dict): A dictionary containing the attributes to set on the Thing instance.
            instance (Thing, optional): An existing instance of Thing to populate. If None, a new instance will be
            created.

        Returns:
            Thing: The populated instance of the Thing class.
        """

        # Check if an existing instance is provided; if not, create a new instance using the name and description from
        # the data.
        if not instance:
            instance = cls(data["name"], data["description"])

        # Iterate over the list of commands from the data and add each command hint to the instance.
        for c in data["commands"]:
            instance.add_command_hint(c)

        # Iterate over the properties in the data dictionary and set each property on the instance.
        for key, value in data["properties"].items():
            instance.set_property(key, value)

        # TODO: shouldn't this function return the instance object if it's newly created? Otherwise, it gets lost.
        return instance

    def to_json(self):
        """
        Converts the Thing instance into a JSON string representation.

        This method first transforms the instance's main attributes into a dictionary format and then serializes that
        dictionary into a JSON string. This is useful for data storage or transmission in a standardized format.

        Returns:
            str: A JSON string representation of the Thing instance.
        """

        # Convert the main attributes of the Thing instance into a dictionary representation.
        data = self.to_primitive()

        # Serialize the dictionary into a JSON string format and return it
        return json.dumps(data)

    @classmethod
    def from_json(cls, data_json):
        """
        Creates an instance of the Thing class from a JSON string representation.

        This method deserializes the provided JSON string into a dictionary and then populates a Thing instance using
        that data. It is useful for reconstructing an object from a stored JSON format.

        Args:
            data_json (str): A JSON string containing the attributes to set on the Thing instance.

        Returns:
            Thing: The populated instance of the Thing class.
        """

        # Deserialize the JSON string into a dictionary format.
        data = json.loads(data_json)

        # Create a Thing instance from the dictionary using the from_primitive method and return it.
        return cls.from_primitive(data)

    def set_property(self, property_name: str, property):
        """
        Sets a specified property for the Thing instance.

        This method allows the user to define or update a property associated with the Thing, storing it in the
        properties dictionary. This is useful for dynamically managing the attributes of the Thing during gameplay.

        Args:
            property_name (str): The name of the property to set.
            property: The value to assign to the specified property.

        Returns:
            None
        """

        self.properties[property_name] = property

    def get_property(self, property_name: str):
        """
        Retrieves the value of a specified property for the Thing instance.

        This method checks the properties dictionary for the given property name and returns its value. If the property
        does not exist, it returns None, allowing for safe access to property values.

        Args:
            property_name (str): The name of the property to retrieve.

        Returns:
            The value of the specified property, or None if the property does not exist.
        """

        return self.properties.get(property_name, None)

    def has_property(self, property_name: str):
        """
        Checks if the Thing instance has a specified property.

        This method determines whether the given property name exists in the properties dictionary of the Thing. It
        returns a boolean value indicating the presence of the property.

        Args:
            property_name (str): The name of the property to check for.

        Returns:
            bool: True if the property exists, False otherwise.
        """

        return property_name in self.properties

    def add_command_hint(self, command: str):
        """
        Adds a special command hint to the Thing instance.

        This method allows the user to associate a specific command with the Thing, which can be invoked during
        gameplay. The command is stored in a set to ensure uniqueness.

        Args:
            command (str): The command hint to add to the Thing instance.

        Returns:
            None
        """

        self.commands.add(command)

    def get_command_hints(self):
        """
        Retrieves the set of special command hints associated with the Thing instance.

        This method returns all the command hints that have been added to the Thing, allowing for easy access to the
        commands that can be invoked during gameplay. The commands are returned as a set.

        Returns:
            set: A set of command hints associated with the Thing instance.
        """

        return self.commands

    def remove_command_hint(self, command: str):
        """
        Removes a special command hint from the Thing instance.

        This method allows the user to delete a specific command hint associated with the Thing. If the command exists,
        it is removed from the set of commands, ensuring that it will no longer be available for invocation.

        Args:
            command (str): The command hint to remove from the Thing instance.

        Returns:
            None
        """

        return self.commands.discard(command)
