import traceback

# Print a message showing that the module is being imported
print(f"Importing Blocks Doors")

# Get the call stack and format it
stack = traceback.format_stack()

# Print the stack of modules calling this module
print("Call stack for import:")
for line in stack:
    print(line.strip())

# local imports
print(f"\t{__name__} calling imports for Base Block")
from .base import Block


class Locked_Door(Block):
    """Represents a locked door in a text adventure game.

    This class extends the Block class to define a door that is locked and cannot be passed through. It manages the
    door's properties, its location, and the connection to other areas in the game.

    Args:
        location (Location): The location where the door is situated.
        door (Door): The door object representing the locked door.
        connection (Location): The location that the door connects to.

    Methods:
        is_blocked() -> bool:
            Determines if the door is blocked based on its locked status.

        to_primitive() -> dict:
            Converts the locked door's data into a primitive dictionary format.

        from_primitive(data: dict) -> Locked_Door:
            Creates an instance of Locked_Door from a primitive dictionary representation.
    """

    def __init__(self, location, door, connection):
        """
        Initializes a locked door object within a specified location and connection. This constructor sets up the door's
        properties and adds it to both the location and connection.

        Args:
            location: The location where the door is situated.
            door: The door object that represents the locked door.
            connection: The location that the door connects to.

        Returns:
            None
        """
        
        print(f"-\tInitializing Locked_Door", location, door, connection)

        # Call the parent class's constructor to initialize the door with a name and description
        super().__init__("locked door", "The door is locked")

        # Store the provided location, door, and connection in instance variables
        self.location = location
        self.door = door
        self.connection = connection

        # Determine the direction from the location to the connection
        loc_direction = self.location.get_direction(self.connection)

        # Add the door to the items in the location
        self.location.add_item(self.door)

        # Register this door as a block in the location for the specified direction
        self.location.add_block(loc_direction, self)

        # Determine the direction from the connection back to the location
        con_direction = self.connection.get_direction(self.location)

        # Add the door to the items in the connection
        self.connection.add_item(self.door)

        # Register this door as a block in the connection for the specified direction
        self.connection.add_block(con_direction, self)

        # Set the door's property to indicate that it is locked
        self.door.set_property("is_locked", True)

        # Add a command hint for unlocking the door
        self.door.add_command_hint("unlock door")

    def is_blocked(self) -> bool:
        """
        Determines if the door is blocked based on its locked status. This function checks if a door exists and whether
        it is locked, returning a boolean value accordingly.

        Returns:
            bool: True if the door is present and locked, otherwise False.
        """

        # Conditions of block:
        # * There is a door
        # * The door locked
        return bool(self.door and self.door.get_property("is_locked"))

    def to_primitive(self):
        """
        Converts the door object and its associated properties into a primitive dictionary format. This method gathers
        relevant information about the door, its location, and connection, ensuring that names are included when
        available.

        Returns:
            dict: A dictionary representation of the door object, including its location, door, and connection names if
            they exist.
        """

        # Call the parent class's method to get the primitive representation of the object
        data = super().to_primitive()

        # Check if the location exists and has a name attribute, then add it to the data dictionary
        if self.location and hasattr(self.location, "name"):
            data["location"] = self.location.name
        # If the location is already in data, ensure it is included
        elif "location" in data:
            data["location"] = self.location

        # Check if the door exists and has a name attribute, then add it to the data dictionary
        if self.door and hasattr(self.door, "name"):
            data["door"] = self.door.name
        # If the door is already in data, ensure it is included
        elif "door" in data:
            data["door"] = self.door

        # Check if the connection exists and has a name attribute, then add it to the data dictionary
        if self.connection and hasattr(self.connection, "name"):
            data["connection"] = self.connection.name
        # If the connection is already in data, ensure it is included
        elif "connection" in data:
            data["connection"] = self.connection

        # Return the constructed dictionary representation of the door object
        return data

    @classmethod
    def from_primitive(cls, data):
        """
        Creates an instance of the class from a primitive dictionary representation. This class method extracts the
        location, door, and connection from the provided data and initializes a new instance accordingly.

        Args:
            data (dict): A dictionary containing the keys "location", "door", and "connection" to reconstruct the
            instance.

        Returns:
            instance: A new instance of the class initialized with the provided data.
        """

        location = data["location"]
        door = data["door"]
        connection = data["connection"]
        return cls(location, door, connection)
