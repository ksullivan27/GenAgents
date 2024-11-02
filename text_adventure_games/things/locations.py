"""
Locations

Locations are the places in the game that a player can visit.  They are connected to other locations and contain items
that the player can interact with.  A connection to an adjacent location can be blocked (often introducing a puzzle for
the player to solve before making progress).
"""

print("Importing Locations")

print(f"\t{__name__} calling imports for Base")
from .base import Thing
print(f"\t{__name__} calling imports for Items")   
from .items import Item

class Location(Thing):
    """
    Represents a location in a text adventure game, inheriting from the Thing class.

    This class allows for the creation of locations that can connect to other locations, contain items and characters,
    and manage travel descriptions and obstacles. It provides methods for serialization, connection management, and item
    and character handling within the location.

    Locations are the places in the game that a player can visit. Internally they are represented as nodes in a graph.
    Each location stores a description of the location, any items in the location, its connections to adjacent
    locations, and any blocks that prevent movement to an adjacent location. The connections is a dictionary whose keys
    are directions and whose values are the location that is the result of traveling in that direction. The
    travel_descriptions also has directions as keys, and its values are an optional short description of traveling to
    that location.

    Attributes:
        connections (dict): A dictionary mapping directions to other Location objects.
        travel_descriptions (dict): A dictionary mapping directions to text describing travel in that direction.
        blocks (dict): A dictionary mapping directions to Block objects that may obstruct movement.
        items (dict): A dictionary mapping item names to Item objects present in the location.
        characters (dict): A dictionary mapping character names to Character objects present in the location.
        has_been_visited (bool): A flag indicating whether the location has been visited by the player.

    Methods:
        to_primitive(): Converts the Location instance into a dictionary representation for serialization.
        from_primitive(data): Creates a Location instance from a dictionary of primitive values.
        add_connection(direction, connected_location, travel_description): Adds a connection to another location in a
        specified direction.
        get_connection(direction): Retrieves the connected location in the specified direction.
        get_direction(location): Finds the direction to a specified connected location.
        here(thing, describe_error): Checks if a specified thing is at the location.
        get_item(name): Retrieves an item by name from the location.
        add_item(item): Adds an item to the location.
        remove_item(item): Removes an item from the location.
        add_character(character): Adds a character to the location.
        remove_character(character): Removes a character from the location.
        is_blocked(direction): Checks if there is an obstacle in the specified direction.
        get_block_description(direction): Retrieves the description of the block in the specified direction.
        add_block(blocked_direction, block): Adds an obstacle in a specified direction.
        remove_block(block): Removes a specified block from the location.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes a new instance of the Location class with a name and description.

        This constructor sets up the basic attributes of the location, including connections to other locations, travel
        descriptions, blocks, items, and characters present in the location. It also initializes a flag to track whether
        the location has been visited by the player.

        Args:
            name (str): The name of the location.
            description (str): A brief description of the location.

        Returns:
            None
        """
        
        print(f"-\tInitializing Location", name)

        # Call the constructor of the parent Thing class to initialize the name and description attributes.
        super().__init__(name, description)

        # Initialize a dictionary to map directions to other Location objects for navigation.
        self.connections = {}

        # Initialize a dictionary to map directions to text that describes traveling in those directions.
        self.travel_descriptions = {}

        # Initialize a dictionary to map directions to Block objects that may obstruct movement.
        self.blocks = {}

        # Initialize a dictionary to map item names to Item objects present in this location.
        self.items = {}

        # Initialize a dictionary to map character names to Character objects present in this location.
        self.characters = {}

        # Set a flag to indicate whether this location has been visited by the player.
        self.has_been_visited = False

    def to_primitive(self):
        """
        Converts the Location instance into a dictionary representation for serialization.

        This method extends the base class's to_primitive method by adding additional attributes specific to the
        Location, such as travel descriptions, blocks, connections, items, and characters. It prepares the data in a
        format suitable for JSON serialization while avoiding circular references.

        Notice that object instances are replaced with their name. This prevents circular references that interfere with
        recursive serialization.

        Returns:
            dict: A dictionary representation of the Location instance, including its attributes.
        """

        # Call the parent class's to_primitive method to get the base dictionary representation.
        thing_data = super().to_primitive()

        # Add the travel descriptions to the dictionary representation of the Location.
        thing_data["travel_descriptions"] = self.travel_descriptions

        # Convert each Block object in the blocks dictionary to its primitive representation.
        blocks = {k: v.to_primitive() for k, v in self.blocks.items()}
        thing_data["blocks"] = blocks

        # Prepare a dictionary of connections, storing the name of each connected location if available.
        connections = {
            k: v.name if v and hasattr(v, "name") else v
            for k, v in self.connections.items()
        }
        thing_data["connections"] = connections

        # Convert each Item object in the items dictionary to its primitive representation.
        items = {k: Item.to_primitive(v) for k, v in self.items.items()}
        thing_data["items"] = items

        # Prepare a dictionary of characters, storing the name of each character if available.
        characters = {
            k: v.name if v and hasattr(v, "name") else v
            for k, v in self.characters.items()
        }
        thing_data["characters"] = characters

        # Add the visited flag to the dictionary representation.
        thing_data["has_been_visited"] = self.has_been_visited

        # Return the complete dictionary representation of the Location instance.
        return thing_data

    @classmethod
    def from_primitive(cls, data):
        """
        Creates a Location instance from a dictionary of primitive values.

        This method reconstructs a Location object using the provided data, including its name, description, travel
        descriptions, blocks, connections, items, characters, and visited status. It ensures that all relevant
        attributes are set correctly, allowing for the restoration of the location's state from serialized data.

        Args:
            data (dict): A dictionary containing the attributes to set on the Location instance.

        Returns:
            Location: The newly created Location instance populated with the provided data.
        """

        # Create a new instance of the Location class using the name and description from the provided data.
        instance = cls(data["name"], data["description"])

        # Call the parent class's from_primitive method to populate the base attributes of the instance.
        instance = super().from_primitive(data, instance)

        # Set the travel descriptions for the instance from the provided data.
        instance.travel_descriptions = data["travel_descriptions"]

        # Assign the blocks from the provided data to the instance; note that blocks are not instantiated here.
        instance.blocks = data["blocks"]  # skeleton doesn't instantiate blocks

        # Set the connections for the instance from the provided data.
        instance.connections = data["connections"]

        # Convert each Item object in the items dictionary from its primitive representation and assign it to the
        # instance.
        instance.items = {k: Item.from_primitive(v) for k, v in data["items"].items()}

        # Set the characters for the instance from the provided data.
        instance.characters = data["characters"]

        # Set the visited status for the instance from the provided data.
        instance.has_been_visited = data["has_been_visited"]

        # Set the properties for the instance from the provided data.
        instance.properties = data["properties"]

        # Return the fully populated Location instance.
        return instance

    def add_connection(
        self, direction: str, connected_location, travel_description: str = ""
    ):
        """
        Adds a connection from the current location to a specified connected location.

        This method establishes a bidirectional connection between the current location and the connected location,
        allowing for navigation in the specified direction. It also sets a travel description for the connection,
        enhancing the player's experience when moving between locations. If the direction is a cardinal direction, then
        we also automatically make a connection in the reverse direction.

        Args:
            direction (str): The direction to connect to the connected location (e.g., 'north', 'south').
            connected_location: The location to which the current location is being connected.
            travel_description (str, optional): A description of the travel experience in that direction. Defaults to an
            empty string.

        Returns:
            None
        """

        # Convert the direction to lowercase to ensure consistency in key naming.
        direction = direction.lower()

        # Add the connected location to the connections dictionary for the specified direction.
        self.connections[direction] = connected_location

        # Set the travel description for the specified direction.
        self.travel_descriptions[direction] = travel_description

        # Establish a bidirectional connection for the north direction.
        if direction == "north":
            connected_location.connections["south"] = (
                self  # Connect the south direction back to this location.
            )
            connected_location.travel_descriptions["south"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the south direction.
        elif direction == "south":
            connected_location.connections["north"] = (
                self  # Connect the north direction back to this location.
            )
            connected_location.travel_descriptions["north"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the east direction.
        elif direction == "east":
            connected_location.connections["west"] = (
                self  # Connect the west direction back to this location.
            )
            connected_location.travel_descriptions["west"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the west direction.
        elif direction == "west":
            connected_location.connections["east"] = (
                self  # Connect the east direction back to this location.
            )
            connected_location.travel_descriptions["east"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the up direction.
        elif direction == "up":
            connected_location.connections["down"] = (
                self  # Connect the down direction back to this location.
            )
            connected_location.travel_descriptions["down"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the down direction.
        elif direction == "down":
            connected_location.connections["up"] = (
                self  # Connect the up direction back to this location.
            )
            connected_location.travel_descriptions["up"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the in direction.
        elif direction == "in":
            connected_location.connections["out"] = (
                self  # Connect the out direction back to this location.
            )
            connected_location.travel_descriptions["out"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the out direction.
        elif direction == "out":
            connected_location.connections["in"] = (
                self  # Connect the in direction back to this location.
            )
            connected_location.travel_descriptions["in"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the inside direction.
        elif direction == "inside":
            connected_location.connections["outside"] = (
                self  # Connect the outside direction back to this location.
            )
            connected_location.travel_descriptions["outside"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

        # Establish a bidirectional connection for the outside direction.
        elif direction == "outside":
            connected_location.connections["inside"] = (
                self  # Connect the inside direction back to this location.
            )
            connected_location.travel_descriptions["inside"] = (
                ""  # Set an empty travel description for the reverse direction.
            )

    def get_connection(self, direction: str):
        """
        Retrieves the connected location in the specified direction.

        This method checks the connections of the current location and returns the location associated with the given
        direction. If there is no connection in that direction, it returns None.

        Args:
            direction (str): The direction for which to retrieve the connected location.

        Returns:
            Location or None: The connected location in the specified direction, or None if no connection exists.
        """

        return self.connections.get(direction, None)

    def get_direction(self, location):
        """
        Finds the direction to a specified connected location.

        This method iterates through the connections of the current location and returns the direction associated with
        the given location. If the location is not found among the connections, it returns None.

        Args:
            location: The location for which to determine the direction.

        Returns:
            str or None: The direction to the specified location, or None if the location is not connected.
        """

        # Iterate through the connections dictionary, where k is the direction and v is the connected location.
        return next((k for k, v in self.connections.items() if v == location), None)

    def here(self, thing: Thing, describe_error: bool = True) -> bool:
        """
        Checks if a specified thing is present at the current location.

        This method verifies whether the given thing is located at the current location. It returns True if the thing is
        at the location, and False otherwise.

        Args:
            thing (Thing): The thing to check for presence at the location.
            describe_error (bool, optional): A flag indicating whether to describe an error if the thing is not present.
            Defaults to True.

        Returns:
            bool: True if the thing is at the location, False otherwise.
        """

        # TODO: implement describe_error

        # Check if the specified thing's location matches the current location.
        return thing.location == self

    def get_item(self, name: str):
        """
        Retrieves an item by name from the current location.

        This method checks the items present in the location and returns the item associated with the specified name.
        If the item is not found, it returns None.

        Args:
            name (str): The name of the item to retrieve.

        Returns:
            Item or None: The item with the specified name, or None if the item is not present in the location.
        """

        # The character must be at the location
        return self.items.get(name, None)

    def add_item(self, item):
        """
        Adds an item to the current location.

        This method places the specified item in the location's inventory and updates the item's location attribute to
        reflect its new position. It also sets the item's owner to None, indicating that it is not currently held by any
        character.

        Args:
            item: The item to be added to the location.

        Returns:
            None
        """

        # Add the item to the location's inventory, using the item's name as the key.
        self.items[item.name] = item

        # Update the item's location attribute to reflect that it is now in this location.
        item.location = self

        # Set the item's owner to None, indicating that it is not currently owned by any character.
        item.owner = None

    def remove_item(self, item):
        """
        Removes an item from the current location.

        This method deletes the specified item from the location's inventory and updates the item's location attribute
        to indicate that it is no longer in this location. This is typically used when a player picks up an item or when
        an item is otherwise removed from the scene.

        Args:
            item: The item to be removed from the location.

        Returns:
            None
        """

        # Remove the item from the location's inventory using its name as the key.
        self.items.pop(item.name)

        # Update the item's location attribute to None, indicating that it is no longer in this location.
        item.location = None

    def add_character(self, character):
        """
        Adds a character to the current location.

        This method places the specified character in the location's character inventory and updates the character's
        location attribute to reflect its new position. This is typically used when a character enters a location in
        the game.

        Args:
            character: The character to be added to the location.

        Returns:
            None
        """

        # Add the character to the location's character inventory, using the character's name as the key.
        self.characters[character.name] = character

        # Update the character's location attribute to reflect that they are now in this location.
        character.location = self

    def remove_character(self, character):
        """
        Removes a character from the current location.

        This method deletes the specified character from the location's character inventory and updates the character's
        location attribute to indicate that they are no longer in this location. This is typically used when a character
        leaves the location or is otherwise removed from the scene.

        Args:
            character: The character to be removed from the location.

        Returns:
            None
        """

        # Remove the character from the location's character inventory using their name as the key.
        self.characters.pop(character.name)

        # Update the character's location attribute to None, indicating that they are no longer in this location.
        character.location = None

    def is_blocked(self, direction: str) -> bool:
        """
        Checks if there is an obstacle in the specified direction.

        This method determines whether movement in the given direction is blocked by checking the blocks dictionary. If
        there is no block in that direction, it returns False; otherwise, it returns the result of the block's
        is_blocked method.

        Args:
            direction (str): The direction to check for obstacles.

        Returns:
            bool: True if movement in the specified direction is blocked, False otherwise.
        """

        # Check if the specified direction is not present in the blocks dictionary.
        if direction not in self.blocks:  # JD logical change
            return False  # Return False if there are no blocks in that direction, indicating movement is not blocked.

        # Retrieve the block object associated with the specified direction.
        block = self.blocks[direction]

        # Return the result of the block's is_blocked method to determine if movement is obstructed.
        return block.is_blocked()

    def get_block_description(self, direction: str):
        """
        Retrieves the description of the block in the specified direction.

        This method checks if there is a block in the given direction and returns its description. If there is no block
        in that direction, it returns an empty string.

        Args:
            direction (str): The direction for which to retrieve the block description.

        Returns:
            str: The description of the block in the specified direction, or an empty string if no block exists.
        """

        # Check if there is no block in the specified direction.
        if direction not in self.blocks:
            return ""  # Return an empty string if there is no block in that direction.

        # Retrieve the block object associated with the specified direction.
        block = self.blocks[direction]

        # Return the description of the block.
        return block.description

    def add_block(self, blocked_direction: str, block):
        """
        Adds a block to the specified direction to prevent movement.

        This method associates a block with a given direction, effectively creating an obstacle that restricts movement
        in that direction. This is useful for implementing barriers or challenges within the game environment.

        Args:
            blocked_direction (str): The direction in which the block is placed.
            block: The block object that will be added to the specified direction.

        Returns:
            None
        """

        self.blocks[blocked_direction] = block

    def remove_block(self, block):
        """
        Removes a specified block from the current location.

        This method searches through the blocks associated with the location and deletes the specified block if it is
        found. This is useful for clearing obstacles that may no longer be relevant in the game environment.

        Args:
            block: The block object to be removed from the location.

        Returns:
            None
        """

        # Iterate through the blocks dictionary, where k is the direction and b is the block object.
        for k, b in self.blocks.items():
            # Check if the current block matches the specified block to be removed.
            if b == block:
                del self.blocks[k]  # Delete the block from the blocks dictionary.
                break  # Exit the loop after removing the block.
