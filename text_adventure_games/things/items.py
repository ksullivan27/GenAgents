circular_import_prints = False

if circular_import_prints:
    print("Importing Items")

if circular_import_prints:
    print(f"\t{__name__} calling imports for Things Items")   
from .base import Thing

class Item(Thing):
    """
    Represents an item in a text adventure game, inheriting from the Thing class.

    This class allows for the creation of items that have a name, description, and additional attributes such as
    examine text, location, and ownership. It provides methods for serialization and deserialization to facilitate
    saving and loading item states.

    Attributes:
        examine_text (str): A detailed description of the item when examined by the player.
        location (str or None): The location of the item in the game world, if applicable.
        owner (str or None): The character or entity that owns the item, if applicable.

    Methods:
        to_primitive(): Converts the Item instance into a dictionary representation for serialization.
        from_primitive(data): Creates an Item instance from a dictionary of primitive values.
    """

    def __init__(self, name: str, description: str, examine_text: str = ""):
        """
        Initializes a new instance of the Item class with a name, description, and examine text.

        This constructor sets up the basic attributes of the item, including its detailed examine text, and initializes
        properties such as whether the item is gettable. It also prepares the item for potential location and ownership
        tracking within the game.

        Args:
            name (str): The name of the item.
            description (str): A brief description of the item.
            examine_text (str, optional): A detailed description shown when the player examines the item. Defaults to an
            empty string.

        Returns:
            None
        """
        
        if circular_import_prints:
            print(f"-\tInitializing Item", name)

        # Call the constructor of the parent Thing class to initialize the name, description, properties, and command
        # attributes.
        super().__init__(name, description)

        # Set the detailed description that will be displayed when the player examines the item.
        self.examine_text = examine_text

        # Mark the item as gettable, allowing the player to pick it up and add it to their inventory.
        self.set_property("gettable", True)

        # Initialize the location attribute to None, indicating that the item may not be placed anywhere initially.
        self.location = None

        # Initialize the owner attribute to None, indicating that the item may not belong to any character at the start.
        self.owner = None

    def to_primitive(self):
        """
        Converts the Item instance into a dictionary representation for serialization.

        This method extends the base class's to_primitive method by adding additional attributes specific to the Item,
        such as examine text, location, and owner. It prepares the data in a format suitable for JSON serialization
        while avoiding circular references (object instances are replaced with their name).

        Returns:
            dict: A dictionary representation of the Item instance, including its attributes.
        """

        # Call the parent class's to_primitive method to get the base dictionary representation.
        thing_data = super().to_primitive()

        # Add the examine text to the dictionary representation of the Item.
        thing_data["examine_text"] = self.examine_text

        # Check if the item has a location.
        if self.location:
            # If it has a name attribute, add the location name to the dictionary.
            if hasattr(self.location, "name"):
                thing_data["location"] = self.location.name
            # If the location is a string, add it directly to the dictionary.
            elif isinstance(self.location, str):
                thing_data["location"] = self.location

        # Check if the item has an owner.
        if self.owner:
            # If it has a name attribute, add the owner's name to the dictionary.
            if hasattr(self.owner, "name"):
                thing_data["owner"] = self.owner.name
            # If the owner is a string, add it directly to the dictionary.
            elif isinstance(self.owner, str):
                thing_data["owner"] = self.owner

        # Return the complete dictionary representation of the Item instance.
        return thing_data

    @classmethod
    def from_primitive(cls, data):
        """
        Creates an Item instance from a dictionary of primitive values.

        This method reconstructs an Item object using the provided data, including its name, description, examine text,
        location, and owner. It ensures that all relevant attributes are set correctly, allowing for the restoration of
        the item's state from serialized data.

        Args:
            data (dict): A dictionary containing the attributes to set on the Item instance.

        Returns:
            Item: The newly created Item instance populated with the provided data.
        """

        # Create a new instance of the Item class using the name, description, and examine text from the provided data.
        instance = cls(data["name"], data["description"], data["examine_text"])

        # Call the parent class's from_primitive method to populate the base attributes of the instance.
        instance = super().from_primitive(data, instance)

        # If the location attribute is present in the data, set it on the instance.
        if "location" in data:
            instance.location = data["location"]

        # If the owner attribute is present in the data, set it on the instance.
        if "owner" in data:
            instance.owner = data["owner"]

        # Return the fully populated Item instance.
        return instance
