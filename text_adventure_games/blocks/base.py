"""Blocks

Blocks are things that prevent movement in a direction - for examlpe a locked
door may prevent you from entering a room, or a hungry troll might block you
from crossing the drawbridge.  We implement them similarly to how we did
Special Actions.

CCB - todo - consider refactoring Block to be Connection that join two
locations.  Connection could support the is_blocked() method, and also be a
subtype of Item which might make it easier to create items that are shared
between two locations (like doors).
"""

import traceback

# Print a message showing that the module is being imported
print(f"Importing Blocks Base")

# Get the call stack and format it
stack = traceback.format_stack()

# Print the stack of modules calling this module
print("Call stack for import:")
for line in stack:
    print(line.strip())
class Block:
    """Represents a basic block in a text adventure game.

    This class encapsulates the properties and behaviors of a block, including its name and description. It provides
    methods to determine if the block is blocked and to convert the block's data into a primitive format.

    Args:
        name (str): The name of the block.
        description (str): A description of the block.

    Methods:
        is_blocked() -> bool:
            Returns whether the block is blocked.

        to_primitive() -> dict:
            Converts the block's data into a primitive dictionary format.
    """

    def __init__(self, name, description):
        """Initializes a new block with a name and description.

        This constructor sets the name and description attributes for the block, allowing it to be identified and
        described in the context of a text adventure game.

        Args:
            name (str): The name of the block.
            description (str): A description of the block.
        """
        
        print(f"-\tInitializing Block", name)

        # Assign the provided name to the block's name attribute
        self.name = name

        # Assign the provided description to the block's description attribute
        self.description = description

    def is_blocked(self) -> bool:
        """Determines if the block is currently blocked.

        This method always returns True, indicating that the block is considered blocked in the context of the game.

        Returns:
            bool: Always returns True, signifying that the block is blocked.
        """

        return True

    def to_primitive(self):
        """Converts the block's data into a primitive dictionary format.

        This method creates a dictionary representation of the block, including its class type. It is intended to
        facilitate serialization or data transfer by providing a simplified view of the block's attributes.

        Returns:
            dict: A dictionary containing the class type of the block.
        """

        # Retrieve the name of the class of the current instance
        cls_type = self.__class__.__name__

        # Return a dictionary containing the class type of the block
        return {
            "_type": cls_type,  # The type of the block, determined by its class name
            # subclasses hardcode these
            # 'name': self.name,  # The name of the block (to be included in subclasses)
            # 'description': self.description,  # The description of the block (to be included in subclasses)
        }
