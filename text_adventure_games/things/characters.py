import os
from typing import List, Union

# Local imports from the current package and parent packages to access various classes and functions.
# Import the base class 'Thing' from the base module.
from .base import Thing

# Import the 'Item' class from the items module to represent items in the game.
from .items import Item

# Import MemoryStream and MemoryType from the agent.memory_stream module for managing agent memory.
from ..agent.memory_stream import MemoryStream, MemoryType

# Import the Act class from the agent_cognition module to handle actions performed by agents.
from ..agent.agent_cognition.act import Act

# Import the Impressions class from the agent_cognition module to manage agent impressions of other characters.
from ..agent.agent_cognition.impressions import Impressions

# Import the Goals class from the agent_cognition module to manage agent goals.
from ..agent.agent_cognition.goals import Goals

# Import the perceive_location function from the agent_cognition module to help agents perceive their surroundings.
from ..agent.agent_cognition.perceive import perceive_location

# Import the context_list_to_string function from the gpt_helpers module to convert context lists into string format.
from ..gpt.gpt_helpers import context_list_to_string


# Used to map group to use_goals and use_impressions
GROUP_MAPPING = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
    "E": (False, False),
}


class Character(Thing):
    """
    Represents a character in the game, inheriting from the Thing class. This class manages the character's attributes,
    including its name, description, persona, inventory, and location, while providing methods for inventory management
    and serialization.

    Args:
        name (str): The name of the character.
        description (str): A description of the character.
        persona (str): The persona associated with the character (written in the first person).

    Returns:
        None
    """

    def __init__(
        self,
        name: str,
        description: str,
        persona: str,
    ):
        """
        Initializes a character with specified attributes, including name, description, and persona. This constructor
        method sets up the character's properties, inventory, location, and memory, allowing for the management of the
        character's state within the game.

        Args:
            name (str): The name of the character.
            description (str): A description of the character.
            persona (str): The persona associated with the character.

        Returns:
            None
        """

        # Call the initializer of the parent class (Thing) to set up the name and description attributes for the
        # character.
        super().__init__(name, description)

        # Set the character type property to a default value of "notset".
        self.set_property("character_type", "notset")

        # Initialize the is_dead property to False, indicating that the character is alive.
        self.set_property("is_dead", False)

        # Assign the provided persona to the character's persona attribute.
        self.persona = persona

        # Initialize an empty dictionary to hold the character's inventory items.
        self.inventory = {}

        # Set the initial location of the character to None, indicating that the character has not been placed in a
        # location yet.
        self.location = None

        # Initialize an empty list to store the character's memories.
        self.memory = []

    def __hash__(self):
        """
        Returns the hash value of the character based on its unique identifier. This method allows instances of the
        character to be used in hash-based collections, such as sets and dictionaries.

        Returns:
            int: The hash value of the character's ID.
        """

        return hash(self.id)

    def __eq__(self, other):
        """
        Compares two character instances for equality based on their unique identifiers.
        This method ensures that only instances of the Character class are compared, returning True if their IDs match
        and False otherwise.

        Args:
            other (object): The object to compare against the current character.

        Returns:
            bool: True if the characters are equal (same ID), False otherwise.

        Raises:
            NotImplemented: If the other object is not an instance of Character.
        """

        # Check if the object being compared is not an instance of the Character class.
        return self.id == other.id if isinstance(other, Character) else NotImplemented

    def to_primitive(self):
        """
        Convert the character instance into a primitive dictionary representation, whose values can be safely
        serialized to JSON. This representation includes the character's persona, inventory, and location. Notice that
        object instances are replaced with their name. This prevents circular references that interfere with recursive
        serialization.

        The method gathers data from the character's attributes and formats it into a
        dictionary that can be easily serialized or used for other purposes.

        Args:
            self: The instance of the character.

        Returns:
            dict: A dictionary containing the character's persona, inventory, and location.
        """

        # Call the parent class's to_primitive method to get the base data representation.
        thing_data = super().to_primitive()

        # Add the character's persona to the data representation for serialization.
        thing_data["persona"] = self.persona

        # Create a dictionary for the character's inventory, converting each item to its primitive representation if it
        # has one.
        inventory = {
            k: (
                v.to_primitive() if hasattr(v, "to_primitive") else v
            )  # Check if the item has a to_primitive method.
            for k, v in self.inventory.items()
        }

        # Add the inventory dictionary to the character's data representation.
        thing_data["inventory"] = inventory

        # Check if the character has a location assigned and if it has a name attribute.
        if self.location:
            if hasattr(self.location, "name"):
                # If the location has a name, add the location's name to the data representation.
                thing_data["location"] = self.location.name
            else:
                # If the location doesn't have a name, add the location object directly to the data representation.
                thing_data["location"] = self.location

        # Return the complete data representation of the character, ready for serialization.
        return thing_data

    @classmethod
    def from_primitive(cls, data):
        """
        Create a character instance from a primitive dictionary representation.
        This method reconstructs the character's attributes, including its name, description, persona, location, and
        inventory.

        The method initializes a new instance of the character class using the provided data and populates its
        attributes accordingly.

        Notice that the from_primitive method is called for items.

        Args:
            cls: The class that is being instantiated.
            data (dict): A dictionary containing the character's attributes.

        Returns:
            Character: An instance of the character class populated with the provided data.
        """

        # Create a new instance of the character class using the provided name, description, and persona
        instance = cls(data["name"], data["description"], data["persona"])

        # Call the parent class's from_primitive method to populate common attributes of the instance
        instance = super().from_primitive(data, instance=instance)

        # Set the character's location, defaulting to None if not provided in the data
        instance.location = data.get("location", None)

        # Populate the character's inventory by converting each item from its primitive representation
        instance.inventory = {
            k: Item.from_primitive(v) for k, v in data["inventory"].items()
        }

        # Return the fully constructed character instance
        return instance

    def add_to_inventory(self, item):
        """
        Add an item to the character's inventory.
        This method updates the item's location and ownership when it is added to the inventory.

        The function first checks if the item is currently located somewhere; if so, it removes the item
        from that location before adding it to the character's inventory and setting the item's owner.

        Args:
            self: The instance of the character.
            item: The item to be added to the inventory.

        Returns:
            None
        """

        # Check if the item has a location
        if item.location is not None:
            # Remove the item from its current location
            item.location.remove_item(item)
            # Set the item's location to None
            item.location = None

        # Add the item to the inventory using its name as the key
        self.inventory[item.name] = item

        # Set the owner of the item to the current instance
        item.owner = self

    def is_in_inventory(self, item):
        """
        Check if an item is present in the character's inventory.
        This method verifies the existence of the item by its name within the inventory.

        The function returns a boolean indicating whether the specified item is currently held by the character.

        Args:
            self: The instance of the character.
            item: The item to check for in the inventory.

        Returns:
            bool: True if the item is in the inventory, False otherwise.
        """

        return item.name in self.inventory

    def remove_from_inventory(self, item):
        """
        Remove an item from the character's inventory.
        This method updates the item's ownership and removes it from the inventory.

        The function sets the item's owner to None and removes the item from the character's inventory
        using its name as the key.

        Args:
            self: The instance of the character.
            item: The item to be removed from the inventory.

        Returns:
            None
        """

        item.owner = None
        self.inventory.pop(item.name)

    def get_item_by_name(self, item_name):
        """
        Retrieve an item from the character's inventory by its name.
        This method returns the item if it exists in the inventory, or None if it does not.

        The function checks the inventory dictionary for the specified item name and returns the corresponding item
        or None if the item is not found.

        Args:
            self: The instance of the character.
            item_name (str): The name of the item to retrieve.

        Returns:
            Item or None: The item if found in the inventory, otherwise None.
        """

        return self.inventory.get(item_name, None)


class GenerativeAgent(Character):
    """
    A class representing a generative agent that extends the Character class.
    This agent has cognitive capabilities, including managing goals, impressions, and memory, and can interact with the
    game world.

    The GenerativeAgent initializes with a persona and a group, setting up its cognitive settings, memory, and tracking
    interactions with other characters. It provides methods for engaging with the game, managing goals, and updating
    impressions based on the agent's surroundings.

    Attributes:
        group (str): The group classification of the agent.
        use_goals (bool): Indicates if the agent uses goals.
        use_impressions (bool): Indicates if the agent uses impressions.
        persona: The persona of the agent.
        impressions: The agent's impressions of other characters.
        goals: The agent's goals.
        memory: The agent's memory stream.
        last_location_observations: Observations from the last location.
        last_talked_to: The last character the agent interacted with.
        idol_search_count (int): Count of idol searches performed by the agent.
    """

    def __init__(self, persona, group: str = "D"):
        """
        Initialize a GenerativeAgent with a specified persona and cognitive group.
        This constructor sets up the agent's attributes, including its persona, cognitive settings, memory, and tracking
        of interactions.

        The agent's group determines its cognitive capabilities, such as the use of goals and impressions.
        The constructor also initializes the agent's memory stream and tracks the last character it interacted with.

        Args:
            self: The instance of the agent.
            persona: The persona object containing the agent's characteristics.
            group (str, optional): The group classification of the agent, default is "D" (the full cognitive
            architecture).

        Returns:
            None
        """

        # Call the parent class's constructor to initialize the character with the persona's name, description, and
        # summary.
        super().__init__(
            persona.facts["Name"], persona.description, persona=persona.summary
        )

        # Set the cognitive group for the agent and determine if goals and impressions will be used based on the group.
        self.group = group
        self.use_goals, self.use_impressions = GROUP_MAPPING[self.group]

        # Assign the agent's persona and initialize impressions and goals based on cognitive settings.
        self.persona = persona
        if self.use_impressions:
            # If impressions are used, create an Impressions object for the agent to track interactions.
            self.impressions = Impressions(self)
        else:
            # If impressions are not used, set the impressions attribute to None.
            self.impressions = None

        # Initialize the goals for the agent based on whether goals are used.
        self.goals = Goals(self) if self.use_goals else None

        # Initialize the agent's memory stream to track observations made by the agent.
        self.memory = MemoryStream(self)

        # Set the last location observations to None initially, indicating no observations have been made yet.
        self.last_location_observations = None

        # Track the last character the agent interacted with, initialized to None.
        self.last_talked_to = None

        # Initialize a counter for the number of idol searches performed by the agent, starting at zero.
        self.idol_search_count = 0

    def set_dialogue_participant(self, talked_to):
        """
        Set the last character that the agent has interacted with during dialogue.
        This method updates the agent's record of the last dialogue participant based on the provided input.

        If the input is None, it clears the last participant. If the input is a valid Character instance, it updates the
        record; otherwise, it raises a ValueError for invalid input.

        Args:
            self: The instance of the agent.
            talked_to: The character that the agent has interacted with, or None to clear the record.

        Raises:
            ValueError: If the provided input is not a Character instance or None.
        """

        # Check if the input is None; if so, clear the record of the last dialogue participant
        if not talked_to:
            self.last_talked_to = None
        # Check if the input is a valid Character instance
        elif isinstance(talked_to, Character):
            # Update the last dialogue participant to the provided character
            self.last_talked_to = talked_to
        # If the input is neither None nor a valid Character, raise an error
        else:
            raise ValueError(f"{talked_to} is invalid.")

    def get_last_dialogue_target(self):
        """
        Retrieve the last character that the agent interacted with during dialogue.
        This method returns the record of the last dialogue participant, which can be used for context in future
        interactions.

        Args:
            self: The instance of the agent.

        Returns:
            Character or None: The last character the agent talked to, or None if no interaction has occurred.
        """

        # Return the last character that the agent interacted with during dialogue
        return self.last_talked_to

    def _parse_perceptions(self):
        """
        Generate a descriptive summary of the agent's perceptions based on the last location observations.
        This method processes the observations to create a human-readable format that reflects what the agent perceives
        in its environment.

        The function analyzes the types of perceptions and their values, handling cases where there are multiple
        perceptions or none at all. It constructs a list of perception descriptions and returns them as a formatted
        string.

        Args:
            self: The instance of the agent.

        Returns:
            str: A formatted string summarizing the agent's perceptions.
        """

        # Initialize an empty list to hold descriptions of perceptions.
        perception_descriptions = []

        # Iterate over each type of perception and its corresponding values from the last location observations.
        for ptype, percept in self.last_location_observations.items():
            # Check if there is only one perception for the current type.
            if len(percept) == 1:
                # If the perception indicates absence (starts with "No"), append a corresponding message.
                if percept[0].startswith("No "):
                    perception_descriptions.append(f"{self.name} has no {ptype}.")
                else:
                    # Otherwise, append the single perception value to the descriptions.
                    perception_descriptions.append(percept[0])
            # Check for a common prefix among multiple perceptions.
            elif common_prefix := os.path.commonprefix(percept):
                # Strip the common prefix from each perception and capitalize the first character.
                unique_parts = [p[len(common_prefix) :].capitalize() for p in percept]
                # Append a description that includes the common prefix and unique parts.
                perception_descriptions.append(
                    f"{common_prefix.strip()}: {', '.join(unique_parts)}"
                )
            else:
                # If there is no common prefix, join the perceptions with ', ' and append to the descriptions.
                perception_descriptions.append(", ".join(percept))

        # Return the formatted string of perception descriptions, joining them with new lines.
        return context_list_to_string(perception_descriptions, sep="\n")

    def get_standard_info(self, game, include_goals=True, include_perceptions=True):
        """
        Generate a standard summary of the agent's current context within the game.
        This method provides information about the game world, the agent's persona, and optionally includes goals and
        perceptions.

        The summary includes world information, a personal summary of the agent, and, if specified, the agent's current
        goals and perceptions.
        This is useful for providing the agent with relevant context for decision-making.

        Args:
            self: The instance of the agent.
            game: The current game object containing world information.
            include_goals (bool, optional): Whether to include the agent's goals in the summary. Default is True.
            include_perceptions (bool, optional): Whether to include the agent's perceptions in the summary. Default is
            True.

        Returns:
            str: A formatted summary string containing the agent's context.
        """

        # Start building the summary with world information from the game
        summary = f"WORLD INFO: {game.world_info}\n"
        # Add the agent's personal summary to the summary
        summary += f"You are {self.persona.get_personal_summary()}.\n"

        # Check if the agent uses goals and if goals should be included in the summary
        if self.use_goals and include_goals:
            # Retrieve the current goals for the agent based on the current game round, appending them to the summary.
            if goals := self.get_goals(round=game.round, as_str=True):
                summary += f"Your current GOALS:\n{goals}\n"

        # Check if perceptions should be included and if there are any last location observations
        if include_perceptions and self.last_location_observations:
            # Parse the agent's perceptions into a readable format, appending them to the summary.
            if perceptions := self._parse_perceptions():
                summary += f"Your current perceptions are:\n{perceptions}\n"

        # Return the complete summary string
        return summary

    def get_goals(self, round=-1, priority="all", as_str=False):
        """
        Retrieve the agent's current goals based on the specified round and priority.
        This method returns the goals if the agent is configured to use them; otherwise, it returns None.

        The function allows for filtering goals by round and priority, and can return the goals in a string format if
        specified.

        Args:
            self: The instance of the agent.
            round (int, optional): The game round for which to retrieve goals. Default is -1, which typically indicates
            the current round.
            priority (str, optional): The priority level for filtering goals. Default is "all".
            as_str (bool, optional): If True, returns the goals as a string. Default is False.

        Returns:
            list or str or None: A list of goals for the specified round and priority, or None if goals are not used.
        """

        # Check if the agent is configured to use goals.
        return (
            # If goals are enabled, retrieve the goals for the current round with the specified priority and format.
            self.goals.get_goals(round=round, priority=priority, as_str=as_str)
            # If goals are not enabled, return None.
            if self.use_goals
            else None
        )

    def generate_goals(self, game):
        """
        Generate new goals for the agent at the start of a game round.
        This method checks if the current game tick indicates the beginning of a round and, if so, invokes the goal
        generation process.

        The function is designed to set up the agent's goals based on the game's current state, ensuring that the agent
        has relevant objectives to pursue.

        Args:
            self: The instance of the agent.
            game: The current game object, which contains information about the game state.

        Returns:
            None
        """

        # Check if the current game tick indicates the start of a new round and if the agent is configured to use goals
        if game.tick == 0 and self.use_goals:

            # Uncomment the following line to debug and see which agent's goals are being set
            # print(f"Setting goal for {self.name}")

            # Generate new goals for the agent based on the current game state
            self.goals.gpt_generate_goals(game)

    def engage(self, game) -> Union[str, int]:
        """
        Manage the agent's actions and interactions within the game during a round.
        This method handles perception, updates impressions, and executes actions based on the current game state.
        It's a wrapper method for all agent cognition: perceive, retrieve, act, reflect, set goals.

        If the current game tick indicates the end of a round, the method forces reflection and evaluates the agent's
        goals. Otherwise, it perceives the surroundings, updates impressions of nearby characters, and performs the
        appropriate action.

        Args:
            self: The instance of the agent.
            game: The current game object containing the game state.

        Returns:
            Union[str, int]: An action string representing the agent's decision or an integer flag (-999) to indicate a
            skipped action.
        """

        # Check if the current game tick is the last one in the round
        if game.tick == (game.max_ticks_per_round - 1):
            # Force the agent to reflect on the round's events
            self.memory.reflect(game)
            # If the agent uses goals, evaluate them at the end of the round
            if self.use_goals:
                self.goals.evaluate_goals(game)
            # Return -999 to indicate that the action was skipped
            return -999

        # Allow the agent to perceive its surroundings and gather information
        self.perceive(game)

        # If the agent uses impressions, update its perceptions of nearby characters
        if self.use_impressions:
            self.update_character_impressions(game)

        # Execute the appropriate action based on the agent's current state and decisions
        return Act(game, self).act()

    def perceive(self, game):
        """
        Gather information about the agent's surroundings in the game.
        This method updates the agent's perception of its location and identifies other characters present in the same
        area.

        The function calls an external method to assess the current location and updates the agent's view of nearby
        characters, allowing for informed decision-making in subsequent actions.

        Args:
            self: The instance of the agent.
            game: The current game object containing the game state.

        Returns:
            None
        """

        # Assess the current location and update the agent's perception of the environment
        perceive_location(game, self)

        # Retrieve and store the list of characters that are currently in view of the agent
        self.chars_in_view = self.get_characters_in_view(game)

    def get_characters_in_view(self, game):
        """
        Identify and retrieve a list of characters that are currently in the same location as the agent.
        This method filters the characters in the game to find those that share the same location ID as the agent,
        excluding itself.

        The function iterates through all characters in the game and compiles a list of those present in the agent's
        location, which can be useful for interactions and decision-making.

        Args:
            self: The instance of the agent.
            game: The current game object containing all characters and their states.

        Returns:
            list: A list of characters that are in view of the agent.
        """

        # Return a list of characters that are located in the same location as the current character.
        return [
            char  # Include the character in the list if the following conditions are met.
            for char in game.characters.values()  # Iterate over all characters in the game.
            if char.location.id == self.location.id
            and char.id
            != self.id  # Check if the character's location matches and is not the current character.
        ]

    def update_character_impressions(self, game):
        """
        Update the agent's impressions of nearby characters based on their interactions.
        This method assesses the characters currently in view and refreshes the agent's impressions of them.

        The function iterates through the characters that the agent can see and updates the impressions
        using the agent's current context and interactions, allowing for dynamic changes in perception.

        Args:
            self: The instance of the agent.
            game: The current game object containing the state of the game and characters.

        Returns:
            None
        """

        # Iterate through each character that the agent can currently see
        for target in self.get_characters_in_view(game):
            # Update the agent's impression of the target character based on the current game context
            self.impressions.update_impression(game, self, target)

    def to_primitive(self):
        """
        Convert the agent's state into a primitive dictionary representation.
        This method gathers relevant data about the agent, including its memory, goals, and impressions, for
        serialization or external use.

        The function first calls the parent class's method to obtain a base representation and then adds additional
        attributes specific to the agent's state, allowing for a comprehensive view of the agent's current context.

        Args:
            self: The instance of the agent.

        Returns:
            dict: A dictionary containing the agent's memory stream, goals, and impressions.
        """

        # Call the parent class's to_primitive method to get the base data representation
        thing_data = super().to_primitive()

        # Add the agent's memory stream observations to the data representation
        thing_data["memory_stream"] = self.memory.get_observations_after_round(0, True)

        thing_data["group"] = self.group

        # If the agent has goals, include them in the data representation
        if self.goals:
            thing_data["use_goals"] = True
            thing_data["goals"] = self.goals.get_goals()
        else:
            thing_data["use_goals"] = False
            thing_data["goals"] = None

        # If the agent has impressions, include them in the data representation
        if self.impressions:
            thing_data["use_impressions"] = True
            thing_data["impressions"] = self.impressions.impressions
        else:
            thing_data["use_impressions"] = False
            thing_data["impressions"] = None

        thing_data["search_idol_count"] = self.idol_search_count

    @classmethod
    def from_primitive(cls, data):
        """
        Create a character instance from a primitive data structure.

        This class method constructs a new character instance using the provided data,
        populating its attributes based on the input dictionary. It initializes various
        properties such as goals, impressions, memory, and interaction history.

        Args:
            data (dict): A dictionary containing the character's attributes, including
                        name, description, persona, use_goals, use_impressions, and
                        search_idol_count.

        Returns:
            Character: A fully constructed character instance with initialized attributes.
        """

        # Create a new instance of the character class using the provided name, description, and persona
        instance = cls(data["name"], data["description"], data["persona"])

        # Call the parent class's from_primitive method to populate common attributes of the instance
        instance = super().from_primitive(data, instance=instance)

        # Set the group attribute based on the provided data, defaulting to "D"
        instance.group = data.get("group", "D")

        # Set the use_goals attribute based on the provided data, defaulting to True
        instance.use_goals = data.get("use_goals", True)

        # Set the use_impressions attribute based on the provided data, defaulting to True
        instance.use_impressions = data.get("use_impressions", True)

        # Initialize the goals for the agent if goals are enabled
        instance.goals = Goals(instance) if instance.use_goals else None

        # Initialize the impressions for the agent if impressions are enabled
        instance.impressions = (
            Impressions(instance) if instance.use_impressions else None
        )

        # Initialize a memory stream to track observations made by the agent
        instance.memory = MemoryStream(instance)

        # Set the last location observations to None, indicating no observations have been made yet
        instance.last_location_observations = None

        # Track the last character the agent interacted with, initialized to None
        instance.last_talked_to = None

        # Initialize a counter for the number of idol searches performed by the agent, starting at zero
        instance.idol_search_count = data.get("search_idol_count", 0)

        # Return the fully constructed character instance
        return instance

    def get_idol_searches(self):
        """
        Retrieve the count of idol searches performed by the agent.
        This method provides a way to access the number of times the agent has searched for idols, which can be useful
        for tracking behavior.

        Args:
            self: The instance of the agent.

        Returns:
            int: The number of idol searches conducted by the agent.
        """

        return self.idol_search_count

    def increment_idol_search(self):
        """
        Increment the count of idol searches performed by the agent.
        This method updates the internal counter each time the agent conducts an idol search, allowing for tracking of
        search activity.

        Args:
            self: The instance of the agent.

        Returns:
            None
        """

        self.idol_search_count += 1


class DiscoveryAgent(GenerativeAgent):
    """
    A class representing a discovery agent that extends the GenerativeAgent class.
    This agent is designed to interact with teammates, manage a score, and engage in cognitive processes within the
    game.

    The DiscoveryAgent initializes with a persona and a group, setting up its score and allowing for the management of
    teammates. It provides methods for setting teammates, retrieving their names, updating the agent's score, and
    engaging in actions based on the game state.

    Attributes:
        score (int): The current score of the agent.
        teammates (List[GenerativeAgent]): A list of the agent's teammates.
    """

    def __init__(self, persona, group: str = "D"):
        """
        Initialize a DiscoveryAgent with a specified persona and cognitive group.
        This constructor sets the initial score for the agent and calls the parent class's constructor to initialize
        inherited attributes.

        The agent's group determines its cognitive capabilities, and the score is initialized to zero, allowing for
        tracking of performance in the game.

        Args:
            self: The instance of the agent.
            persona: The persona object containing the agent's characteristics.
            group (str, optional): The group classification of the agent, default is "D".

        Returns:
            None
        """

        # Call the parent class's constructor to initialize the agent with the provided persona and group
        super().__init__(persona, group)

        # Initialize the agent's score to zero
        self.score = 0

    def set_teammates(self, members: List[GenerativeAgent]):
        """
        Assign a list of teammates to the agent, filtering out invalid entries.
        This method ensures that only instances of GenerativeAgent that are not the agent itself are included in the
        teammates list.

        The function iterates through the provided list of members and populates the agent's teammates attribute
        with valid GenerativeAgent instances, allowing for effective collaboration within the game.

        Args:
            self: The instance of the agent.
            members (List[GenerativeAgent]): A list of potential teammates to be assigned.

        Returns:
            None
        """

        # Assign the list of teammates by filtering the provided members
        # Only include members that are instances of GenerativeAgent and are not the agent itself
        self.teammates = [
            m for m in members if isinstance(m, GenerativeAgent) and m.id != self.id
        ]

    def get_teammates(self, names_only=False, as_str=False):
        """
        Retrieve the list of teammates associated with the agent. This method can return either the full teammate
        objects or just their names, depending on the specified parameters.

        If the `names_only` flag is set to True, the method will return either a list of names or a single string of
        names, based on the `as_str` parameter. If `names_only` is False, the full list of teammate objects is returned.

        Args:
            self: The instance of the agent.
            names_only (bool, optional): If True, return only the names of the teammates. Default is False.
            as_str (bool, optional): If True and `names_only` is also True, return names as a single string. Default is
            False.

        Returns:
            list or str: A list of teammate names, a single string of names, or the full list of teammate objects.
        """

        # Check if only names should be returned; if not, return the full teammates list.
        if not names_only:
            return self.teammates

        # If as_str is True, join the names of the teammates into a single string separated by commas.
        if as_str:
            return ", ".join([agent.name for agent in self.teammates])
        else:
            # Otherwise, return a list of the names of the teammates.
            return [agent.name for agent in self.teammates]

    def update_score(self, add_on: int):
        """
        Update the agent's score by adding a specified value.
        This method allows for dynamic adjustments to the agent's score, which can be used to reflect performance or
        achievements in the game.

        The function takes an integer value and increments the agent's score accordingly, enabling score tracking
        throughout gameplay.

        Args:
            self: The instance of the agent.
            add_on (int): The value to be added to the current score.

        Returns:
            None
        """

        self.score += add_on

    # TODO: I don't think this is necessary seeing that the same method is implemented the same way in the parent class.
    def engage(self, game) -> Union[str, int]:
        """
        Manage the agent's actions and interactions within the game during a round.
        This method handles perception, updates impressions, and executes actions based on the current game state.
        It's a wrapper method for all agent cognition: perceive, retrieve, act, reflect, set goals.

        If the current game tick indicates the end of a round, the method forces reflection and evaluates the agent's
        goals. Otherwise, it perceives the surroundings, updates impressions of nearby characters, and performs the
        appropriate action.

        Args:
            self: The instance of the agent.
            game: The current game object containing the game state.

        Returns:
            Union[str, int]: An action string representing the agent's decision or an integer flag (-999) to indicate a
            skipped action.
        """

        # Check if the current game tick is the last one in the round and the agent's group is not "E"
        if game.tick == (game.max_ticks_per_round - 1) and self.group != "E":
            # Force the agent to reflect on the round's events
            self.memory.reflect(game)
            # If the agent uses goals, evaluate them at the end of the round
            if self.use_goals:
                self.goals.evaluate_goals(game)
            # Return -999 to indicate that the action was skipped
            return -999

        # Allow the agent to perceive its surroundings and gather information
        self.perceive(game)

        # If the agent uses impressions, update its perceptions of nearby characters
        if self.use_impressions:
            self.update_character_impressions(game)

        # Execute the appropriate action based on the agent's current state and decisions
        return Act(game, self).act()
