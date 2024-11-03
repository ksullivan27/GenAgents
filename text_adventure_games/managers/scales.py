
circular_import_prints = False

if circular_import_prints:
    print("Importing Scales")

# Imports
from collections import namedtuple
from typing import List

# local imports
if circular_import_prints:
    print(f"\t{__name__} calling imports for GptAgentSetup")
from ..gpt.gpt_agent_setup import get_target_adjective
if circular_import_prints:
    print(f"\t{__name__} calling imports for Character")
from ..things.characters import Character


class BaseScale:
    """
    Represents a base scale for measuring a trait with defined minimum and maximum descriptors. This class allows for
    the initialization of a score within specified limits and provides methods to update and retrieve the score.

    Args:
        dichotomy (Dichotomy): A named tuple containing the minimum and maximum descriptors for the scale.
        score (int, optional): The initial score for the scale, defaulting to 50.
        min (int, optional): The minimum value of the scale, defaulting to 0.
        max (int, optional): The maximum value of the scale, defaulting to 100.

    Returns:
        None
    """

    # Define a named tuple called 'Dichotomy' to represent a scale with minimum and maximum descriptors.
    # This structure allows for easy access to the min and max values by name, enhancing code readability.
    Dichotomy = namedtuple("Scale", ["min", "max"])
    
    # The default GPT model to use.
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self, dichotomy: Dichotomy, score: int = 50, min: int = 0, max: int = 100
    ):
        """
        Initializes a scale with specified minimum and maximum descriptors, along with an initial score. This
        constructor method sets up the scale's boundaries and score, allowing for the measurement of traits within
        defined limits.

        Args:
            dichotomy (Dichotomy): A named tuple containing the minimum and maximum descriptors for the scale.
            score (int, optional): The initial score for the scale, defaulting to 50.
            min (int, optional): The minimum value of the scale, defaulting to 0.
            max (int, optional): The maximum value of the scale, defaulting to 100.

        Returns:
            None
        """

        # Assign the minimum and maximum descriptors from the provided dichotomy named tuple to the instance variables.
        self.min_descriptor, self.max_descriptor = dichotomy

        # Set the minimum value of the scale to the provided minimum argument.
        self.min = min

        # Set the maximum value of the scale to the provided maximum argument.
        self.max = max

        # Initialize the score of the scale with the provided score argument.
        self.score = score

    def __str__(self):
        """
        Returns a string representation of the scale, detailing its range and current score. This method provides a
        formatted description that includes the minimum and maximum values, their descriptors, and the current score on
        the scale.

        Returns:
            str: A formatted string that describes the scale's range and current score.
        """

        # On a scale from 0 to 100, where 0 is evil and 100 is good, a score of 50
        # This wording was ranked most interpretable by GPT 3.5 and 4
        # over a number of trials
        return (
            f"""On a scale from {self.min} to {self.max}, where {self.min} is {self.min_descriptor} and {self.max} """
            f"""is {self.max_descriptor}, a score of {self.score}"""
        )

    def update_score(self, delta: int):
        """
        Updates the score of the scale by a specified delta, ensuring that the score remains within defined minimum and
        maximum limits. This method raises an error if the provided delta is not an integer and adjusts the score
        accordingly.

        Args:
            delta (int): The amount by which to change the current score.

        Returns:
            None

        Raises:
            TypeError: If delta is not of type int.
        """

        # Check if the provided delta is an integer; raise a TypeError if it is not.
        if not isinstance(delta, int):
            raise TypeError("delta must be of type int")

        # Calculate the new score by adding the delta to the current score.
        update = self.score + delta

        # Update the score, ensuring it remains within the defined minimum and maximum limits.
        # The min function ensures the score does not exceed the maximum, while the max function ensures it does not
        # fall below the minimum.
        self.score = max(self.min, min(update, self.max))

    def get_score(self):
        """
        Retrieves the current score of the scale. This method provides access to the score attribute, allowing other
        components to read the current value of the scale.

        Returns:
            int: The current score of the scale.
        """

        return self.score


class AffinityScale(BaseScale):
    """
    Represents an affinity scale that measures the relationship between a character and an agent. This class extends the
    BaseScale to include a target character and visibility settings, allowing for the management of affinity scores
    within defined limits.

    Args:
        character (Character): The target character for whom the affinity is being measured.
        dichotomy (BaseScale.Dichotomy): A named tuple containing the minimum and maximum descriptors for the scale.
        score (int, optional): The initial score for the scale, defaulting to 50.
        min (int, optional): The minimum value of the scale, defaulting to 0.
        max (int, optional): The maximum value of the scale, defaulting to 100.
        visibile (bool, optional): A flag indicating whether the scale is visible, defaulting to False.

    Returns:
        None
    """

    def __init__(
        self,
        character: Character,
        dichotomy: BaseScale.Dichotomy,
        score: int = 50,
        min: int = 0,
        max: int = 100,
        visibile: bool = False,
    ):
        """
        Initializes an affinity scale that measures the relationship between an agent and a specified character. This
        constructor method sets up the scale's boundaries, initial score, and visibility, allowing for the management of
        affinity scores within defined limits.

        Args:
            character (Character): The target character for whom the affinity is being measured.
            dichotomy (BaseScale.Dichotomy): A named tuple containing the minimum and maximum descriptors for the scale.
            score (int, optional): The initial score for the scale, defaulting to 50.
            min (int, optional): The minimum value of the scale, defaulting to 0.
            max (int, optional): The maximum value of the scale, defaulting to 100.
            visibile (bool, optional): A flag indicating whether the scale is visible, defaulting to False.

        Returns:
            None
        """

        # Call the constructor of the parent class (BaseScale) to initialize the scale with the provided dichotomy,
        # score, min, and max values.
        super().__init__(dichotomy, score, min, max)

        # Set the target character for this affinity scale to the specified character.
        self.target = character

        # Initialize the visibility attribute to indicate whether the scale is visible or not.
        self.visible = visibile

    def get_visibility(self):
        """
        Retrieves the visibility status of the affinity scale. This method returns a boolean value indicating whether
        the scale is currently visible.

        Returns:
            bool: The visibility status of the scale.
        """

        return self._visibile

    def set_visibility(self):
        """
        Toggles the visibility status of the affinity scale. This method changes the current visibility state to its
        opposite, allowing the scale to be shown or hidden.

        Returns:
            None
        """

        self.visible = not self.visible


class TraitScale(BaseScale):
    """
    Represents a trait scale that measures specific personality traits within defined limits. This class extends the
    BaseScale to include trait-specific attributes and methods for managing and retrieving trait information.

    Args:
        trait_name (str): The name of the trait being measured.
        trait_dichotomy (BaseScale.Dichotomy): A named tuple containing the minimum and maximum descriptors for the
        trait scale.
        score (int, optional): The initial score for the trait, defaulting to 50.
        min (int, optional): The minimum value of the scale, defaulting to 0.
        max (int, optional): The maximum value of the scale, defaulting to 100.
        adjective (str, optional): An optional adjective describing the trait.

    Returns:
        None
    """

    # A dictionary that defines monitored traits and their corresponding dichotomies.
    # Each key represents a trait, while the value is a tuple containing the minimum descriptor
    # and the maximum descriptor for that trait, allowing for a range of evaluation.
    MONITORED_TRAITS = {
        "judgement": ("Prejudicial", "Unbiased"),
        "cooperation": ("Stubborn", "Agreeable"),
        "outlook": ("Pessimistic", "Optimistic"),
        "initiative": ("Passive", "Assertive"),
        "generosity": ("Selfish", "Generous"),
        "social": ("Follower", "Leader"),
        "mind": ("Creative", "Logical"),
        "openness": ("Close-minded", "Open-minded"),
        "stress": ("Anxious", "Calm"),
    }

    def __init__(
        self,
        trait_name: str,
        trait_dichotomy: BaseScale.Dichotomy,
        score: int = 50,
        min: int = 0,
        max: int = 100,
        adjective: str = None,
    ):
        """
        Initializes a TraitScale object that measures a specific personality trait with defined boundaries. This
        constructor method sets up the trait's name, dichotomy, initial score, and optional adjective, allowing for the
        management of trait evaluations within specified limits.

        Args:
            trait_name (str): The name of the trait being measured.
            trait_dichotomy (BaseScale.Dichotomy): A named tuple containing the minimum and maximum descriptors for the
            trait scale.
            score (int, optional): The initial score for the trait, defaulting to 50.
            min (int, optional): The minimum value of the scale, defaulting to 0.
            max (int, optional): The maximum value of the scale, defaulting to 100.
            adjective (str, optional): An optional adjective describing the trait.

        Returns:
            None
        """

        # Call the constructor of the parent class (BaseScale) to initialize the trait scale with the provided
        # dichotomy, score, min, and max values.
        super().__init__(trait_dichotomy, score, min, max)

        # Set the name of the trait being measured to the specified trait_name.
        self.name = trait_name

        # Assign the optional adjective describing the trait to the instance variable.
        self.adjective = adjective

    @classmethod
    def get_monitored_traits(cls) -> List:
        """
        Retrieves the list of traits that are monitored by the TraitScale class. This class method provides access to
        the predefined traits and their corresponding dichotomies, allowing other components to reference the monitored
        traits.

        Returns:
            List: A list of monitored traits defined in the class.
        """

        return cls.MONITORED_TRAITS

    def get_name(self) -> str:
        """
        Retrieves the name of the trait associated with the TraitScale instance. This method provides access to the
        trait's name, allowing other components to reference it as needed.

        Returns:
            str: The name of the trait.
        """

        return self.name

    def update_score(self, new_score) -> None:
        """
        Updates the score of the scale to a new specified value. This method directly sets the score attribute to the
        provided new score.

        Args:
            new_score (int): The new score to be assigned to the scale.

        Returns:
            None
        """

        self.score = new_score

    def set_adjective(self, model=None):
        """
        Sets the adjective for the trait based on the current score and defined descriptors. This method utilizes a
        specified model to determine the appropriate adjective that reflects the trait's position within its defined
        range.

        Args:
            model (str, optional): The model to be used for generating the adjective, defaulting to None to trigger it
            being set to the default model value (e.g., "gpt-4o-mini").

        Returns:
            None
        """

        # If no model is provided, use the default model from the class
        if model is None:
            model = self.DEFAULT_MODEL

        # Extract the minimum and maximum descriptors for the trait from the instance variables.
        low, high = self.min_descriptor, self.max_descriptor

        # Call the function to get the target adjective based on the trait's score and defined limits.
        # The function uses the specified model to determine the appropriate adjective.
        adj = get_target_adjective(
            low=low,  # The minimum descriptor for the trait.
            high=high,  # The maximum descriptor for the trait.
            target=self.score,  # The current score of the trait.
            low_int=self.min,  # The minimum value of the scale.
            high_int=self.max,  # The maximum value of the scale.
        )

        # Assign the retrieved adjective to the instance variable for the trait.
        self.adjective = adj

    def get_adjective(self):
        """
        Retrieves the adjective associated with the trait. This method provides access to the adjective attribute,
        allowing other components to reference the descriptive term for the trait.

        Returns:
            str: The adjective describing the trait.
        """

        return self.adjective
