"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: agent/persona.py
Description: Defines the Persona of an agent
"""

circular_import_prints = False

if circular_import_prints:
    print("Importing Persona")

# local imports
from text_adventure_games.gpt.gpt_agent_setup import summarize_agent_facts
from text_adventure_games.managers.scales import TraitScale
import json
import os

class Persona:
    """
    Represents an agent (persona) with specific traits, facts, and characteristics, allowing for the management of its
    attributes and behaviors. This class provides methods to add traits, retrieve scores, generate summaries, and manage
    the agent's game theory strategies. It also facilitates importing/exporting agents.

    A persona is an instance of an archetype. It's initialized with a list of one or more archetypes to base the
    persona off of.

    Traits (dict): {trait name: TraitScale} on a scale from 0-100
    Facts (dict) these include the agent's name, age, occupation, likes, dislikes, and home city
    summary: str

    Args:
        None

    Returns:
        None
    """

    def __init__(self, facts):
        """
        Initializes an agent with specified facts and sets up its traits and characteristics. This constructor method
        establishes the agent's basic attributes, including its summary, description, game theory strategy, and speaking
        style.

        Args:
            facts (dict): A dictionary containing the foundational facts about the agent, such as name, age, and
            occupation.

        Returns:
            None
        """

        # Agent traits
        self.facts = facts
        self.traits = {}
        self.summary = summarize_agent_facts(str(self.facts))
        self.description = f'A {self.facts["Age"]} year old {self.facts["Occupation"]} named {self.facts["Name"]}'
        self.game_theory_strategy = "nothing specific yet"
        self.strategy_in_effect = False
        self.speaking_style = ""
        self.archetype_base = None

    def add_trait(self, trait):
        """
        Adds a trait to the agent and updates the agent's game theory strategy based on the trait's characteristics.
        This method ensures that each trait is unique and adjusts the strategy according to the score of the
        "cooperation" trait.

        Args:
            trait (Trait): The trait object to be added to the agent's traits.

        Returns:
            None
        """

        # Check if the trait is not already present in the agent's traits dictionary.
        if trait.name not in self.traits:
            # If the trait is new, add it to the traits dictionary.
            self.traits[trait.name] = trait

        # Set the default game theory strategy based on the score of the "cooperation" trait.
        if trait.name == "cooperation":
            # If the cooperation trait score is less than 33, set the strategy to "backstab".
            if trait.score < 33:
                self.game_theory_strategy = "backstab"
            # If the cooperation trait score is between 34 and 66, set the strategy to "tit-for-tat".
            elif trait.score >= 34 and trait.score <= 66:
                self.game_theory_strategy = "tit-for-tat"
            # If the cooperation trait score is greater than 66, set the strategy to "cooperate".
            else:
                self.game_theory_strategy = "cooperate"

    def get_trait_score(self, name: str):
        """
        Retrieves the score of a specified trait from the agent's traits. This method checks if the trait exists and
        returns its score if found.

        Args:
            name (str): The name of the trait whose score is to be retrieved.

        Returns:
            int: The score of the specified trait if it exists; otherwise, None.
        """

        if name in self.traits:
            return self.traits[name].score

    def get_trait_summary(self):
        """
        Generates a summary of the agent's traits, describing each trait along with its associated adjective. This
        method compiles a list of formatted strings that convey the nature of each trait in a readable format.

        Returns:
            list: A list of strings summarizing each trait and its corresponding adjective.
        """

        return [
            f"{tname} which *tends* to be {self.traits[tname].get_adjective()}"
            for tname in self.traits
        ]

    def set_archetype(self, archetype: str):
        """
        Sets the archetype for the agent, defining its foundational characteristics and behavior. This method updates
        the agent's archetype base with the specified archetype string.

        Args:
            archetype (str): The name of the archetype to be assigned to the agent.

        Returns:
            None
        """

        self.archetype_base = archetype

    def start_strategy(self):
        """
        Activates the agent's strategy by setting the strategy_in_effect flag to True. This method indicates that the
        agent is currently employing its defined strategy in the context of the game or simulation.

        Returns:
            None
        """

        self.strategy_in_effect = True

    def stop_strategy(self):
        """
        Deactivates the agent's current strategy by setting the strategy_in_effect flag to False. This method indicates
        that the agent is no longer employing its defined strategy in the context of the game or simulation.

        Returns:
            None
        """

        self.strategy_in_effect = False

    def set_game_theory_strategy(self, strategy):
        """
        Sets the agent's game theory strategy and activates it. This method updates the strategy_in_effect flag to True
        and assigns the specified strategy to the agent, indicating that the agent will now operate under this strategy.

        Args:
            strategy (str): The game theory strategy to be assigned to the agent.

        Returns:
            None
        """

        self.strategy_in_effect = True
        self.game_theory_strategy = strategy

    # Export a persona into a json file ({person's name}.json)
    def export_persona(self):
        """
        Exports the agent's persona to a JSON file, uniquely identifying it by concatenating trait scores. This method
        creates a directory for storing persona files if it does not exist, constructs a filename based on the agent's
        name and scores, and writes the persona's attributes to the specified file.

        Returns:
            None
        """

        # Create a unique identifier for the persona by concatenating the scores of its traits.
        # Each score is formatted as a two-digit number to ensure consistent length.
        unique_id = "".join(
            "{:02d}".format(int(trait.score)) for trait in self.traits.values()
        )

        # Define the directory where persona files will be saved.
        # filedir = '../../SurvivorWorld/text_adventure_games/assets/personas/'  # Original path commented out.
        filedir = "./game_personas/"  # Set the current directory for persona files.

        # Check if the specified directory exists; if not, create it.
        if not os.path.isdir(filedir):
            os.makedirs(
                filedir, exist_ok=True
            )  # Create the directory, allowing existing directories to be ignored.

        # Construct the filename for the persona JSON file using the agent's name and the unique ID.
        filename = self.facts["Name"] + "_" + unique_id + ".json"

        # Create the full file path by combining the directory and filename.
        filepath = filedir + filename

        # Convert the persona's attributes into a dictionary format for easy serialization.
        persona_dict = {  # Convert the persona to a dictionary
            "archetype_base": self.archetype_base,  # Include the base archetype of the persona.
            "traits": {
                tname: {"score": trait.score, "adjective": trait.adjective}
                for tname, trait in self.traits.items()
            },  # Include traits with their scores and adjectives.
            "facts": self.facts,  # Include the foundational facts of the persona.
            "fact_summary": self.summary,  # Include a summary of the facts.
            "persona_summary": self.get_personal_summary(),  # Include a personal summary of the persona.
            "description": self.description,  # Include a description of the persona.
            "strategy_in_effect": self.strategy_in_effect,  # Include the current strategy status.
            "game_theory_strategy": self.game_theory_strategy,  # Include the game theory strategy.
            "speaking_style": self.get_speaking_style(),  # Include the speaking style of the persona.
        }

        # Open the file for writing and serialize the persona dictionary to JSON format.
        with open(filepath, "w") as f:
            json.dump(
                persona_dict, f, indent=4
            )  # Write the persona dictionary to the file with indentation for readability.

        # Print a success message indicating the persona has been exported to the specified file path.
        print(f"Successfully exported persona to {filepath}")

    # Import a persona from a file.
    # NOTE: This does *not* allow you to cross-import personas from different people, since
    # it copies their facts as well.
    @classmethod
    def import_persona(cls, filename):
        """
        Imports a persona from a specified JSON file and creates an instance of the Persona class. This class method
        reads the persona data, initializes a new Persona object with the loaded facts, and populates its attributes,
        including traits and strategies.

        Args:
            filename (str): The path to the JSON file containing the persona data.

        Returns:
            Persona: An instance of the Persona class populated with the data from the JSON file.

        Raises:
            FileNotFoundError: If the specified file does not exist or cannot be opened.
            KeyError: If the expected keys are not found in the loaded persona data.
        """

        # Load the persona data from the JSON file
        with open(filename, "r") as f:
            persona_dict = json.load(f)

        # Create a new Persona instance using the facts loaded from the persona dictionary.
        persona = cls(persona_dict["facts"])

        # The following lines are commented out; they represent alternative methods for setting the traits of the
        # persona.
        # persona.traits = {tname: {'score': trait['score'], 'adjective': trait['adjective']} for tname, trait in
        # persona_dict['traits'].items()}
        # persona.traits = {tname: TraitScale(**trait) for tname, trait in persona_dict['traits'].items()}

        # Set the base archetype for the persona from the loaded data.
        persona.archetype_base = persona_dict["archetype_base"]

        # Assign the summary of facts to the persona.
        persona.summary = persona_dict["fact_summary"]

        # Set the description of the persona based on the loaded data.
        persona.description = persona_dict["description"]

        # Update the strategy_in_effect status for the persona.
        persona.strategy_in_effect = persona_dict["strategy_in_effect"]

        # Set the game theory strategy for the persona.
        persona.game_theory_strategy = persona_dict["game_theory_strategy"]

        # Assign the speaking style of the persona from the loaded data.
        persona.speaking_style = persona_dict["speaking_style"]

        # Retrieve the monitored traits that are relevant for the persona.
        monitored_traits = TraitScale.get_monitored_traits()

        # Iterate through the traits in the persona dictionary to create TraitScale instances for each.
        for tname, trait in persona_dict["traits"].items():
            # Get the dichotomy for the current trait from the monitored traits.
            trait_dichotomy = monitored_traits[tname]

            # Create a TraitScale instance for the trait and add it to the persona's traits.
            persona.traits[tname] = TraitScale(
                tname,
                trait_dichotomy,
                score=trait["score"],
                adjective=trait["adjective"],
            )

        # Return the fully constructed persona object.
        return persona

    # Make a string describing the persona's speaking style, instructing chatGPT how to speak.
    def get_speaking_style(self):
        """
        Generates and retrieves the speaking style of the agent based on its characteristics and traits. This method
        constructs a description of how the agent speaks, taking into account its age, occupation, and personality
        traits such as outlook and stress levels.

        Returns:
            str: A description of the agent's speaking style, including tone, word choice, and overall demeanor.

        Raises:
            None
        """

        # Check if the speaking style has already been generated; if so, return it to avoid redundant calculations.
        if self.speaking_style:
            return self.speaking_style

        # Construct the initial speaking style description based on the agent's facts, including name, age, occupation,
        # and home city.
        style = (
            f"""{self.facts['Name']} speaks in the style of a {self.facts['Age']} year old """
            f"""{self.facts['Occupation']} from {self.facts.get('Home city', 'no place in particular')}."""
        )

        # Retrieve the scores for the 'outlook' and 'stress' traits to incorporate into the speaking style.
        outlook = self.get_trait_score("outlook")
        stress = self.get_trait_score("stress")

        # Determine the speaking style based on the outlook trait score.
        if outlook > 66:
            style += (
                " They're generally optimistic and have a positive outlook on life."
            )
        elif outlook < 33:
            style += (
                " They're generally pessimistic and have a negative outlook on life."
            )
        else:
            style += " They have a balanced outlook on life."

        # Determine the speaking style based on the stress trait score.
        if stress > 66:
            style += " They're often stressed and anxious."
        elif stress < 33:
            style += " They're often calm and relaxed."
        else:
            style += " They're generally balanced in their stress levels."

        # Append additional guidance on how the speaking style should influence the agent's communication.
        style += (
            "Use this information to guide how they speak, in terms of tone, word choice, sentence structure, "
            "phrasing, terseness, verbosity, and overall demeanor."
        )

        # Store the generated speaking style in the agent's attributes for future reference.
        self.speaking_style = style

        # Return the constructed speaking style description.
        return style

    # Made more natural-language friendly, not just a list of facts/traits, etc.
    def get_personal_summary(self):
        """
        Generates a personal summary of the agent based on its facts and traits. This method constructs a descriptive
        string that includes the agent's name, age, occupation, home city, likes, dislikes, and overall game strategy,
        providing a comprehensive overview of the agent's characteristics.

        Returns:
            str: A formatted summary string that encapsulates the agent's personal details and traits.

        Raises:
            None
        """

        # Construct the initial summary string using the agent's name, age, and occupation.
        summary = f"{self.facts['Name']}, a {self.facts['Age']}-year-old {self.facts['Occupation']}."

        # If the agent's home city is provided, append it to the summary.
        if "Home city" in self.facts and self.facts["Home city"]:
            summary += f" You hail from {self.facts['Home city']}."

        # If the agent has specified likes, include them in the summary, limited to the first three.
        if "Likes" in self.facts and self.facts["Likes"]:
            summary += (
                f" You are passionate about {', '.join(self.facts['Likes'][:3])}."
            )

        # If the agent has specified dislikes, include them in the summary, limited to the first three.
        if "Dislikes" in self.facts and self.facts["Dislikes"]:
            summary += (
                f" You have aversions to {', '.join(self.facts['Dislikes'][:3])}."
            )

        # Iterate through the facts to include any additional key facts not already mentioned.
        for key in self.facts:
            if key not in [
                "Name",
                "Age",
                "Occupation",
                "Likes",
                "Dislikes",
                "Home city",
            ]:
                summary += f" Some key facts about yourself are: {key.lower()}, {self.facts[key]}."

        # If the agent has a defined game theory strategy, include it in the summary.
        if self.game_theory_strategy:
            summary += f" Your overall game strategy is {self.game_theory_strategy},"
            # Retrieve the agent's traits and append them to the summary.
            traits = self.get_trait_summary()
            summary += f" reflecting your {', '.join(traits[:-1])}, and {traits[-1]}."
        else:  # If no strategy is defined, just list the agent's traits.
            traits = self.get_trait_summary()
            summary += f" Your traits are {', '.join(traits[:-1])}, and {traits[-1]}."

        # Append the agent's speaking style to the summary.
        summary += f" {self.get_speaking_style()}"

        # Return the complete summary string.
        return summary

    def __str__(self):
        """
        Returns a string representation of the agent by providing a personal summary. This method overrides the default
        string conversion to present a comprehensive overview of the agent's characteristics and traits.

        Returns:
            str: A summary string that describes the agent's personal details and traits.
        """

        return self.get_personal_summary()
