action_system_mid = (
    """Given the context of your environment, past memories, and interpretation of relationships """
    """with other characters, select a next action that advances your goals or strategy. You can only select one """
    """action; selecting multiple would cause an error."""
)

action_system_end = (
    """Using the information provided, generate a short action statement in the present tense from your perspective. """
    """Examples could be:\nGo outside to the garden.\n"""
    """\"Talk to Tom about his strategy\"\n"""
    """\"Pick up the stone from the ground\"\n"""
    """\"Give your food to the guard\"\n"""
    """\"Climb up the tree\"\n\n"""
    """Notes to keep in mind:\n"""
    """You can only use items that are in your possession. """
    """If you want to go somewhere, state the direction or the location in which you want to travel. """
    """Actions should be atomic, not general, and should interact with your immediate environment. """
    """Be sure to mention any characters you wish to interact with by name. """
    """Aim to keep action statements to 10 words or less. """
    """Here is list of valid action verbs to use:\n"""
)

action_incentivize_exploration = (
    """Remember, exploring the island helps you complete the goal of this game."""
)
