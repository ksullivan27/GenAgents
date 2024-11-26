action_system_mid = (
    """Given the context of your environment, past memories, and interpretation of relationships """
    """with other characters, select a next action that advances your goals or strategy. You can only select one """
    """action; selecting multiple would cause an error."""
)

action_system_end = (
    "Using the information provided, generate a short action statement in the present tense from your perspective.\n\n"
    "Examples could be:\n"
    "\t- \"Go outside to the garden.\"\n"
    "\t- \"Talk to Tom and Joe about their strategy\"\n"
    "\t- \"Pick up the stone from the ground\"\n"
    "\t- \"Give your food to the guard\"\n"
    "\t- \"Climb up the tree\"\n\n"
    "Notes to keep in mind:\n"
    "\t- You can only use items that are in your possession.\n"
    "\t- If you want to go somewhere, state the direction or the location in which you want to travel.\n"
    "\t- Actions should be atomic, not general, and should interact with your immediate environment.\n"
    "\t- Be sure to mention any characters you wish to interact with by name.\n"
    "\t- Aim to keep action statements to 10 words or less.\n"
    "\t- Here is list of valid action verbs to use:\n"
)

action_incentivize_exploration = (
    """Remember, exploring the island helps you complete the goal of this game."""
)
