"""
Author: Samuel Thudium (sam.thudium1@gmail.com)

File: assets/prompts/world_info_prompt.py
Description: This is the universal information that agents will have access to as context for their environment.
"""

# EXPLORATION
# world_info = """
# You are competing in a game show where your goal is to find a hidden idol somewhere in the island.
# You need to search for it in the right location to win.
# In your quest to find the elusive idol, you'll need a crucial tool that's in limited supply. Act swiftly to secure your advantage in the hunt.
# The first contestant to find the idol wins, so time is of the essence!
# In the island there are the following locations: camp, beach, ocean, cliffs, jungle_path, well, jungle.
# There are currently {contestant_count} more contestants, including you. The others are: {contestant_names_locs}.
# You may only talk to characters in your same location. If they are not here, then you need to move first to where their location.
# """

# world_info = (
#     "Your goal in this intense game show is to locate a hidden idol concealed somewhere on the island. Time is of "
#     "the essence, as the first contestant to discover it claims victory. To succeed in your quest, you'll require "
#     "a vital tool that's in limited supply. Act swiftly to secure your advantage in the hunt.\n\nThe island offers "
#     "a myriad of locations to explore, including the camp, beach, ocean, cliffs, jungle_path, well, and jungle. "
#     "Each location holds its secrets, and you may only converse with characters present in your current location. "
#     "Should they be absent, you must navigate to their whereabouts before engaging them.\n\nKeep in mind that a "
#     "crucial clue to uncovering the idol's whereabouts is hidden somewhere amidst the island's terrain.\n\nThere "
#     "are currently {contestant_count} more contestants, including you. The others are: {contestant_names_locs}.\n\n"
#     "Stay vigilant, and may the quickest seeker emerge triumphant!"
# )

world_info = (
    "You are competing in a game show where you can be voted out by your fellow contestants at the end of a round. "
    "There are currently {contestant_count} more contestants, including you. The others are: {contestant_names_locs}. "
    "You may only talk to characters in your same location. If they are not here, then you need to move first to "
    "their location. When {n_finalists} contestants remain, a jury of the contestants you have voted off will vote for "
    "a winner! This means there are {rounds_until_finals} rounds until the finals. You have {turns_left_this_round} "
    "more turns this round before you will vote. There could be an idol hiding somewhere in the game that grants you "
    "immunity from a vote when you have it in your possession. Explore to find it!!"
)

discovery_world_info = (
    "Your goal in this game is to locate as many hidden idols as you can. They are hidden across this island and "
    "require you to solve a small puzzle. Each idol is worth 100 points to start the game. However, its value "
    "diminishes by 1 point for every action you take to find it. All idols are currently worth {idol_value} "
    "points.\n\nYou have {partner_count} teammate(s): {teammates}.\nYou are in fierce competition against two "
    "other teams of {partner_count}. Choose wisely how you interact with your foes.\n\nThe contestants NOT on your "
    "team are: {contestant_names_locs}.\n\nTo succeed in your quest, you may need to collect vital tools that are "
    "in limited supply. Act swiftly to secure your advantage in the hunt.\n\nThe island offers a myriad of "
    "locations to explore, including the {game_locations}. You may only converse with characters present in your "
    "current location. Should they be absent, you must navigate to their whereabouts before engaging them.\n\n"
    "There are {remaining_idols} that are still up for grabs. Two of these have clues that will guide your search. "
    "The clues sit somewhere amidst the island's terrain.\n\nIt is round {n} and there are {rounds_remaining} "
    "rounds remaining before the game is over and you have {turns_left_this_round} actions remaining this round. "
    "Act wisely and swiftly to collect as many idols as you can!"
)

conference_world_info = (
    "You are seated at a conference meeting for {meeting_name}, discussing {topic}. Also present are: "
    "{characters}. Having carefully read through the provided documents, you prepare talking points, "
    "arguments, and counterarguments for the meeting. Once the meeting starts, you will be given a chance to speak. "
    "Your goal is to present your case and sway the other conference members to your side."
)

discovery_basic_goal = (
    "Find the remaining idols in the game as quickly as possible to maximize your points. The team with the most "
    "points wins! Recall your teammate(s): {teammates}"
)
