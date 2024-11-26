gpt_goals_prompt = (
    "Using the context above that describes the world and yourself, as well as information you'll be provided "
    "about your past reflections and impressions of others, create high level goals at several priority levels "
    "for the next round. Do not exceed 8 goals in total. You can keep previous goals, update them, or create new ones "
    "based on your strategy. These should be goals that will be used as a guide for the actions you take in the "
    "future. A well-thought-out goal positions you advantageously against other competitors. Remember your "
    "overarching objective is to win the game by not being voted out by the remaining contestants at the end of the "
    "round. Focus on developing a strategy with your goal, rather than planning out each action. Keep the goals "
    "concise.\n\nYou may see your previous goals as well as an accompanying goal progress score, which is on a "
    "scale of 1 to {max_progress_score}. This score tells you how well your actions advanced completion of the "
    "goal, with 1 meaning almost no progress was made towards the goal and {max_progress_score} meaning the "
    "goal was completely achieved. If you see these scores, consider updating your goals accordingly. A low score could "
    "indicate you haven't been working toward the goal or that the goal is very difficult to achieve. A high score "
    "indicates you've successfully achieved the goal and should think about your next strategic target. You must only "
    "use the scoring information in deciding how to update your goals. Do not add a score to your goals when picking "
    "goals. Do not add a score to your goals when picking new ones.\n\nIf you want to keep working toward a previous "
    "goal, just write it out again."
)

impartial_evaluate_goals_prompt = (
    "You are an impartial evaluator. On a scale of 1 to {max_progress_score}, give an integer for each "
    "priority tier that evaluates whether the goal was achieved or not with 1 being almost no progress towards the "
    "whether the goal was achieved or not with 1 being almost no progress towards the goal and 5 being goal "
    "completely achieved."
)

persona_evaluate_goals_prompt = (
    "On a scale of 1 to {max_progress_score}, give an integer for each priority tier that evaluates whether "
    "the goal was achieved or not with 1 being almost no progress towards the goal and {max_progress_score} "
    "being the goal completely achieved."
)