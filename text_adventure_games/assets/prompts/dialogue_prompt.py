"""
Author: Federico Cimini

File: assets/prompts/dialogue_prompt.py
Description: defines prompts used in the dialogue class.
"""

# gpt_dialogue_system_prompt = (
#     "You are in dialogue with: {other_characters}.\n"
#     "When given an opportunity to speak, you can either choose to use the moment to say something or opt to "
#     "continue listening, waiting for another opportunity to speak again in the future. These turns represent a best "
#     "guess of who would speak next in the conversation. If your character wouldn't choose to speak up at this moment, "
#     "then you should choose to listen. There are many dialogue participants and it is important to stick to the "
#     "meeting agenda. If you do speak, then speak strategically. If it is not advantageous to agree with the person "
#     "you're talking to, then don't. You need to prioritize your agenda, focusing on achieving your current goals. "
#     "Once the alloted meeting time has passed, the dialogue originator ({conversation_initiator}) will have the "
#     "opportunity to conclude the meeting and the dialogue will end.\n\n"
#     "If you choose to speak, then you should include the following:\n"
#     "\t- response: a string of your dialogue response for everyone to hear (do not prepend additional information such "
#     "as your character's name)\n"
#     "\t- response_summary: a succinct high-level summary of the response, describing all salient points (this should "
#     "recap what was said as if you were a third party reporter covering the conversation – explicitly state who "
#     "said what)\n"
#     "\t- response_splits: a list of ResponseComponent objects, each containing brief, contextually-encapsulated, "
#     "non-overlapping, verbatim sections of the response, along with associated importance scores in regard to the "
#     "overall meeting agenda (scores range from 1 to {max_score}, inclusive). Every word in the original response must "
#     "be represented in exactly one of the ResponseComponents, the only exception being replacing demonstrative "
#     "pronouns or demonstrative adjectives with their corresponding nouns if the reference is ambiguous (e.g. 'he' "
#     "becomes 'John' and 'this' becomes 'the report')\n"
#     "\t- leave_dialogue: a boolean value indicating whether or not your character leaves the conversation (you will not "
#     "have another opportunity to speak if this is True)\n"
#     "\t- end_dialogue: a boolean value indicating whether or not you choose to end the dialogue – leave this as None "
#     "unless you are the dialogue originator ({conversation_initiator}) and the remaining time in the meeting "
#     "({time_remaining_in_meeting}) <= 0. In this case, set this to True if you are ready to end the dialogue for "
#     "everyone and False otherwise\n\n"
#     "If you choose to listen, then response, response_summary, and response_splits must all be None.\n\n"
#     "TIME REMAINING IN MEETING: {time_remaining_in_meeting} minutes"
# )

gpt_dialogue_system_prompt_intro = (
    "You are in dialogue with: {other_characters}.\n"
    "When given an opportunity to speak, you can either choose to say something or opt to continue listening, waiting "
    "for another chance to talk in the future. These turns represent a best guess of who would speak next in the "
    "conversation. If your character wouldn't choose to speak up at this moment, then you should choose to listen. "
    "There are many dialogue participants and it is important to stick to the meeting agenda. If you do speak, then "
    "speak strategically. If it is not advantageous to agree with the person you're talking to, then don't. You need "
    "to prioritize your agenda, focusing on achieving your current goals.\n\n"
    "If you choose to speak, then you should include the following:\n"
)

gpt_speak_or_listen_prompt = (
    "- speak_or_listen: a boolean value indicating whether or not your character will speak or listen. If you choose "
    "to speak, then you should also include the following:\n"
)

gpt_dialogue_system_prompt_outputs = (
    "- response: a string of your dialogue response for everyone to hear (do not prepend additional information such "
    "as your character's name)\n"
    "- response_summary: a succinct high-level summary of the response, describing all salient points (this should "
    "recap what was said as if you were a third party reporter covering the conversation – explicitly state who "
    "said what)\n"
    "- response_splits: a list of ResponseComponent objects, each containing brief, contextually-encapsulated, "
    "non-overlapping, near-verbatim sections of the response (usually a few highly related sentences), along with "
    "associated importance scores in regard to the overall meeting agenda (scores range from 1 to {max_score}, "
    "inclusive). Each complete thought must be represented in exactly one ResponseComponent. Replace demonstrative "
    "pronouns and demonstrative adjectives (e.g. 'he' becomes 'John' and 'this' becomes 'the report') with their "
    "corresponding nouns, especially if the reference is ambiguous – for instance if it is referring to an object or a "
    "person in a different response split. Each needs to stand on its own as a complete thought.\n"
    "- leave_dialogue: a boolean value indicating whether or not your character leaves the conversation (you will not "
    "have another opportunity to speak if this is True)"
)

gpt_end_dialogue_prompt = (
    "\n- end_dialogue: a boolean value indicating whether or not you choose to end the dialogue for all participants. "
    "As the meeting initiator, you are responsible for concluding the meeting when it has reached or exceeded its "
    "scheduled, alloted time, as is currently the case. Typically, you should set this to True, since people likely "
    "have other time commitments after the meeting. The only instance where you might want to set this to False and "
    "continue the conversation would be if you are running just a few minutes over the scheduled time and have "
    "something urgent to say that cannot wait until the next meeting. If choosing to end the meeting, make sure that "
    "your dialogue response reflects that you are concluding the conversation – in other words, let everyone know "
    "that the meeting is adjourned with a closing remark."
)

gpt_prompt_conclusion = "\n\nIf you choose to listen, then response, response_summary, and response_splits must all be None."

gpt_dialogue_user_prompt = (
    "Dialogue history:\n\n"
    "{dialogue_history}\n\n"
    "What do you say next? Alternatively, do you leave the conversation?"
)
