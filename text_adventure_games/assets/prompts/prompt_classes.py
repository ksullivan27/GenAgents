from pydantic import BaseModel
from typing import Literal, List

class Goal(BaseModel):
    description: str

class Goals(BaseModel):
    low_priority: list[Goal]
    medium_priority: list[Goal]
    high_priority: list[Goal]

class Score(BaseModel):
    node_id: int
    progress_score: int

class Scores(BaseModel):
    scores: list[Score]

class Character(BaseModel):
    Name: str
    Age: int
    Likes: list[str]
    Dislikes: list[str]
    Occupation: str
    Home_City: str

class Impressions(BaseModel):
    key_strategies: str
    probable_next_moves: str
    impressions_of_me: str
    information_to_keep_secret: str

# class ResponseSummary(BaseModel):
#     summary: str
#     importance_score: int


class ResponseComponent(BaseModel):
    component: str
    importance_score: int


class Dialogue(BaseModel):
    speak_or_listen: Literal["speak", "listen"]
    response: str | None
    response_summary: str | None
    response_splits: list[ResponseComponent] | None
    leave_dialogue: bool

class ForcedDialogue(BaseModel):
    response: str
    response_summary: str
    response_splits: list[ResponseComponent]
    leave_dialogue: bool

class DialogueInitiator(BaseModel):
    speak_or_listen: Literal["speak", "listen"]
    end_dialogue: bool
    response: str | None
    response_summary: str | None
    response_splits: list[ResponseComponent] | None
    leave_dialogue: bool


class ForcedDialogueInitiator(BaseModel):
    end_dialogue: bool
    response: str
    response_summary: str
    response_splits: list[ResponseComponent]
    leave_dialogue: bool
