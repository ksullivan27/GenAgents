from pydantic import BaseModel

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
