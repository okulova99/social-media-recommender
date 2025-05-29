from pydantic import BaseModel
from datetime import datetime

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True