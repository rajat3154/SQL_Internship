from pydantic import BaseModel,Field
from datetime import datetime
from typing import List
from uuid import UUID
from decimal import Decimal
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str


class UserResponse(BaseModel):
    id: UUID = Field(alias="user_id")  # 👈 KEY FIX
    username: str
    email: str
    full_name: str
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class PostCreate(BaseModel):
      user_id:int
      content:str

class PostResponse(BaseModel):
    id: UUID = Field(alias="post_id")      # 👈 FIX
    user_id: UUID                          # 👈 FIX
    content: str
    media_urls: List[str] = []
    like_count: int
    comment_count: int
    engagement_score: Decimal
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class LikeCreate(BaseModel):
      post_id:int
      user_id:int

class CommentCreate(BaseModel):
      post_id:int
      user_id:int
      content:str

class AnalyticsResponse(BaseModel):
      top_posts:List
      user_summary:List
      engagement_stats:dict

