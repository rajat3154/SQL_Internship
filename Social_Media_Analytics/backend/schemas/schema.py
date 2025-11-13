from pydantic import BaseModel
from datetime import datetime
from typing import List
class UserCreate(BaseModel):
      username:str
      email:str
      full_name:str

class UserResponse(BaseModel):
      id:int
      username:str
      email:str
      full_name:str
      created_at:datetime
      class Config:
            from_attributes=True

class PostCreate(BaseModel):
      user_id:int
      content:str

class PostResponse(BaseModel):
      id:int
      user_id:int
      content:str
      like_count:int
      comment_count:int
      engagement_score:float

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

