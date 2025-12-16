# main.py - COMPLETE FIXED VERSION FOR SUPABASE
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from uuid import UUID
from decimal import Decimal
from dotenv import load_dotenv
import os
import io
import csv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Social Media Analytics API",
    description="Production API for social media analytics",
    version="2.0.0"
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://social-media-analytics-liard.vercel.app",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== SUPABASE DATABASE CONFIGURATION ==========
def get_database_url():
    """Get and format database URL for Supabase"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # For Supabase, use port 6543 for connection pooling
    if "pooler.supabase.com" in DATABASE_URL:
        # Replace port 5432 with 6543 for connection pooling
        DATABASE_URL = DATABASE_URL.replace(":5432/", ":6543/")
        logger.info("Using Supabase connection pooler on port 6543")
    
    return DATABASE_URL

# Initialize database engine with NullPool
DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # CRITICAL: Use NullPool for Supabase
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
    },
    echo=False,
    future=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

def get_db():
    """Database dependency with proper cleanup"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()  # IMPORTANT: Always close connection

# ========== PYDANTIC MODELS ==========
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    created_at: datetime
    profile_data: Optional[Dict[str, Any]] = {}
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)

class PostCreate(BaseModel):
    user_id: str
    content: str
    media_urls: Optional[List[str]] = []

class PostResponse(BaseModel):
    post_id: str
    user_id: str
    username: str
    content: str
    media_urls: List[str] = []
    like_count: int = 0
    comment_count: int = 0
    engagement_score: float = 0.0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class LikeCreate(BaseModel):
    post_id: str
    user_id: str

class CommentCreate(BaseModel):
    post_id: str
    user_id: str
    content: str
    parent_comment_id: Optional[str] = None

class CommentResponse(BaseModel):
    comment_id: str
    post_id: str
    user_id: str
    username: str
    content: str
    parent_comment_id: Optional[str] = None
    like_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AnalyticsResponse(BaseModel):
    top_posts: List[Dict[str, Any]]
    user_summary: List[Dict[str, Any]]
    engagement_stats: Dict[str, Any]
    union_activities: List[Dict[str, Any]] = []
    grouped_engagement: List[Dict[str, Any]] = []

# ========== UTILITY FUNCTIONS ==========
def execute_query(db: Session, query: str, params: Dict = None):
    """Execute raw SQL query with UUID conversion"""
    try:
        result = db.execute(text(query), params or {})
        if result.returns_rows:
            columns = result.keys()
            rows = result.fetchall()
            
            # Convert each row to dict with proper type handling
            converted_rows = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert UUID to string
                    if isinstance(value, UUID):
                        value = str(value)
                    # Convert Decimal to float
                    elif isinstance(value, Decimal):
                        value = float(value)
                    row_dict[col] = value
                converted_rows.append(row_dict)
            return converted_rows
        db.commit()
        return None
    except Exception as e:
        db.rollback()
        logger.error(f"Query execution error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")

# ========== USER ENDPOINTS ==========
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        query = """
        INSERT INTO Users (username, email, full_name)
        VALUES (:username, :email, :full_name)
        RETURNING 
            user_id::text as user_id, 
            username, email, full_name, 
            created_at, profile_data, is_active
        """
        result = execute_query(db, query, {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name
        })
        
        if result:
            return result[0]
        raise HTTPException(status_code=400, detail="User creation failed")
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.get("/users/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    """Get all users"""
    query = """
    SELECT 
        user_id::text as user_id, 
        username, email, full_name, 
        created_at, profile_data, is_active
    FROM Users 
    ORDER BY created_at DESC
    """
    return execute_query(db, query)

# ========== POST ENDPOINTS ==========
@app.post("/posts/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreate, db: Session = Depends(get_db)):
    """Create a new post"""
    try:
        query = """
        INSERT INTO Posts (user_id, content, media_urls)
        VALUES (:user_id::uuid, :content, :media_urls)
        RETURNING 
            post_id::text as post_id, 
            user_id::text as user_id, 
            content, media_urls, like_count, 
            comment_count, engagement_score, created_at, updated_at
        """
        result = execute_query(db, query, {
            "user_id": post.user_id,
            "content": post.content,
            "media_urls": post.media_urls or []
        })
        
        if result:
            # Get username for response
            user_query = """
            SELECT username 
            FROM Users 
            WHERE user_id = :user_id::uuid
            """
            user_result = execute_query(db, user_query, {"user_id": post.user_id})
            if user_result:
                result[0]["username"] = user_result[0]["username"]
            return result[0]
        raise HTTPException(status_code=400, detail="Post creation failed")
    except Exception as e:
        logger.error(f"Error creating post: {e}")
        raise HTTPException(status_code=500, detail="Failed to create post")

@app.get("/posts/", response_model=List[PostResponse])
def get_posts(db: Session = Depends(get_db)):
    """Get all posts with user info"""
    query = """
    SELECT 
        p.post_id::text as post_id, 
        p.user_id::text as user_id, 
        p.content, p.media_urls, p.like_count,
        p.comment_count, p.engagement_score, p.created_at, p.updated_at,
        u.username
    FROM Posts p
    JOIN Users u ON p.user_id = u.user_id
    ORDER BY p.created_at DESC
    """
    return execute_query(db, query)

# ========== LIKE ENDPOINTS ==========
@app.post("/likes/", status_code=status.HTTP_201_CREATED)
def create_like(like: LikeCreate, db: Session = Depends(get_db)):
    """Add a like to a post"""
    try:
        query = """
        INSERT INTO Likes (post_id, user_id)
        VALUES (:post_id::uuid, :user_id::uuid)
        ON CONFLICT (user_id, post_id) DO NOTHING
        RETURNING created_at
        """
        result = execute_query(db, query, {
            "post_id": like.post_id,
            "user_id": like.user_id
        })
        
        if result:
            return {"message": "Like added successfully", "liked_at": result[0]["created_at"]}
        return {"message": "Like already exists"}
    except Exception as e:
        logger.error(f"Error adding like: {e}")
        raise HTTPException(status_code=500, detail="Failed to add like")

# ========== COMMENT ENDPOINTS ==========
@app.post("/comments/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
def create_comment(comment: CommentCreate, db: Session = Depends(get_db)):
    """Add a comment to a post"""
    try:
        query = """
        INSERT INTO Comments (post_id, user_id, content, parent_comment_id)
        VALUES (
            :post_id::uuid, 
            :user_id::uuid, 
            :content, 
            CASE WHEN :parent_comment_id IS NOT NULL AND :parent_comment_id != '' 
                 THEN :parent_comment_id::uuid 
                 ELSE NULL END
        )
        RETURNING 
            comment_id::text as comment_id, 
            post_id::text as post_id, 
            user_id::text as user_id, 
            content, 
            CASE WHEN parent_comment_id IS NOT NULL 
                 THEN parent_comment_id::text 
                 ELSE NULL END as parent_comment_id,
            like_count, created_at, updated_at
        """
        
        result = execute_query(db, query, {
            "post_id": comment.post_id,
            "user_id": comment.user_id,
            "content": comment.content,
            "parent_comment_id": comment.parent_comment_id or None
        })
        
        if result:
            # Get username
            user_query = """
            SELECT username 
            FROM Users 
            WHERE user_id = :user_id::uuid
            """
            user_result = execute_query(db, user_query, {"user_id": comment.user_id})
            if user_result:
                result[0]["username"] = user_result[0]["username"]
            return result[0]
        raise HTTPException(status_code=400, detail="Comment creation failed")
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create comment")

# ========== ANALYTICS ENDPOINTS ==========
@app.get("/analytics/top-posts")
def get_top_posts(limit: int = 10, db: Session = Depends(get_db)):
    """Get top posts using window functions"""
    try:
        query = """
        SELECT 
            post_id::text as post_id,
            content,
            username,
            like_count,
            comment_count,
            engagement_score,
            created_at,
            engagement_rank,
            engagement_percentile,
            engagement_rate
        FROM top_posts 
        ORDER BY engagement_rank
        LIMIT :limit
        """
        return execute_query(db, query, {"limit": limit})
    except Exception as e:
        logger.error(f"Error fetching top posts: {e}")
        return []

@app.get("/analytics/user-summary")
def get_user_summary(limit: int = 20, db: Session = Depends(get_db)):
    """Get user engagement analytics"""
    try:
        query = """
        SELECT 
            user_id::text as user_id,
            username,
            total_posts,
            total_likes_given,
            total_comments,
            total_likes_received,
            total_comments_received,
            avg_engagement_score,
            user_rank,
            engagement_quartile
        FROM user_engagement_analytics 
        ORDER BY user_rank
        LIMIT :limit
        """
        return execute_query(db, query, {"limit": limit})
    except Exception as e:
        logger.error(f"Error fetching user summary: {e}")
        return []

@app.get("/analytics/engagement-stats")
def get_engagement_stats(db: Session = Depends(get_db)):
    """Get overall engagement statistics"""
    try:
        # Overall stats
        stats_query = """
        SELECT 
            COUNT(DISTINCT u.user_id) as total_users,
            COUNT(DISTINCT p.post_id) as total_posts,
            COALESCE(SUM(p.like_count), 0) as total_likes,
            COALESCE(SUM(p.comment_count), 0) as total_comments,
            COALESCE(AVG(p.engagement_score), 0) as avg_engagement,
            COALESCE(MAX(p.engagement_score), 0) as max_engagement
        FROM Users u
        LEFT JOIN Posts p ON u.user_id = p.user_id
        """
        stats = execute_query(db, stats_query)
        
        # Top users
        top_users_query = """
        SELECT 
            username, 
            total_likes_received,
            RANK() OVER (ORDER BY total_likes_received DESC) as rank
        FROM user_engagement_analytics
        LIMIT 5
        """
        top_users = execute_query(db, top_users_query)
        
        return {
            "overall_stats": stats[0] if stats else {},
            "top_engaged_users": top_users
        }
    except Exception as e:
        logger.error(f"Error fetching engagement stats: {e}")
        return {
            "overall_stats": {},
            "top_engaged_users": []
        }

@app.get("/analytics/union-activities")
def get_union_activities(db: Session = Depends(get_db)):
    """Get combined activities using UNION"""
    try:
        query = """
        SELECT 
            'POST' as activity_type, 
            u.username, 
            p.content, 
            p.created_at as activity_date
        FROM Users u 
        JOIN Posts p ON u.user_id = p.user_id
        UNION ALL
        SELECT 
            'LIKE' as activity_type, 
            u.username, 
            'Liked a post' as content, 
            l.created_at as activity_date
        FROM Users u 
        JOIN Likes l ON u.user_id = l.user_id
        UNION ALL
        SELECT 
            'COMMENT' as activity_type, 
            u.username, 
            c.content, 
            c.created_at as activity_date
        FROM Users u 
        JOIN Comments c ON u.user_id = c.user_id
        ORDER BY activity_date DESC
        LIMIT 20
        """
        return execute_query(db, query)
    except Exception as e:
        logger.error(f"Error fetching union activities: {e}")
        return []

@app.get("/analytics/group-by-engagement")
def group_by_engagement(db: Session = Depends(get_db)):
    """Group users by engagement level using GROUP BY and HAVING"""
    try:
        query = """
        SELECT 
            u.username,
            COUNT(p.post_id) as post_count,
            COALESCE(AVG(p.engagement_score), 0) as avg_engagement,
            COALESCE(SUM(p.like_count), 0) as total_likes,
            CASE 
                WHEN COALESCE(AVG(p.engagement_score), 0) > 2 THEN 'High'
                WHEN COALESCE(AVG(p.engagement_score), 0) > 1 THEN 'Medium'
                ELSE 'Low'
            END as engagement_level
        FROM Users u
        LEFT JOIN Posts p ON u.user_id = p.user_id
        GROUP BY u.user_id, u.username
        HAVING COUNT(p.post_id) >= 0
        ORDER BY avg_engagement DESC
        """
        return execute_query(db, query)
    except Exception as e:
        logger.error(f"Error fetching grouped engagement: {e}")
        return []

@app.get("/analytics/full-report", response_model=AnalyticsResponse)
def get_full_report(db: Session = Depends(get_db)):
    """Get complete analytics report"""
    try:
        top_posts = get_top_posts(limit=10, db=db)
        user_summary = get_user_summary(limit=20, db=db)
        engagement_stats = get_engagement_stats(db=db)
        union_activities = get_union_activities(db=db)
        grouped_engagement = group_by_engagement(db=db)
        
        return AnalyticsResponse(
            top_posts=top_posts,
            user_summary=user_summary,
            engagement_stats=engagement_stats,
            union_activities=union_activities,
            grouped_engagement=grouped_engagement
        )
    except Exception as e:
        logger.error(f"Error generating full report: {e}")
        # Return empty response instead of error
        return AnalyticsResponse(
            top_posts=[],
            user_summary=[],
            engagement_stats={},
            union_activities=[],
            grouped_engagement=[]
        )

# ========== HEALTH ENDPOINTS ==========
@app.get("/")
def root():
    return {
        "message": "Social Media Analytics API",
        "status": "running",
        "version": "2.0.0",
        "docs": "https://social-media-analytics-7nx4.onrender.com/docs"
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check with database connection"""
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat(),
            "service": "social-media-analytics-api"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 503

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"
    )