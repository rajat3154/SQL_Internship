# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, date
from uuid import UUID
from decimal import Decimal
from dotenv import load_dotenv
import os
import io
import csv
import json
from sqlalchemy.pool import NullPool
load_dotenv()

app = FastAPI(title="Social Media Analytics API", 
              description="Advanced PostgreSQL social media analytics with triggers and views",
              version="1.0.0")

origins = [
    "http://localhost:5173",  # Local development
    "http://localhost:3000",  # Alternative local
    "https://social-media-analytics-liard.vercel.app",  # Your Vercel frontend
    "https://social-media-analytics-liard.vercel.app/",  # With trailing slash
    "https://*.vercel.app",  # All Vercel deployments
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers
    max_age=600,  # Cache preflight requests for 10 minutes
)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(
    DATABASE_URL,
    # Use NullPool for Supabase to avoid connection limits
    poolclass=NullPool,  # This is crucial for Supabase
    # Or use very small pool if you must use pooling
    # pool_size=1,
    # max_overflow=0,
    pool_pre_ping=False,  # Disable for Supabase
    connect_args={
        "connect_timeout": 10,
        "application_name": "social_media_analytics",
        "options": "-c statement_timeout=30000"  # 30 second timeout
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database Connected Successfully")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from decimal import Decimal
# ========== PYDANTIC MODELS ==========
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str

class UserResponse(BaseModel):
    user_id: UUID
    username: str
    email: str
    full_name: str
    created_at: datetime
    profile_data: Optional[Dict[str, Any]] = {}
    is_active: bool = True

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }
    )

class PostCreate(BaseModel):
    user_id: UUID
    content: str
    media_urls: Optional[List[str]] = []

class PostResponse(BaseModel):
    post_id: UUID
    user_id: UUID
    username: str
    content: str
    media_urls: List[str] = []
    like_count: int = 0
    comment_count: int = 0
    engagement_score: float = 0.0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }
    )

class LikeCreate(BaseModel):
    post_id: UUID
    user_id: UUID

class CommentCreate(BaseModel):
    post_id: UUID
    user_id: UUID
    content: str
    parent_comment_id: Optional[UUID] = None

class CommentResponse(BaseModel):
    comment_id: UUID
    post_id: UUID
    user_id: UUID
    username: str
    content: str
    parent_comment_id: Optional[UUID] = None
    like_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }
    )

class AnalyticsResponse(BaseModel):
    top_posts: List[Dict[str, Any]]
    user_summary: List[Dict[str, Any]]
    engagement_stats: Dict[str, Any]
    union_activities: List[Dict[str, Any]] = []
    grouped_engagement: List[Dict[str, Any]] = []

    model_config = ConfigDict(
        json_encoders={
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }
    )

# ========== UTILITY FUNCTIONS ==========
def execute_query(db: Session, query: str, params: Dict = None):
    """Execute raw SQL query and return results"""
    try:
        result = db.execute(text(query), params or {})
        if result.returns_rows:
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        db.commit()
        return None
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

# ========== USER ENDPOINTS ==========
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        query = """
        INSERT INTO Users (username, email, full_name)
        VALUES (:username, :email, :full_name)
        RETURNING user_id, username, email, full_name, created_at, profile_data, is_active
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/users/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    """Get all users"""
    query = """
    SELECT user_id, username, email, full_name, created_at, profile_data, is_active
    FROM Users 
    ORDER BY created_at DESC
    """
    return execute_query(db, query)

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user by ID"""
    query = """
    SELECT user_id, username, email, full_name, created_at, profile_data, is_active
    FROM Users 
    WHERE user_id = :user_id
    """
    result = execute_query(db, query, {"user_id": user_id})
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result[0]

@app.delete("/users/{user_id}")
def delete_user(user_id: str, db: Session = Depends(get_db)):
    """Delete user by ID"""
    try:
        # First check if user exists
        check_query = "SELECT user_id FROM Users WHERE user_id = :user_id"
        user_exists = execute_query(db, check_query, {"user_id": user_id})
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Delete user (cascade will handle related data)
        delete_query = "DELETE FROM Users WHERE user_id = :user_id"
        execute_query(db, delete_query, {"user_id": user_id})
        
        return {"message": "User deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

# ========== POST ENDPOINTS ==========
@app.post("/posts/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreate, db: Session = Depends(get_db)):
    """Create a new post"""
    try:
        # Insert post
        query = """
        INSERT INTO Posts (user_id, content, media_urls)
        VALUES (:user_id, :content, :media_urls)
        RETURNING 
            post_id, user_id, content, media_urls, like_count, 
            comment_count, engagement_score, created_at, updated_at
        """
        result = execute_query(db, query, {
            "user_id": post.user_id,
            "content": post.content,
            "media_urls": post.media_urls or []
        })
        
        if result:
            # Get username for response
            user_query = "SELECT username FROM Users WHERE user_id = :user_id"
            user_result = execute_query(db, user_query, {"user_id": post.user_id})
            if user_result:
                result[0]["username"] = user_result[0]["username"]
            return result[0]
        raise HTTPException(status_code=400, detail="Post creation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/posts/", response_model=List[PostResponse])
def get_posts(db: Session = Depends(get_db)):
    """Get all posts with user info"""
    query = """
    SELECT 
        p.post_id, p.user_id, p.content, p.media_urls, p.like_count,
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
        VALUES (:post_id, :user_id)
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/likes/")
def remove_like(post_id: str, user_id: str, db: Session = Depends(get_db)):
    """Remove a like from a post"""
    try:
        query = """
        DELETE FROM Likes 
        WHERE post_id = :post_id AND user_id = :user_id
        RETURNING post_id
        """
        result = execute_query(db, query, {"post_id": post_id, "user_id": user_id})
        
        if result:
            return {"message": "Like removed successfully"}
        raise HTTPException(status_code=404, detail="Like not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ========== COMMENT ENDPOINTS ==========
@app.post("/comments/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
def create_comment(comment: CommentCreate, db: Session = Depends(get_db)):
    """Add a comment to a post"""
    try:
        query = """
        INSERT INTO Comments (post_id, user_id, content, parent_comment_id)
        VALUES (:post_id, :user_id, :content, :parent_comment_id)
        RETURNING 
            comment_id, post_id, user_id, content, parent_comment_id,
            like_count, created_at, updated_at
        """
        result = execute_query(db, query, {
            "post_id": comment.post_id,
            "user_id": comment.user_id,
            "content": comment.content,
            "parent_comment_id": comment.parent_comment_id
        })
        
        if result:
            # Get username
            user_query = "SELECT username FROM Users WHERE user_id = :user_id"
            user_result = execute_query(db, user_query, {"user_id": comment.user_id})
            if user_result:
                result[0]["username"] = user_result[0]["username"]
            return result[0]
        raise HTTPException(status_code=400, detail="Comment creation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/comments/", response_model=List[CommentResponse])
def get_comments(post_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Get all comments or comments for specific post"""
    if post_id:
        query = """
        SELECT 
            c.comment_id, c.post_id, c.user_id, c.content, c.parent_comment_id,
            c.like_count, c.created_at, c.updated_at, u.username
        FROM Comments c
        JOIN Users u ON c.user_id = u.user_id
        WHERE c.post_id = :post_id
        ORDER BY c.created_at DESC
        """
        params = {"post_id": post_id}
    else:
        query = """
        SELECT 
            c.comment_id, c.post_id, c.user_id, c.content, c.parent_comment_id,
            c.like_count, c.created_at, c.updated_at, u.username
        FROM Comments c
        JOIN Users u ON c.user_id = u.user_id
        ORDER BY c.created_at DESC
        LIMIT 100
        """
        params = {}
    
    return execute_query(db, query, params)

# ========== ANALYTICS ENDPOINTS ==========
@app.get("/analytics/top-posts")
def get_top_posts(limit: int = 10, db: Session = Depends(get_db)):
    """Get top posts using window functions"""
    query = """
    SELECT * FROM top_posts 
    ORDER BY engagement_rank
    LIMIT :limit
    """
    return execute_query(db, query, {"limit": limit})

@app.get("/analytics/user-summary")
def get_user_summary(limit: int = 20, db: Session = Depends(get_db)):
    """Get user engagement analytics"""
    query = """
    SELECT * FROM user_engagement_analytics 
    ORDER BY user_rank
    LIMIT :limit
    """
    return execute_query(db, query, {"limit": limit})

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
        SELECT username, total_likes_received,
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/analytics/union-activities")
def get_union_activities(db: Session = Depends(get_db)):
    """Get combined activities using UNION"""
    query = """
    SELECT 'POST' as activity_type, u.username, p.content, p.created_at as activity_date
    FROM Users u JOIN Posts p ON u.user_id = p.user_id
    UNION ALL
    SELECT 'LIKE' as activity_type, u.username, 'Liked a post' as content, l.created_at as activity_date
    FROM Users u JOIN Likes l ON u.user_id = l.user_id
    UNION ALL
    SELECT 'COMMENT' as activity_type, u.username, c.content, c.created_at as activity_date
    FROM Users u JOIN Comments c ON u.user_id = c.user_id
    ORDER BY activity_date DESC
    LIMIT 20
    """
    return execute_query(db, query)

@app.get("/analytics/search-posts")
def search_posts(query: str = "", db: Session = Depends(get_db)):
    """Search posts using LIKE operator"""
    if not query:
        return []
    
    search_query = """
    SELECT p.*, u.username 
    FROM Posts p 
    JOIN Users u ON p.user_id = u.user_id 
    WHERE p.content ILIKE :query
    ORDER BY p.created_at DESC
    LIMIT 20
    """
    return execute_query(db, search_query, {"query": f"%{query}%"})

@app.post("/analytics/refresh-materialized")
def refresh_materialized_view(db: Session = Depends(get_db)):
    """Refresh materialized views (simulated)"""
    try:
        # Simulate refresh by updating engagement scores
        refresh_query = """
        UPDATE Posts 
        SET engagement_score = (like_count * 2.5 + comment_count * 3.0) * 
            (1 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0),
            updated_at = NOW()
        """
        execute_query(db, refresh_query)
        return {"message": "Views and metrics refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/analytics/group-by-engagement")
def group_by_engagement(db: Session = Depends(get_db)):
    """Group users by engagement level using GROUP BY and HAVING"""
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

@app.get("/analytics/export-report")
def export_report(db: Session = Depends(get_db)):
    """Export analytics report as CSV"""
    try:
        # Get top posts data
        query = "SELECT * FROM top_posts ORDER BY engagement_rank LIMIT 50"
        result = execute_query(db, query)
        
        if not result:
            raise HTTPException(status_code=404, detail="No data to export")
        
        # Create CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=engagement_report.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

# ========== HEALTH ENDPOINTS ==========
@app.get("/")
def root():
    return {"message": "Social Media Analytics API", "status": "running"}

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check with database connection"""
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)