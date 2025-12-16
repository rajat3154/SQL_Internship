# main.py - COMPLETE WORKING VERSION
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
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Social Media Analytics API",
    description="Production-ready API for social media analytics",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ========== CORS CONFIGURATION ==========
# Allow all origins for now to debug
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ========== DATABASE CONFIGURATION ==========
def format_database_url():
    """Format database URL for Supabase"""
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    
    if not DATABASE_URL:
        logger.error("❌ DATABASE_URL not found in environment variables")
        return ""
    
    logger.info(f"🔧 Original DATABASE_URL: {DATABASE_URL[:50]}...")
    
    # Ensure we're using port 6543 for Supabase connection pooling
    if "pooler.supabase.com" in DATABASE_URL:
        if ":5432/" in DATABASE_URL:
            DATABASE_URL = DATABASE_URL.replace(":5432/", ":6543/")
            logger.info("✅ Using Supabase connection pooler on port 6543")
        else:
            logger.info("ℹ️ Already using connection pooler port")
    
    logger.info(f"🔧 Formatted DATABASE_URL: {DATABASE_URL[:50]}...")
    return DATABASE_URL

DATABASE_URL = format_database_url()

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")

try:
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
        },
        echo=False,
    )
    
    # Test connection immediately
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )
    
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    raise

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
        db.close()

# ========== PYDANTIC MODELS ==========
class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    created_at: datetime
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)

class PostResponse(BaseModel):
    post_id: str
    user_id: str
    username: str
    content: str
    like_count: int = 0
    comment_count: int = 0
    engagement_score: float = 0.0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AnalyticsResponse(BaseModel):
    top_posts: List[Dict[str, Any]]
    user_summary: List[Dict[str, Any]]
    engagement_stats: Dict[str, Any]

# ========== UTILITY FUNCTIONS ==========
def execute_safe_query(db: Session, query: str, params: Dict = None):
    """Safely execute SQL query with error handling"""
    try:
        result = db.execute(text(query), params or {})
        
        if result.returns_rows:
            columns = result.keys()
            rows = result.fetchall()
            
            converted_rows = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert problematic types
                    if isinstance(value, UUID):
                        value = str(value)
                    elif isinstance(value, Decimal):
                        value = float(value)
                    elif isinstance(value, datetime):
                        value = value.isoformat()
                    row_dict[col] = value
                converted_rows.append(row_dict)
            return converted_rows
        
        db.commit()
        return []
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Query failed: {str(e)[:200]}")
        logger.error(f"❌ Query was: {query[:200]}")
        raise

# ========== HEALTH & ROOT ENDPOINTS ==========
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Social Media Analytics API",
        "status": "running",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "users": "/users",
            "posts": "/posts",
            "analytics": "/analytics"
        }
    }

@app.get("/health", include_in_schema=False)
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        
        # Get some basic stats
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM users) as user_count,
            (SELECT COUNT(*) FROM posts) as post_count,
            (SELECT COUNT(*) FROM likes) as like_count,
            (SELECT COUNT(*) FROM comments) as comment_count
        """
        stats = execute_safe_query(db, stats_query)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.now().isoformat(),
                "stats": stats[0] if stats else {}
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# ========== BASIC ENDPOINTS ==========
@app.get("/users", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    """Get all users"""
    try:
        query = """
        SELECT 
            user_id::text as user_id,
            username,
            email,
            full_name,
            created_at,
            is_active
        FROM users 
        ORDER BY created_at DESC
        LIMIT 50
        """
        users = execute_safe_query(db, query)
        return JSONResponse(content=users)
    except Exception as e:
        logger.error(f"Error in /users: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch users", "details": str(e)[:100]}
        )

@app.get("/posts", response_model=List[PostResponse])
def get_posts(db: Session = Depends(get_db)):
    """Get all posts"""
    try:
        query = """
        SELECT 
            p.post_id::text as post_id,
            p.user_id::text as user_id,
            u.username,
            p.content,
            p.like_count,
            p.comment_count,
            p.engagement_score,
            p.created_at,
            p.updated_at
        FROM posts p
        JOIN users u ON p.user_id = u.user_id
        ORDER BY p.created_at DESC
        LIMIT 50
        """
        posts = execute_safe_query(db, query)
        return JSONResponse(content=posts)
    except Exception as e:
        logger.error(f"Error in /posts: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch posts", "details": str(e)[:100]}
        )

# ========== ANALYTICS ENDPOINTS ==========
@app.get("/analytics/user-summary")
def get_user_summary(limit: int = 20, db: Session = Depends(get_db)):
    """Get user engagement analytics - SIMPLIFIED VERSION"""
    try:
        # First check if view exists
        check_view_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name = 'user_engagement_analytics'
        ) as view_exists
        """
        view_check = execute_safe_query(db, check_view_query)
        
        if not view_check or not view_check[0].get('view_exists'):
            logger.warning("View 'user_engagement_analytics' does not exist, using fallback query")
            # Fallback query
            query = """
            SELECT 
                u.user_id::text,
                u.username,
                COUNT(p.post_id) as total_posts,
                COALESCE(SUM(p.like_count), 0) as total_likes_received,
                COALESCE(SUM(p.comment_count), 0) as total_comments_received,
                COALESCE(AVG(p.engagement_score), 0) as avg_engagement_score,
                ROW_NUMBER() OVER (ORDER BY COALESCE(SUM(p.engagement_score), 0) DESC) as user_rank
            FROM users u
            LEFT JOIN posts p ON u.user_id = p.user_id
            GROUP BY u.user_id, u.username
            ORDER BY total_likes_received DESC
            LIMIT :limit
            """
        else:
            # Use the view
            query = """
            SELECT 
                user_id::text as user_id,
                username,
                total_posts,
                total_likes_received,
                total_comments_received,
                avg_engagement_score,
                user_rank
            FROM user_engagement_analytics 
            ORDER BY user_rank
            LIMIT :limit
            """
        
        result = execute_safe_query(db, query, {"limit": limit})
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in /analytics/user-summary: {e}")
        # Return empty array instead of error
        return JSONResponse(content=[])

@app.get("/analytics/engagement-stats")
def get_engagement_stats(db: Session = Depends(get_db)):
    """Get overall engagement statistics - SIMPLIFIED VERSION"""
    try:
        query = """
        SELECT 
            COUNT(DISTINCT u.user_id) as total_users,
            COUNT(DISTINCT p.post_id) as total_posts,
            COALESCE(SUM(p.like_count), 0) as total_likes,
            COALESCE(SUM(p.comment_count), 0) as total_comments,
            COALESCE(AVG(p.engagement_score), 0) as avg_engagement,
            COALESCE(MAX(p.engagement_score), 0) as max_engagement
        FROM users u
        LEFT JOIN posts p ON u.user_id = p.user_id
        """
        
        stats = execute_safe_query(db, query)
        
        # Top users simple query
        top_users_query = """
        SELECT 
            u.username,
            COALESCE(SUM(p.like_count), 0) as total_likes_received,
            ROW_NUMBER() OVER (ORDER BY COALESCE(SUM(p.like_count), 0) DESC) as rank
        FROM users u
        LEFT JOIN posts p ON u.user_id = p.user_id
        GROUP BY u.user_id, u.username
        ORDER BY total_likes_received DESC
        LIMIT 5
        """
        
        top_users = execute_safe_query(db, top_users_query)
        
        response_data = {
            "overall_stats": stats[0] if stats else {},
            "top_engaged_users": top_users
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in /analytics/engagement-stats: {e}")
        # Return empty stats instead of error
        return JSONResponse(content={
            "overall_stats": {},
            "top_engaged_users": []
        })

@app.get("/analytics/top-posts")
def get_top_posts(limit: int = 10, db: Session = Depends(get_db)):
    """Get top posts - SIMPLIFIED VERSION"""
    try:
        # Check if view exists
        check_view_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name = 'top_posts'
        ) as view_exists
        """
        view_check = execute_safe_query(db, check_view_query)
        
        if not view_check or not view_check[0].get('view_exists'):
            logger.warning("View 'top_posts' does not exist, using fallback query")
            # Fallback query
            query = """
            SELECT 
                p.post_id::text,
                p.content,
                u.username,
                p.like_count,
                p.comment_count,
                p.engagement_score,
                p.created_at,
                ROW_NUMBER() OVER (ORDER BY p.engagement_score DESC) as engagement_rank
            FROM posts p
            JOIN users u ON p.user_id = u.user_id
            WHERE p.created_at >= NOW() - INTERVAL '30 days'
            ORDER BY p.engagement_score DESC
            LIMIT :limit
            """
        else:
            # Use the view
            query = """
            SELECT 
                post_id::text,
                content,
                username,
                like_count,
                comment_count,
                engagement_score,
                created_at,
                engagement_rank
            FROM top_posts 
            ORDER BY engagement_rank
            LIMIT :limit
            """
        
        result = execute_safe_query(db, query, {"limit": limit})
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in /analytics/top-posts: {e}")
        return JSONResponse(content=[])

@app.get("/analytics/full-report")
def get_full_report(db: Session = Depends(get_db)):
    """Get complete analytics report"""
    try:
        import asyncio
        
        # Fetch all analytics in parallel (simulated)
        top_posts = get_top_posts(limit=10, db=db).body
        user_summary = get_user_summary(limit=20, db=db).body
        engagement_stats = get_engagement_stats(db=db).body
        
        # Parse JSON responses
        top_posts_data = json.loads(top_posts) if isinstance(top_posts, (bytes, str)) else top_posts
        user_summary_data = json.loads(user_summary) if isinstance(user_summary, (bytes, str)) else user_summary
        engagement_stats_data = json.loads(engagement_stats) if isinstance(engagement_stats, (bytes, str)) else engagement_stats
        
        response_data = {
            "top_posts": top_posts_data,
            "user_summary": user_summary_data,
            "engagement_stats": engagement_stats_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in /analytics/full-report: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to generate report",
                "timestamp": datetime.now().isoformat()
            }
        )

# ========== OPTIONS HANDLER ==========
@app.options("/{path:path}")
async def options_handler():
    """Handle OPTIONS requests for CORS preflight"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# ========== ERROR HANDLER ==========
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("ENVIRONMENT") == "development" else "Something went wrong"
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default
    port = int(os.getenv("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )