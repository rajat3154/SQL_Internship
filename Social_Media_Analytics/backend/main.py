from fastapi import FastAPI,Depends,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy import  create_engine,text,func
from sqlalchemy.orm import sessionmaker,Session
from fastapi.responses import StreamingResponse
from typing import List,Optional
from pydantic import BaseModel
from datetime import datetime
from schemas.schema import UserCreate,UserResponse,PostCreate,PostResponse,LikeCreate,CommentCreate,AnalyticsResponse
import os,io,csv

load_dotenv()

app=FastAPI()

app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"]
)

DATABASE_URL=os.getenv("DATABASE_URL")
engine=create_engine(DATABASE_URL,pool_pre_ping=True)
SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)

def get_db():
      db=SessionLocal()
      try:
            yield db
      finally:
            db.close()

@app.on_event("startup")
def startup_event():
      try:
            with engine.connect() as conn:
                  conn.execute(text("SELECT 1"))
                  print("Database Connected Succesfully")
      except Exception as e:
            print("Database connection failed: ", e)

@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("INSERT INTO users (username, email, full_name) VALUES (:username, :email, :full_name) RETURNING *"),
            {"username": user.username, "email": user.email, "full_name": user.full_name}
        )
        db.commit()
        return result.mappings().first()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM users ORDER BY created_at DESC"))
    return result.mappings().all()

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM users WHERE id = :user_id"), {"user_id": user_id})
    user = result.mappings().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserCreate, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("UPDATE users SET username=:username, email=:email, full_name=:full_name, updated_at=CURRENT_TIMESTAMP WHERE id=:id RETURNING *"),
            {"id": user_id, "username": user.username, "email": user.email, "full_name": user.full_name}
        )
        db.commit()
        updated_user = result.mappings().first()
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")
        return updated_user
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    try:
        result = db.execute(text("DELETE FROM users WHERE id = :user_id RETURNING id"), {"user_id": user_id})
        db.commit()
        if not result.mappings().first():
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/posts/", response_model=PostResponse)
def create_post(post: PostCreate, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("INSERT INTO posts (user_id, content) VALUES (:user_id, :content) RETURNING *"),
            {"user_id": post.user_id, "content": post.content}
        )
        db.commit()
        return result.mappings().first()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/posts/", response_model=List[PostResponse])
def get_posts(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM posts ORDER BY created_at DESC"))
    return result.mappings().all()


@app.post("/likes/")
def create_like(like: LikeCreate, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("INSERT INTO likes (post_id, user_id) VALUES (:post_id, :user_id) RETURNING *"),
            {"post_id": like.post_id, "user_id": like.user_id}
        )
        db.commit()
        return {"message": "Post liked successfully", "like": result.mappings().first()}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Already liked or invalid data")

@app.delete("/likes/")
def remove_like(like: LikeCreate, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("DELETE FROM likes WHERE post_id = :post_id AND user_id = :user_id RETURNING id"),
            {"post_id": like.post_id, "user_id": like.user_id}
        )
        db.commit()
        if not result.mappings().first():
            raise HTTPException(status_code=404, detail="Like not found")
        return {"message": "Like removed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/comments/")
def create_comment(comment: CommentCreate, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("INSERT INTO comments (post_id, user_id, content) VALUES (:post_id, :user_id, :content) RETURNING *"),
            {"post_id": comment.post_id, "user_id": comment.user_id, "content": comment.content}
        )
        db.commit()
        return {"message": "Comment added successfully", "comment": result.mappings().first()}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/top-posts")
def get_top_posts(limit: int = 10, db: Session = Depends(get_db)):
    result = db.execute(
        text("SELECT * FROM top_posts LIMIT :limit"),
        {"limit": limit}
    )
    return result.mappings().all()

@app.get("/analytics/user-summary")
def get_user_summary(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM user_engagement_summary"))
    return result.mappings().all()

@app.get("/analytics/engagement-stats")
def get_engagement_stats(db: Session = Depends(get_db)):

    result = db.execute(text("""
        SELECT 
            COUNT(*) as total_posts,
            SUM(like_count) as total_likes,
            SUM(comment_count) as total_comments,
            AVG(engagement_score) as avg_engagement,
            MAX(engagement_score) as max_engagement
        FROM posts
    """))
    stats = result.mappings().first()
    
    top_engaged = db.execute(text("""
        SELECT username, total_likes_received,
               RANK() OVER (ORDER BY total_likes_received DESC) as rank
        FROM user_engagement_summary
        LIMIT 5
    """))
    
    return {
        "overall_stats": stats,
        "top_engaged_users": top_engaged.mappings().all()
    }

@app.get("/analytics/union-activities")
def get_union_activities(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT 'POST' as activity_type, u.username, p.content, p.created_at as activity_date
        FROM users u JOIN posts p ON u.id = p.user_id
        UNION ALL
        SELECT 'LIKE' as activity_type, u.username, 'Liked a post' as content, l.created_at as activity_date
        FROM users u JOIN likes l ON u.id = l.user_id
        UNION ALL
        SELECT 'COMMENT' as activity_type, u.username, c.content, c.created_at as activity_date
        FROM users u JOIN comments c ON u.id = c.user_id
        ORDER BY activity_date DESC
        LIMIT 20
    """))
    return result.mappings().all()

@app.get("/analytics/search-posts")
def search_posts(query: str, db: Session = Depends(get_db)):
    result = db.execute(
        text("SELECT p.*, u.username FROM posts p JOIN users u ON p.user_id = u.id WHERE p.content LIKE :query"),
        {"query": f"%{query}%"}
    )
    return result.mappings().all()

@app.post("/analytics/refresh-materialized")
def refresh_materialized_view(db: Session = Depends(get_db)):
    try:
        db.execute(text("REFRESH MATERIALIZED VIEW post_engagement_mv"))
        db.execute(text("CALL refresh_engagement_metrics()"))
        db.commit()
        return {"message": "Materialized views and metrics refreshed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/group-by-engagement")
def group_by_engagement(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT 
            u.username,
            COUNT(p.id) as post_count,
            AVG(p.engagement_score) as avg_engagement,
            SUM(p.like_count) as total_likes,
            CASE 
                WHEN AVG(p.engagement_score) > 2 THEN 'High'
                WHEN AVG(p.engagement_score) > 1 THEN 'Medium'
                ELSE 'Low'
            END as engagement_level
        FROM users u
        JOIN posts p ON u.id = p.user_id
        GROUP BY u.id, u.username
        HAVING COUNT(p.id) >= 1
        ORDER BY avg_engagement DESC
    """))
    return result.mappings().all()

@app.get("/analytics/export-report")
def export_report(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM top_posts"))
    
    output = io.StringIO()
    writer = csv.writer(output)
    if result.returns_rows:
        writer.writerow(result.keys())
        for row in result:
            writer.writerow(row)
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=engagement_report.csv"}
    )

@app.get("/")
def health_check():
    return {"message": "Social Media Analytics API is running"}

@app.get("/health")
def detailed_health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected", "timestamp": datetime.now()}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}