
CREATE DATABASE social_media_analytics;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    engagement_score DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE likes (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, user_id)
);

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO users (username, email, full_name) VALUES
('john_doe', 'john@example.com', 'John Doe'),
('jane_smith', 'jane@example.com', 'Jane Smith'),
('mike_wilson', 'mike@example.com', 'Mike Wilson'),
('sarah_jones', 'sarah@example.com', 'Sarah Jones'),
('alex_brown', 'alex@example.com', 'Alex Brown');

INSERT INTO posts (user_id, content) VALUES
(1, 'Just launched my new startup! Excited for this journey. #entrepreneurship'),
(2, 'Beautiful sunset at the beach today 🌅 #nature #photography'),
(3, 'Just finished reading an amazing book about AI and machine learning.'),
(1, 'Working on some exciting new features for our platform. Stay tuned!'),
(4, 'Recipe for the perfect chocolate chip cookies: ... #baking #food'),
(5, 'My thoughts on the future of renewable energy and sustainability.'),
(2, 'Just completed a 10k run! Feeling great. #fitness #health'),
(3, 'New blog post about database optimization techniques is live!'),
(4, 'Gardening tips for growing organic vegetables at home.'),
(5, 'The impact of social media on modern communication. #sociology');

INSERT INTO likes (post_id, user_id) VALUES
(1,2), (1,3), (1,4), (1,5),
(2,1), (2,3), (2,5),
(3,1), (3,2), (3,4),
(4,2), (4,3),
(5,1), (5,3), (5,4), (5,5),
(6,1), (6,2),
(7,3), (7,4), (7,5),
(8,1), (8,2), (8,4),
(9,1), (9,3),
(10,2), (10,4);

INSERT INTO comments (post_id, user_id, content) VALUES
(1, 2, 'Congratulations! Wishing you all the success.'),
(1, 3, 'This is amazing! What industry are you in?'),
(2, 1, 'Stunning photo! Where was this taken?'),
(3, 4, 'Which book was it? I am looking for recommendations.'),
(4, 5, 'Can you give us a hint about the new features?'),
(5, 2, 'Tried this recipe and it was delicious!'),
(6, 3, 'Great insights on renewable energy future.');


CREATE OR REPLACE VIEW top_posts AS
SELECT 
    p.id as post_id,
    u.username,
    p.content,
    p.like_count,
    p.comment_count,
    p.engagement_score,
    RANK() OVER (ORDER BY p.engagement_score DESC) as rank
FROM posts p
JOIN users u ON p.user_id = u.id
WHERE p.engagement_score > 0
ORDER BY p.engagement_score DESC;

CREATE OR REPLACE VIEW user_engagement_summary AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(DISTINCT p.id) as total_posts,
    COALESCE(SUM(p.like_count), 0) as total_likes_received,
    COALESCE(SUM(p.comment_count), 0) as total_comments_received,
    COUNT(DISTINCT l.id) as likes_given,
    COUNT(DISTINCT c.id) as comments_given
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
LEFT JOIN likes l ON u.id = l.user_id
LEFT JOIN comments c ON u.id = c.user_id
GROUP BY u.id, u.username;


SELECT 
    username,
    total_posts,
    total_likes_received,
    RANK() OVER (ORDER BY total_likes_received DESC) as likes_rank,
    DENSE_RANK() OVER (ORDER BY total_posts DESC) as posts_rank,
    PERCENT_RANK() OVER (ORDER BY total_likes_received) as percentile_rank
FROM user_engagement_summary;


CREATE OR REPLACE FUNCTION update_post_like_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE posts SET like_count = like_count + 1 WHERE id = NEW.post_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE posts SET like_count = like_count - 1 WHERE id = OLD.post_id;
    END IF;
    

    UPDATE posts 
    SET engagement_score = (like_count * 0.6 + comment_count * 0.4)
    WHERE id = COALESCE(NEW.post_id, OLD.post_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_like_count
    AFTER INSERT OR DELETE ON likes
    FOR EACH ROW EXECUTE FUNCTION update_post_like_count();


CREATE OR REPLACE FUNCTION update_post_comment_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE posts SET comment_count = comment_count + 1 WHERE id = NEW.post_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE posts SET comment_count = comment_count - 1 WHERE id = OLD.post_id;
    END IF;
    
    UPDATE posts 
    SET engagement_score = (like_count * 0.6 + comment_count * 0.4)
    WHERE id = COALESCE(NEW.post_id, OLD.post_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_comment_count
    AFTER INSERT OR DELETE ON comments
    FOR EACH ROW EXECUTE FUNCTION update_post_comment_count();


CREATE OR REPLACE PROCEDURE refresh_engagement_metrics()
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE posts 
    SET engagement_score = (like_count * 0.6 + comment_count * 0.4);
    
    RAISE NOTICE 'Engagement metrics refreshed successfully';
END;
$$;

SELECT 
    u.username,
    COUNT(p.id) as post_count,
    AVG(p.engagement_score) as avg_engagement,
    SUM(p.like_count) as total_likes
FROM users u
JOIN posts p ON u.id = p.user_id
GROUP BY u.id, u.username
HAVING COUNT(p.id) >= 2
ORDER BY avg_engagement DESC;


SELECT u.username FROM users u
WHERE EXISTS (SELECT 1 FROM posts p WHERE p.user_id = u.id)
INTERSECT
SELECT u.username FROM users u
WHERE EXISTS (SELECT 1 FROM comments c WHERE c.user_id = u.id);


SELECT 'POST' as activity_type, u.username, p.created_at as activity_date
FROM users u JOIN posts p ON u.id = p.user_id
UNION ALL
SELECT 'LIKE' as activity_type, u.username, l.created_at as activity_date
FROM users u JOIN likes l ON u.id = l.user_id
UNION ALL
SELECT 'COMMENT' as activity_type, u.username, c.created_at as activity_date
FROM users u JOIN comments c ON u.id = c.user_id
ORDER BY activity_date DESC;

SELECT * FROM posts 
WHERE content LIKE '%startup%' OR content LIKE '%AI%';

SELECT * FROM users 
WHERE username IN ('john_doe', 'jane_smith') 
   OR email LIKE '%@example.com';

-- 8. Materialized View for performance
CREATE MATERIALIZED VIEW post_engagement_mv AS
SELECT 
    p.id,
    p.content,
    u.username,
    p.like_count,
    p.comment_count,
    p.engagement_score,
    p.created_at
FROM posts p
JOIN users u ON p.user_id = u.id
ORDER BY p.engagement_score DESC;

CREATE UNIQUE INDEX ON post_engagement_mv (id);