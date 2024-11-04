from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from typing import Optional, Tuple
import hashlib
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Database connection parameters
DB_USER = os.getenv('db_user')
DB_PASSWORD = os.getenv('db_password')
DB_HOST = os.getenv('db_host')
DB_PORT = os.getenv('db_port')
DB_NAME = os.getenv('db_name')

# Create Base class for declarative models
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(64), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=True)
    role = Column(String(50), nullable=True)
    status = Column(String(20), nullable=False, default='pending')  # 'pending' or 'approved'
    created_at = Column(DateTime, nullable=False, default=func.now())
    approved_at = Column(DateTime, nullable=True)
    last_signin = Column(DateTime, nullable=True)  # Tracks the user's last sign-in time

class AdminLog(Base):
    __tablename__ = 'admin_logs'
    
    id = Column(Integer, primary_key=True)
    admin_user = Column(String(255), nullable=False)
    action = Column(String(100), nullable=False)
    affected_user = Column(String(255), nullable=True)
    details = Column(String(500), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())

class BlogPost(Base):
    __tablename__ = 'blog_posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    summary = Column(String(500), nullable=False)
    link = Column(String(1000), nullable=False)
    author_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
    likes_count = Column(Integer, default=0)
    
class BlogComment(Base):
    __tablename__ = 'blog_comments'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, nullable=False)  # Links to BlogPost.id
    author_id = Column(Integer, nullable=False)  # Links to User.id
    content = Column(String(1000), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())

class BlogLike(Base):
    __tablename__ = 'blog_likes'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, nullable=False)  # Links to BlogPost.id
    user_id = Column(Integer, nullable=False)  # Links to User.id
    created_at = Column(DateTime, nullable=False, default=func.now())

class DatabaseManager:
    def __init__(self):
        # Construct connection string using the defined variables
        connection_string = (
            f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        )
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, email: str, password: str, first_name: str, last_name: str, 
                   department: str = None, role: str = None) -> bool:
        session = self.Session()
        try:
            user = User(
                email=email,
                password_hash=self._hash_password(password),
                first_name=first_name,
                last_name=last_name,
                department=department,
                role=role
            )
            session.add(user)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False
        finally:
            session.close()
    
    def verify_user(self, email: str, password: str) -> Optional[Tuple[str, str]]:
        session = self.Session()
        try:
            user = session.query(User).filter_by(
                email=email,
                password_hash=self._hash_password(password)
            ).first()
            return (user.first_name, user.last_name,user.department,user.role,user.status) if user else None
        finally:
            session.close()
    
    def update_last_signin(self, username):
        with self.Session() as session:
            user = session.query(User).filter_by(email=username).first()
            if user:
                user.last_signin = func.now()
                session.commit()
    
    def approve_user(self, user_id: int, role: str, department: str) -> bool:
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.status = 'approved'
                user.role = role
                user.department = department
                user.approved_at = datetime.now()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error approving user: {str(e)}")
            return False
        finally:
            session.close()

    def is_admin(self, username: str) -> bool:
        if not username:
            return False
        
        session = self.Session()
        try:
            user = session.query(User).filter_by(email=username).first()
            return user is not None and user.role == 'Admin' and user.status == 'approved'
        finally:
            session.close()

    def get_pending_users(self):
        session = self.Session()
        try:
            return session.query(User).filter_by(status='pending').all()
        finally:
            session.close()

    def get_all_users(self):
        session = self.Session()
        try:
            return session.query(User).all()
        finally:
            session.close()

    def delete_user(self, user_id: int) -> bool:
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                session.delete(user)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting user: {str(e)}")
            return False
        finally:
            session.close()

    def update_user(self, user_id: int, department: str = None, role: str = None, status: str = None) -> bool:
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                if department is not None:
                    user.department = department
                if role is not None:
                    user.role = role
                if status is not None:
                    user.status = status
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error updating user: {str(e)}")
            return False
        finally:
            session.close()

    def log_admin_action(self, action_data):
        """Log admin actions to the database"""
        session = self.Session()
        try:
            admin_log = AdminLog(
                admin_user=action_data['admin_user'],
                action=action_data['action'],
                affected_user=action_data['affected_user'],
                details=action_data['details'],
                timestamp=action_data['timestamp']
            )
            session.add(admin_log)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error logging admin action: {str(e)}")
            return False
        finally:
            session.close()

    def create_blog_post(self, title: str, summary: str, link: str, author_email: str) -> Optional[int]:
        session = self.Session()
        try:
            # Get author's ID
            author = session.query(User).filter_by(email=author_email).first()
            if not author:
                return None
            
            post = BlogPost(
                title=title,
                summary=summary,
                link=link,
                author_id=author.id
            )
            session.add(post)
            session.commit()
            return post.id
        except Exception as e:
            session.rollback()
            print(f"Error creating blog post: {str(e)}")
            return None
        finally:
            session.close()

    def get_blog_posts(self, limit: int = None):
        session = self.Session()
        try:
            query = session.query(
                BlogPost,
                User.first_name,
                User.last_name,
                User.email
            ).join(User, BlogPost.author_id == User.id)\
              .order_by(BlogPost.created_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        finally:
            session.close()

    def add_blog_comment(self, post_id: int, author_email: str, content: str) -> bool:
        session = self.Session()
        try:
            author = session.query(User).filter_by(email=author_email).first()
            if not author:
                return False
            
            comment = BlogComment(
                post_id=post_id,
                author_id=author.id,
                content=content
            )
            session.add(comment)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error adding comment: {str(e)}")
            return False
        finally:
            session.close()

    def get_blog_comments(self, post_id: int):
        session = self.Session()
        try:
            return session.query(
                BlogComment,
                User.first_name,
                User.last_name
            ).join(User, BlogComment.author_id == User.id)\
              .filter(BlogComment.post_id == post_id)\
              .order_by(BlogComment.created_at.desc())\
              .all()
        finally:
            session.close()

    def toggle_blog_like(self, post_id: int, user_email: str) -> Tuple[bool, int]:
        session = self.Session()
        try:
            user = session.query(User).filter_by(email=user_email).first()
            if not user:
                return False, 0
            
            existing_like = session.query(BlogLike)\
                .filter_by(post_id=post_id, user_id=user.id)\
                .first()
            
            if existing_like:
                session.delete(existing_like)
                session.query(BlogPost).filter_by(id=post_id)\
                    .update({"likes_count": BlogPost.likes_count - 1})
            else:
                new_like = BlogLike(post_id=post_id, user_id=user.id)
                session.add(new_like)
                session.query(BlogPost).filter_by(id=post_id)\
                    .update({"likes_count": BlogPost.likes_count + 1})
            
            session.commit()
            
            # Get updated likes count
            post = session.query(BlogPost).filter_by(id=post_id).first()
            return True, post.likes_count
        except Exception as e:
            session.rollback()
            print(f"Error toggling like: {str(e)}")
            return False, 0
        finally:
            session.close()

    def has_user_liked_post(self, post_id: int, user_email: str) -> bool:
        session = self.Session()
        try:
            user = session.query(User).filter_by(email=user_email).first()
            if not user:
                return False
            
            return session.query(BlogLike)\
                .filter_by(post_id=post_id, user_id=user.id)\
                .first() is not None
        finally:
            session.close()

    def delete_blog_post(self, post_id: int) -> bool:
        session = self.Session()
        try:
            # First delete associated likes and comments
            session.query(BlogLike).filter_by(post_id=post_id).delete()
            session.query(BlogComment).filter_by(post_id=post_id).delete()
            
            # Then delete the post
            post = session.query(BlogPost).filter_by(id=post_id).first()
            if post:
                session.delete(post)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting blog post: {str(e)}")
            return False
        finally:
            session.close()