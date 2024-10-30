from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from typing import Optional, Tuple
import hashlib
import os
from dotenv import load_dotenv

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
            return (user.first_name, user.last_name,user.department,user.role) if user else None
        finally:
            session.close()