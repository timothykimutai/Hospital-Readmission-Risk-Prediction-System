from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class Database:
    def __init__(self, env='development'):
        self.env = env
        self.engine = self._get_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def _get_engine(self):
        if self.env == 'test':
            return create_engine("sqlite:///:memory:", echo=True)
        elif self.env == 'development':
            return create_engine("sqlite:///db/readmission.db", echo=True)
        else:  # production
            db_url = (
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
                f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            )
            return create_engine(db_url)
    
    def init_db(self):
        """Initialize database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()

# Singleton database instance
db = Database(env=os.getenv("ENV", "development"))