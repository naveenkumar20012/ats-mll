import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_db_session():
    engine = create_engine(os.getenv("DATABASE_URL"))
    Session = sessionmaker(engine)
    return Session
