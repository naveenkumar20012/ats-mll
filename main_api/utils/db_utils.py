import os

from fastapi_sqlalchemy import db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.db_models import Company as ModelCompany

from utils.logging_helpers import error_logger

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW"))


def get_db_session():
    engine = create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
    )
    Session = sessionmaker(engine)
    return Session


Session = get_db_session()


def get_or_create_company(company_uuid, Session=Session):
    try:
        with Session() as session:
            company_obj = (
                session.query(ModelCompany)
                .filter(ModelCompany.uuid == company_uuid)
                .first()
            )
            if company_obj:
                return company_obj.id, company_obj.num_calls_to_third_party
            else:
                company_obj = ModelCompany(uuid=company_uuid)
                session.add(company_obj)
                session.commit()
                return company_obj.id, 0
    except Exception as e:
        if session:
            session.rollback()
            session.close()
        error_logger(
            f"Error while fetching and storing data of company in database {e}"
        )
        return None, None
