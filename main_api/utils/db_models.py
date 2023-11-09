from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    ARRAY,
    Numeric,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String)
    num_calls_to_third_party = Column(Integer, default=0)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())


class JDescriptions(Base):
    __tablename__ = "jdescriptions"
    id = Column(Integer, primary_key=True, index=True)
    gcp_filepath = Column(String)
    json_output = Column(JSON)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    resume_process_id = Column(String)
    gcp_filepath = Column(String)
    local_filepath = Column(String)
    json_output = Column(JSON)
    rchilli_output = Column(JSON)
    is_self_parsed = Column(
        Boolean, default=True
    )  # If the resume was parsed on our server or not
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
    company_id = Column(Integer, ForeignKey("companies.id"))
    resume_text = Column(String)
    resume_entities = Column(ARRAY(String))
    ats_response_code = Column(Integer)
    author = relationship("Company")


class ReviewedResume(Base):
    __tablename__ = "candidate_data"
    id = Column(Integer, primary_key=True, index=True)
    resume_filepath = Column(String)
    parser_output = Column(JSON)
    reviewed_data = Column(JSON)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
    reviewer_id = Column(Integer)
    resume_process_id = Column(Integer)


class SyncResume(Base):
    __tablename__ = "sync_resumes"
    id = Column(Integer, primary_key=True, index=True)
    gcp_filepath = Column(String)
    json_output = Column(JSON)
    third_party_output = Column(JSON)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
    company_id = Column(Integer, ForeignKey("companies.id"))
    resume_text = Column(String)
    resume_entities = Column(ARRAY(String))
    hash_value = Column(String)
    author = relationship("Company")


class Scoring(Base):
    __tablename__ = "scoring"
    id = Column(Integer, primary_key=True)
    job_candidate_id = Column(Integer)
    cloud_filepath = Column(String(collation="default"))
    bucket_name = Column(String(collation="default"))
    callback_url = Column(String(collation="default"))
    job_data = Column(String(collation="default"))
    resume_text = Column(String(collation="default"))
    base_resume_summary = Column(String(collation="default"))
    json_resume_summary = Column(String(collation="default"))
    base_answer = Column(String(collation="default"))
    json_answer = Column(JSON)
    taken_time = Column(Numeric)
    ats_response_code = Column(Integer)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
    status = Column(Integer)


class Scoring_server(Base):
    __tablename__ = "scoring_servers"
    id = Column(Integer, primary_key=True)
    token = Column(String(collation="default"))
    availability = Column(Integer)
    status = Column(Integer)
    request_count = Column(Integer)
    success_count = Column(Integer)
    failure_count = Column(Integer)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())
