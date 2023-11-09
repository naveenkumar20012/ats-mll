"""add scoring table

Revision ID: 6a873e5881e7
Revises: a6d21395502b
Create Date: 2023-05-29 18:48:38.204704

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6a873e5881e7"
down_revision = None
branch_labels = None


def upgrade() -> None:
    op.create_table(
        "scoring",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("resume_process_id", sa.String(collation="default")),
        sa.Column("resume_object_id", sa.Integer, nullable=False),
        sa.Column("job_id", sa.Integer, nullable=False),
        sa.Column("job_title", sa.String(collation="default")),
        sa.Column("job_summary", sa.String(collation="default")),
        sa.Column("resume_text", sa.String(collation="default")),
        sa.Column("base_resume_summary", sa.String(collation="default")),
        sa.Column("json_resume_summary", sa.String(collation="default")),
        sa.Column("base_answer", sa.String(collation="default")),
        sa.Column("json_answer", sa.JSON()),
        sa.Column(
            "time_created",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("time_updated", sa.TIMESTAMP(timezone=True)),
        sa.PrimaryKeyConstraint("id"),
        schema="public",
    )

    # op.create_index("scoring_id_seq", "scoring", ["id"], unique=False)


def downgrade() -> None:
    op.drop_table("scoring", schema="public")
    op.drop_index("scoring_id_seq", table_name="scoring")
