"""add column ats_response_code

Revision ID: 09a87061d288
Revises: b7cfd7364d81
Create Date: 2023-05-11 13:24:32.310090

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "09a87061d288"
down_revision = "b7cfd7364d81"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "resumes",
        sa.Column(
            "ats_response_code",
            sa.Integer(),
            nullable=True,
            server_default=None,
        ),
    )


def downgrade() -> None:
    op.drop_column("resumes", "ats_response_code")
