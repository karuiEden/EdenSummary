"""add jobs.summq_eval column for SummQ QA-consistency check

Revision ID: f2c7d019b3e8
Revises: e1b6c9d24a7f
Create Date: 2026-06-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f2c7d019b3e8'
down_revision: Union[str, Sequence[str], None] = 'e1b6c9d24a7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'jobs',
        sa.Column('summq_eval', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('jobs', 'summq_eval')
