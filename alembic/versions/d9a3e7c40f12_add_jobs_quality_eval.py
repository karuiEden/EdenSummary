"""add jobs.quality_eval column for Tier-2 faithfulness eval

Revision ID: d9a3e7c40f12
Revises: c8f5a2b1d3e6
Create Date: 2026-06-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd9a3e7c40f12'
down_revision: Union[str, Sequence[str], None] = 'c8f5a2b1d3e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'jobs',
        sa.Column('quality_eval', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('jobs', 'quality_eval')
