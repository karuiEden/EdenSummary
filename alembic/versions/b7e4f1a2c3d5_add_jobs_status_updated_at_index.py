"""add jobs (status, updated_at) index for reaper

Revision ID: b7e4f1a2c3d5
Revises: 3deba96a8578
Create Date: 2026-06-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7e4f1a2c3d5'
down_revision: Union[str, Sequence[str], None] = '3deba96a8578'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index('ix_jobs_status_updated_at', 'jobs', ['status', 'updated_at'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_jobs_status_updated_at', table_name='jobs')
