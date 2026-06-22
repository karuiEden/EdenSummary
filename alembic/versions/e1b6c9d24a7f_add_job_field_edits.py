"""add job_field_edits table for edit-calibration labels

Revision ID: e1b6c9d24a7f
Revises: d9a3e7c40f12
Create Date: 2026-06-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e1b6c9d24a7f'
down_revision: Union[str, Sequence[str], None] = 'd9a3e7c40f12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'job_field_edits',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('job_id', sa.String(), nullable=False),
        sa.Column('field', sa.String(), nullable=False),
        sa.Column('edited', sa.Boolean(), nullable=False),
        sa.Column('original', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('corrected', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', 'field', name='uq_job_field_edits_job_field'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('job_field_edits')
