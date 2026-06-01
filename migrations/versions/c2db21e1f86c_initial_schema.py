"""initial_schema

Revision ID: c2db21e1f86c
Revises:
Create Date: 2026-06-01 19:42:35.880008

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = 'c2db21e1f86c'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'agentrunlog',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('run_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('triggered_by', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('model', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('pii_detected', sa.Boolean(), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False),
        sa.Column('completion_tokens', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.Column('tool_calls', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('outcome', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('result_summary', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('ip_address', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_agentrunlog_agent_id', 'agentrunlog', ['agent_id'], unique=False)
    op.create_index('ix_agentrunlog_run_id', 'agentrunlog', ['run_id'], unique=False)
    op.create_index('ix_agentrunlog_triggered_by', 'agentrunlog', ['triggered_by'], unique=False)

    op.create_table(
        'workflow',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', 'timeout',
                                    name='workflowstatus'), nullable=False),
        sa.Column('run_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('result_json', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('error', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_workflow_agent_id', 'workflow', ['agent_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_workflow_agent_id', table_name='workflow')
    op.drop_table('workflow')
    op.drop_index('ix_agentrunlog_triggered_by', table_name='agentrunlog')
    op.drop_index('ix_agentrunlog_run_id', table_name='agentrunlog')
    op.drop_index('ix_agentrunlog_agent_id', table_name='agentrunlog')
    op.drop_table('agentrunlog')
