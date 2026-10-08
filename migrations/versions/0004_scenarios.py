"""Budget scenarios bound to verified evaluation evidence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scenarios",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "model_run_id",
            sa.Uuid(),
            sa.ForeignKey("model_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_run_id",
            sa.Uuid(),
            sa.ForeignKey("evaluation_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("evaluation_record_hash", sa.String(), nullable=False),
        sa.Column("budget", sa.Float(), nullable=False),
        sa.Column("constraints", postgresql.JSONB(), nullable=False),
        sa.Column("effective_bounds", postgresql.JSONB(), nullable=False),
        sa.Column("release", postgresql.JSONB(), nullable=False),
        sa.Column("horizon_weeks", sa.Integer(), nullable=False),
        sa.Column("risk_policy", sa.String(), nullable=False),
        sa.Column("code_version", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("budget > 0"),
        sa.CheckConstraint("horizon_weeks BETWEEN 1 AND 52"),
        sa.CheckConstraint("risk_policy IN ('expected', 'conservative')"),
    )
    op.create_index("ix_scenarios_model_run_id", "scenarios", ["model_run_id"])


def downgrade() -> None:
    op.drop_table("scenarios")
