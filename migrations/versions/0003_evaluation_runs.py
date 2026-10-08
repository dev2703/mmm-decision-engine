"""Registered external evaluation lifecycle and immutable policy evidence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "model_run_id",
            sa.Uuid(),
            sa.ForeignKey("model_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "sensitivity_run_id",
            sa.Uuid(),
            sa.ForeignKey("model_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("evaluator", sa.String(), nullable=False),
        sa.Column("configuration", postgresql.JSONB(), nullable=False),
        sa.Column("artifact_uri", sa.String(), nullable=False, unique=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("summary", postgresql.JSONB()),
        sa.Column("record_hash", sa.String()),
        sa.Column("error_type", sa.String()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')"),
    )
    op.create_index(
        "ix_evaluation_runs_model_run_id", "evaluation_runs", ["model_run_id"]
    )


def downgrade() -> None:
    op.drop_table("evaluation_runs")
