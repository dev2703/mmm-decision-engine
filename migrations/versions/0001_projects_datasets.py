"""Projects and immutable dataset metadata."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("current_dataset_id", sa.Uuid()),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("artifact_uri", sa.String(), nullable=False, unique=True),
        sa.Column("schema_version", sa.String(), nullable=False),
        sa.Column("date_start", sa.Date(), nullable=False),
        sa.Column("date_end", sa.Date(), nullable=False),
        sa.Column("hash", sa.String(), nullable=False),
        sa.Column("quality_status", sa.String(), nullable=False),
        sa.Column("configuration", postgresql.JSONB(), nullable=False),
        sa.Column("quality", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("project_id", "version"),
        sa.CheckConstraint("version > 0"),
        sa.CheckConstraint("quality_status IN ('RESOLVED', 'WARNING', 'BLOCKER')"),
        sa.CheckConstraint("date_end >= date_start"),
    )
    op.create_index("ix_datasets_project_id", "datasets", ["project_id"])
    op.create_foreign_key(
        "fk_projects_current_dataset",
        "projects",
        "datasets",
        ["current_dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_projects_current_dataset", "projects", type_="foreignkey")
    op.drop_table("datasets")
    op.drop_table("projects")
