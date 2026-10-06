"""Migrations use the same explicit database setting as the application."""

from alembic import context
from sqlalchemy import create_engine, pool

from decisionguard.api.database import Base
from decisionguard.config import Settings

configuration = context.config
url = (
    configuration.attributes.get("database_url")
    or Settings.from_environment().database_url
)
if context.is_offline_mode():
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
