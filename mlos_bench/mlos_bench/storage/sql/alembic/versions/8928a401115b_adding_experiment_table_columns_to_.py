#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
"""Adding Experiment table columns to support mlos_benchd service - See #732

Revision ID: 8928a401115b
Revises: f83fb8ae7fc4
Create Date: 2025-01-14 17:06:36.181503+00:00

"""
# pylint: disable=invalid-name
# pylint: disable=no-member

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8928a401115b"
down_revision: str | None = "f83fb8ae7fc4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """The schema upgrade script for this revision."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("experiment", sa.Column("ts_start", sa.DateTime(), nullable=True))
    op.add_column("experiment", sa.Column("ts_end", sa.DateTime(), nullable=True))
    op.add_column("experiment", sa.Column("status", sa.String(length=16), nullable=True))
    op.add_column(
        "experiment",
        sa.Column(
            "driver_name",
            sa.String(length=40),
            nullable=True,
            comment="Driver Host/Container Name",
        ),
    )
    op.add_column(
        "experiment",
        sa.Column("driver_pid", sa.Integer(), nullable=True, comment="Driver Process ID"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """The schema downgrade script for this revision."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("experiment", "driver_pid")
    op.drop_column("experiment", "driver_name")
    op.drop_column("experiment", "status")
    op.drop_column("experiment", "ts_end")
    op.drop_column("experiment", "ts_start")
    # ### end Alembic commands ###
