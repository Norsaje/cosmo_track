"""initial schema

Revision ID: 9f99468df5c4
Revises: 
Create Date: 2026-09-05 09:06:18.895374
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# autogenerate подставил geoalchemy2.types.Geometry в колонку polygons.geometry,
# но импорт не добавил — без этой строки upgrade падает с NameError.
import geoalchemy2

revision: str = '9f99468df5c4'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # PostGIS в образе postgis/postgis уже установлен, но clean install по README
    # может подниматься и на голом PostgreSQL: без расширения тип geometry
    # не существует и первая же таблица не создастся.
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table('polygons',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=True),
    sa.Column('geometry', geoalchemy2.types.Geometry(geometry_type='MULTIPOLYGON', srid=4326, dimension=2, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry', nullable=False), nullable=False),
    sa.Column('geometry_hash', sa.String(length=64), nullable=False),
    sa.Column('area_ha', sa.Float(), nullable=True),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('crop_type', sa.String(length=120), nullable=True),
    sa.Column('anon_polygon_id', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source IN ('manual', 'fields_world', 'osm', 'worldcereal', 'demo')", name='ck_polygon_source'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_polygons_anon_polygon_id'), 'polygons', ['anon_polygon_id'], unique=False)
    op.create_index('ix_polygons_geometry', 'polygons', ['geometry'], unique=False, postgresql_using='gist')
    op.create_index('ix_polygons_geometry_hash', 'polygons', ['geometry_hash'], unique=False)
    op.create_table('analyses',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('polygon_id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('geometry_hash', sa.String(length=64), nullable=False),
    sa.Column('date_from', sa.Date(), nullable=False),
    sa.Column('date_to', sa.Date(), nullable=False),
    sa.Column('pipeline_version', sa.String(length=32), nullable=False),
    sa.Column('model_version', sa.String(length=120), nullable=True),
    sa.Column('state', sa.String(length=20), nullable=False),
    sa.Column('partial', sa.Boolean(), nullable=False),
    sa.Column('cached', sa.Boolean(), nullable=False),
    sa.Column('data_retrieved_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['polygon_id'], ['polygons.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('geometry_hash', 'date_from', 'date_to', 'pipeline_version', name='uq_analysis_idempotency')
    )
    op.create_table('observations',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('polygon_id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('processing_version', sa.String(length=32), nullable=False),
    sa.Column('ndvi_raw', sa.Float(), nullable=True),
    sa.Column('ndvi_cleaned', sa.Float(), nullable=True),
    sa.Column('ndvi_invalid_flag', sa.Boolean(), nullable=False),
    sa.Column('evi_raw', sa.Float(), nullable=True),
    sa.Column('evi_cleaned', sa.Float(), nullable=True),
    sa.Column('evi_invalid_flag', sa.Boolean(), nullable=False),
    sa.Column('ndwi_raw', sa.Float(), nullable=True),
    sa.Column('ndwi_cleaned', sa.Float(), nullable=True),
    sa.Column('ndwi_invalid_flag', sa.Boolean(), nullable=False),
    sa.Column('temp_c', sa.Float(), nullable=True),
    sa.Column('temp_available', sa.Boolean(), nullable=False),
    sa.Column('precip_mm', sa.Float(), nullable=True),
    sa.Column('precip_available', sa.Boolean(), nullable=False),
    sa.Column('valid_pixel_fraction', sa.Float(), nullable=True),
    sa.Column('pixel_count', sa.Integer(), nullable=True),
    sa.Column('qa_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('asset_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.ForeignKeyConstraint(['polygon_id'], ['polygons.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('polygon_id', 'date', 'source', 'processing_version', name='uq_observation_key')
    )
    op.create_index('ix_observations_polygon_date', 'observations', ['polygon_id', 'date'], unique=False)
    op.create_table('analysis_jobs',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('analysis_id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('state', sa.String(length=20), nullable=False),
    sa.Column('progress', sa.Integer(), nullable=False),
    sa.Column('partial', sa.Boolean(), nullable=False),
    sa.Column('stage_started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('retry_count', sa.Integer(), nullable=False),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('error_message_safe', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('progress BETWEEN 0 AND 100', name='ck_job_progress_range'),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_analysis_jobs_analysis', 'analysis_jobs', ['analysis_id'], unique=False)
    op.create_table('anomaly_events',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('analysis_id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('end_date', sa.Date(), nullable=False),
    sa.Column('severity', sa.String(length=32), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('min_robust_z', sa.Float(), nullable=False),
    sa.Column('negative_area', sa.Float(), nullable=False),
    sa.Column('observed_points', sa.Integer(), nullable=False),
    sa.Column('reconstructed_points', sa.Integer(), nullable=False),
    sa.Column('reason_codes', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('explanation_ru', sa.Text(), nullable=False),
    sa.Column('algorithm_version', sa.String(length=64), nullable=False),
    sa.Column('schema_version', sa.String(length=16), nullable=True),
    sa.Column('duration_days', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_anomaly_events_analysis', 'anomaly_events', ['analysis_id'], unique=False)
    op.create_table('provenance',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('analysis_id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('provider', sa.String(length=64), nullable=False),
    sa.Column('collection_id', sa.String(length=128), nullable=False),
    sa.Column('collection_version', sa.String(length=64), nullable=True),
    sa.Column('item_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('queried_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('crs', sa.String(length=32), nullable=True),
    sa.Column('resolution_m', sa.Float(), nullable=True),
    sa.Column('qa_definition', sa.Text(), nullable=True),
    sa.Column('processing_params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('license_url', sa.String(length=400), nullable=True),
    sa.Column('fingerprint', sa.String(length=64), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('cached', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_provenance_analysis', 'provenance', ['analysis_id'], unique=False)
    op.create_index(op.f('ix_provenance_fingerprint'), 'provenance', ['fingerprint'], unique=False)
    op.create_table('reconstructions',
    sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('analysis_id', sa.UUID(as_uuid=False), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('primary_ndvi_raw', sa.Float(), nullable=True),
    sa.Column('primary_ndvi_reconstructed', sa.Float(), nullable=True),
    sa.Column('ndvi_harmonized', sa.Float(), nullable=True),
    sa.Column('is_observed', sa.Boolean(), nullable=False),
    sa.Column('is_reconstructed', sa.Boolean(), nullable=False),
    sa.Column('lower', sa.Float(), nullable=True),
    sa.Column('upper', sa.Float(), nullable=True),
    sa.Column('selected_source', sa.String(length=32), nullable=True),
    sa.Column('method', sa.String(length=64), nullable=True),
    sa.Column('p_s2', sa.Float(), nullable=True),
    sa.Column('p_landsat', sa.Float(), nullable=True),
    sa.Column('p_modis', sa.Float(), nullable=True),
    sa.Column('p_unknown', sa.Float(), nullable=True),
    sa.Column('left_distance_days', sa.Float(), nullable=True),
    sa.Column('right_distance_days', sa.Float(), nullable=True),
    sa.Column('model_disagreement', sa.Float(), nullable=True),
    sa.Column('fallback_reason', sa.String(length=64), nullable=True),
    sa.Column('context_quality', sa.Float(), nullable=True),
    sa.Column('source_confidence', sa.Float(), nullable=True),
    sa.Column('quality_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('interval_status', sa.String(length=64), nullable=True),
    sa.Column('interval_level', sa.Float(), nullable=True),
    sa.Column('harmonization_status', sa.String(length=64), nullable=True),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('analysis_id', 'date', name='uq_reconstruction_key')
    )
    op.create_index('ix_reconstructions_analysis_date', 'reconstructions', ['analysis_id', 'date'], unique=False)


def downgrade() -> None:
    # Расширение postgis намеренно не удаляется: оно может использоваться
    # не только нашей схемой, и снос по downgrade — потеря чужих данных.
    op.drop_index('ix_reconstructions_analysis_date', table_name='reconstructions')
    op.drop_table('reconstructions')
    op.drop_index(op.f('ix_provenance_fingerprint'), table_name='provenance')
    op.drop_index('ix_provenance_analysis', table_name='provenance')
    op.drop_table('provenance')
    op.drop_index('ix_anomaly_events_analysis', table_name='anomaly_events')
    op.drop_table('anomaly_events')
    op.drop_index('ix_analysis_jobs_analysis', table_name='analysis_jobs')
    op.drop_table('analysis_jobs')
    op.drop_index('ix_observations_polygon_date', table_name='observations')
    op.drop_table('observations')
    op.drop_table('analyses')
    op.drop_index('ix_polygons_geometry_hash', table_name='polygons')
    op.drop_index('ix_polygons_geometry', table_name='polygons', postgresql_using='gist')
    op.drop_index(op.f('ix_polygons_anon_polygon_id'), table_name='polygons')
    op.drop_table('polygons')
