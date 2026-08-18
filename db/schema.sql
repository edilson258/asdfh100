-- PostgreSQL Schema for H100/RTX 5090 Massive Inference Batch Pipeline

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_dir       TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    tile_size       INTEGER NOT NULL,
    tile_overlap_pct NUMERIC(4,2) NOT NULL,
    conf_threshold  NUMERIC(4,3) NOT NULL,
    iou_threshold   NUMERIC(4,3) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running','completed','aborted')),
    images_total    INTEGER,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID REFERENCES runs(id) ON DELETE SET NULL,
    image_path      TEXT NOT NULL UNIQUE,
    file_hash       TEXT,
    file_size_bytes BIGINT,
    width           INTEGER,
    height          INTEGER,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN
                            ('pending','processing','tiling_done',
                             'inference_done','draw_done',
                             'uploaded','done','failed')),
    tiles_total     INTEGER,
    tiles_done      INTEGER NOT NULL DEFAULT 0,
    tiles_failed    INTEGER NOT NULL DEFAULT 0,
    detections_count INTEGER NOT NULL DEFAULT 0,
    annotated_local_path TEXT,
    gcp_bucket      TEXT,
    gcp_object_path TEXT,
    gcp_url         TEXT,
    uploaded        BOOLEAN NOT NULL DEFAULT FALSE,
    attempts        INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_images_status ON images (status);
CREATE INDEX IF NOT EXISTS idx_images_run_status ON images (run_id, status);
CREATE INDEX IF NOT EXISTS idx_images_updated_at ON images (updated_at);

CREATE TABLE IF NOT EXISTS tiles (
    id              BIGSERIAL PRIMARY KEY,
    image_id        BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tile_index      INTEGER NOT NULL,
    x_offset        INTEGER NOT NULL,
    y_offset        INTEGER NOT NULL,
    tile_width      INTEGER NOT NULL,
    tile_height     INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','done','failed','skipped')),
    detections_count INTEGER NOT NULL DEFAULT 0,
    inference_ms    INTEGER,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (image_id, tile_index)
);

CREATE INDEX IF NOT EXISTS idx_tiles_image_id ON tiles (image_id);
CREATE INDEX IF NOT EXISTS idx_tiles_status ON tiles (status);

CREATE TABLE IF NOT EXISTS detections (
    id              BIGSERIAL PRIMARY KEY,
    image_id        BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    tile_id         BIGINT REFERENCES tiles(id) ON DELETE SET NULL,
    class_name      TEXT NOT NULL,
    confidence      NUMERIC(5,4) NOT NULL,
    bbox_x1         INTEGER NOT NULL,
    bbox_y1         INTEGER NOT NULL,
    bbox_x2         INTEGER NOT NULL,
    bbox_y2         INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections (image_id);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections (class_name);
