"""Single source of truth for the TACO dataset schema.

This module only declares column names, PyArrow types and English
descriptions. The pipeline modules import it so the schema stays consistent
between ``build_taco.py`` and the verification tests.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["LEVEL0_SCHEMA", "LEVEL0_COLUMNS"]


LEVEL0_SCHEMA = pa.schema(
    [
        pa.field(
            "id",
            pa.string(),
            metadata={"description": "Scene identifier, equal to the folder name."},
        ),
        pa.field(
            "stac:crs",
            pa.string(),
            metadata={"description": "Coordinate reference system of the image, EPSG:4326."},
        ),
        pa.field(
            "stac:geotransform",
            pa.string(),
            metadata={"description": "Affine transform as a JSON array."},
        ),
        pa.field(
            "stac:tensor_shape",
            pa.string(),
            metadata={"description": "Band, height and width as a JSON array."},
        ),
        pa.field(
            "stac:centroid",
            pa.string(),
            metadata={"description": "Centroid longitude and latitude."},
        ),
        pa.field(
            "stac:time_start",
            pa.int64(),
            metadata={"description": "Segment start time in microseconds since epoch (UTC)."},
        ),
        pa.field(
            "stac:time_end",
            pa.int64(),
            metadata={"description": "Segment end time in microseconds since epoch (UTC)."},
        ),
        pa.field(
            "meta:time_source",
            pa.string(),
            metadata={"description": "Source of the timestamp: product_id or spot_log."},
        ),
        pa.field(
            "scene:sensor",
            pa.string(),
            metadata={"description": "Sensor family: PROBA-V, SPOT-VGT1 or SPOT-VGT2."},
        ),
        pa.field(
            "scene:platform",
            pa.string(),
            metadata={"description": "Platform: PROBA-V, SPOT-4 or SPOT-5."},
        ),
        pa.field(
            "scene:collection",
            pa.string(),
            metadata={"description": "Product collection: V101 or V003."},
        ),
        pa.field(
            "scene:camera",
            pa.int8(),
            metadata={"description": "Proba-V camera (1 left, 2 center, 3 right); null for SPOT."},
        ),
        pa.field(
            "scene:nominal_resolution_m",
            pa.int16(),
            metadata={"description": "Nominal spatial resolution in metres: 1000 or 333."},
        ),
        pa.field(
            "scene:gsd_deg",
            pa.float64(),
            metadata={
                "description": "Ground sample distance in decimal degrees from the transform."
            },
        ),
        pa.field(
            "scene:valid_fraction",
            pa.float32(),
            metadata={"description": "Fraction of non-nodata image pixels."},
        ),
        pa.field(
            "scene:orbit_number",
            pa.int32(),
            metadata={"description": "SPOT orbit number from the product LOG; null otherwise."},
        ),
        pa.field(
            "scene:geom_quality_flag",
            pa.string(),
            metadata={"description": "SPOT geometric quality flag from the LOG; null otherwise."},
        ),
        pa.field(
            "annotation:type",
            pa.string(),
            metadata={"description": "Annotation type: pixel or point."},
        ),
        pa.field(
            "annotation:labeled_fraction",
            pa.float32(),
            metadata={"description": "Fraction of labelled pixels within the valid area."},
        ),
        pa.field(
            "annotation:cloud_fraction",
            pa.float32(),
            metadata={"description": "Cloud fraction over labelled pixels."},
        ),
        pa.field(
            "annotation:class_counts",
            pa.string(),
            metadata={"description": "JSON counts of clear, cloud and nodata pixels."},
        ),
        pa.field(
            "annotation:n_points",
            pa.int32(),
            metadata={"description": "Number of point annotations; 0 for pixel scenes."},
        ),
        pa.field(
            "split:role",
            pa.string(),
            metadata={
                "description": "Role in this release: finetune_train, finetune_val, test_pixel, heldout_unused, test_points."
            },
        ),
        pa.field(
            "split:paper",
            pa.string(),
            metadata={"description": "Role used in the paper: train, val, test or none."},
        ),
        pa.field(
            "split:clean",
            pa.string(),
            metadata={"description": "Role in the E2 cleaned split: train, val, test or none."},
        ),
        pa.field(
            "qa:weak_db_split",
            pa.string(),
            metadata={
                "description": "Weak pretraining DB split from weak_clean.csv: train, val, test or none."
            },
        ),
        pa.field(
            "qa:in_original_test89",
            pa.bool_(),
            metadata={
                "description": "True if the scene was part of the original 89-scene test table."
            },
        ),
        pa.field(
            "qa:mask_variant_differs",
            pa.bool_(),
            metadata={"description": "True for the SPOT scene whose mask has a known variant."},
        ),
        pa.field(
            "qa:product_leak",
            pa.bool_(),
            metadata={
                "description": "True if another crop of the same product appears in a different split."
            },
        ),
        pa.field(
            "qa:source_image_md5",
            pa.string(),
            metadata={"description": "MD5 of the source image file."},
        ),
        pa.field(
            "qa:source_mask_md5",
            pa.string(),
            metadata={"description": "MD5 of the source human_mask.tif file."},
        ),
    ]
)


LEVEL0_COLUMNS = [field.name for field in LEVEL0_SCHEMA]
