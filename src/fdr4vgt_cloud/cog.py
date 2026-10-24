"""COG helpers used by the release pipeline.

The public release uses plain Cloud Optimized GeoTIFFs, not TACOTIFF. COGs
are validated with ``rio_cogeo.cog_validate`` and written with
``rio_cogeo.cog_translate``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import rasterio as rio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

__all__ = ["to_cog", "validate_cog", "image_profile", "mask_profile"]


def image_profile() -> dict[str, Any]:
    """COG profile for a 4-band TOA reflectance image."""
    profile = cog_profiles.get("zstd")
    profile.update(
        {
            "blockxsize": 512,
            "blockysize": 512,
            "predictor": 3,
            "overview_resampling": "average",
            "nodata": 0,
        }
    )
    return profile


def mask_profile() -> dict[str, Any]:
    """COG profile for a single-band uint8 mask."""
    profile = cog_profiles.get("zstd")
    profile.update(
        {
            "blockxsize": 512,
            "blockysize": 512,
            "predictor": 2,
            "overview_resampling": "nearest",
            "nodata": 255,
        }
    )
    return profile


def to_cog(
    src_path: str | pathlib.Path,
    dst_path: str | pathlib.Path,
    profile: dict[str, Any] | None = None,
    overview_level: int | None = None,
) -> pathlib.Path:
    """Convert a GeoTIFF to a validated COG.

    Parameters
    ----------
    src_path:
        Source raster path.
    dst_path:
        Destination path for the COG.
    profile:
        COG profile (see ``image_profile`` or ``mask_profile``).
    overview_level:
        Maximum overview level. ``None`` lets ``rio_cogeo`` decide.

    Returns
    -------
    Path to the written COG.
    """
    src_path = pathlib.Path(src_path)
    dst_path = pathlib.Path(dst_path)

    if profile is None:
        with rio.open(src_path) as src:
            profile = image_profile() if src.count > 1 else mask_profile()

    config = {"GDAL_TIFF_INTERNAL_MASK": True, "GDAL_NUM_THREADS": "ALL_CPUS"}
    cog_translate(
        src_path,
        dst_path,
        profile,
        overview_level=overview_level,
        config=config,
        quiet=True,
    )
    return dst_path


def validate_cog(path: str | pathlib.Path, strict: bool = True) -> tuple[bool, list[str]]:
    """Validate a COG and return ``(is_valid, warnings_and_errors)``."""
    from rio_cogeo.cogeo import cog_validate

    is_valid, errors, warnings = cog_validate(path, strict=strict)
    return is_valid, errors + warnings


def ensure_crs_4326(path: str | pathlib.Path) -> bool:
    """Return True if the raster reports EPSG:4326."""
    with rio.open(path) as src:
        crs = src.crs
        if crs is None:
            return False
        return crs.to_epsg() == 4326
