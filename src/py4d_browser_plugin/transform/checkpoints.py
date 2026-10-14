"""RAM checkpoint records and datacube snapshot helpers."""

from dataclasses import dataclass


CHECKPOINT_WARNING_BYTES = 1024**3
CHECKPOINT_TOTAL_WARNING_BYTES = 4 * 1024**3


@dataclass
class DatacubeCheckpoint:
    checkpoint_id: int
    name: str
    created_at: object
    shape: tuple
    dtype: str
    estimated_bytes: int
    datacube: object
    metadata: dict


def copy_datacube(datacube):
    if hasattr(datacube, "copy"):
        copied = datacube.copy()
    else:
        copied = type(datacube)(datacube.data.copy())
        if hasattr(datacube, "calibration"):
            copied.calibration = datacube.calibration.copy()

    # py4DSTEM's DataCube.copy() rebuilds a fresh Root, and constructing a
    # DataCube with an explicit calibration object names that root
    # "py4DSTEM_root" instead of preserving the original's name. That
    # silently breaks py4D-browser's calibration reload for any file whose
    # root name it hardcodes (e.g. "datacube_root"), so keep the original
    # root's name on the copy.
    original_root = getattr(datacube, "root", None)
    copied_root = getattr(copied, "root", None)
    if original_root is not None and copied_root is not None:
        copied_root.name = original_root.name

    return copied


def datacube_nbytes(datacube):
    return int(getattr(datacube.data, "nbytes", 0))


def format_bytes(nbytes):
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024


def checkpoint_metadata(datacube):
    return {
        "shape": tuple(datacube.data.shape),
        "dtype": str(datacube.data.dtype),
        "estimated_bytes": datacube_nbytes(datacube),
    }
