"""TICI - Time-series Image Continuum format.

.tíci is an HDF5-based medical imaging file format for storing
time-series 4D CT/MRI data along with AI analysis results
(segmentation masks, IGTV target volumes, motion trajectories).
"""

from .tici_io import TiciWriter, TiciReader, TiciMetadata, TiciIgtvResult, TiciTrajectoryResult
from .converter import from_dicom_dir, from_nifti, from_study

__version__ = "1.0.0"

__all__ = [
    "TiciWriter",
    "TiciReader",
    "TiciMetadata",
    "TiciIgtvResult",
    "TiciTrajectoryResult",
    "from_dicom_dir",
    "from_nifti",
    "from_study",
]
