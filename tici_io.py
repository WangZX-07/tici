"""TICI core I/O: TiciWriter and TiciReader for .tiCi HDF5 files.

Magic number: \\x89TICI\\r\\n\\x1a\\n
Format version: 1.0 (stored as /format_version attribute)

Hierarchical HDF5 layout:
  /patient/         - patient metadata (attributes)
  /study/           - study metadata (attributes)
  /phases/phase_N/  - per-phase images (float32) and masks (uint8)
  /igtv/            - IGTV mask, mesh, volume (optional)
  /trajectory/      - trajectory data (optional)
  /provenance/      - source trace (attributes)
"""

import os
import json
import h5py
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


MAGIC = b"\x89TICI\r\n\x1a\n"
FORMAT_VERSION = "1.0"
CONVERTER_VERSION = "LungTic-1.0.0"


# ---------------------------------------------------------------------------
# Dataclasses for return types
# ---------------------------------------------------------------------------

@dataclass
class TiciPatientInfo:
    patient_id: str = ""
    patient_name: str = ""
    patient_sex: str = ""
    patient_age: str = ""


@dataclass
class TiciStudyInfo:
    study_uid: str = ""
    study_date: str = ""
    modality: str = "CT"
    phase_count: int = 10
    rows: int = 512
    cols: int = 512
    slice_count: int = 0
    pixel_spacing: list = field(default_factory=lambda: [1.0, 1.0])
    slice_thickness: float = 1.0
    slice_locations: list = field(default_factory=list)


@dataclass
class TiciProvenance:
    source_type: str = "dicom"
    source_files_count: int = 0
    converter_version: str = CONVERTER_VERSION
    conversion_date: str = ""
    model_version: str = ""


@dataclass
class TiciMetadata:
    patient: TiciPatientInfo = field(default_factory=TiciPatientInfo)
    study: TiciStudyInfo = field(default_factory=TiciStudyInfo)
    provenance: TiciProvenance = field(default_factory=TiciProvenance)


@dataclass
class TiciIgtvResult:
    volume_cc: float = 0.0
    method: str = "union"
    mask: Optional[np.ndarray] = None
    mesh: str = ""


@dataclass
class TiciTrajectoryResult:
    prediction_model: str = ""
    centroids: Optional[np.ndarray] = None   # (N, 3)
    observed: Optional[np.ndarray] = None    # (7, 3)
    predicted: Optional[np.ndarray] = None   # (3, 3)


# ---------------------------------------------------------------------------
# TiciWriter -- progressive .tiCi writer
# ---------------------------------------------------------------------------

class TiciWriter:
    """Context-managed progressive writer for .tiCi files.

    Modes:
      - "w" : create new file (overwrites existing)
      - "a" : open existing file and append/update data

    Usage::

        # Create new
        with TiciWriter("output.tici", "w") as w:
            w.write_metadata(patient_dict, study_dict)
            w.write_phase_images(0, images)

        # Append to existing
        with TiciWriter("output.tici", "a") as w:
            w.append_phase_masks(0, masks)
            w.write_igtv(itv_mask, mesh, 12.5, "union")
    """

    def __init__(self, path: str, mode: str = "w"):
        if mode not in ("w", "a"):
            raise ValueError("mode must be 'w' or 'a'")
        self.path = path
        self._mode = mode
        self._file: Optional[h5py.File] = None

    def __enter__(self) -> "TiciWriter":
        if self._mode == "a":
            self._file = h5py.File(self.path, "r+")
        else:
            self._file = h5py.File(self.path, "w")
            self._file.attrs["MAGIC"] = np.bytes_(MAGIC)
            self._file.attrs["format_version"] = FORMAT_VERSION
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    # -- Metadata --

    def write_metadata(self, patient: dict, study: dict) -> None:
        """Write patient and study metadata groups with attributes."""
        self._require_open()

        pg = self._file.require_group("patient")
        for k in ("patient_id", "patient_name", "patient_sex", "patient_age"):
            val = patient.get(k, "")
            pg.attrs[k] = str(val) if val is not None else ""

        sg = self._file.require_group("study")
        for k in ("study_uid", "study_date", "modality"):
            val = study.get(k, "")
            sg.attrs[k] = str(val) if val is not None else ""

        sg.attrs["phase_count"] = int(study.get("phase_count", 10))
        sg.attrs["rows"] = int(study.get("rows", 512))
        sg.attrs["cols"] = int(study.get("cols", 512))
        sg.attrs["slice_count"] = int(study.get("slice_count", 0))

        ps = study.get("pixel_spacing", [1.0, 1.0])
        sg.attrs["pixel_spacing"] = np.array(ps, dtype=np.float32)

        sg.attrs["slice_thickness"] = float(study.get("slice_thickness", 1.0))

        slocs = study.get("slice_locations", [])
        if slocs:
            sg.attrs["slice_locations"] = np.array(slocs, dtype=np.float32)

    # -- Phase images --

    def write_phase_images(
        self,
        phase_idx: int,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        """Write images (Z,H,W float32) and optional masks (Z,H,W uint8)."""
        self._require_open()

        phases_grp = self._file.require_group("phases")
        phase_grp = phases_grp.require_group(f"phase_{phase_idx}")

        if "images" in phase_grp:
            del phase_grp["images"]
        phase_grp.create_dataset(
            "images",
            data=images.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )

        if masks is not None:
            if "masks" in phase_grp:
                del phase_grp["masks"]
            phase_grp.create_dataset(
                "masks",
                data=masks.astype(np.uint8),
                compression="lzf",
            )

    def append_phase_masks(self, phase_idx: int, masks: np.ndarray) -> None:
        """Append or update masks for a specific phase without touching images."""
        self._require_open()

        phases_grp = self._file.require_group("phases")
        phase_grp = phases_grp.require_group(f"phase_{phase_idx}")

        if "masks" in phase_grp:
            del phase_grp["masks"]
        phase_grp.create_dataset(
            "masks",
            data=masks.astype(np.uint8),
            compression="lzf",
        )

    # -- IGTV --

    def write_igtv(
        self,
        mask: np.ndarray,
        mesh_text: str,
        volume_cc: float,
        method: str = "union",
    ) -> None:
        """Write IGTV reconstruction result."""
        self._require_open()

        ig_grp = self._file.require_group("igtv")
        ig_grp.attrs["volume_cc"] = float(volume_cc)
        ig_grp.attrs["method"] = str(method)

        if "mask" in ig_grp:
            del ig_grp["mask"]
        ig_grp.create_dataset(
            "mask",
            data=mask.astype(np.uint8),
            compression="lzf",
        )

        if "mesh" in ig_grp:
            del ig_grp["mesh"]
        ig_grp.create_dataset("mesh", data=np.bytes_(mesh_text))

    # -- Trajectory --

    def write_trajectory(
        self,
        centroids: np.ndarray,
        observed: np.ndarray,
        predicted: np.ndarray,
        model_name: str = "",
    ) -> None:
        """Write tumor motion trajectory data."""
        self._require_open()

        tr_grp = self._file.require_group("trajectory")
        tr_grp.attrs["prediction_model"] = str(model_name)

        for name, arr in [
            ("centroids", centroids),
            ("observed", observed),
            ("predicted", predicted),
        ]:
            if name in tr_grp:
                del tr_grp[name]
            if arr is not None:
                tr_grp.create_dataset(name, data=arr.astype(np.float32))

    # -- Provenance --

    def write_provenance(self, info: dict) -> None:
        """Write provenance / source trace metadata."""
        self._require_open()

        pv = self._file.require_group("provenance")
        pv.attrs["source_type"] = str(info.get("source_type", "dicom"))
        pv.attrs["source_files_count"] = int(info.get("source_files_count", 0))
        pv.attrs["converter_version"] = str(info.get("converter_version", CONVERTER_VERSION))
        pv.attrs["conversion_date"] = str(info.get("conversion_date", datetime.now().isoformat()))
        pv.attrs["model_version"] = str(info.get("model_version", ""))

    # -- helpers --

    def _require_open(self):
        if self._file is None:
            raise RuntimeError("TiciWriter is not open. Use `with TiciWriter(...)` context.")


# ---------------------------------------------------------------------------
# TiciReader -- on-demand .tiCi reader
# ---------------------------------------------------------------------------

class TiciReader:
    """Context-managed on-demand reader for .tiCi files.

    Usage::

        with TiciReader("input.tici") as r:
            meta = r.get_metadata()
            phase3 = r.get_phase_images(3)
            masks3 = r.get_phase_masks(3)
            igtv = r.get_igtv()
            traj = r.get_trajectory()
    """

    def __init__(self, path: str):
        self.path = path
        self._file: Optional[h5py.File] = None

    def __enter__(self) -> "TiciReader":
        self._file = h5py.File(self.path, "r")

        # Validate magic number
        magic = self._file.attrs.get("MAGIC", b"")
        if isinstance(magic, np.ndarray):
            magic = bytes(magic)
        if magic != MAGIC:
            self._file.close()
            raise ValueError(
                f"Not a valid .tici file (bad magic). "
                f"Expected {MAGIC!r}, got {magic!r}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    # -- Metadata --

    def get_metadata(self) -> TiciMetadata:
        self._require_open()
        meta = TiciMetadata()

        if "patient" in self._file:
            pg = self._file["patient"]
            meta.patient = TiciPatientInfo(
                patient_id=str(pg.attrs.get("patient_id", "")),
                patient_name=str(pg.attrs.get("patient_name", "")),
                patient_sex=str(pg.attrs.get("patient_sex", "")),
                patient_age=str(pg.attrs.get("patient_age", "")),
            )

        if "study" in self._file:
            sg = self._file["study"]
            ps = sg.attrs.get("pixel_spacing", [1.0, 1.0])
            if hasattr(ps, "tolist"):
                ps = ps.tolist()
            slocs = sg.attrs.get("slice_locations", [])
            if hasattr(slocs, "tolist"):
                slocs = slocs.tolist()

            meta.study = TiciStudyInfo(
                study_uid=str(sg.attrs.get("study_uid", "")),
                study_date=str(sg.attrs.get("study_date", "")),
                modality=str(sg.attrs.get("modality", "CT")),
                phase_count=int(sg.attrs.get("phase_count", 10)),
                rows=int(sg.attrs.get("rows", 512)),
                cols=int(sg.attrs.get("cols", 512)),
                slice_count=int(sg.attrs.get("slice_count", 0)),
                pixel_spacing=ps,
                slice_thickness=float(sg.attrs.get("slice_thickness", 1.0)),
                slice_locations=slocs,
            )

        if "provenance" in self._file:
            pv = self._file["provenance"]
            meta.provenance = TiciProvenance(
                source_type=str(pv.attrs.get("source_type", "dicom")),
                source_files_count=int(pv.attrs.get("source_files_count", 0)),
                converter_version=str(pv.attrs.get("converter_version", "")),
                conversion_date=str(pv.attrs.get("conversion_date", "")),
                model_version=str(pv.attrs.get("model_version", "")),
            )

        return meta

    @property
    def phase_count(self) -> int:
        self._require_open()
        if "phases" not in self._file:
            return 0
        return len(self._file["phases"])

    # -- Phase data --

    def get_phase_images(self, phase_idx: int) -> np.ndarray:
        """Load images for a single phase. Returns (Z, H, W) float32."""
        self._require_open()
        path = f"phases/phase_{phase_idx}/images"
        if path not in self._file:
            raise KeyError(f"Phase {phase_idx} images not found")
        return self._file[path][:].astype(np.float32)

    def get_phase_slice(self, phase_idx: int, slice_idx: int) -> np.ndarray:
        """Load one image slice without materializing the full phase volume."""
        self._require_open()
        path = f"phases/phase_{phase_idx}/images"
        if path not in self._file:
            raise KeyError(f"Phase {phase_idx} images not found")
        dataset = self._file[path]
        if slice_idx < 0 or slice_idx >= dataset.shape[0]:
            raise IndexError(
                f"Slice {slice_idx} out of range for phase {phase_idx} "
                f"(0-{dataset.shape[0] - 1})"
            )
        return dataset[slice_idx].astype(np.float32)

    def get_phase_shape(self, phase_idx: int) -> tuple[int, ...]:
        """Return a phase image shape without loading its pixel data."""
        self._require_open()
        path = f"phases/phase_{phase_idx}/images"
        if path not in self._file:
            raise KeyError(f"Phase {phase_idx} images not found")
        return tuple(self._file[path].shape)

    def get_phase_masks(self, phase_idx: int) -> Optional[np.ndarray]:
        """Load masks for a single phase. Returns (Z, H, W) uint8 or None."""
        self._require_open()
        path = f"phases/phase_{phase_idx}/masks"
        if path not in self._file:
            return None
        return self._file[path][:].astype(np.uint8)

    def get_phase_mask_slice(self, phase_idx: int, slice_idx: int) -> Optional[np.ndarray]:
        """Load one mask slice, or return None when the phase has no masks."""
        self._require_open()
        path = f"phases/phase_{phase_idx}/masks"
        if path not in self._file:
            return None
        dataset = self._file[path]
        if slice_idx < 0 or slice_idx >= dataset.shape[0]:
            raise IndexError(
                f"Mask slice {slice_idx} out of range for phase {phase_idx} "
                f"(0-{dataset.shape[0] - 1})"
            )
        return dataset[slice_idx].astype(np.uint8)

    def get_all_phases(self) -> list[np.ndarray]:
        """Load all phase images at once. Use with care for large datasets."""
        self._require_open()
        n = self.phase_count
        return [self.get_phase_images(i) for i in range(n)]

    # -- IGTV --

    def get_igtv(self) -> Optional[TiciIgtvResult]:
        """Load IGTV result if present."""
        self._require_open()
        if "igtv" not in self._file:
            return None

        ig = self._file["igtv"]
        result = TiciIgtvResult(
            volume_cc=float(ig.attrs.get("volume_cc", 0.0)),
            method=str(ig.attrs.get("method", "union")),
        )
        if "mask" in ig:
            result.mask = ig["mask"][:].astype(np.uint8)
        if "mesh" in ig:
            mesh_data = ig["mesh"][()]
            if isinstance(mesh_data, (bytes, np.bytes_)):
                result.mesh = mesh_data.decode("utf-8", errors="replace")
            else:
                result.mesh = str(mesh_data)

        return result

    # -- Trajectory --

    def get_trajectory(self) -> Optional[TiciTrajectoryResult]:
        """Load trajectory prediction if present."""
        self._require_open()
        if "trajectory" not in self._file:
            return None

        tr = self._file["trajectory"]
        result = TiciTrajectoryResult(
            prediction_model=str(tr.attrs.get("prediction_model", "")),
        )
        for name in ("centroids", "observed", "predicted"):
            if name in tr:
                setattr(result, name, tr[name][:].astype(np.float32))
        return result

    # -- Format version --

    @property
    def format_version(self) -> str:
        self._require_open()
        return str(self._file.attrs.get("format_version", "unknown"))

    @property
    def has_igtv(self) -> bool:
        self._require_open()
        return "igtv" in self._file

    @property
    def has_trajectory(self) -> bool:
        self._require_open()
        return "trajectory" in self._file

    # -- Phase metadata for previews --

    def get_phase_meta(self) -> list[dict]:
        """Return metadata for each phase (has_images, has_masks, slice_count)."""
        self._require_open()
        result = []
        if "phases" not in self._file:
            return result
        for name in sorted(self._file["phases"].keys(), key=lambda n: int(n.split("_")[1])):
            grp = self._file["phases"][name]
            info = {
                "phase": name,
                "has_images": "images" in grp,
                "has_masks": "masks" in grp,
            }
            if "images" in grp:
                info["shape"] = grp["images"].shape
            result.append(info)
        return result

    def _require_open(self):
        if self._file is None:
            raise RuntimeError("TiciReader is not open. Use `with TiciReader(...)` context.")
