"""Tests for .tiCi format I/O and conversion."""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add parent to path so we can import tici package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tici.tici_io import (
    TiciWriter, TiciReader, TiciMetadata, TiciIgtvResult, TiciTrajectoryResult,
    MAGIC, FORMAT_VERSION,
)


class TestTiciIO(unittest.TestCase):
    """Core I/O tests for .tiCi format."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.tmpdir, "test.tici")

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- Basic create/read --

    def test_create_empty_and_read_metadata(self):
        """Write empty file and verify magic + format version."""
        patient = {"patient_id": "P001", "patient_name": "Test"}
        study = {"study_uid": "1.2.3", "phase_count": 10, "rows": 256, "cols": 256}

        with TiciWriter(self.test_path) as w:
            w.write_metadata(patient, study)

        self.assertTrue(os.path.exists(self.test_path))
        self.assertGreater(os.path.getsize(self.test_path), 0)

        with TiciReader(self.test_path) as r:
            self.assertEqual(r.format_version, FORMAT_VERSION)
            meta = r.get_metadata()
            self.assertEqual(meta.patient.patient_id, "P001")
            self.assertEqual(meta.study.phase_count, 10)
            self.assertEqual(meta.study.rows, 256)

    def test_magic_number_validation(self):
        """Ensure bad magic raises ValueError."""
        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {})

        # Corrupt the magic manually
        import h5py
        with h5py.File(self.test_path, "r+") as f:
            f.attrs["MAGIC"] = np.bytes_(b"GARBAGE")

        with self.assertRaises(ValueError) as ctx:
            TiciReader(self.test_path).__enter__()
        self.assertIn("bad magic", str(ctx.exception))

    # -- Images round-trip --

    def test_write_and_read_phase_images(self):
        """Write phase images and verify round-trip."""
        patient = {"patient_id": "P002"}
        study = {"study_uid": "1.2.4", "phase_count": 3}

        # Synthetic data: 3 phases, each 5 slices of 64x64
        original = {}
        with TiciWriter(self.test_path) as w:
            w.write_metadata(patient, study)
            for p in range(3):
                vol = np.random.rand(5, 64, 64).astype(np.float32)
                original[p] = vol
                w.write_phase_images(p, vol)

        with TiciReader(self.test_path) as r:
            self.assertEqual(r.phase_count, 3)
            for p in range(3):
                loaded = r.get_phase_images(p)
                self.assertTrue(np.allclose(original[p], loaded, atol=1e-6))
            all_phases = r.get_all_phases()
            self.assertEqual(len(all_phases), 3)

    def test_write_and_read_masks(self):
        """Write images + masks and verify."""
        patient = {"patient_id": "P003"}
        study = {"study_uid": "1.2.5", "phase_count": 1}
        vol = np.random.rand(5, 32, 32).astype(np.float32)
        mask = (vol > 0.5).astype(np.uint8)

        with TiciWriter(self.test_path) as w:
            w.write_metadata(patient, study)
            w.write_phase_images(0, vol, masks=mask)

        with TiciReader(self.test_path) as r:
            loaded_mask = r.get_phase_masks(0)
            self.assertIsNotNone(loaded_mask)
            self.assertTrue(np.array_equal(mask, loaded_mask))
            # Phase without masks returns None
            self.assertIsNone(r.get_phase_masks(99))

    # -- IGTV --

    def test_write_and_read_igtv(self):
        """Write IGTV result and read back."""
        patient = {"patient_id": "P004"}
        study = {"study_uid": "1.2.6", "phase_count": 1}
        itv_mask = np.ones((10, 32, 32), dtype=np.uint8)
        mesh_text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"

        with TiciWriter(self.test_path) as w:
            w.write_metadata(patient, study)
            w.write_phase_images(0, np.random.rand(10, 32, 32).astype(np.float32))
            w.write_igtv(itv_mask, mesh_text, volume_cc=12.5, method="union")

        with TiciReader(self.test_path) as r:
            igtv = r.get_igtv()
            self.assertIsNotNone(igtv)
            self.assertEqual(igtv.volume_cc, 12.5)
            self.assertEqual(igtv.method, "union")
            self.assertTrue(np.array_equal(itv_mask, igtv.mask))
            self.assertEqual(igtv.mesh, mesh_text)

    def test_read_igtv_when_missing(self):
        """Reader gracefully returns None when IGTV not present."""
        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {})

        with TiciReader(self.test_path) as r:
            self.assertIsNone(r.get_igtv())

    # -- Trajectory --

    def test_write_and_read_trajectory(self):
        """Write trajectory data and read back."""
        patient = {"patient_id": "P005"}
        study = {"study_uid": "1.2.7", "phase_count": 10}
        centroids = np.random.rand(10, 3).astype(np.float32)
        observed = centroids[:7]
        predicted = np.random.rand(3, 3).astype(np.float32)

        with TiciWriter(self.test_path) as w:
            w.write_metadata(patient, study)
            w.write_trajectory(centroids, observed, predicted, model_name="BiLSTM-v2")

        with TiciReader(self.test_path) as r:
            traj = r.get_trajectory()
            self.assertIsNotNone(traj)
            self.assertEqual(traj.prediction_model, "BiLSTM-v2")
            self.assertTrue(np.allclose(centroids, traj.centroids))
            self.assertTrue(np.allclose(observed, traj.observed))
            self.assertTrue(np.allclose(predicted, traj.predicted))

    # -- Provenance --

    def test_provenance(self):
        """Write and read provenance."""
        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {})
            w.write_provenance({
                "source_type": "dicom",
                "source_files_count": 42,
                "converter_version": "test-0.1",
            })

        with TiciReader(self.test_path) as r:
            meta = r.get_metadata()
            self.assertEqual(meta.provenance.source_type, "dicom")
            self.assertEqual(meta.provenance.source_files_count, 42)
            self.assertEqual(meta.provenance.converter_version, "test-0.1")

    # -- Phase metadata --

    def test_phase_meta(self):
        """Test get_phase_meta helper."""
        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {"phase_count": 2})
            w.write_phase_images(0, np.random.rand(5, 32, 32).astype(np.float32))
            w.write_phase_images(1, np.random.rand(3, 32, 32).astype(np.float32),
                                masks=np.ones((3, 32, 32), dtype=np.uint8))

        with TiciReader(self.test_path) as r:
            meta_list = r.get_phase_meta()
            self.assertEqual(len(meta_list), 2)
            self.assertTrue(meta_list[0]["has_images"])
            self.assertFalse(meta_list[0]["has_masks"])
            self.assertTrue(meta_list[1]["has_images"])
            self.assertTrue(meta_list[1]["has_masks"])

    # -- Progressive write (overwrite) --

    def test_progressive_write_overwrite(self):
        """Verify that re-writing a phase replaces old data."""
        vol1 = np.ones((3, 16, 16), dtype=np.float32) * 0.1
        vol2 = np.ones((3, 16, 16), dtype=np.float32) * 0.9

        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {"phase_count": 1})
            w.write_phase_images(0, vol1)
            w.write_phase_images(0, vol2)  # overwrite

        with TiciReader(self.test_path) as r:
            loaded = r.get_phase_images(0)
            self.assertTrue(np.allclose(loaded, vol2))

    # -- Compression verification --

    def test_compression_images(self):
        """Verify gzip-compressed images decompress correctly."""
        vol = np.random.rand(10, 64, 64).astype(np.float32)
        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {})
            w.write_phase_images(0, vol)

        # File should be smaller than raw size
        raw_size = vol.nbytes
        file_size = os.path.getsize(self.test_path)
        # With gzip, a random float32 array will compress poorly,
        # but HDF5 overhead + compression should still be reasonable
        self.assertLess(file_size, raw_size * 1.5)  # Not exploding

        with TiciReader(self.test_path) as r:
            loaded = r.get_phase_images(0)
            self.assertTrue(np.allclose(vol, loaded, atol=1e-6))

    # -- Boundary: empty study --

    def test_empty_study_phase_count(self):
        """Phase count is 0 when no phases written."""
        with TiciWriter(self.test_path) as w:
            w.write_metadata({}, {})

        with TiciReader(self.test_path) as r:
            self.assertEqual(r.phase_count, 0)

    # -- Boundary: access outside context --

    def test_writer_outside_context_raises(self):
        w = TiciWriter.__new__(TiciWriter)
        w._file = None
        with self.assertRaises(RuntimeError):
            w.write_metadata({}, {})

    def test_reader_outside_context_raises(self):
        r = TiciReader.__new__(TiciReader)
        r._file = None
        with self.assertRaises(RuntimeError):
            _ = r.phase_count

    def test_lazy_slice_access_and_content_flags(self):
        """Viewer helpers read one slice and expose optional content cheaply."""
        images = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        masks = (images % 2 == 0).astype(np.uint8)

        with TiciWriter(self.test_path) as writer:
            writer.write_metadata({}, {"phase_count": 1})
            writer.write_phase_images(0, images, masks)

        with TiciReader(self.test_path) as reader:
            self.assertEqual(reader.get_phase_shape(0), (3, 4, 5))
            np.testing.assert_array_equal(reader.get_phase_slice(0, 1), images[1])
            np.testing.assert_array_equal(reader.get_phase_mask_slice(0, 1), masks[1])
            self.assertFalse(reader.has_igtv)
            self.assertFalse(reader.has_trajectory)

            with self.assertRaises(IndexError):
                reader.get_phase_slice(0, 3)


class TestTiciConverter(unittest.TestCase):
    """Tests for DICOM/NIfTI conversion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_minimal_dicom(self, path, patient_id="TEST001",
                              instance_number=1, study_uid="1.2.840.10008.5.1"):
        """Create a minimal valid DICOM file for testing."""
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset

        # Create file meta
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = "1.2.840.10008.1.2"

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.PatientID = patient_id
        ds.PatientName = "Test^Patient"
        ds.PatientSex = "M"
        ds.PatientAge = "050Y"
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = "20260604"
        ds.Modality = "CT"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.Rows = 64
        ds.Columns = 64
        ds.PixelSpacing = [0.5, 0.5]
        ds.SliceThickness = 2.0
        ds.InstanceNumber = instance_number
        ds.SliceLocation = float(instance_number)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Pixel data: uint16 random noise
        pixels = np.random.randint(0, 2000, (64, 64), dtype=np.uint16)
        ds.PixelData = pixels.tobytes()

        ds.save_as(path, write_like_original=False)

    def test_from_dicom_dir_basic(self):
        """Convert a directory of DICOM files to .tici and verify."""
        from tici.converter import from_dicom_dir

        dicom_dir = os.path.join(self.tmpdir, "dicom")
        os.makedirs(dicom_dir)

        # Create 10 DICOM files (one per phase, same z=0)
        for i in range(10):
            self._create_minimal_dicom(
                os.path.join(dicom_dir, f"slice_{i:04d}.dcm"),
                instance_number=i,
            )

        output = os.path.join(self.tmpdir, "output.tici")
        result = from_dicom_dir(dicom_dir, output)
        self.assertEqual(result, output)

        with TiciReader(output) as r:
            meta = r.get_metadata()
            self.assertEqual(meta.patient.patient_id, "TEST001")
            self.assertEqual(meta.study.modality, "CT")
            self.assertEqual(r.phase_count, 10)  # instance_number % 10 => each unique

    def test_from_dicom_dir_empty_raises(self):
        """Empty DICOM directory raises ValueError."""
        from tici.converter import from_dicom_dir

        empty_dir = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty_dir)
        output = os.path.join(self.tmpdir, "output.tici")

        with self.assertRaises(ValueError):
            from_dicom_dir(empty_dir, output)

    def test_from_nifti_4d_uses_tici_volume_order(self):
        """4D NIfTI (X,Y,Z,T) becomes TICI phases in (Z,Y,X) order."""
        import nibabel as nib
        from tici.converter import from_nifti

        data = np.arange(4 * 5 * 6 * 2, dtype=np.float32).reshape(4, 5, 6, 2)
        affine = np.diag([1.5, 2.5, 3.5, 1.0])
        nifti_path = os.path.join(self.tmpdir, "timeseries.nii.gz")
        nib.save(nib.Nifti1Image(data, affine), nifti_path)

        output = os.path.join(self.tmpdir, "timeseries.tici")
        from_nifti(nifti_path, output)

        expected = (data - data.min()) / (data.max() - data.min())
        expected_phase_0 = np.transpose(expected[:, :, :, 0], (2, 1, 0))

        with TiciReader(output) as reader:
            meta = reader.get_metadata()
            phase_0 = reader.get_phase_images(0)

            self.assertEqual(reader.phase_count, 2)
            self.assertEqual(meta.study.rows, 5)
            self.assertEqual(meta.study.cols, 4)
            self.assertEqual(meta.study.slice_count, 6)
            np.testing.assert_allclose(meta.study.pixel_spacing, [2.5, 1.5])
            self.assertEqual(phase_0.shape, (6, 5, 4))
            np.testing.assert_allclose(phase_0, expected_phase_0)


if __name__ == "__main__":
    unittest.main()
