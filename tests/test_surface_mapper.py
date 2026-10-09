"""Surface-mapper helpers: command construction and PySurfer script generation.

Earlier versions of this file replaced ``numpy`` and ``jax`` in ``sys.modules``
with MagicMocks at import time, which poisoned every test collected after it.
The real modules import fine; only FreeSurfer's binary is mocked.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from dmipy_jax.viz.surface_mapper import map_to_surface, generate_pysurfer_script


class TestSurfaceMapper(unittest.TestCase):
    @patch('dmipy_jax.viz.surface_mapper.subprocess.run')
    @patch('dmipy_jax.viz.surface_mapper.Path')
    def test_map_to_surface_command(self, mock_path, mock_subprocess):
        mock_volume = MagicMock()
        mock_volume.resolve.return_value = mock_volume
        mock_volume.exists.return_value = True
        mock_volume.parent = MagicMock()
        mock_volume.stem = "test_vol"
        mock_volume.__str__.return_value = "/path/to/volume.nii.gz"
        mock_path.return_value = mock_volume

        map_to_surface("/path/to/volume.nii.gz", "lh", subject="BigMac")

        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        self.assertEqual(cmd[:5], ["mri_vol2surf", "--mov", "/path/to/volume.nii.gz", "--hemi", "lh"])
        self.assertIn("--surf", cmd)
        self.assertIn("white", cmd)
        self.assertIn("--projfrac", cmd)
        self.assertIn("0.5", cmd)
        self.assertIn("--regheader", cmd)
        self.assertIn("BigMac", cmd)

    def test_generate_pysurfer_script(self):
        script_path = generate_pysurfer_script("overlay.mgh", "lh", subject="BigMac")
        try:
            with open(script_path, 'r') as f:
                content = f.read()
        finally:
            os.remove(script_path)
        self.assertIn("from surfer import Brain", content)
        self.assertIn("subject = 'BigMac'", content)
        self.assertIn("brain.add_overlay('overlay.mgh'", content)


if __name__ == '__main__':
    unittest.main()
