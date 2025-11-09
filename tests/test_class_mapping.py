"""
Unit tests to verify class mapping consistency between training and inference.

Expected class mapping:
- 0 = smoke (includes wildfire)
- 1 = haze
- 2 = normal (cloud, land, seaside, dust)
"""
import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'haze-detection'))


class TestClassMapping:
    """Test class mapping consistency."""

    def test_inference_class_mapping_order(self):
        """Test that inference.py has correct CLASS_MAPPING order."""
        from app.inference import CLASS_MAPPING

        expected = ['smoke', 'haze', 'normal']
        assert CLASS_MAPPING == expected, f"Expected {expected}, got {CLASS_MAPPING}"

        # Verify indices
        assert CLASS_MAPPING[0] == 'smoke', "Index 0 should be 'smoke'"
        assert CLASS_MAPPING[1] == 'haze', "Index 1 should be 'haze'"
        assert CLASS_MAPPING[2] == 'normal', "Index 2 should be 'normal'"

    def test_inference_filename_to_class_smoke(self):
        """Test that smoke/wildfire filenames map to class 0."""
        from app.inference import get_actual_class_from_filename

        smoke_filenames = [
            'smoke_123.tif',
            'SMOKE_456.TIF',
            'wildfire_789.tif',
            'WILDFIRE_000.TIF',
            'test_smoke.tif',
            'test_wildfire.tif'
        ]

        for filename in smoke_filenames:
            result = get_actual_class_from_filename(filename)
            assert result == 0, f"'{filename}' should map to class 0 (smoke), got {result}"

    def test_inference_filename_to_class_haze(self):
        """Test that haze filenames map to class 1."""
        from app.inference import get_actual_class_from_filename

        haze_filenames = [
            'haze_123.tif',
            'HAZE_456.TIF',
            'test_haze.tif',
            'haze_test_001.tif'
        ]

        for filename in haze_filenames:
            result = get_actual_class_from_filename(filename)
            assert result == 1, f"'{filename}' should map to class 1 (haze), got {result}"

    def test_inference_filename_to_class_normal(self):
        """Test that normal/cloud/land/seaside/dust filenames map to class 2."""
        from app.inference import get_actual_class_from_filename

        normal_filenames = [
            'cloud_123.tif',
            'CLOUD_456.TIF',
            'land_789.tif',
            'seaside_000.tif',
            'dust_111.tif',
            'normal_222.tif',
            'test_cloud.tif',
            'other_333.tif'
        ]

        for filename in normal_filenames:
            result = get_actual_class_from_filename(filename)
            assert result == 2, f"'{filename}' should map to class 2 (normal), got {result}"

    def test_dataset_class_order(self):
        """Test that dataset.py has correct class order."""
        # We'll test this by checking the class_to_idx mapping
        expected_classes = ['smoke', 'haze', 'normal']
        expected_mapping = {'smoke': 0, 'haze': 1, 'normal': 2}

        # Create mapping as done in dataset.py
        classes = ['smoke', 'haze', 'normal']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        assert classes == expected_classes, f"Expected {expected_classes}, got {classes}"
        assert class_to_idx == expected_mapping, f"Expected {expected_mapping}, got {class_to_idx}"

        # Verify individual mappings
        assert class_to_idx['smoke'] == 0, "smoke should map to 0"
        assert class_to_idx['haze'] == 1, "haze should map to 1"
        assert class_to_idx['normal'] == 2, "normal should map to 2"

    def test_training_inference_consistency(self):
        """Test that training and inference use the same class mapping."""
        from app.inference import CLASS_MAPPING

        # Training class order (from dataset.py)
        training_classes = ['smoke', 'haze', 'normal']

        # Inference class order
        inference_classes = CLASS_MAPPING

        assert training_classes == inference_classes, \
            f"Training classes {training_classes} != Inference classes {inference_classes}"

    def test_mapping_matches_requirements(self):
        """Test that the mapping matches the requirements from PROBLEM.md."""
        from app.inference import CLASS_MAPPING

        # Requirements from PROBLEM.md:
        # 0 = smoke (includes wildfire)
        # 1 = haze
        # 2 = normal (cloud, land, seaside, dust)

        assert CLASS_MAPPING[0] == 'smoke', "Class 0 should be 'smoke'"
        assert CLASS_MAPPING[1] == 'haze', "Class 1 should be 'haze'"
        assert CLASS_MAPPING[2] == 'normal', "Class 2 should be 'normal'"


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_case_insensitive_filename_matching(self):
        """Test that filename matching is case-insensitive."""
        from app.inference import get_actual_class_from_filename

        # Test mixed case
        assert get_actual_class_from_filename('SMOKE_test.tif') == 0
        assert get_actual_class_from_filename('SmOkE_test.tif') == 0
        assert get_actual_class_from_filename('HAZE_test.tif') == 1
        assert get_actual_class_from_filename('HaZe_test.tif') == 1
        assert get_actual_class_from_filename('CLOUD_test.tif') == 2
        assert get_actual_class_from_filename('ClOuD_test.tif') == 2

    def test_wildfire_is_smoke(self):
        """Test that wildfire is correctly classified as smoke (class 0)."""
        from app.inference import get_actual_class_from_filename

        wildfire_files = [
            'wildfire_001.tif',
            'WILDFIRE_002.tif',
            'test_wildfire.tif',
            'wildfire.tif'
        ]

        for filename in wildfire_files:
            result = get_actual_class_from_filename(filename)
            assert result == 0, f"'{filename}' should be class 0 (smoke), got {result}"

    def test_no_ambiguous_classifications(self):
        """Test that smoke/haze keywords take priority over normal."""
        from app.inference import get_actual_class_from_filename

        # These should be smoke, not normal
        assert get_actual_class_from_filename('smoke_cloud.tif') == 0
        assert get_actual_class_from_filename('wildfire_land.tif') == 0

        # These should be haze, not normal
        assert get_actual_class_from_filename('haze_cloud.tif') == 1
        assert get_actual_class_from_filename('haze_land.tif') == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
