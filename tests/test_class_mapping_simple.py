"""
Simple unit tests to verify class mapping without heavy dependencies.
Tests the logic directly by reading the source files.
"""
import pytest
import re


class TestInferenceClassMapping:
    """Test inference.py class mapping by parsing the source file."""

    def test_class_mapping_constant(self):
        """Verify CLASS_MAPPING is correct in inference.py."""
        with open('app/inference.py', 'r') as f:
            content = f.read()

        # Find CLASS_MAPPING definition
        match = re.search(r"CLASS_MAPPING\s*=\s*\[(.*?)\]", content)
        assert match, "Could not find CLASS_MAPPING in inference.py"

        mapping_str = match.group(1)
        # Extract the class names
        classes = [s.strip().strip("'\"") for s in mapping_str.split(',')]

        expected = ['smoke', 'haze', 'normal']
        assert classes == expected, f"Expected {expected}, got {classes}"
        print(f"✓ CLASS_MAPPING = {classes}")

    def test_get_actual_class_from_filename_logic(self):
        """Verify get_actual_class_from_filename function logic."""
        with open('app/inference.py', 'r') as f:
            content = f.read()

        # Check that smoke/wildfire returns 0
        assert "return 0  # smoke" in content or "return 0 # smoke" in content, \
            "smoke should return 0"

        # Check that haze returns 1
        assert "return 1  # haze" in content or "return 1 # haze" in content, \
            "haze should return 1"

        # Check that normal returns 2
        assert "return 2  # normal" in content or "return 2 # normal" in content, \
            "normal should return 2"

        # Verify the order: smoke check comes before haze check
        smoke_pos = content.find("'smoke' in filename_lower or 'wildfire' in filename_lower")
        haze_pos = content.find("'haze' in filename_lower")

        assert smoke_pos < haze_pos, \
            "smoke/wildfire check should come before haze check to avoid misclassification"

        print("✓ get_actual_class_from_filename logic is correct")
        print("  - smoke/wildfire → 0")
        print("  - haze → 1")
        print("  - normal → 2")


class TestDatasetClassMapping:
    """Test dataset.py class mapping by parsing the source file."""

    def test_class_order(self):
        """Verify classes are defined in correct order in dataset.py."""
        with open('haze-detection/dataset.py', 'r') as f:
            content = f.read()

        # Find self.classes definition
        match = re.search(r"self\.classes\s*=\s*\[(.*?)\]", content)
        assert match, "Could not find self.classes in dataset.py"

        classes_str = match.group(1)
        classes = [s.strip().strip("'\"") for s in classes_str.split(',')]

        expected = ['smoke', 'haze', 'normal']
        assert classes == expected, f"Expected {expected}, got {classes}"
        print(f"✓ Dataset classes = {classes}")

    def test_custom_mapping_used(self):
        """Verify that custom class_to_idx mapping is used, not LabelEncoder."""
        with open('haze-detection/dataset.py', 'r') as f:
            content = f.read()

        # Should have class_to_idx
        assert 'class_to_idx' in content, "Should use class_to_idx mapping"

        # Should NOT be using LabelEncoder transform
        assert 'self.label_encoder.transform' not in content, \
            "Should not use LabelEncoder.transform (it sorts alphabetically)"

        # Should use dictionary-based mapping
        assert 'self.class_to_idx[' in content, \
            "Should use class_to_idx dictionary for mapping"

        print("✓ Dataset uses custom class_to_idx mapping (not LabelEncoder)")

    def test_labelencoder_not_imported(self):
        """Verify LabelEncoder is not imported (we use custom mapping instead)."""
        with open('haze-detection/dataset.py', 'r') as f:
            content = f.read()

        # LabelEncoder should not be imported
        assert 'from sklearn.preprocessing import LabelEncoder' not in content, \
            "LabelEncoder should not be imported"
        assert 'import LabelEncoder' not in content, \
            "LabelEncoder should not be imported"

        print("✓ LabelEncoder is not imported (using custom mapping)")


class TestConsistency:
    """Test consistency between training and inference."""

    def test_training_inference_same_order(self):
        """Verify training and inference use the same class order."""
        # Read inference.py
        with open('app/inference.py', 'r') as f:
            inference_content = f.read()

        inference_match = re.search(r"CLASS_MAPPING\s*=\s*\[(.*?)\]", inference_content)
        assert inference_match, "Could not find CLASS_MAPPING in inference.py"

        inference_classes = [s.strip().strip("'\"")
                           for s in inference_match.group(1).split(',')]

        # Read dataset.py
        with open('haze-detection/dataset.py', 'r') as f:
            dataset_content = f.read()

        dataset_match = re.search(r"self\.classes\s*=\s*\[(.*?)\]", dataset_content)
        assert dataset_match, "Could not find self.classes in dataset.py"

        dataset_classes = [s.strip().strip("'\"")
                         for s in dataset_match.group(1).split(',')]

        assert inference_classes == dataset_classes, \
            f"Inference classes {inference_classes} != Dataset classes {dataset_classes}"

        print(f"✓ Training and inference use same class order: {inference_classes}")

    def test_matches_requirements(self):
        """Verify mapping matches PROBLEM.md requirements."""
        # Requirements: 0=smoke, 1=haze, 2=normal
        with open('app/inference.py', 'r') as f:
            content = f.read()

        match = re.search(r"CLASS_MAPPING\s*=\s*\[(.*?)\]", content)
        assert match, "Could not find CLASS_MAPPING"

        classes = [s.strip().strip("'\"") for s in match.group(1).split(',')]

        assert classes[0] == 'smoke', "Class 0 should be smoke"
        assert classes[1] == 'haze', "Class 1 should be haze"
        assert classes[2] == 'normal', "Class 2 should be normal"

        print("✓ Mapping matches requirements:")
        print("  0 = smoke (includes wildfire)")
        print("  1 = haze")
        print("  2 = normal (cloud, land, seaside, dust)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
