# Class Mapping Fix - Solution Summary

## Problem Identified

The model was misclassifying smoke images as cloud/normal due to a class mapping mismatch between training and the expected output format.

### Original Issue
- **Training mapping** (dataset.py with LabelEncoder + sorted classes):
  - 0 = haze
  - 1 = normal
  - 2 = smoke

- **Expected mapping** (per feedback):
  - 0 = smoke (includes wildfire)
  - 1 = haze
  - 2 = normal (cloud, land, seaside, dust)

## Changes Made

### 1. Fixed `app/inference.py`
- Updated `CLASS_MAPPING` from `['haze', 'normal', 'smoke']` to `['smoke', 'haze', 'normal']`
- Updated `get_actual_class_from_filename()` function to match new encoding:
  - smoke/wildfire → 0
  - haze → 1
  - normal (cloud, land, seaside, dust) → 2

### 2. Fixed `haze-detection/dataset.py`
- Replaced `LabelEncoder` with custom `class_to_idx` mapping
- Explicitly set class order to `['smoke', 'haze', 'normal']`
- Added validation to ensure only expected classes are present
- **Note**: LabelEncoder always sorts alphabetically, which was causing the mismatch

## Class Mapping (Now Consistent)

```
0 = smoke (includes wildfire)
1 = haze
2 = normal (cloud, land, seaside, dust)
```

## Next Steps

1. **Retrain the model** using the updated dataset.py to ensure correct class encoding
2. Run inference with the updated inference script
3. Verify that smoke images are no longer misclassified as cloud/normal

## Files Modified

- `/app/inference.py` - Updated class mapping and filename-to-class function
- `/haze-detection/dataset.py` - Replaced LabelEncoder with custom mapping

## Impact

After retraining, the model should correctly learn:
- Class 0 = smoke features
- Class 1 = haze features
- Class 2 = normal features

This should resolve the issue where ~40% (196/488) of smoke images were misclassified.
