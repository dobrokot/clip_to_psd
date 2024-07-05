# Test Files

This folder contains files related to testing the script's functionality, particularly focusing on the export of various layer properties.

## Testing Export Correctness

The `testing_export_correctness.md` file provides detailed instructions on how to run tests that verify the script's ability to export all supported features.

Reference Files

The following files are used in conjunction with the export correctness tests:

- `exported_raster_current_output.png`: Reference output file for raster mode export
- `exported_vector_current_output.png`: Reference output file for vector mode export
- `test_export_all_features.clip`: Input file for comprehensive feature testing
- `test_export_all_features_reference_output.png`: Reference output file exported directly by Clip Paint Studio

## Additional Tests

PIL Import Test

The `test_image_pil_import.py` file tests how the script performs without the Python Imaging Library (PIL) installed. This helps ensure the script's functionality in environments where PIL might not be available.
