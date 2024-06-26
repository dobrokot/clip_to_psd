
# clip_to_psd

A Python script to convert Clip Studio Paint (.clip) files to PSD format without complicated external dependencies.

## Basic Usage

To convert a .clip file to a .psd file, run the following command:

`python clip_to_psd.py input.clip -o output.psd`

*Note: Python 3 and the Python PIL library are required. The Python PIL library is optional but needed for PSD thumbnail previews or exporting layers as PNG files.*

## Caution

The script exports some advanced features beyond basic pixel layers, such as text or backgrounds defined as solid fill layers without pixels. Clip Studio Paint cannot read these features back from the PSD file; only Photoshop or GIMP can. This often occurs with the background layer, which appears transparent when the PSD is reopened in Clip Studio Paint because it cannot recognize vector solid fill layers in the PSD. Clip Studio Paint itself exports such layers as plain pixel rasterized layers, without attempting to preserve their vector nature.

## Features

- Exports all basic layer properties (visibility, alpha lock, folder structure, all blending modes).
- Supports exporting text as editable vector text layers with common text features (transformation, color, typeface, etc.), a feature not available for export even in the original Clip Studio Paint.
- Supports some filter layer types: HSL, Levels, Brightness/Contrast, and Curves. HSL settings are interpreted differently by PSD and CLIP files, which may require review of the export result.
- With the command-line switch `--blank-psd-preview`, you can avoid dependency on the Python PIL library and export a .clip file to a .psd file using only Python's built-in libraries.

## Additional Options

To see more options, including how to export layers as raw PNG files or how to export the internal SQLite database without exporting pixel data, use the `--help` command-line option.
