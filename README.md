
# clip_to_psd

A Python script to convert Clip Studio Paint (.clip) files to PSD format without dependencies.

## Basic Usage

To convert a .clip file to a .psd file, run the following command:

`python clip_to_psd.py input.clip -o output.psd`

python3 and Python PIL are required. Python PIL is optional, it's required for PSD thumbnail preview or layers export as PNG files.

## Caution
The script exports some advanced features beyond basic pixel layers, such as text or backgrounds defined as solid fill layers without pixels. Clip Studio Paint cannot read these features back from the PSD file; only Photoshop or GIMP can. This often occurs with the background layer, which appears transparent when the PSD is reopened in Clip Studio Paint because it cannot recognize vector solid fill layers in the PSD. Clip Studio Paint itself exports such layers as plain pixel rasterized layers, without attempting to preserve their vector nature.

## Features

- Exports all basic layer properties (visibility, alpha lock, folder structure, all blending modes).
- Supports exporting text as editable vector text layers with common text features set (tranformation, color, type face, etc), a feature not available in the original Clip Studio Paint.
- Supports some filter layer types: HSL, Levels, Brightness/Contrast, Curve. HSL settings are interpreted in different way by .psd and .clip, could require review of the export result.
- With the command-line switch `--blank-psd-preview`, it's possible to avoid dependency on the Image PIL library and export a .clip file to a .psd file without any dependencies outside of Python's built-in libraries.

## Additional Options

To see more options, including how to export layers as raw PNG files or how to export the internal SQLite database without exporting pixel data, use the `--help` command-line option
