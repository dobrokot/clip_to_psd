
# clip_to_psd

A Python script to convert Clip Studio Paint (.clip) files to PSD format without dependencies.

## Basic Usage

To convert a .clip file to a .psd file, run the following command:

`python clip_to_psd.py input.clip -o output.psd`

python3 and Python PIL is required (Python PIL is optional).

## Features

- Exports all basic layer properties (visibility, alpha lock, folder structure, all blending modes).
- Supports exporting text as editable vector text layers with common text features set (tranformation, color, type face, etc), a feature not available in the original Clip Studio Paint.
- Supports some filter layer types: HSL, Levels, Brightness/Contrast, Curve. HSL settings are interpreted in different way by .psd and .clip, could require review of the export result.
- With the command-line switch `--blank-psd-preview`, it's possible to avoid dependency on the Image PIL library and export a .clip file to a .psd file without any dependencies outside of Python's built-in libraries.

## Additional Options

To see more options, including how to export layers as raw PNG files or how to export the internal SQLite database without exporting pixel data, use the `--help` command-line option
