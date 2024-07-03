
# clip_to_psd

A Python script to convert Clip Studio Paint (.clip) files to PSD format without complicated external dependencies.

## Introduction
This script is designed for those who want to convert their Clip Studio Paint files to the widely-used PSD format. It is currently the only existing tool that works as standalone software on a local machine capable of running Python, including Linux. This means you can export .clip files to .psd without needing to install Clip Studio Paint software.Additionally, this script exports several properties that Clip Studio Paint either does not export or exports as rasterized, non-editable layers, such as:  Text as editable text; Filter layers; Gradients and Solid Color layers; Outline effect; Color tags for layers. This script works as command line tool. 

## Basic Usage

To convert a .clip file to a .psd file, run the following command:

`python clip_to_psd.py input.clip -o output.psd`

## Dependencies:
- Python 3 is required.
- The Python PIL library is optional but needed for PSD thumbnail previews or exporting layers as PNG files. Use the `--blank-psd-preview` option to avoid importing the PIL library.

## Installation
Download the script clip_to_psd.py to any directory and run it with Python. The script is not available as a pip or deb package yet.

## Caution

The script exports some advanced features beyond basic pixel layers, such as text or backgrounds defined as solid fill layers without pixels. Clip Studio Paint cannot read these features back from the PSD file; only Photoshop or GIMP can. This often occurs with the background layer, which appears transparent when the PSD is reopened in Clip Studio Paint because it cannot recognize vector solid fill layers in the PSD. Clip Studio Paint itself exports such layers as plain pixel rasterized layers, without attempting to preserve their vector nature.

## Features

- **Layer Properties and folder structure**: Exports all basic layer properties including visibility, alpha lock, folder structure, and blending modes. Exports pixel data of layers with maks and mask editing properties. Draft layers are forced to be invisible in the exported PSD.
- **Text Layers**: Exports text as editable vector text layers with common features like transformation, color, and typeface. This feature is not available for export even in the original Clip Studio Paint.
- **Filter Layers**: Supports some filter layer types: HSL, Levels, Brightness/Contrast, and Curves.
- **Outline Effect**: Exports as the PSD Stroke layer property. This also works with outlined text.
- **Gradient** and **Solid Color Layers**


## Missing features
- **Vector layers** export is not supported; they are always exported as rasterized if the .clip file contains the pixel data. It seems newer versions of CSP drop the pixel data for Vector layers :(
- **Tone** and **Frame Border** export is also not supported, with the same limitations and workarounds as Vector layers.
- Export of **Color Balance**, **Posterization**, **Binarization**, and **Gradient Map** effect layers is not supported.
- **Animation Data**: Timelines and other animation data are outside the scope of this tool.
- **External** layer file references are not exported.
- **Ruler, 3D**, and other advanced types of layers are not exported
- **Gradient layer**  export may be limited for complex settings; the script allows exporting both vector and raster versions of Gradient layers if the .clip file contains pixel data of Gradient type layer.
- **HSL effect layer**: Note that HSL  settings are interpreted differently by PSD and CLIP files, which may require reviewing the export result.

## Additional Options

To see more options, including how to export layers as raw PNG files or how to export the internal SQLite database without exporting pixel data, use the `--help` command-line option.
