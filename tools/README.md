
# Tools

This directory contains supplementary scripts needed for the main project. These scripts assist in parsing and analyzing various file formats related to the project.


### parse_adobe_fonts.py

This script parses Adobe fonts file descriptions, which define font styles for fonts used in PSD files. It's used for identifying font styles (regular, bold, italic, bold-italic) for each font family. The script outputs a table that maps regular font names to their corresponding style variations.

### psd_parse.py

This script is designed to analyze and extract data from PSD (Photoshop Document) files. Its main features include:

- Outputting some PSD data sections as readable text or PNG files.
- Descending into sub-containers of properties (though support is not complete).

Useful for:
1. Reverse engineering of PSD files to assist in the creation of `clip_to_psd.py`. It helps to create redable diff for slightly modified PSD files.
2. Exploring the internals of PSD files.
3. Exporting image data from PSD files.
