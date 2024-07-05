# Testing process for converting .clip Files to .psd Files

Testing the ability of Photoshop to interpret exported PSD files requires some manual work. Attempting to automate this process could introduce additional difficulties, particularly because Photoshop may prompt to re-render text layers. However, raster exports should not face such issues.

Note about GIMP: GIMP has limitations in reading exported PSD files, most notably being unable to read solid color background layer and all filter layers.

## Steps for Testing

### 1. Run the export in raster and/or vector mode:

   - For rasterized versions of text/gradient layers:

     `python ../clip_to_psd.py --log-level warning test_export_all_features.clip -o exported_raster.psd`


   - For vector versions of text/gradient layers:

     `python ../clip_to_psd.py test_export_all_features.clip --log-level warning --text-layer-raster disable --text-layer-vector enable --gradient-layer-raster disable --gradient-layer-vector enable -o exported_vector.psd`


### 2. Open the PSD file in Photoshop and export it:

   - Photoshop may prompt to update and re-render text layers; respond 'yes' to this prompt.
   - Note that GIMP cannot load most of the exported features, especially when exported in vector mode.
   - Copy the Photoshop image result (menu: Edit -> Copy Merged) or export it (menu: File -> Save For Web).

### 3. Compare the exported PSD with the original .clip output:

   - Overlay or switch the copied/exported image in an image viewer to compare against the original .clip output, saved in `test_export_all_features_reference_output.png`.
   - If you switch files in image viwer, it could help to create empty folder to compare files without other image files
     ```
     mkdir tmp; rm tmp/*.png
     cp exported_vector.png exported_raster_current_output.png tmp
      ```


## Expected Results

- Raster Export Mode:
  - The HSL filter layer is expected to differ.
  - All other elements should be identical to the original PNG.

- Vector Export Mode:
  - The gradient and text in vector mode may display slight differences compared to the original.
  - All other elements should be identical to the original PNG.

The output should closely match either `exported_raster_current_output.png` or `exported_vector_current_output.png`. If there are any differences, the source of the differences should be investigated. If the changes are expected, the new output version should be committed. Slight pixel-level micro-differences may exist due to different versions of Photoshop.

## Additional Notes

The preview of the PSD file may differ from the exported PSD result because it is taken from the .clip saved preview of the file. This isn't part of the testing process but could potentially cause confusion if an image viewer loads the PSD. Remember that the preview loaded by image viewers is not accurate and should not be confused with the original or exported files.
