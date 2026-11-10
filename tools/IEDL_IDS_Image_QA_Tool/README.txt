IEDL-IDS IMAGE ARRAY QA TOOL
===========================

Purpose
-------
Validate the already-generated NPZ image files and create PNG previews.
This tool does NOT scan or reopen the original 23 CSV files, and does not
modify the existing image arrays.

Expected input folder
---------------------
iedl_images/
  images_train.npz
  images_validation.npz
  images_test.npz
  preprocessing_metadata.json

Run (PowerShell)
----------------
py verify_image_arrays.py --data-dir "D:\Academics\7th-Sem\Computer And Network Security\Project\IEDL_IDS_Project\iedl_images"

Requirements
------------
NumPy is already required by the preprocessing pipeline.
Pillow is used to save image previews. If missing:
  py -m pip install pillow

Outputs
-------
iedl_images/qa/
  image_array_validation.json
  train_<class>_1.png
  validation_<class>_1.png
  test_<class>_1.png

Upload image_array_validation.json and a few representative PNGs if you
want the arrays and visual encoding checked before model training.
