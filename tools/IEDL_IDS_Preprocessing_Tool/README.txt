IEDL-IDS preprocessing and RGB image-generation tool
====================================================

This tool consumes the already-created 10% sampled CSV. It DOES NOT read the
23 original 3.2 GB shards, so do not rerun the audit or sampling steps.

Requirements (same environment used for the earlier tools):
    py -m pip install pandas numpy

Example PowerShell command (adjust paths if your folder differs):
    py prepare_iedl_images.py --input-csv "D:\Academics\7th-Sem\Computer And Network Security\Project\IEDL_IDS_Project\sampled_data\sampled_raw_10pct.csv" --out-dir "D:\Academics\7th-Sem\Computer And Network Security\Project\IEDL_IDS_Project\iedl_images"

Outputs:
    images_train.npz
    images_validation.npz
    images_test.npz
    preprocessing_metadata.json
    image_generation_summary.csv
    feature_selection.csv

IMPORTANT METHODOLOGY NOTES
---------------------------
1. The IEDL-IDS article specifies 21 ToN_IoT dimensions but does not list their
   names. This tool uses a transparent reconstruction: 14 populated connection/
   statistics fields plus seven documented Boolean indicators.
2. The article specifies 63 flows per 21x21x3 image, with the first 21 flows in
   R, the next 21 in G and the final 21 in B.
3. The article does not fully specify multiclass image-label assignment. This
   tool groups by class, orders each class by timestamp, and assigns each image
   that class. This is an explicit assumption and must be described in the paper.
4. The paper reports 80% train, 20% test and 20% validation (120% total). This
   tool uses a chronological 80/10/10 split within each class, documented in
   preprocessing_metadata.json.
5. Categorical vocabularies and min-max ranges are fitted on training flows only.
6. MITM is excluded from the multiclass experiment; it remains in sampled_raw
   and is counted in the metadata. The 105 sampled MITM flows are too few to
   form meaningful train/validation/test image sets.
7. The script has not been validated on your complete sampled CSV in this
   environment. It should be run locally against the generated sampled file.
