IEDL-IDS — ONE-PASS STRATIFIED SAMPLING TOOL

Purpose
-------
Read the 23 full ToN_IoT Network_dataset_*.csv shards once and create an
approximately 10% random sample, stratified by the `type` class, while
preserving the original timestamp (`ts`) and all feature columns.

The script deliberately does NOT choose the final 21 features yet. The current
audit shows that the paper's stated >80%-zero rule yields 7 removed features
when only numeric zeros are counted, but 29 removed features when '-' and blank
placeholders are also treated as absent. Neither result gives the paper's stated
21 dimensions. We should not guess at the missing selection rule.

Expected input folder
--------------------
A folder containing:
  Network_dataset_1.csv
  Network_dataset_2.csv
  ...
  Network_dataset_23.csv

Run from PowerShell
-------------------
py -m pip install -r requirements.txt

py prepare_sample.py `
  --data-dir "D:\path\to\Processed_Network_dataset" `
  --out-dir "D:\path\to\IEDL_IDS_Project\sampled_data"

Defaults
--------
Sample fraction: 0.10
Random seed:     42
Chunk size:      100,000
Expected rows:   22,339,021

Output files
------------
sampled_data\sampled_raw_10pct.csv
sampled_data\sampled_class_counts.csv
sampled_data\sampled_shard_counts.csv
sampled_data\sampling_manifest.json

Important notes
---------------
1. The script reads the original shards only once. It samples each class
   independently within each chunk. Per-class totals will be very close to 10%,
   but may differ slightly because each chunk/class sample size is rounded.
2. All predictors, ts, label, and type are preserved. The `uid` column is
   dropped because it appears only in shard 6 and is an identifier.
3. The raw sample is NOT timestamp-sorted. Do not generate images yet. After
   feature selection/encoding is finalized, sort by ts and then create groups.
4. MITM is retained in this intermediate sample. It can be excluded later for
   the paper's multiclass experiment.
5. Output is not overwritten by default. Delete the previous sampled_data
   folder or add --overwrite if rerunning intentionally.
6. Keep the original 23 CSV files unchanged.
