IEDL-IDS — Local ToN_IoT Shard Audit
====================================

Purpose
-------
Audit the 23 original processed ToN_IoT Network CSV shards without uploading
or modifying the approximately 3.2 GB dataset. The script scans in chunks.

Requirements
------------
Python 3.10+
pandas 2.x

Install:
    py -m pip install -r requirements.txt

Run (Windows PowerShell):
    py audit_network_shards.py --data-dir "D:\path\to\Processed_Network_dataset" --out-dir "D:\path\to\iedl_audit"

The input directory must contain:
    Network_dataset_1.csv
    Network_dataset_2.csv
    ...
    Network_dataset_23.csv

Outputs (small files):
    audit_summary.json
    class_counts.csv
    feature_presence_audit.csv
    shard_counts.csv

Important:
- This script is READ-ONLY. It does not alter your CSV shards.
- It checks the expected 22,339,021 records and class counts from the
  Statistics of Network Records workbook.
- Timestamp (ts) is not treated as a model feature.
- For sparse categorical Zeek columns, '-' and blank are counted as absent;
  for numeric columns, zero is counted as absent. We will review this feature
  selection against the paper and metadata before freezing the 21-feature set.
- Do not begin 10% sampling or RGB image construction until the audit passes
  and the image-label grouping rule is agreed.
