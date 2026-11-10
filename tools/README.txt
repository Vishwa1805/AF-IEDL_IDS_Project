IEDL-IDS feature audit v2

Purpose
- Re-check the 23 ToN_IoT network CSV shards.
- Fix the previous schema-reporting bug (one warning per chunk instead of per file).
- Treat an optional 'uid' column as an identifier, not a model feature.
- Separately measure numeric zeros, '-' placeholders, blank values and literal F values.

Run from PowerShell:
py -m pip install pandas
py audit_network_features_v2.py --data-dir "D:\path\to\Processed_Network_dataset" --out-dir "D:\path\to\IEDL_IDS_Project\audit_output_v2"

The script is READ-ONLY. It does not modify the original CSV files.

Upload these outputs afterward:
- audit_summary_v2.json
- feature_presence_audit_v2.csv

Important:
The paper says features with >80% zero values are removed. The script intentionally does NOT decide that '-' means zero. We will compare the separate counts with the feature metadata and paper before selecting the final 21 features.
