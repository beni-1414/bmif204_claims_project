# Inovalon project: polypharmacy and AEs
This project explores the risk factors in polypharmacy patients that lead to a higher incidence of AEs, and explores diverse prediction approaches.

## Structure


## Functionality
### Data extraction scripts - `src/sql_extraction`
- `extract_raw_sql_opioid.py`: Connects to the Inovalon SQL Server and extracts member-level opioid prescription and claim data, exporting key datasets as Parquet files for downstream processing (prescription fills, adverse events, demographics, enrollment).
- `opioid_ndcs.py`: This script uses the OpenFDA database (Download from https://open.fda.gov/apis/drug/ndc/download/) to extract a CSV with all the rellevant NDC codes, and uses a library (#https://github.com/eddie-cosma/ndclib) to convert between NDC10 and NDC11. Only need to be rerun if changing the list of drugs.
- `prepare_database_for_extraction.py`: Prepares helper tables and database objects (for example inserting the NDC list) needed by the extraction queries.
- `extract_icd10_codes.py`: Extracts ICD-10 claim records around spell start/change times to support AE and diagnosis postprocessing.
- `test_connection.py`: Small utility to verify database connectivity and credentials.

### Data processing scripts - `src/spell_generation`
#### `spell_generation.py`

This script detects **polypharmacy spells** and labels them with **adverse events (AEs)** using prescription, enrollment, and claims data extracted from SQL.

##### Overview

Each patient’s prescription history is scanned to identify continuous periods (“spells”) with multiple concurrent drugs. These spells are then:

* Restricted to those involving **opioid fills**
* Censored by **enrollment periods** and **washout windows**
* Annotated with **AEs** occurring during follow-up

##### Steps

1. **Input & Arguments**
   Reads Parquet files (`rx_fills`, `adverse_events`, `enrollment`, `demographics`) from the scratch directory.
   Key parameters:

   * `--min_concurrent`: Minimum overlapping drugs (default = 3)
   * `--extend_days`: Grace period after drop below threshold (default = 21)
   * `--min_spell_len`: Minimum total duration (default = 30 + extend_days)

2. **Spell Detection**

   * Merges overlapping fill intervals per drug.
   * Tracks concurrent drug counts over time.
   * Starts a spell when concurrent ≥ `min_concurrent` and an opioid is present; ends after `extend_days` below threshold.
   * Runs in **parallel** across members using up to 8 CPUs.

3. **Filtering & Censoring**

   * Removes spells not covered by enrollment or lacking ≥ 180 days pre-index enrollment.

4. **AE Labeling**

   * Flags spells with at least one **new AE** code (not seen before spell start) during the active or follow-up period.
   * NOTE: The AE codes that do have been seen before are simply ignored, 

5. **Output**

   * Saves final dataset:
     `spells_with_labels_<EXTEND_DAYS>_days<suffix>.parquet`
   * Writes debug sample CSV for quick inspection.

#### `split_spells_from_changes.py`
This code takes the output of the spell_generation script and generates a table that splits the spells by the drug additions, and aligns all of them at time 0.

#### `extract_icd10_codes.py`
This code extracts all the claims with an ICD10 code in the 6 months before the start of a splitted spell (so before a drug change).

#### `icd10_postprocessing.py`
This code clusters the ICD10 codes extracted into a `_clustered` parquet file ready for analysis.

#### `exploration_final.ipynb`
Run the notebook to get descriptive stats and plots. Expand with more plots as needed.

## Usage
### Environment
Create a venv with the requirements.txt file. If running on O2, be sure to load the necessary modules: gcc python unixODBC freetds msodbcsql17.

Queries to SQL can only be run from an interactive session, either on terminal or through jupyter or VSCode. In a terminal, run kinit to connect to innovalon. In case of issues, refer to Inovalon Harvard [documentation](https://github.com/ccb-hms/HarvardInovalonUserGuide).

### Pipeline
First, extract data from SQL. This process is a bit tricky. 

1. Insert the query drugs (in our study, the opioid ndcs) into a table:
```bash
python src/sql_extraction/prepare_database_for_insertion --db 5
```

2. Run this query directly in SQL server (it can take a few hours, on the 5M db it took 4h30)
```sql
SELECT
    r.MemberUID
INTO <YOUR_USERNAME>.dbo.elegible_members
FROM RxClaim r
LEFT JOIN dbo.QueryDrugs o
    ON r.ndc11code = o.ndc11code
WHERE r.supplydayscount IS NOT NULL
GROUP BY r.MemberUID
HAVING
    MAX(CASE WHEN o.ndc11code IS NOT NULL THEN 1 ELSE 0 END) = 1
    AND COUNT(DISTINCT r.ndc11code) >= 3;
```

3. Run the following script (ensure you have sufficient memory, on the 5M db it was necessary to ask for 64GB, for larger scales the code would need to be adapted to do even more chunking and modular processing so the whole dataframe is never loaded in memory):
```bash
python src/sql_extraction/extract_raw_sql_opioid.py --suffix "_sample5M" --db 5
```

4. Run the script to add the drug mappings to the RXfills to be able to cluster by ATC class:
```bash
python src/drug_mapping/join_ndc_to_atc.py input.parquet
```

5. Generate spells and labels. Modify the pipeline.sbatch file as needed, making sure to add a rellevant suffix, and submit via (it takes about 30 mins on the 1M sample, a ~3h on the 5M, be sure to ask enough memory to avoid OOM error, although the code already does many memory management tricks):
```bash
sbatch src/spell_generation/pipeline.sbatch
```

6. Manually run the ICD10 extraction from SQL (it can't be run in a job, and if it takes longer than a few seconds to start it means the db is locked by someone else. Recommend running at night when no one is using the db):
```bash
python src/sql_extraction/extract_icd10_codes.py --suffix "WHATEVER YOU USED IN THE sbatch FILE"
```

7. Run the ICD10 postprocessing (is quite fast):
```bash
python src/icd10_postprocessing.py --suffix "WHATEVER YOU USED IN THE sbatch FILE"
```

8. With this, you are ready to run the exploration notebook, train models, etc. Be sure to use the final_preprocessing script on top of whatever you do to get the final features and apply the necessary filters for data leakage prevention and quality control.