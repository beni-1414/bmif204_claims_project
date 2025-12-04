# Inovalon project: polypharmacy and AEs
This project explores the risk factors in polypharmacy patients that lead to a higher incidence of AEs, and explores diverse prediction approaches.

## Structure
- `data/`: Reference inputs such as `opioid_ndc11_list.csv` and `icd10_codes.csv`.
- `results/`: Generated reports and tables (for example, `results/report/consort_diagram_data.txt`, `results/tables/`).
- `src/sql_extraction/`: SQL helper scripts to prepare tables and pull raw Rx, AE, and ICD-10 claims from Inovalon.
- `src/drug_mapping/`: Utilities to map NDC codes to ATC drug classes.
- `src/spell_generation/`: Pipeline to build spells, split spells by drug changes, and prepare ICD-10 features.
- `src/modelling/`: Feature preprocessing plus baseline and neural model trainers (with associated notebooks and sbatch files).
- `exploration.ipynb`, `final_plots.ipynb`: Notebooks for exploratory analysis and figures.
- `requirements.txt`: Python dependencies for the full pipeline.


## Functionality
### Data extraction scripts - `src/sql_extraction`
- `extract_raw_sql_opioid.py`: Connects to Inovalon SQL Server and exports prescription fills, AEs, demographics, and enrollment as Parquet for downstream use.
- `opioid_ndcs.py`: Builds the query drug list from the OpenFDA NDC download (https://open.fda.gov/apis/drug/ndc/download/) using ndclib conversion; rerun only when changing the drug set.
- `prepare_database_for_extraction.py`: Creates helper tables and inserts the NDC list required by the extraction queries.
- `extract_icd10_codes.py`: Pulls ICD-10 claim records around spell start/change windows to support AE and diagnosis postprocessing.
- `test_connection.py`: Verifies database connectivity and credentials.

### Data processing scripts - `src/spell_generation`
#### `spell_generation.py`
Builds polypharmacy spells and AE labels from the extracted Parquet files.

- Inputs: `rx_fills`, `adverse_events`, `enrollment`, and `demographics` Parquet files plus a query drug list (default `data/opioid_ndc11_list.csv`); key args include `--min_concurrent`, `--grace_period`, `--min_spell_len`, `--input_suffix`, and `--output_suffix`.
- Spell detection: merges overlapping fill windows per `atc_3_code`, tags fills in the query drug list, and starts a spell when concurrent count >= threshold with at least one query drug; ends after `grace_period` days below threshold and keeps spells meeting the minimum length (runs in parallel across members).
- Censoring: retains spells overlapping enrollment and with >=180 days of enrollment before entry.
- AE labeling: shifts AE dates one day earlier to avoid same-day drug-change clashes, keeps spells whose first AE code in-window is new relative to prior history, and stores AE flags/dates/codes.
- Outputs: `spells_with_labels<suffix>.parquet` and `drug_changes<suffix>.parquet` with spell ids and add/drop events.

#### `split_spells_from_changes.py`
Splits each spell at drug add/drop events, aligning timelines so changes start at t=0 for downstream modeling.

#### `drug_combo_labeling.py`
Attaches the concurrent drug combination present at each split-spell start using the `drug_changes` output.

#### `icd10_postprocessing.py`
Clusters extracted ICD-10 codes into a `_clustered` Parquet file ready for analysis.

### Modeling - `src/modelling`
- `final_preprocessing.py`: Merges spells, demographics, and clustered ICD-10 features, builds AE labels within a fixed window, and deduplicates member + drug combos.
- `drug_pair_analysis.py`: Counts drug-pair occurrences and runs chi-square tests versus AE outcomes with multiple-testing correction.
- `logreg.py` / `logreg.sbatch` / `logreg.ipynb`: Logistic regression baseline on multi-hot drug features with scripts/notebook and an O2 sbatch wrapper.
- `mlp_medtok_base.py`: Trains an AE classifier using MedTok embeddings for drugs/diagnoses with an MLP backbone.
- `mlp_pooled_embeddings.py` / `mlp.sbatch`: Variant MLP using pooled embeddings and weighted sampling; sbatch for cluster runs.
- `xgboost_trainer.py` / `xgboost.sbatch` / `xgboost.ipynb`: Gradient-boosted tree baseline with script, tuning notebook, and sbatch config.

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

2. Run this query (``src/sql_extraction/elegible_members_query.sql``) directly in SQL server (it can take a few hours, on the 5M db it took 4h30)
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

6. Manually run the ICD10 extraction from SQL (it can't be easily chained in a job due to the kinit command, and if it takes longer than a few seconds to start it means the db is locked by someone else. Recommend running at night when no one is using the db):
```bash
python src/sql_extraction/extract_icd10_codes.py --suffix "WHATEVER YOU USED IN THE sbatch FILE"
```

7. Run the ICD10 postprocessing (is quite fast):
```bash
python src/sql_extraction/icd10_postprocessing.py --suffix "WHATEVER YOU USED IN THE sbatch FILE"
```

8. With this, you are ready to run the exploration notebook, train models, etc. Be sure to use the final_preprocessing script on top of whatever you do to get the final features and apply the necessary filters for data leakage prevention and quality control.

NOTE: in the future, it would be ideal to get some way of embedding the authentication token into the jobs, so the entire pipeline can be run in a single sbatch job, allowing for easier reproducibility and automation.
