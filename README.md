# Inovalon project: polypharmacy and AEs
This project explores the risk factors in polypharmacy patients that lead to a higher incidence of AEs.

## Usage
### Data extraction
The data is extracted raw from SQL for processing in Python. The selection of ICD codes was done based on this paper https://pmc.ncbi.nlm.nih.gov/articles/PMC3994866, focusing on drugs that had A1, A2, B1, B2 or C class. There are 2 scripts, built in a similar structure:

#### ``extract_raw_sql_opioid.py``
This script connects to the Inovalon SQL Server and extracts member-level data related to opioid prescriptions and adverse events for downstream analysis.

The script builds a cohort of eligible members based on prescription and claim data, filtering for members with:
* At least one opioid prescription (has_opioid = 1)
* At least 3 distinct drug fills (total_drugs >= 3)

It then exports multiple datasets as Parquet files for later processing.

Some notable steps:
* ICD-10 Filter Loading: Reads icd10_codes.csv to build a SQL WHERE clause matching relevant diagnosis codes for adverse event (AE) filtering.
* Opioid NDC Loading: Reads opioid_ndc11_list.csv to create chunked SQL IN clauses identifying opioid drugs.
* Cohort Definition: Executes a CTE (base_cte_create) to populate a temporary table #eligible_members containing qualified subjects.
* Data Extraction:Runs and saves four queries for:
    * rx_fills_opioid*.parquet: Prescription fills
    * adverse_events_opioid*.parquet: AE claims
    * demographics_opioid*.parquet: Member demographics
    * enrollment_opioid*.parquet: Enrollment periods

Each query is also saved to the queries/ folder for transparency.

#### ``extract_raw_sql.py``
Same as above, but for polypharmacy patients. Some rellevant steps:
* ICD-10 Code Loading: Reads icd10_codes.csv and dynamically builds an SQL WHERE clause of CodeValue LIKE filters for identifying adverse event diagnoses.
* Cohort Definition (CTE): Builds a layered SQL common table expression (base_cte) that:
    * Identifies each member’s earliest fill dates per NDC.
    * Ranks drug fills chronologically per member.
    * Selects the 5th filled drug as the polypharmacy index date.
    * Filters to members aged 65 or older at index.
    * Links enrollment data to ensure eligibility (6-month continuous enrollment checked later in Python).
* Data Extraction:Runs and saves four queries for:
    * rx_fills*.parquet: Prescription fills
    * adverse_events*.parquet: AE claims
    * demographics*.parquet: Member demographics
    * enrollment*.parquet: Enrollment periods

Each query is also saved to the queries/ folder for transparency.

#### ``opioid_ndcs.py``
This script uses the OpenFDA database (Download from https://open.fda.gov/apis/drug/ndc/download/) to extract a CSV with all the rellevant NDC codes, and uses a library (#https://github.com/eddie-cosma/ndclib) to convert between NDC10 and NDC11.

Only need to be rerun if the list of opioids needs to be modified.

#### `test.py`
Tests connectivity to SQL

### Data processing:
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

### Whole pipeline execution
First, extract data from SQL. This process is a bit tricky. 

1. Insert the query drugs (in our study, the opioid ndcs) into a table:
```bash
python src/sql_extraction/prepare_database_for_insertion --db 5
```

2. Run this query directly in SQL server (it can take a few hours, on the 5M db it took 4h30)
```sql
SELECT
    r.MemberUID
INTO bef299.dbo.elegible_members
FROM RxClaim r
LEFT JOIN dbo.OpioidNdc o
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

4. Generate spells and labels. Modify the pipeline.sbatch file as needed, making sure to add a rellevant suffix, and submit via (it takes about 30 mins on the 1M sample, a lot more on the 5M, be sure to ask enough memory to avoid OOM error, although the code already does many memory management tricks):
```bash
sbatch src/spell_generation/pipeline.sbatch
```

5. Manually run the ICD10 extraction from SQL (it can't be run in a job, and if it takes longer than a few seconds to start it means the db is locked by someone else. Recommend running at night when no one is using it):
```bash
python src/sql_extraction/extract_icd10_codes.py --suffix "WHATEVER YOU USED IN THE sbatch FILE"
```

6. Run the ICD10 postprocessing (is quite fast):
```bash
python src/icd10_postprocessing.py --suffix "WHATEVER YOU USED IN THE sbatch FILE"
```

You are ready to analyse and run analysis!