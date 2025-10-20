# 🧾 Data Extraction: Cohort & Events from Inovalon Claims

## Overview

This stage extracts **raw data tables** from the Inovalon claims database to support downstream **polypharmacy spell detection** and **adverse event labeling**. The goal is to build a structured, analysis-ready dataset in Parquet format, using reproducible SQL queries and dynamic filtering for ICD-10 adverse event codes.

---

## 📊 SQL Extraction Logic

### 1. **Identifying the Eligible Cohort**

We identify the eligible study population using a series of SQL Common Table Expressions (CTEs):

* **`first_fills`**
  For each patient and each NDC11 drug code, we find the earliest recorded fill date. We exclude fills without `supplydayscount`.

* **`ranked_drugs`**
  Drugs are ordered by their first fill date per patient.

* **`polypharmacy_index`**
  Defines the **index date** as the date of the **5th distinct drug fill**, used as a proxy for entering polypharmacy.

* **`eligible_members`**
  Keeps only patients who:

  * Were **≥65 years old** on their index date (calculated from birth year)
  * Have at least 5 distinct drugs on record (polypharmacy)

  These form the **base study cohort**.

---

### 2. **Extracting Prescription Fills**

The `rx_filtered` CTE retrieves all prescription fills for eligible members:

* Includes **fill date**, **NDC11 code**, and **supplydayscount**.
* Time window: **180 days before** index to **1 year after** index.
* These data are used to reconstruct **polypharmacy spells** in Python (i.e., intervals during which ≥5 distinct drugs are concurrently active).

The output is saved to:

```
rx_fills.parquet
```

---

### 3. **Extracting Adverse Events**

The `adverse_events` CTE retrieves **ICD-10 diagnosis codes** from claims that correspond to **adverse events of interest**, joined to the eligible cohort. We based our selection on this paper https://pmc.ncbi.nlm.nih.gov/articles/PMC3994866.

* **ICD-10 codes are loaded from a CSV** (`icd10_codes.csv`) that lists relevant codes and descriptions (e.g., drug poisonings, confusion, nephropathy, liver toxicity).
* The code list is converted into a dynamic `WHERE` clause using `LIKE` statements to capture all children of prefix codes (e.g., `T36%` matches T36.0, T36.1, etc.).
* Only **ICD-10 code types** (`CodeType` ∈ 17, 18, 22, 23, 24) are used.
* No ICD-9 codes are included.

The output is saved to:

```
adverse_events.parquet
```

This file will later be used to identify **whether an adverse event occurs during each polypharmacy spell**, and can be expanded or reduced easily via the CSV file.

---

### 4. **Demographics and Enrollment**

For each eligible patient, we also extract:

* **Demographics** (birthyear, gender, race, ZIP, state) from the `Member` table
  → `demographics.parquet`

* **Enrollment periods** (effective and termination dates) from `MemberEnrollment`
  → `enrollment.parquet`

These will be used for baseline features and censoring logic.

---

## 📥 Output Files

After running the Python extraction script, the following Parquet files are created in the scratch directory:

```
 /n/scratch/users/b/bef299/polypharmacy_project/
    ├── rx_fills.parquet
    ├── adverse_events.parquet
    ├── demographics.parquet
    └── enrollment.parquet
```

Additionally, a small `rx_fills_subset.csv` file is generated for quick testing and debugging.



## 🧠 Next Steps: Polypharmacy Spell Reconstruction

The next step is to move to **Python-based temporal processing**, using the extracted Rx fills and AE events:

1. **Medication Interval Construction**

   * Convert each Rx fill to an interval `[fill_date, fill_date + supplydayscount - 1]`.
   * Stack all intervals per patient.

2. **Daily Drug Count Timeline**

   * For each patient, construct a timeline of **distinct active drugs per day**.
   * Identify **entry dates** = first day with ≥5 concurrent medications.

3. **Spell Detection**

   * **Exit date** = last day before the patient drops below 5 concurrent meds and remains below for 15 consecutive days.
   * Extend the spell by **15 days after exit**.
   * Minimum spell length = **30 days** (to avoid spurious short periods).

4. **Multiple Spells per Patient**

   * Allow patients to **re-enter polypharmacy** after a gap and detect multiple spells.
   * Each spell is treated as a separate observation.

5. **AE Labeling**

   * For each spell, check if **any AE occurs between entry date and exit+15 days**.
   * AE events outside of these intervals are ignored.

6. **Censoring**

   * Spell follow-up ends at `exit_date + 15 days` or at disenrollment, whichever comes first.
