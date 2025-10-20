import json
import csv
from ndclib import NDC #https://github.com/eddie-cosma/ndclib

# --- STEP 1: Load your openFDA NDC JSON file ---
with open("drug-ndc-0001-of-0001.json", "r", encoding="utf-8") as f:
    ndc_data = json.load(f)
# Download from https://open.fda.gov/apis/drug/ndc/download/

results = ndc_data["results"]

# --- STEP 2: Define opioid ingredient/class keywords ---
opioid_terms = {
    "morphine", "oxycodone", "hydrocodone", "hydromorphone", "oxymorphone",
    "codeine", "fentanyl", "tapentadol", "tramadol", "meperidine",
    "methadone", "buprenorphine", "butorphanol", "nalbuphine",
    "pentazocine", "levorphanol", "opium", "dihydrocodeine"
}

# --- STEP 3: Helper to detect opioids ---
def is_opioid(drug):
    for ingr in drug.get("active_ingredients", []):
        if any(term in ingr["name"].lower() for term in opioid_terms):
            return True
    for field in ("pharm_class", "pharm_class_cs", "pharm_class_epc"):
        for cls in drug.get(field, []):
            if any(term in cls.lower() for term in opioid_terms):
                return True
    return False

# --- STEP 4: Collect opioid package-level NDCs ---
opioid_records = []

for drug in results:
    if is_opioid(drug):
        brand = drug.get("brand_name")
        strength = ", ".join(f"{i['name']} {i['strength']}" for i in drug.get("active_ingredients", []))
        route = ", ".join(drug.get("route", []))
        for pkg in drug.get("packaging", []):
            raw_ndc = pkg.get("package_ndc")
            if not raw_ndc:
                continue
            try:
                ndc_obj = NDC(raw_ndc)
                ndc11 = ndc_obj.to_11()  # 11-digit, no hyphens
                opioid_records.append({
                    "ndc11": ndc11,
                    "brand_name": brand,
                    "strength": strength,
                    "route": route,
                    "labeler_name": drug.get("labeler_name")
                })
            except Exception as e:
                # skip invalid or UPC-like codes
                continue

# --- STEP 5: Deduplicate and sort ---
seen = set()
unique_opioids = []
for rec in opioid_records:
    if rec["ndc11"] not in seen:
        seen.add(rec["ndc11"])
        unique_opioids.append(rec)

# --- STEP 6: Output summary ---
print(f"Found {len(unique_opioids)} unique opioid NDC11s.")
print("Example:", [r["ndc11"] for r in unique_opioids[:10]])

# --- STEP 7: Save to CSV (optional) ---
with open("opioid_ndc11_list.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ndc11", "brand_name", "strength", "route", "labeler_name"])
    writer.writeheader()
    writer.writerows(unique_opioids)

print("Saved to opioid_ndc11_list.csv")
