#!/usr/bin/env python3
import pandas as pd
import requests
import argparse
from tqdm import tqdm

# --- Step 1: Get RxCUI from NDC ---
def fetch_rxcui_for_ndc(ndc_code, session, cache_rxcui):
    """Query RxNav getNDCProperties API and return RxCUI for a given NDC code."""
    if pd.isna(ndc_code):
        return None

    ndc = str(ndc_code).strip()
    if ndc == "":
        return None

    if ndc in cache_rxcui:
        return cache_rxcui[ndc]

    url = f"https://rxnav.nlm.nih.gov/REST/ndcproperties.json?id={ndc}&ndcstatus=ALL"
    try:
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            cache_rxcui[ndc] = None
            return None

        data = response.json()
        ndc_property_list = data.get("ndcPropertyList", {})
        ndc_properties = ndc_property_list.get("ndcProperty", [])

        if isinstance(ndc_properties, dict):
            ndc_properties = [ndc_properties]
        if not ndc_properties:
            cache_rxcui[ndc] = None
            return None

        rxcui = ndc_properties[0].get("rxcui")
        cache_rxcui[ndc] = rxcui
        return rxcui

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching RxCUI for {ndc}: {e}")
        cache_rxcui[ndc] = None
        return None


# --- Step 2: Get concept name from RxCUI ---
def fetch_concept_name(rxcui, session, cache_name):
    """Query RxNav getRxNormName API to get the human-readable concept name."""
    if pd.isna(rxcui) or not str(rxcui).strip():
        return None

    rxcui = str(rxcui).strip()
    if rxcui in cache_name:
        return cache_name[rxcui]

    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json?propName=RxNorm%20Name"
    try:
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            cache_name[rxcui] = None
            return None

        data = response.json()
        prop_concept_group = data.get("propConceptGroup", {})
        properties = prop_concept_group.get("propConcept", [])
        if isinstance(properties, dict):
            properties = [properties]
        if not properties:
            cache_name[rxcui] = None
            return None

        concept_name = properties[0].get("propValue")
        cache_name[rxcui] = concept_name
        return concept_name

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching name for RxCUI {rxcui}: {e}")
        cache_name[rxcui] = None
        return None


# --- Step 3: Get ingredient-level (IN) RxCUI and name ---
def fetch_ingredient_from_rxcui(rxcui, session, cache_in):
    """Return (IN RxCUI, IN Name) for a given RxCUI."""
    if pd.isna(rxcui) or not str(rxcui).strip():
        return None, None

    rxcui = str(rxcui).strip()
    if rxcui in cache_in:
        return cache_in[rxcui]

    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN+PIN"
    try:
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            cache_in[rxcui] = (None, None)
            return None, None

        data = response.json()
        groups = data.get("relatedGroup", {}).get("conceptGroup", [])
        for g in groups:
            if g.get("tty") in ("IN", "PIN"):
                props = g.get("conceptProperties", [])
                if isinstance(props, dict):
                    props = [props]
                if props:
                    in_rxcui = props[0].get("rxcui")
                    in_name = props[0].get("name")
                    cache_in[rxcui] = (in_rxcui, in_name)
                    return in_rxcui, in_name

        cache_in[rxcui] = (None, None)
        return None, None

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching IN for RxCUI {rxcui}: {e}")
        cache_in[rxcui] = (None, None)
        return None, None


# --- Step 4: Main driver ---
def main(input_csv, output_csv):
    df = pd.read_csv(input_csv, dtype=str)
    if "ndc11code" not in df.columns:
        raise ValueError("Input CSV must have a column named 'ndc11code'")

    session = requests.Session()
    cache_rxcui, cache_name, cache_in = {}, {}, {}

    tqdm.pandas(desc="Fetching RxCUI from NDC")
    df["RxNorm"] = df["ndc11code"].progress_apply(lambda x: fetch_rxcui_for_ndc(x, session, cache_rxcui))

    tqdm.pandas(desc="Fetching RxNorm Name")
    df["RxNormName"] = df["RxNorm"].progress_apply(lambda x: fetch_concept_name(x, session, cache_name))

    tqdm.pandas(desc="Fetching Ingredient-level RxCUI and Name")
    ingredient_results = df["RxNorm"].progress_apply(lambda x: fetch_ingredient_from_rxcui(x, session, cache_in))
    df["RxNorm_IN"], df["RxNorm_IN_Name"] = zip(*ingredient_results)

    df.to_csv(output_csv, index=False)
    print(f"✅ Done. Output saved to: {output_csv}")


# --- CLI Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append RxNorm (RxCUI + Name + Ingredient Level) based on NDC11 codes via RxNav APIs"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/n/scratch/users/y/yul736/BMIF204/Inovalon/Data/rx_fills_subset.csv",
        help="Input CSV path (must include ndc11code column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/n/scratch/users/y/yul736/BMIF204/Inovalon/Data/rx_fills_with_rxnorm_and_generic.csv",
        help="Output CSV path (adds RxNorm, RxNormName, RxNorm_IN, RxNorm_IN_Name columns)",
    )
    args = parser.parse_args()
    main(args.input, args.output)
