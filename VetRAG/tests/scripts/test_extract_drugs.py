"""Golden file test for scripts/extract_drugs.py"""
import json
import sys
from collections import defaultdict
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.extract_drugs import normalize_drug_name, split_drug_names


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
MINI_DISEASES = FIXTURE_DIR / "mini_diseases.json"
GOLDEN_FILE = FIXTURE_DIR / "golden_extract_drugs.json"


def _extract_drugs_from_diseases(diseases: list) -> list[dict]:
    """核心提取逻辑（从 extract_drugs.main 提取，保持同步）。"""
    drug_map = defaultdict(lambda: {
        "drug_name": "", "categories": set(), "indications": [],
        "dosages": [], "routes": set(), "frequencies": set(),
        "durations": set(), "treatment_names": [],
    })

    for disease in diseases:
        d_name = disease.get("disease_name", "")
        for t in disease.get("treatment", []):
            drug_field = t.get("drug", "")
            if not drug_field:
                continue
            drug_names = split_drug_names(drug_field)
            dosage = t.get("dosage", "")
            route = t.get("route", "")
            freq = t.get("frequency", "")
            duration = t.get("duration", "")
            category = t.get("category", "")
            treatment_name = t.get("name", "")

            for dn in drug_names:
                dn = normalize_drug_name(dn)
                if len(dn) < 2:
                    continue
                entry = drug_map[dn]
                entry["drug_name"] = dn
                if category:
                    entry["categories"].add(category)
                if d_name and d_name not in entry["indications"]:
                    entry["indications"].append(d_name)
                if dosage and dosage not in entry["dosages"]:
                    entry["dosages"].append(dosage)
                if route:
                    entry["routes"].add(route)
                if freq:
                    entry["frequencies"].add(freq)
                if duration:
                    entry["durations"].add(duration)
                if treatment_name and treatment_name not in entry["treatment_names"]:
                    entry["treatment_names"].append(treatment_name)

    output = []
    for name in sorted(drug_map.keys()):
        entry = drug_map[name]
        output.append({
            "drug_name": entry["drug_name"],
            "drug_class": sorted(entry["categories"]),
            "indications": entry["indications"],
            "dosages": entry["dosages"],
            "routes": sorted(entry["routes"]),
            "frequencies": sorted(entry["frequencies"]),
            "durations": sorted(entry["durations"]),
            "treatment_names": entry["treatment_names"],
        })
    return output


class TestExtractDrugsGolden:
    def test_split_drug_names_basic(self):
        assert split_drug_names("DrugA/DrugB") == ["DrugA", "DrugB"]
        assert split_drug_names("DrugA, DrugB") == ["DrugA", "DrugB"]
        assert split_drug_names("") == []
        # Single short char (< 2) is filtered out
        assert split_drug_names("X") == []

    def test_normalize_drug_name(self):
        assert normalize_drug_name("Amoxicillin-Clavulanate") == "Amoxicillin-Clavulanate"
        # Parentheses content is stripped
        result = normalize_drug_name("Balanced Solution (with additives)")
        assert result == "Balanced Solution"

    def test_golden_extraction(self):
        with open(MINI_DISEASES, encoding="utf-8") as f:
            diseases = json.load(f)
        with open(GOLDEN_FILE, encoding="utf-8") as f:
            expected = json.load(f)

        result = _extract_drugs_from_diseases(diseases)

        assert len(result) == len(expected), f"drug count: {len(result)} != {len(expected)}"
        for i, (r, e) in enumerate(zip(result, expected, strict=False)):
            assert r["drug_name"] == e["drug_name"], f"drug {i}: name mismatch"
            assert r["drug_class"] == e["drug_class"], f"drug {i}: class mismatch"
            assert r["dosages"] == e["dosages"], f"drug {i}: dosage mismatch"
            assert r["indications"] == e["indications"], f"drug {i}: indications mismatch"

    def test_golden_not_empty(self):
        with open(GOLDEN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0
        for drug in data:
            assert drug.get("drug_name"), "every drug must have a name"
