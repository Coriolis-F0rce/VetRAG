"""VetRAG fixtures"""
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parents[1]
# __file__ = D:\Backup\PythonProject2\VetRAG\tests\conftest.py
# parents[0] = VetRAG/tests/, [1] = VetRAG/, [2] = PythonProject2/
# project_root = D:\Backup\PythonProject2\VetRAG
# src/ lives at project_root / "src"
# data/ lives at project_root / "data"
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def behaviors_file(data_dir):
    return data_dir / "behaviors.json"


@pytest.fixture
def breeds_file(data_dir):
    return data_dir / "breeds.json"


@pytest.fixture
def cares_file(data_dir):
    return data_dir / "cares.json"


@pytest.fixture
def diseases_file(data_dir):
    return data_dir / "diseases.json"


@pytest.fixture
def surgeries_file(data_dir):
    return data_dir / "surgeries.json"


@pytest.fixture
def sample_behavior_data():
    return {
        "behavior": {
            "name": "测试行为",
            "category": "测试类别",
            "description": "描述",
            "antecedents": "前因",
            "consequences": "后果",
            "meaning": "含义",
            "evaluation": "评估",
            "function": "功能",
            "intervention_level": "无/观察",
            "resource": "资源"
        }
    }


@pytest.fixture
def sample_disease_data():
    return {
        "disease_name": "犬瘟热",
        "disease_type": "病毒性",
        "disease_category": "传染病",
        "urgency_level": 3,
        "zoonotic": False,
        "affected_species": ["犬"],
        "symptoms": {
            "primary": ["发热", "咳嗽"],
            "secondary": ["食欲不振"]
        },
        "treatment": [
            {"name": "支持治疗", "category": "常规", "drug": "", "dosage": ""}
        ],
        "prognosis": "预后不良",
        "prevention": ["疫苗接种"]
    }


@pytest.fixture
def sample_breed_data():
    return {
        "chinese_name": "金毛寻回犬",
        "english_name": "Golden Retriever",
        "akc_group": "运动犬组",
        "size_category": "大型",
        "average_weight_kg": {"male": "29-34kg", "female": "25-29kg"},
        "life_expectancy": "10-12年",
        "coat_type": "双层被毛",
        "primary_traits": ["友善", "耐心", "聪明"],
        "energy_level": 8,
        "intelligence_rank": 4,
        "common_health_issues": ["髋关节发育不良", "眼部疾病"]
    }


@pytest.fixture
def sample_care_data():
    return {
        "vaccine_schedule": {
            "data": [
                {
                    "age": "6-8周",
                    "vaccine_name": "DHPP",
                    "disease_protected": ["犬瘟热"],
                    "importance": "核心",
                    "notes": "首次接种",
                    "repeat_interval": "每3-4周"
                }
            ]
        }
    }
