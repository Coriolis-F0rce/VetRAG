"""
测试 JSON 数据加载与解析模块
覆盖 src/json_loader.py 的所有解析方法
"""
import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parents[2]
# __file__ = D:\Backup\PythonProject2\VetRAG\tests\rag\test_json_loader.py
# parents[0] = VetRAG/tests/rag/, [1] = VetRAG/tests/, [2] = VetRAG/
# project_root = D:\Backup\PythonProject2\VetRAG
sys.path.insert(0, str(project_root))

from src.json_loader import VetRAGDataLoader


class TestVetRAGDataLoaderInit:
    def test_loader_initializes(self):
        loader = VetRAGDataLoader()
        assert hasattr(loader, "documents")
        assert hasattr(loader, "content_type_stats")
        assert isinstance(loader.documents, list)


class TestParseBehaviors:
    def test_parse_single_behavior(self, sample_behavior_data):
        loader = VetRAGDataLoader()
        chunks = loader._parse_behaviors([sample_behavior_data], "behaviors.json")
        assert len(chunks) == 1
        assert chunks[0]["content_type"] == "behaviors"
        assert "测试行为" in chunks[0]["content"]
        assert chunks[0]["metadata"]["behavior_name"] == "测试行为"
        assert chunks[0]["metadata"]["behavior_category"] == "测试类别"

    def test_parse_behavior_missing_fields(self):
        loader = VetRAGDataLoader()
        incomplete_data = [{"behavior": {"name": "只有关键字段"}}]
        chunks = loader._parse_behaviors(incomplete_data, "behaviors.json")
        assert len(chunks) == 1
        assert chunks[0]["content_type"] == "behaviors"

    def test_parse_behavior_empty_list(self):
        loader = VetRAGDataLoader()
        chunks = loader._parse_behaviors([], "behaviors.json")
        assert chunks == []

    def test_parse_behavior_real_file(self, behaviors_file):
        loader = VetRAGDataLoader()
        with open(behaviors_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = loader._parse_behaviors(data, str(behaviors_file))
        assert len(chunks) > 0
        assert all(c["content_type"] == "behaviors" for c in chunks)
        assert all("behavior_name" in c["metadata"] for c in chunks)


class TestParseBreeds:
    def test_parse_breed_direct(self, sample_breed_data):
        loader = VetRAGDataLoader()
        chunks = loader._parse_breeds(sample_breed_data, "breeds.json")
        assert len(chunks) == 1
        assert chunks[0]["content_type"] == "breeds"
        assert "金毛寻回犬" in chunks[0]["content"]
        assert chunks[0]["metadata"]["breed_name"] == "金毛寻回犬"

    def test_parse_breed_wrapped_in_breed_key(self):
        loader = VetRAGDataLoader()
        data = {
            "breed": {
                "chinese_name": "边牧",
                "english_name": "Border Collie",
                "akc_group": "牧羊犬组",
                "size_category": "中型"
            }
        }
        chunks = loader._parse_breeds(data, "breeds.json")
        assert len(chunks) == 1
        assert "边牧" in chunks[0]["content"]

    def test_parse_breed_wrapped_in_data_key(self):
        loader = VetRAGDataLoader()
        data = {
            "data": [
                {"breed": {"chinese_name": "柯基", "size_category": "小型"}}
            ]
        }
        chunks = loader._parse_breeds(data, "breeds.json")
        assert len(chunks) == 1

    def test_parse_breed_list_format(self):
        loader = VetRAGDataLoader()
        data = [
            {"chinese_name": "哈士奇", "size_category": "中型"},
            {"chinese_name": "拉布拉多", "size_category": "大型"}
        ]
        chunks = loader._parse_breeds(data, "breeds.json")
        assert len(chunks) == 2

    def test_parse_breed_empty_data(self):
        loader = VetRAGDataLoader()
        chunks = loader._parse_breeds({}, "breeds.json")
        assert isinstance(chunks, list)

    def test_parse_breed_real_file(self, breeds_file):
        loader = VetRAGDataLoader()
        with open(breeds_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = loader._parse_breeds(data, str(breeds_file))
        assert len(chunks) > 0
        assert all(c["content_type"] == "breeds" for c in chunks)


class TestParseDiseases:
    def test_parse_disease_basic(self, sample_disease_data):
        loader = VetRAGDataLoader()
        # 包装在 diseases 键下以匹配代码逻辑
        wrapped_data = {"diseases": [sample_disease_data]}
        chunks = loader._parse_diseases(wrapped_data, "diseases.json")
        assert len(chunks) > 0
        assert all(c["content_type"] == "diseases_professional" for c in chunks)
        assert any("犬瘟热" in c["content"] for c in chunks)

    def test_parse_disease_list_format(self):
        loader = VetRAGDataLoader()
        data = [
            {"disease_name": "测试病1", "symptoms": {"primary": ["发烧"]}},
            {"disease_name": "测试病2", "symptoms": {"primary": ["咳嗽"]}}
        ]
        chunks = loader._parse_diseases(data, "diseases.json")
        assert len(chunks) >= 2

    def test_parse_disease_wrapped_in_diseases_key(self):
        loader = VetRAGDataLoader()
        data = {
            "diseases": [
                {"disease_name": "犬细小病毒", "symptoms": {"primary": ["呕吐"]}}
            ]
        }
        chunks = loader._parse_diseases(data, "diseases.json")
        assert len(chunks) > 0

    def test_parse_disease_empty_list(self):
        loader = VetRAGDataLoader()
        chunks = loader._parse_diseases([], "diseases.json")
        assert chunks == []

    def test_parse_disease_real_file(self, diseases_file):
        loader = VetRAGDataLoader()
        with open(diseases_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = loader._parse_diseases(data, str(diseases_file))
        assert len(chunks) > 0


class TestParseCare:
    def test_parse_vaccine_schedule(self, sample_care_data):
        loader = VetRAGDataLoader()
        chunks = loader._parse_vaccine_schedule(sample_care_data["vaccine_schedule"], "cares.json")
        assert len(chunks) == 1
        assert "DHPP" in chunks[0]["content"]
        assert chunks[0]["metadata"]["content_type"] == "cares"

    def test_parse_cleaned_dog_care_basic(self):
        loader = VetRAGDataLoader()
        data = {
            "results": {
                "vaccine_schedule": {
                    "data": [
                        {"age": "10-12周", "vaccine_name": "狂犬病", "disease_protected": ["狂犬病"], "importance": "核心", "notes": "", "repeat_interval": ""}
                    ]
                },
                "daily_care": {
                    "care_categories": [
                        {
                            "category": "毛发护理",
                            "frequency": "每周",
                            "procedures": ["刷毛"],
                            "tips": ["使用耙梳"]
                        }
                    ]
                }
            }
        }
        chunks = loader._parse_cleaned_dog_care(data, "cares.json")
        assert len(chunks) > 0
        assert all(c["content_type"] == "cares" for c in chunks)

    def test_parse_joint_care_guide(self):
        loader = VetRAGDataLoader()
        data = {
            "content_type": "老年犬关节护理指南",
            "common_joint_problems": [
                {
                    "problem": "关节炎",
                    "symptoms": ["跛行"],
                    "risk_factors": ["年龄"]
                }
            ],
            "daily_care_recommendations": {"exercise": "适度运动"},
            "nutritional_supplements": [
                {"supplement": "软骨素", "benefits": ["保护关节"], "dosage_notes": "按体重"}
            ],
            "when_to_see_vet": ["跛行超过3天"],
            "data_sources": ["VCA动物医院"]
        }
        chunks = loader._parse_joint_care_guide(data, "cares.json")
        assert len(chunks) == 2
        assert chunks[0]["content_type"] == "cares"
        assert chunks[1]["content_type"] == "cares"

    def test_parse_real_care_file(self, cares_file):
        loader = VetRAGDataLoader()
        with open(cares_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = loader._parse_cleaned_dog_care(data, "cares.json")
        assert len(chunks) > 0


class TestParseSurgeries:
    def test_parse_surgery_basic(self):
        loader = VetRAGDataLoader()
        data = {
            "data": [
                {
                    "surgery": {
                        "chinese_name": "绝育手术",
                        "english_name": "Spay/Neuter",
                        "category": "常规手术",
                        "indications": ["防止繁殖"],
                        "preoperative_preparation": ["禁食12小时"],
                        "postoperative_care": ["观察恢复"],
                        "prognosis": "良好"
                    }
                }
            ]
        }
        chunks = loader._parse_surgeries(data, "surgeries.json")
        assert len(chunks) == 2
        assert all(c["content_type"] == "surgeries" for c in chunks)
        assert any("绝育手术" in c["content"] for c in chunks)

    def test_parse_surgery_list_without_wrapper(self):
        loader = VetRAGDataLoader()
        data = [
            {"chinese_name": "骨折修复", "category": "骨科手术"}
        ]
        chunks = loader._parse_surgeries(data, "surgeries.json")
        assert len(chunks) >= 1

    def test_parse_surgery_real_file(self, surgeries_file):
        loader = VetRAGDataLoader()
        with open(surgeries_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = loader._parse_surgeries(data, str(surgeries_file))
        assert len(chunks) > 0


class TestParseGeneric:
    def test_parse_generic_with_dict(self):
        loader = VetRAGDataLoader()
        data = {"key": "value", "nested": {"a": 1}}
        chunks = loader._parse_generic(data, "unknown.json")
        assert len(chunks) == 1
        assert chunks[0]["content_type"] == "generic"
        assert chunks[0]["source_file"] == "unknown.json"

    def test_parse_generic_unknown_filename(self):
        loader = VetRAGDataLoader()
        data = {"results": "some data"}
        chunks = loader._parse_file_based_on_type("unknown.json", data, "unknown.json")
        assert len(chunks) >= 1


class TestLoadAllFiles:
    def test_load_nonexistent_file_raises(self):
        loader = VetRAGDataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_all_files(["nonexistent_file_12345.json"])

    def test_load_invalid_json_raises(self, tmp_path):
        loader = VetRAGDataLoader()
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json }", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_all_files([str(bad_file)])

    def test_load_all_data_files(self, behaviors_file, breeds_file, cares_file, diseases_file, surgeries_file):
        loader = VetRAGDataLoader()
        chunks = loader.load_all_files([
            str(behaviors_file),
            str(breeds_file),
            str(cares_file),
            str(diseases_file),
            str(surgeries_file)
        ])
        assert len(chunks) > 0
        content_types = set(c.get("content_type", "") for c in chunks)
        assert len(content_types) > 1
