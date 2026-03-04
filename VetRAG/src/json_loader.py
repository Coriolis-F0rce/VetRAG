"""兽医RAG系统架构：1. 数据加载模块 - 加载所有JSON文件2. 数据解析模块 - 解析不同格式的JSON结构3. 文本分块模块 - 将长文本分割为适合检索的块4. 向量化模块 - 使用嵌入模型生成向量5. 向量存储模块 - 存储和检索向量6. 检索模块 - 相似度搜索和结果融合7. 接口模块 - 提供查询接口"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    source_file: str = ""
    content_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    content_hash: str = ""

class VetRAGDataLoader:
    """兽医RAG数据加载器"""

    def __init__(self):
        self.documents = []
        self.content_type_stats = {}

    def load_all_files(self, file_paths: List[str]) -> List[Dict]:
        """加载并解析所有JSON文件"""
        all_chunks = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                chunks = self._parse_file_based_on_type(os.path.basename(file_path), data, file_path)
                all_chunks.extend(chunks)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
        return all_chunks

    def _parse_file_based_on_type(self, filename: str, data: Dict, file_path: str) -> List[Dict]:
        filename_lower = filename.lower()
        if "cleaned_dog_care_data" in filename_lower or "cares.json" in filename_lower:
            return self._parse_cleaned_dog_care(data, file_path)
        elif "diseases" in filename_lower:
            return self._parse_diseases(data, file_path)
        elif "breeds" in filename_lower:
            return self._parse_breeds(data, file_path)
        elif "behavior" in filename_lower:
            return self._parse_behaviors(data, file_path)
        elif "surgery" in filename_lower or "surgeries" in filename_lower:
            return self._parse_surgeries(data, file_path)
        else:
            # 通用解析
            return self._parse_generic(data, file_path)

    def _parse_generic(self, data: Dict, source_file: str) -> List[Dict]:
        chunks = []
        try:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            chunk = {
                "content": content,  # 移除长度限制
                "metadata": {"content_type": "generic", "source": source_file},
                "source_file": source_file,
                "content_type": "generic"
            }
            chunks.append(chunk)
        except:
            pass
        return chunks

    def _parse_cleaned_dog_care(self, data: Dict, source_file: str) -> List[Dict]:
        chunks = []

        if not isinstance(data.get("results"), dict):
            return chunks

        results = data["results"]

        if "vaccine_schedule" in results:
            vaccine_chunks = self._parse_vaccine_schedule(results["vaccine_schedule"], source_file)
            chunks.extend(vaccine_chunks)

        if "joint_care_guide" in results:
            joint_care_chunks = self._parse_joint_care_guide(results["joint_care_guide"], source_file)
            chunks.extend(joint_care_chunks)

        daily_care = results.get("daily_care", {})
        if isinstance(daily_care, dict) and "care_categories" in daily_care:
            for category in daily_care["care_categories"]:
                category_name = category.get("category", "")
                frequency = category.get("frequency", "")

                procedures = category.get("procedures", [])
                procedures_text = "\n".join([f"  {i+1}. {proc}" for i, proc in enumerate(procedures)]) if procedures else "暂无步骤说明"

                tips = category.get("tips", [])
                tips_text = "\n".join([f"  - {tip}" for tip in tips]) if tips else "暂无小贴士"

                content = f"""
护理类别: {category_name}
频率: {frequency}

步骤:
{procedures_text}

小贴士:
{tips_text}
"""
                metadata = {
                    "content_type": "cares",
                    "category": "daily_care",
                    "sub_category": category_name,
                    "source_file": source_file
                }

                chunks.append({
                    "content": content,
                    "metadata": metadata,
                    "source_file": source_file,
                    "content_type": "cares"
                })

        dog_regulations = results.get("dog_regulations", {})
        if isinstance(dog_regulations, dict) and "cities" in dog_regulations:
            for city in dog_regulations["cities"]:
                city_name = city.get("city_name", "")
                regulation_name = city.get("regulation_name", "")
                key_points = city.get("key_points", {})
                registration_requirements = key_points.get("registration_requirements", [])
                vaccination_requirements = key_points.get("vaccination_requirements", [])
                walking_restrictions = key_points.get("walking_restrictions", [])
                prohibited_breeds = key_points.get("prohibited_breeds", [])
                penalty_standards = key_points.get("penalty_standards", [])
                data_source = city.get("data_source", "")

                breeds_text = "\n".join([f"  - {breed}" for breed in prohibited_breeds]) if prohibited_breeds else "无禁养犬种"

                content = f"""
城市: {city_name}
相关规定: {regulation_name}
主要内容:
注册要求: {registration_requirements}
疫苗接种要求: {vaccination_requirements}
遛狗限制: {walking_restrictions}
禁止犬种:
{breeds_text}
惩罚措施: {penalty_standards}
数据来源: {data_source}
"""
                metadata = {
                    "content_type": "cares",
                    "category": "dog_regulations",
                    "sub_category": city_name,
                    "source_file": source_file
                }

                chunks.append({
                    "content": content,
                    "metadata": metadata,
                    "source_file": source_file,
                    "content_type": "cares"
                })

        return chunks

    def _parse_vaccine_schedule(self, vaccine_data: Dict, source_file: str) -> List[Dict]:
        chunks = []

        if not isinstance(vaccine_data, dict):
            return chunks

        # 疫苗数据列表
        vaccines = vaccine_data.get("data", [])
        if not isinstance(vaccines, list):
            return chunks

        general_notes = vaccine_data.get("general_notes", "")
        data_sources = vaccine_data.get("data_sources", [])
        last_updated = vaccine_data.get("last_updated", "")

        for i, vaccine in enumerate(vaccines):
            if not isinstance(vaccine, dict):
                continue

            age = vaccine.get("age", "")
            vaccine_name = vaccine.get("vaccine_name", "")
            disease_protected = vaccine.get("disease_protected", [])
            importance = vaccine.get("importance", "")
            notes = vaccine.get("notes", "")
            repeat_interval = vaccine.get("repeat_interval", "")

            diseases_text = "\n".join([f"  - {disease}" for disease in disease_protected]) if disease_protected else "暂无信息"

            content = f"""
疫苗名称: {vaccine_name}
疫苗阶段: {age}
重要性: {importance}
保护疾病:
{diseases_text}
注意事项: {notes}
重复接种间隔: {repeat_interval}

通用说明和注意事项:
{general_notes}

数据来源: {', '.join(data_sources) if data_sources else '暂无来源'}
最后更新: {last_updated}
"""
            metadata = {
                "content_type": "cares",
                "category": "vaccine_schedule",
                "sub_category": f"疫苗",
                "vaccine_type": vaccine_name,
                "age_group": age,
                "importance_level": importance,
                "source_file": source_file
            }

            chunks.append({
                "content": content,
                "metadata": metadata,
                "source_file": source_file,
                "content_type": "cares"
            })

        return chunks

    def _parse_joint_care_guide(self, joint_care_data: Dict, source_file: str) -> List[Dict]:
        """解析关节护理指南 - 分为2-3个语义块"""
        chunks = []

        if not isinstance(joint_care_data, dict):
            return chunks

        # 基础元数据（所有chunk共享）
        base_metadata = {
            "content_type": "cares",
            "category": "joint_care_guide",
            "source_file": source_file,
            "guide_type": joint_care_data.get("content_type", "关节护理指南")
        }

        # ========== CHUNK 1: 常见关节问题 + 日常护理建议 ==========
        chunk1_parts = []

        # 标题
        guide_title = joint_care_data.get("content_type", "老年犬关节护理指南")
        chunk1_parts.append(f"【{guide_title}】")

        # 常见关节问题
        common_problems = joint_care_data.get("common_joint_problems", [])
        if isinstance(common_problems, list) and common_problems:
            chunk1_parts.append("\n一、常见关节问题:")
            for i, problem in enumerate(common_problems, 1):
                if not isinstance(problem, dict):
                    continue

                problem_name = problem.get("problem", "")
                symptoms = problem.get("symptoms", [])
                risk_factors = problem.get("risk_factors", [])

                chunk1_parts.append(f"\n{i}. {problem_name}")

                if symptoms:
                    chunk1_parts.append("   症状表现:")
                    for symptom in symptoms:
                        chunk1_parts.append(f"     • {symptom}")

                if risk_factors:
                    chunk1_parts.append("   风险因素:")
                    for risk in risk_factors:
                        chunk1_parts.append(f"     • {risk}")

        # 日常护理建议
        daily_care = joint_care_data.get("daily_care_recommendations", {})
        if isinstance(daily_care, dict) and daily_care:
            chunk1_parts.append("\n二、日常护理建议:")

            exercise = daily_care.get("exercise", "")
            if exercise:
                chunk1_parts.append(f"1. 运动管理:\n   {exercise}")

            weight_management = daily_care.get("weight_management", "")
            if weight_management:
                chunk1_parts.append(f"2. 体重管理:\n   {weight_management}")

            environment_adjustments = daily_care.get("environment_adjustments", "")
            if environment_adjustments:
                chunk1_parts.append(f"3. 环境调整:\n   {environment_adjustments}")

        chunk1_content = "\n".join(chunk1_parts)

        # 添加元数据统计
        chunk1_metadata = {
            **base_metadata,
            "chunk_type": "joint_problems_daily_care",
            "problem_count": len(common_problems) if isinstance(common_problems, list) else 0,
            "has_daily_care": bool(daily_care)
        }

        chunks.append({
            "content": chunk1_content,
            "metadata": chunk1_metadata,
            "source_file": source_file,
            "content_type": "cares"
        })

        # ========== CHUNK 2: 营养补充剂 + 何时看兽医 ==========
        chunk2_parts = []

        # 营养补充剂
        supplements = joint_care_data.get("nutritional_supplements", [])
        if isinstance(supplements, list) and supplements:
            chunk2_parts.append("三、营养补充剂:")
            for i, supplement in enumerate(supplements, 1):
                if not isinstance(supplement, dict):
                    continue

                supplement_name = supplement.get("supplement", "")
                benefits = supplement.get("benefits", [])
                dosage_notes = supplement.get("dosage_notes", "")

                chunk2_parts.append(f"\n{i}. {supplement_name}")

                if benefits:
                    chunk2_parts.append("   主要益处:")
                    for benefit in benefits:
                        chunk2_parts.append(f"     • {benefit}")

                if dosage_notes:
                    chunk2_parts.append(f"   剂量说明: {dosage_notes}")

        # 何时看兽医
        when_to_see_vet = joint_care_data.get("when_to_see_vet", [])
        if isinstance(when_to_see_vet, list) and when_to_see_vet:
            chunk2_parts.append("\n四、何时需要看兽医（关节问题预警）:")
            for i, point in enumerate(when_to_see_vet, 1):
                chunk2_parts.append(f"{i}. {point}")

        # 免责声明
        disclaimer = joint_care_data.get("disclaimer", "")
        if disclaimer:
            chunk2_parts.append(f"\n免责声明: {disclaimer}")

        chunk2_content = "\n".join(chunk2_parts)

        # 数据来源（可以作为元数据的一部分，也可以单独一个chunk，这里放在内容中）
        data_sources = joint_care_data.get("data_sources", [])
        if isinstance(data_sources, list) and data_sources:
            chunk2_content += "\n\n数据来源:"
            for i, source in enumerate(data_sources, 1):
                chunk2_content += f"\n{i}. {source}"

        chunk2_metadata = {
            **base_metadata,
            "chunk_type": "supplements_vet_when",
            "supplement_count": len(supplements) if isinstance(supplements, list) else 0,
            "warning_points": len(when_to_see_vet) if isinstance(when_to_see_vet, list) else 0,
            "has_disclaimer": bool(disclaimer)
        }

        chunks.append({
            "content": chunk2_content,
            "metadata": chunk2_metadata,
            "source_file": source_file,
            "content_type": "cares"
        })

        return chunks

    def _parse_diseases(self, data: Dict, source_file: str) -> List[Dict]:
        """解析疾病数据 - 分为4个语义块"""
        chunks = []

        # 提取疾病列表
        if isinstance(data, list):
            diseases = data
        elif "diseases" in data and isinstance(data["diseases"], list):
            diseases = data["diseases"]
        else:
            return chunks

        print(f"找到 {len(diseases)} 个疾病数据")

        for i, disease in enumerate(diseases):
            if not isinstance(disease, dict):
                continue

            disease_name = disease.get("disease_name", f"疾病_{i}")

            # 基础元数据（共享）
            base_metadata = {
                "content_type": "diseases_professional",
                "category": "health",
                "disease_name": disease_name,
                "disease_type": disease.get("disease_type", ""),
                "disease_category": disease.get("disease_category", ""),
                "urgency_level": disease.get("urgency_level", ""),
                "zoonotic": disease.get("zoonotic", False),
                "source_file": source_file,
                "disease_index": i
            }

            # ========== CHUNK 1: 基本信息、症状、诊断 ==========
            chunk1_parts = []

            # 基础信息
            chunk1_parts.append(f"疾病: {disease_name}")
            chunk1_parts.append(f"类型: {disease.get('disease_type', '')}")
            chunk1_parts.append(f"类别: {disease.get('disease_category', '')}")

            standard_codes = disease.get("standard_codes", {})
            if isinstance(standard_codes, dict):
                icd11 = standard_codes.get("icd11_vet", "")
                snomed = standard_codes.get("snomed_ct", "")
                if icd11 or snomed:
                    chunk1_parts.append(f"标准编码: ICD-11-Vet: {icd11}, SNOMED-CT: {snomed}")

            zoonotic = disease.get("zoonotic", False)
            chunk1_parts.append(f"人畜共患: {'是' if zoonotic else '否'}")

            # 受影响物种
            affected_species = disease.get("affected_species", [])
            if affected_species:
                species_text = "、".join(affected_species)
                chunk1_parts.append(f"受影响物种: {species_text}")

            # 紧急等级和程度
            urgency = disease.get("urgency_level", "")
            if urgency:
                urgency_map = {1: "轻度", 2: "中度", 3: "重度"}
                urgency_text = urgency_map.get(int(urgency), f"等级{urgency}")
                chunk1_parts.append(f"紧急程度: {urgency_text}")

            contagious = disease.get("contagious_level", "")
            if contagious:
                chunk1_parts.append(f"传染性: {contagious * 100:.0f}%")

            severity = disease.get("severity_level", "")
            if severity:
                chunk1_parts.append(f"严重程度: {severity * 100:.0f}%")

            # 症状
            symptoms = disease.get("symptoms", {})
            if isinstance(symptoms, dict):
                primary = symptoms.get("primary", [])
                secondary = symptoms.get("secondary", [])

                if primary:
                    chunk1_parts.append("\n主要症状:")
                    for j, symptom in enumerate(primary, 1):
                        chunk1_parts.append(f"{j}. {symptom}")

                if secondary:
                    chunk1_parts.append("\n次要症状:")
                    for j, symptom in enumerate(secondary, 1):
                        chunk1_parts.append(f"{j}. {symptom}")

            # 症状权重
            symptom_weights = disease.get("symptom_weights", {})
            if isinstance(symptom_weights, dict) and symptom_weights:
                chunk1_parts.append("\n症状权重（重要程度）:")
                for symptom, weight in list(symptom_weights.items()):
                    chunk1_parts.append(f"• {symptom}: {weight:.1%}")

            # 鉴别症状
            differential_symptoms = disease.get("differential_symptoms", [])
            if differential_symptoms:
                chunk1_parts.append("\n鉴别诊断关键症状:")
                for symptom in differential_symptoms:
                    chunk1_parts.append(f"• {symptom}")

            # 诊断方法
            diagnosis = disease.get("diagnosis", [])
            if diagnosis:
                chunk1_parts.append("\n诊断方法:")
                for j, method in enumerate(diagnosis, 1):
                    chunk1_parts.append(f"{j}. {method}")

            chunk1_content = "\n".join(chunk1_parts)
            chunk1_metadata = {
                **base_metadata,
                "chunk_type": "disease_overview_symptoms",
                "has_symptoms": bool(symptoms),
                "has_diagnosis": bool(diagnosis),
                "symptom_count": len("primary") + len("secondary") if isinstance(symptoms, dict) else 0
            }

            chunks.append({
                "content": chunk1_content,
                "metadata": chunk1_metadata,
                "source_file": source_file,
                "content_type": "diseases_professional"
            })

            # ========== CHUNK 2: 治疗、预后、费用 ==========
            chunk2_parts = []

            chunk2_parts.append(f"疾病: {disease_name}")
            # 治疗方案
            treatments = disease.get("treatment", [])
            if treatments:
                chunk2_parts.append("治疗方案:")
                for j, treatment in enumerate(treatments, 1):
                    if isinstance(treatment, dict):
                        name = treatment.get("name", "")
                        category = treatment.get("category", "")
                        drug = treatment.get("drug", "")
                        dosage = treatment.get("dosage", "")

                        chunk2_parts.append(f"{j}. {name}")
                        if category:
                            chunk2_parts.append(f"   类别: {category}")
                        if drug:
                            chunk2_parts.append(f"   药物: {drug}")
                        if dosage:
                            chunk2_parts.append(f"   剂量: {dosage}")
                        chunk2_parts.append("")

            # 预后
            prognosis = disease.get("prognosis", "")
            if prognosis:
                chunk2_parts.append(f"预后评估: {prognosis}")

            # 费用估算
            cost_estimation = disease.get("cost_estimation", "")
            if cost_estimation:
                chunk2_parts.append(f"费用估算: {cost_estimation}")

            # 紧急阈值
            emergency_threshold = disease.get("emergency_threshold", "")
            if emergency_threshold:
                chunk2_parts.append(f"\n紧急就医阈值（出现以下情况需立即就医）:")
                chunk2_parts.append(f"  {emergency_threshold}")

            # 紧急指南
            emergency_guidelines = disease.get("emergency_guidelines", [])
            if emergency_guidelines:
                chunk2_parts.append("\n紧急处理指南:")
                for j, guideline in enumerate(emergency_guidelines, 1):
                    chunk2_parts.append(f"{j}. {guideline}")

            if len(chunk2_parts) > 0:
                chunk2_content = "\n".join(chunk2_parts)
                chunk2_metadata = {
                    **base_metadata,
                    "chunk_type": "treatment_prognosis_emergency",
                    "treatment_count": len(treatments),
                    "has_emergency_guide": bool(emergency_guidelines)
                }

                chunks.append({
                    "content": chunk2_content,
                    "metadata": chunk2_metadata,
                    "source_file": source_file,
                    "content_type": "diseases_professional"
                })

            # ========== CHUNK 3: 流行病学、预防、误诊风险 ==========
            chunk3_parts = []

            chunk3_parts.append(f"疾病: {disease_name}")
            # 流行病学信息
            chunk3_parts.append("流行病学特征:")

            incidence = disease.get("incidence_level", "")
            if incidence:
                chunk3_parts.append(f"• 发病率: {incidence * 100:.0f}%")

            prevalence_by_age = disease.get("prevalence_by_age", {})
            if isinstance(prevalence_by_age, dict):
                puppy_rate = prevalence_by_age.get("puppy", "")
                adult_rate = prevalence_by_age.get("adult", "")
                senior_rate = prevalence_by_age.get("senior", "")

                if puppy_rate:
                    chunk3_parts.append(f"• 幼犬患病率: {puppy_rate * 100:.0f}%")
                if adult_rate:
                    chunk3_parts.append(f"• 成年犬患病率: {adult_rate * 100:.0f}%")
                if senior_rate:
                    chunk3_parts.append(f"• 老年犬患病率: {senior_rate * 100:.0f}%")

            onset_pattern = disease.get("onset_pattern", "")
            if onset_pattern:
                chunk3_parts.append(f"• 发病模式: {onset_pattern}")

            seasonality = disease.get("seasonality", "")
            if seasonality:
                chunk3_parts.append(f"• 季节性: {seasonality}")

            # 常见触发因素
            common_triggers = disease.get("common_triggers", [])
            if common_triggers:
                chunk3_parts.append("\n常见触发因素:")
                for trigger in common_triggers:
                    chunk3_parts.append(f"• {trigger}")

            # 误诊风险
            misdiagnosis_risks = disease.get("misdiagnosis_risks", [])
            if misdiagnosis_risks:
                chunk3_parts.append("\n易混淆疾病（误诊风险）:")
                for risk in misdiagnosis_risks:
                    chunk3_parts.append(f"• {risk}")

            # 关键诊断词
            critical_keywords = disease.get("critical_keywords", [])
            if critical_keywords:
                chunk3_parts.append("\n关键诊断词:")
                chunk3_parts.append(f"  {', '.join(critical_keywords)}")

            # 预防措施
            prevention = disease.get("prevention", [])
            if prevention:
                chunk3_parts.append("\n预防措施:")
                for j, prevention_item in enumerate(prevention, 1):
                    chunk3_parts.append(f"{j}. {prevention_item}")

            if len(chunk3_parts) > 1:  # 不止有标题
                chunk3_content = "\n".join(chunk3_parts)
                chunk3_metadata = {
                    **base_metadata,
                    "chunk_type": "epidemiology_prevention",
                    "prevention_count": len(prevention),
                    "has_misdiagnosis_risks": bool(misdiagnosis_risks)
                }

                chunks.append({
                    "content": chunk3_content,
                    "metadata": chunk3_metadata,
                    "source_file": source_file,
                    "content_type": "diseases_professional"
                })

            # ========== CHUNK 4: FAQ、参考文献 ==========
            chunk4_parts = []

            chunk4_parts.append(f"疾病: {disease_name}")
            # FAQ
            faq = disease.get("faq", [])
            if faq:
                chunk4_parts.append("常见问题解答 (FAQ):")
                for j, qa in enumerate(faq, 1):
                    if isinstance(qa, dict):
                        question = qa.get("question", "")
                        answer = qa.get("answer", "")
                        if question and answer:
                            chunk4_parts.append(f"\nQ{j}: {question}")
                            chunk4_parts.append(f"A: {answer}")

            # 行为标志
            behavioral_flag = disease.get("behavioral_flag", "")
            if behavioral_flag:
                chunk4_parts.append("\n行为标志: 是" if behavioral_flag else "\n行为标志: 否")

            # 参考文献
            source_refs = disease.get("source_refs", [])
            if source_refs:
                chunk4_parts.append("\n参考文献:")
                for j, ref in enumerate(source_refs, 1):
                    chunk4_parts.append(f"{j}. {ref}")

            if len(chunk4_parts) > 0:
                chunk4_content = "\n".join(chunk4_parts)
                chunk4_metadata = {
                    **base_metadata,
                    "chunk_type": "faq_references",
                    "faq_count": len(faq),
                    "has_references": bool(source_refs)
                }

                chunks.append({
                    "content": chunk4_content,
                    "metadata": chunk4_metadata,
                    "source_file": source_file,
                    "content_type": "diseases_professional"
                })

        return chunks

    def _parse_breeds(self, data: Dict, source_file: str) -> List[Dict]:
        """解析犬种数据 - 完整版本"""
        chunks = []

        # 提取犬种列表
        breeds_list = []

        # 1. 处理标准嵌套结构：{"breed": {...}, "breed_name": "...", "processed_date": "..."}
        if isinstance(data, dict) and 'breed' in data:
            breed_data = data['breed']
            if isinstance(breed_data, dict):
                breeds_list.append(breed_data)

        # 2. 处理包含data字段的嵌套结构：{"data": [{"breed": {...}}, ...]}
        elif isinstance(data, dict) and 'data' in data:
            data_content = data['data']
            if isinstance(data_content, list):
                for item in data_content:
                    if isinstance(item, dict) and 'breed' in item:
                        breed_data = item['breed']
                        if isinstance(breed_data, dict):
                            breeds_list.append(breed_data)
                    elif isinstance(item, dict):
                        # 也可能直接包含犬种数据
                        breeds_list.append(item)
            elif isinstance(data_content, dict) and 'breed' in data_content:
                breed_data = data_content['breed']
                if isinstance(breed_data, dict):
                    breeds_list.append(breed_data)

        # 3. 处理直接是列表的情况
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'breed' in item:
                    breed_data = item['breed']
                    if isinstance(breed_data, dict):
                        breeds_list.append(breed_data)
                elif isinstance(item, dict):
                    breeds_list.append(item)

        # 4. 处理breeds字段
        elif isinstance(data, dict) and 'breeds' in data:
            breeds_data = data['breeds']
            if isinstance(breeds_data, list):
                breeds_list = breeds_data
            else:
                breeds_list = [breeds_data]

        # 5. 如果以上都不匹配，尝试将整个data作为犬种数据
        elif isinstance(data, dict):
            breeds_list = [data]

        if not breeds_list:
            print(f"警告: 无法从数据中提取犬种信息")
            return chunks

        print(f"找到 {len(breeds_list)} 个犬种数据")

        for i, breed in enumerate(breeds_list):
            if not isinstance(breed, dict):
                continue

            # 获取犬种名称（尝试多个可能的字段名）
            breed_name = (
                    breed.get('chinese_name') or
                    breed.get('breed_name') or
                    breed.get('name') or
                    breed.get('english_name') or
                    f"犬种_{i}"
            )

            # 获取英文名
            english_name = breed.get('english_name', '')

            # 构建元数据
            metadata = {
                "content_type": "breeds",
                "category": "breed_info",
                "breed_name": breed_name,
                "english_name": english_name,
                "akc_group": breed.get('akc_group', ''),
                "size_category": breed.get('size_category', ''),
                "origin": breed.get('origin', ''),
                "source_file": source_file
            }

            # 构建内容字符串
            content_parts = []

            # 1. 基本信息
            if breed_name:
                content_parts.append(f"犬种: {breed_name}")
            if english_name:
                content_parts.append(f"英文名: {english_name}")

            akc_group = breed.get('akc_group', '')
            if akc_group:
                content_parts.append(f"AKC组别: {akc_group}")

            origin = breed.get('origin', '')
            if origin:
                content_parts.append(f"原产地: {origin}")

            original_purpose = breed.get('original_purpose', '')
            if original_purpose:
                content_parts.append(f"原始用途: {original_purpose}")

            size_category = breed.get('size_category', '')
            if size_category:
                content_parts.append(f"体型分类: {size_category}")

            # 2. 体型数据
            weight_data = breed.get('average_weight_kg', {})
            height_data = breed.get('average_height_cm', {})

            if isinstance(weight_data, dict) or isinstance(height_data, dict):
                content_parts.append("体型数据:")

                if isinstance(weight_data, dict):
                    male_weight = weight_data.get('male') or weight_data.get('Male')
                    female_weight = weight_data.get('female') or weight_data.get('Female')

                    if male_weight:
                        content_parts.append(f"  • 公犬体重: {male_weight} kg")
                    if female_weight:
                        content_parts.append(f"  • 母犬体重: {female_weight} kg")

                if isinstance(height_data, dict):
                    male_height = height_data.get('male') or height_data.get('Male')
                    female_height = height_data.get('female') or height_data.get('Female')

                    if male_height:
                        content_parts.append(f"  • 公犬肩高: {male_height} cm")
                    if female_height:
                        content_parts.append(f"  • 母犬肩高: {female_height} cm")

            # 3. 寿命和被毛信息
            life_expectancy = breed.get('life_expectancy', '')
            if life_expectancy:
                content_parts.append(f"寿命: {life_expectancy}")

            coat_type = breed.get('coat_type', '')
            if coat_type:
                content_parts.append(f"被毛类型: {coat_type}")

            # 4. 被毛颜色
            coat_colors = breed.get('coat_colors', [])
            if coat_colors and isinstance(coat_colors, list):
                colors_str = "、".join(coat_colors)
                content_parts.append(f"被毛颜色: {colors_str}")

            # 5. 主要特征
            primary_traits = breed.get('primary_traits', [])
            if primary_traits and isinstance(primary_traits, list):
                content_parts.append("主要特征:")
                for trait in primary_traits:
                    content_parts.append(f"  - {trait}")

            # 6. 评分信息
            energy_level = breed.get('energy_level')
            if energy_level is not None:
                content_parts.append(f"能量等级: {energy_level}/10")

            intelligence_rank = breed.get('intelligence_rank')
            if intelligence_rank is not None:
                content_parts.append(f"智商分数: {intelligence_rank}/10")

            trainability = breed.get('trainability')
            if trainability is not None:
                content_parts.append(f"可训练性: {trainability}/10")

            # 7. 社交性
            with_family = breed.get('with_family', '')
            if with_family:
                content_parts.append(f"家庭适应性: {with_family}")

            with_children = breed.get('with_children', '')
            if with_children:
                content_parts.append(f"与儿童相处: {with_children}")

            with_other_dogs = breed.get('with_other_dogs', '')
            if with_other_dogs:
                content_parts.append(f"与其他犬相处: {with_other_dogs}")

            # 8. 运动需求
            daily_exercise_minutes = breed.get('daily_exercise_minutes')
            if daily_exercise_minutes is not None:
                content_parts.append(f"日常运动需求: {daily_exercise_minutes} 分钟/天")

            # 9. 居住适应性
            apartment_friendly = breed.get('apartment_friendly')
            if apartment_friendly is not None:
                content_parts.append(f"适合公寓: {'是' if apartment_friendly else '否'}")

            # 10. 护理需求
            grooming_needs = breed.get('grooming_needs', '')
            if grooming_needs:
                content_parts.append(f"美容需求: {grooming_needs}")

            shedding_level = breed.get('shedding_level')
            if shedding_level is not None:
                content_parts.append(f"掉毛程度: {shedding_level}/10")

            # 11. 健康问题
            common_health_issues = breed.get('common_health_issues', [])
            if common_health_issues and isinstance(common_health_issues, list):
                content_parts.append("常见健康问题:")
                for issue in common_health_issues:
                    content_parts.append(f"  - {issue}")

            # 12. 适宜性评估
            suitability_assessment = breed.get('suitability_assessment', '')
            if suitability_assessment:
                content_parts.append(f"适宜性评估: {suitability_assessment}")

            # 13. 知识来源
            knowledge_sources = breed.get('knowledge_sources', [])
            if knowledge_sources and isinstance(knowledge_sources, list):
                content_parts.append("参考文献:")
                for source in knowledge_sources:
                    content_parts.append(f"  - {source}")

            # 组合内容
            content = "\n".join(content_parts)

            # 更新元数据以包含更多信息
            metadata.update({
                "akc_group": akc_group,
                "size_category": size_category,
                "origin": origin,
                "life_expectancy": life_expectancy,
                "coat_type": coat_type,
                "primary_traits_count": len(primary_traits) if isinstance(primary_traits, list) else 0,
                "energy_level": energy_level,
                "intelligence_rank": intelligence_rank,
                "trainability": trainability,
                "daily_exercise_minutes": daily_exercise_minutes,
                "apartment_friendly": apartment_friendly,
                "shedding_level": shedding_level,
                "health_issues_count": len(common_health_issues) if isinstance(common_health_issues, list) else 0
            })

            chunks.append({
                "content": content,
                "metadata": metadata,
                "source_file": source_file,
                "content_type": "breeds"
            })

        return chunks

    def _parse_behaviors(self, data: Dict, source_file: str) -> List[Dict]:
        """解析行为数据 - 修复版本"""
        chunks = []

        # 情况1：数据是列表（你的JSON格式）
        if isinstance(data, list):
            behaviors = []
            for item in data:
                if isinstance(item, dict):
                    # 每个item应该是 {"behavior": {...}}
                    if "behavior" in item and isinstance(item["behavior"], dict):
                        behaviors.append(item["behavior"])
                    else:
                        # 如果没有behavior键，可能直接就是行为对象
                        behaviors.append(item)

        # 情况2：数据是字典且包含behavior字段
        elif isinstance(data, dict) and "behavior" in data:
            behaviors = [data["behavior"]]

        # 情况3：数据是字典但直接就是行为对象
        elif isinstance(data, dict):
            behaviors = [data]

        # 情况4：其他格式
        else:
            behaviors = []
            print(f"警告：无法解析的行为数据格式: {type(data)}")

        # 处理每个行为对象
        for behavior in behaviors:
            if not isinstance(behavior, dict):
                continue

            # 构建内容字符串
            content = f"""
行为名称: {behavior.get('name', '')}
行为类别: {behavior.get('category', '')}
描述: {behavior.get('description', '')}

前因: {behavior.get('antecedents', '')}
后果: {behavior.get('consequences', '')}
意义: {behavior.get('meaning', '')}

评估: {behavior.get('evaluation', '')}
功能: {behavior.get('function', '')}
干预等级: {behavior.get('intervention_level', '')}

资源参考: {behavior.get('resource', '')}
"""
            # 创建元数据
            metadata = {
                "content_type": "behaviors",
                "category": "behavior",
                "behavior_name": behavior.get('name', ''),
                "behavior_category": behavior.get('category', ''),
                "intervention_level": behavior.get('intervention_level', ''),
                "source_file": source_file
            }

            chunks.append({
                "content": content,
                "metadata": metadata,
                "source_file": source_file,
                "content_type": "behaviors"
            })

        return chunks

    def _parse_surgeries(self, data: Dict, source_file: str) -> List[Dict]:
        """解析手术数据 - 语义分块版本"""
        chunks = []

        # 提取手术数据（保持原有逻辑）
        if isinstance(data, dict) and "data" in data:
            surgeries_data = data.get("data", [])
            if not isinstance(surgeries_data, list):
                return chunks
        elif isinstance(data, list):
            surgeries_data = data
        else:
            surgeries_data = [data]

        for surgery_item in surgeries_data:
            if isinstance(surgery_item, dict) and "surgery" in surgery_item:
                surgery_data = surgery_item["surgery"]
                surgery_name = surgery_item.get("surgery_name", "")
            elif isinstance(surgery_item, dict):
                surgery_data = surgery_item
                surgery_name = surgery_data.get("chinese_name", surgery_data.get("english_name", ""))
            else:
                continue

            if not isinstance(surgery_data, dict):
                continue

            # ========== CHUNK 1: 手术基本信息、适应症、手术概述 ==========
            chunk1_parts = []

            # 基本信息
            chinese_name = surgery_data.get("chinese_name", "")
            english_name = surgery_data.get("english_name", "")
            category = surgery_data.get("category", "")
            sub_category = surgery_data.get("sub_category", "")

            chunk1_parts.append(f"手术名称: {chinese_name}")
            if english_name:
                chunk1_parts.append(f"英文名: {english_name}")
            if category:
                chunk1_parts.append(f"类别: {category}")
            if sub_category:
                chunk1_parts.append(f"子类别: {sub_category}")

            # 适应症
            indications = surgery_data.get("indications", [])
            if indications:
                chunk1_parts.append("\n适应症:")
                for i, indication in enumerate(indications, 1):
                    chunk1_parts.append(f"{i}. {indication}")

            # 替代疗法
            alternative_therapies = surgery_data.get("alternative_therapies", [])
            if alternative_therapies:
                chunk1_parts.append("\n替代疗法:")
                for i, therapy in enumerate(alternative_therapies, 1):
                    chunk1_parts.append(f"{i}. {therapy}")

            # 手术概述
            surgical_overview = surgery_data.get("surgical_overview", "")
            if surgical_overview:
                chunk1_parts.append(f"\n手术概述: {surgical_overview}")

            # 术前准备
            preoperative_preparation = surgery_data.get("preoperative_preparation", [])
            if preoperative_preparation:
                chunk1_parts.append("\n术前准备:")
                for i, prep in enumerate(preoperative_preparation, 1):
                    chunk1_parts.append(f"{i}. {prep}")

            # 手术技术简介
            surgical_technique_brief = surgery_data.get("surgical_technique_brief", "")
            if surgical_technique_brief:
                chunk1_parts.append(f"\n手术技术简介: {surgical_technique_brief}")

            chunk1_content = "\n".join(chunk1_parts)

            chunks.append({
                "content": chunk1_content,
                "metadata": {
                    "content_type": "surgeries",
                    "chunk_type": "surgery_overview",
                    "surgery_name": surgery_name,
                    "chinese_name": chinese_name,
                    "english_name": english_name,
                    "category": category,
                    "has_indications": bool(indications),
                    "has_alternative_therapies": bool(alternative_therapies),
                    "source_file": source_file
                },
                "source_file": source_file,
                "content_type": "surgeries"
            })

            # ========== CHUNK 2: 术后护理、并发症、预后、费用 ==========
            chunk2_parts = []

            chunk2_parts.append(f"手术名称: {chinese_name}")
            # 术后护理
            postoperative_care = surgery_data.get("postoperative_care", [])
            if postoperative_care:
                chunk2_parts.append("术后护理:")
                for i, care in enumerate(postoperative_care, 1):
                    chunk2_parts.append(f"{i}. {care}")

            # 常见并发症
            common_complications = surgery_data.get("common_complications", [])
            if common_complications:
                chunk2_parts.append("\n常见并发症:")
                for i, complication in enumerate(common_complications, 1):
                    chunk2_parts.append(f"{i}. {complication}")

            # 预后
            prognosis = surgery_data.get("prognosis", "")
            if prognosis:
                chunk2_parts.append(f"\n预后评估: {prognosis}")

            # 费用估算
            cost_estimation = surgery_data.get("cost_estimation", {})
            if isinstance(cost_estimation, dict):
                range_cny = cost_estimation.get("range_cny", "")
                notes = cost_estimation.get("notes", "")
                if range_cny:
                    chunk2_parts.append(f"\n费用估算: {range_cny}")
                if notes:
                    chunk2_parts.append(f"费用说明: {notes}")

            # 品种和年龄考虑
            breed_age_considerations = surgery_data.get("breed_age_considerations", [])
            if breed_age_considerations:
                chunk2_parts.append("\n品种和年龄考虑:")
                for i, consideration in enumerate(breed_age_considerations, 1):
                    chunk2_parts.append(f"{i}. {consideration}")

            # 知识来源
            knowledge_sources = surgery_data.get("knowledge_sources", [])
            if knowledge_sources:
                chunk2_parts.append("\n参考文献:")
                for i, source in enumerate(knowledge_sources, 1):
                    chunk2_parts.append(f"{i}. {source}")

            if chunk2_parts:  # 只有当有内容时才创建第二个chunk
                chunk2_content = "\n".join(chunk2_parts)

                chunks.append({
                    "content": chunk2_content,
                    "metadata": {
                        "content_type": "surgeries",
                        "chunk_type": "postoperative_complications_prognosis",
                        "surgery_name": surgery_name,
                        "chinese_name": chinese_name,
                        "has_postoperative_care": bool(postoperative_care),
                        "has_complications": bool(common_complications),
                        "has_prognosis": bool(prognosis),
                        "source_file": source_file
                    },
                    "source_file": source_file,
                    "content_type": "surgeries"
                })

        return chunks