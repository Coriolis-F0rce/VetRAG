import requests
import json
import time
import os
from typing import Dict, List, Any
from datetime import datetime


class BatchAIContentGenerator:
    """批量AI内容生成器（分批处理）"""

    def __init__(self, api_key: str, output_dir: str = "generated_content"):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _make_api_request(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """单次API调用"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """你是一位专业的兽医内容编辑。请基于可靠来源生成内容。
                    1. 只陈述事实，不提供医疗建议
                    2. 标注信息来源
                    3. 不确定的信息注明'待核实'
                    4. 使用清晰的结构化格式"""
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # 如果返回的不是JSON，尝试包装为JSON
                        return {"raw_content": content}
                else:
                    return {"error": "API响应格式异常", "raw": str(result)[:200]}
            else:
                return {"error": f"HTTP {response.status_code}", "raw": response.text[:200]}

        except requests.exceptions.Timeout:
            return {"error": "请求超时"}
        except Exception as e:
            return {"error": f"请求异常: {str(e)}"}

    def generate_single_content(self, content_id: str, prompt: str,
                                temperature: float = 0.3, delay: int = 2) -> Dict[str, Any]:
        """生成单个内容并立即保存"""

        print(f"开始生成内容: {content_id}")

        # 调用API
        result = self._make_api_request(prompt, temperature)

        # 添加元数据
        result["metadata"] = {
            "content_id": content_id,
            "generated_at": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "temperature": temperature
        }

        # 立即保存为独立文件
        filename = f"{self.output_dir}/{content_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  已保存到: {filename}")

        # 延迟，避免速率限制
        time.sleep(delay)

        return result

    def generate_batch_content(self, content_list: List[Dict[str, str]],
                               batch_size: int = 3, delay_between_batches: int = 10) -> Dict[str, Any]:
        """批量生成内容，分批处理"""

        print(f"开始批量生成 {len(content_list)} 个内容，每批 {batch_size} 个")

        all_results = {}
        total_batches = (len(content_list) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(content_list))
            current_batch = content_list[start_idx:end_idx]

            print(f"\n处理第 {batch_num + 1}/{total_batches} 批 ({len(current_batch)} 个内容)")

            batch_results = {}
            for content_item in current_batch:
                content_id = content_item.get("id", f"content_{start_idx}")
                prompt = content_item["prompt"]

                result = self.generate_single_content(
                    content_id=content_id,
                    prompt=prompt,
                    temperature=content_item.get("temperature", 0.3),
                    delay=content_item.get("delay", 2)
                )

                batch_results[content_id] = result

            # 保存批次结果
            batch_filename = f"{self.output_dir}/batch_{batch_num + 1}_results.json"
            with open(batch_filename, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)

            # 添加到总结果
            all_results.update(batch_results)

            # 批次间延迟（如果有更多批次）
            if batch_num < total_batches - 1:
                print(f"等待 {delay_between_batches} 秒后处理下一批...")
                time.sleep(delay_between_batches)

        # 保存完整结果
        summary = {
            "total_contents": len(content_list),
            "batches_processed": total_batches,
            "batch_size": batch_size,
            "completion_time": datetime.now().isoformat(),
            "results": all_results
        }

        summary_filename = f"{self.output_dir}/all_results_summary.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n批量处理完成！")
        print(f"汇总文件: {summary_filename}")
        print(f"总内容数: {len(content_list)}")

        return summary


# ==================== 定义要生成的内容 ====================

def create_content_list() -> List[Dict[str, Any]]:
    """创建要生成的内容列表"""

    # 1. 幼犬疫苗时间表
    vaccine_prompt = """请生成犬幼犬标准疫苗接种时间表的JSON格式数据，包含以下字段：
{
  "content_type": "幼犬疫苗时间表",
  "data": [
    {
      "age": "6-8周",
      "vaccine_name": "疫苗名称",
      "disease_protected": ["防护疾病1", "防护疾病2"],
      "importance": "核心/非核心",
      "notes": "注意事项",
      "repeat_interval": "重复间隔"
    }
  ],
  "general_notes": "通用注意事项",
  "data_sources": ["来源1", "来源2"],
  "last_updated": "更新时间"
}

请填写完整的JSON数据，确保信息准确、完整。"""

    # 2. 老年犬关节护理
    joint_care_prompt = """请生成老年犬关节护理指南的JSON格式数据，包含以下字段：
{
  "content_type": "老年犬关节护理指南",
  "common_joint_problems": [
    {
      "problem": "问题名称",
      "symptoms": ["症状1", "症状2"],
      "risk_factors": ["风险因素1", "风险因素2"]
    }
  ],
  "daily_care_recommendations": {
    "exercise": "运动建议",
    "weight_management": "体重管理",
    "environment_adjustments": "环境调整"
  },
  "nutritional_supplements": [
    {
      "supplement": "补充剂名称",
      "benefits": ["益处1", "益处2"],
      "dosage_notes": "剂量说明"
    }
  ],
  "when_to_see_vet": ["就医信号1", "就医信号2"],
  "data_sources": ["来源1", "来源2"],
  "disclaimer": "此为一般信息，具体治疗需咨询兽医。"
}

请填写完整的JSON数据，确保信息准确、完整。"""

    # 3. 文明养犬法规
    legal_prompt = """请生成中国主要城市养犬管理规定的JSON格式数据，包含以下字段：
{
  "content_type": "中国主要城市养犬管理规定",
  "cities": [
    {
      "city_name": "城市名称",
      "regulation_name": "法规名称",
      "key_points": {
        "registration_requirements": "登记要求",
        "vaccination_requirements": "免疫要求",
        "walking_restrictions": "遛狗限制",
        "prohibited_breeds": ["禁养犬种1", "禁养犬种2"],
        "penalty_standards": "处罚标准"
      },
      "data_source": "数据来源"
    }
  ],
  "last_updated": "更新时间",
  "note": "各城市规定不同，具体以当地最新法规为准。"
}

请至少包含北京、上海、广州、深圳、成都5个城市的信息。请填写完整的JSON数据。"""

    # 4. 犬只常见疾病症状（额外示例）
    diseases_prompt = """请生成犬只常见疾病症状对照表的JSON格式数据，包含以下字段：
{
  "content_type": "犬只常见疾病症状对照表",
  "diseases": [
    {
      "disease_name": "疾病名称",
      "common_symptoms": ["症状1", "症状2", "症状3"],
      "emergency_level": "紧急程度（低/中/高）",
      "first_aid_measures": ["急救措施1", "急救措施2"],
      "when_to_see_vet": "何时就医"
    }
  ],
  "data_sources": ["来源1", "来源2"],
  "disclaimer": "此信息仅供参考，不能替代专业兽医诊断。"
}

请至少包含犬瘟热、细小病毒、犬传染性肝炎、狂犬病、皮肤病5种疾病。请填写完整的JSON数据。"""

    # 5. 犬只日常护理要点（额外示例）
    daily_care_prompt = """请生成犬只日常护理要点的JSON格式数据，包含以下字段：
{
  "content_type": "犬只日常护理要点",
  "care_categories": [
    {
      "category": "护理类别",
      "frequency": "频率",
      "procedures": ["步骤1", "步骤2"],
      "tips": ["小贴士1", "小贴士2"]
    }
  ],
  "data_sources": ["来源1", "来源2"],
  "last_updated": "更新时间"
}

请至少包含梳毛、洗澡、剪指甲、牙齿护理、耳朵清洁5个类别。请填写完整的JSON数据。"""

    # 返回内容列表
    return [
        {
            "id": "vaccine_schedule",
            "name": "幼犬疫苗时间表",
            "prompt": vaccine_prompt,
            "temperature": 0.2,
            "delay": 3
        },
        {
            "id": "joint_care_guide",
            "name": "老年犬关节护理指南",
            "prompt": joint_care_prompt,
            "temperature": 0.2,
            "delay": 3
        },
        {
            "id": "dog_regulations",
            "name": "中国主要城市养犬管理规定",
            "prompt": legal_prompt,
            "temperature": 0.2,
            "delay": 3
        },
        {
            "id": "common_diseases",
            "name": "犬只常见疾病症状对照表",
            "prompt": diseases_prompt,
            "temperature": 0.2,
            "delay": 3
        },
        {
            "id": "daily_care",
            "name": "犬只日常护理要点",
            "prompt": daily_care_prompt,
            "temperature": 0.2,
            "delay": 3
        }
    ]


# ==================== 使用示例 ====================

def main():
    """主函数"""

    # 配置
    API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"  #
    OUTPUT_DIR = "generated_pet_content"

    print("=" * 50)
    print("犬类知识内容批量生成系统")
    print("=" * 50)

    # 创建生成器
    generator = BatchAIContentGenerator(api_key=API_KEY, output_dir=OUTPUT_DIR)

    # 创建内容列表
    content_list = create_content_list()

    print(f"\n准备生成 {len(content_list)} 个内容:")
    for i, content in enumerate(content_list):
        print(f"  {i + 1}. {content['name']} ({content['id']})")

    # 询问用户是否开始
    user_input = input("\n是否开始生成内容？(y/n): ").strip().lower()

    if user_input != 'y':
        print("已取消。")
        return

    # 批量生成内容
    print("\n开始生成内容...")
    print("=" * 50)

    # 配置分批参数
    batch_size = 2  # 每批处理2个内容
    delay_between_batches = 15  # 批次间延迟15秒

    # 执行批量生成
    try:
        summary = generator.generate_batch_content(
            content_list=content_list,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches
        )

        # 显示统计信息
        print("\n" + "=" * 50)
        print("生成统计信息:")
        print(f"  总内容数: {summary['total_contents']}")
        print(f"  处理批次: {summary['batches_processed']}")
        print(f"  完成时间: {summary['completion_time']}")

        # 检查成功/失败
        successful = 0
        failed = 0

        for content_id, result in summary['results'].items():
            if 'error' in result:
                failed += 1
            else:
                successful += 1

        print(f"  成功: {successful}")
        print(f"  失败: {failed}")

        # 显示输出文件
        print(f"\n输出文件位置: {os.path.abspath(OUTPUT_DIR)}")
        print("生成的文件:")
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith('.json'):
                print(f"  - {file}")

    except Exception as e:
        print(f"批量生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()