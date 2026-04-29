import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"
MODEL_NAME = "deepseek-chat"
DISEASE_LIST_FILE = "find_diseases/dog_diseases.txt"
OUTPUT_JSON_FILE = "find_diseases/dog_diseases_knowledge_base.json"


# ========== 核心处理器 ==========
class VeterinaryDiseaseProcessor:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _construct_prompt(self, disease_name: str) -> str:
        FINAL_PROMPT_TEMPLATE = """【角色】兽医信息结构化专家
【任务】你是一个专业的兽医信息处理系统。你的核心任务是：**运用你自身训练数据中涵盖的权威兽医学知识（教科书、专业文献、权威网站等）**，为犬病“{疾病名称}”生成一份严格、标准化的医疗档案JSON。**绝对禁止编造知识**。

【输出格式与规则】
1.  输出必须是且仅是一个合法的JSON对象。
2.  严格遵循以下字段结构。所有信息必须源自你确信的、可靠的兽医学公开知识。对于任何不确定或未知的信息，字段值必须填写为“未知”。
{
  "disease_name": "疾病的标准医学名称",
  "disease_type": "疾病的标准医学分类"
  "affected_species": ["物种1（如：犬）", "物种2（如：猫、狐、人...）,..."], // 列出所有已知的易感动物
  "zoonotic": "是/否/条件性/未知",
  "infectiousness_details": "描述。若非传染性疾病，则填‘无传染性’。若为传染性疾病，必须说明：a) 传播途径（如直接接触、粪口、呼吸道气溶胶、媒介传播等）；b) 传染强度（高/中/低，基于基本流行病学特征推断）；c)是否有人畜传染性",
  "key_symptoms": ["症状1", "症状2", "症状3", ...], // 尽可能详尽地列出典型临床症状
  "diagnosis": ["诊断方法1（如：临床体格检查结合典型症状）", "诊断方法2（如：血液学检查-白细胞减少）", "诊断方法3（如：粪抗原ELISA检测）", "鉴别诊断（如：需与犬冠状病毒感染、急性出血性胃肠炎鉴别）"], // 详细说明诊断步骤、所需检查及鉴别要点
  "treatment": ["治疗方案或药物A（含具体剂量和用法，如：静脉输液，乳酸林格氏液，60ml/kg/天）", "治疗方案或药物B（如：止吐，马罗皮坦，1mg/kg，皮下注射，每日一次）", "支持疗法（如：肠道外营养）"],
  "prevention": ["预防措施1（如：接种核心疫苗-犬细小病毒疫苗）", "预防措施2（如：患病动物严格隔离）", "预防措施3（如：环境使用1:30稀释漂白水彻底消毒）"],
  "source_refs": ["知识来源标识1（如：兽医学教科书《小动物内科学》）", "知识来源标识2（如：MSD兽医手册在线版）", "知识来源标识3（如：CDC人畜共患病指南）"] // **关键：为你提供的信息标注1-3个你认为最权威的公开知识来源。**
}
3.  **关键判定与生成逻辑**：
    - `affected_species`：基于疾病病原体的宿主范围常识作答。
    - `zoonotic`与`zoonotic_details`：
        *   若为人畜共患病，必须明确写出传播途径，并根据病原体在种间传播的难易度推断强度（如狂犬病通过咬伤传播，**强度高**；某些寄生虫病需特定媒介，**强度低**）。
        *   若为非传染性疾病（如退化性疾病、肿瘤），`zoonotic`填“否”，`zoonotic_details`填“无传染性”。
    - `diagnosis`与`treatment`：必须包含具体的、可操作的步骤或药物名称及**典型剂量范围**（这是专业性的体现）。
    - `source_refs`：**这是衡量信息可靠性的关键**。请引用你知识库中存在的、公认的权威来源名称。**不要编造不存在的书名或URL**，可以用通用名称（如“兽医学通用知识”、“权威兽医免疫学指南”）。

**最终指令**：现在，请开始为疾病“{疾病名称}”生成JSON。请确保每一个字段都有具体、专业的内容，而不是泛泛而谈。"""

        # 替换疾病名称占位符
        full_prompt = FINAL_PROMPT_TEMPLATE.replace("{疾病名称}", disease_name)
        return full_prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_deepseek_api(self, session: aiohttp.ClientSession, prompt: str) -> Optional[Dict[str, Any]]:
        """调用DeepSeek API，并带有重试机制。"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # 低温度保证输出稳定、一致
            "max_tokens": 2048  # 确保有足够空间生成详细JSON
        }
        try:
            async with session.post(self.api_url, headers=self.headers, json=payload, timeout=45) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # 清理可能的Markdown格式
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()

                json_data = json.loads(content)
                return json_data
        except json.JSONDecodeError as e:
            self.logger.error(f"API返回内容JSON解析失败: {e}\n原始内容片段: {content[:300]}...")
            # 可在此处添加简单的内容提取逻辑作为备选
            return None
        except Exception as e:
            self.logger.error(f"调用API时发生网络或服务器错误: {e}")
            raise

    async def process_single_disease(self, session: aiohttp.ClientSession, disease_name: str) -> Optional[
        Dict[str, Any]]:
        """处理单个疾病的完整流程。"""
        self.logger.info(f"开始处理: {disease_name}")

        # 1. 构造自包含Prompt
        prompt = self._construct_prompt(disease_name)

        # 2. 调用API
        json_result = await self._call_deepseek_api(session, prompt)

        if json_result:
            # 3. 简单验证：确保必要字段存在
            required_fields = ["disease_name", "affected_species", "zoonotic"]
            if all(field in json_result for field in required_fields):
                self.logger.info(f"成功处理: {disease_name}")
                return json_result
            else:
                self.logger.error(f"疾病 '{disease_name}' 的返回JSON缺少必要字段。")
        return None

    async def process_all_diseases(self, disease_list: List[str], max_concurrent: int = 1,
                                   delay_seconds: float = 3.0) -> List[Dict[str, Any]]:
        '''断点续传'''
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            valid_results = []
            failed_diseases = []

            # 尝试加载已有的进度，实现断点续传
            try:
                with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    processed_names = {item.get('disease_name') for item in existing_data}
                    print(f"📂 检测到已有{len(processed_names)}条记录，将跳过已处理的疾病。")
                    valid_results = existing_data
            except (FileNotFoundError, json.JSONDecodeError):
                processed_names = set()
                print("📂 未找到有效进度文件，将从头开始处理。")

            for index, disease in enumerate(disease_list):
                if disease in processed_names:
                    self.logger.info(f"跳过已处理疾病: {disease}")
                    continue

                self.logger.info(f"处理进度: {index + 1}/{len(disease_list)} - {disease}")
                try:
                    result = await self.process_single_disease(session, disease)
                    if result:
                        valid_results.append(result)
                        # 每成功处理一个就立即保存，防止数据丢失
                        save_results(valid_results, OUTPUT_JSON_FILE)
                        self.logger.info(f"已保存进度。")
                    else:
                        failed_diseases.append(disease)
                except Exception as e:
                    self.logger.error(f"处理疾病 '{disease}' 时发生异常: {e}")
                    failed_diseases.append(disease)

                # 关键：在请求之间强制添加延迟，严格遵守API限制
                if index < len(disease_list) - 1:  # 最后一个请求后不需要等待
                    await asyncio.sleep(delay_seconds)

            if failed_diseases:
                self.logger.warning(f"以下疾病处理失败，建议稍后重试: {failed_diseases}")
                with open('failed_diseases.txt', 'w', encoding='utf-8') as f:
                    for d in failed_diseases:
                        f.write(d + '\n')

            return valid_results

# ========== 工具函数 ==========
def load_disease_list(file_path: str) -> List[str]:
    """从文本文件加载疾病列表，过滤掉分类标题。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        diseases = []
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 核心过滤规则：跳过所有以“一、”到“十、”开头的分类标题
            # 这个正则表达式匹配“一、”、“二、”……“十、”开头
            import re
            if re.match(r'^[一二三四五六七八九十]、', line):
                print(f"跳过分类标题: {line}")
                continue
            diseases.append(line)
    return diseases


def save_results(results: List[Dict[str, Any]], output_path: str):
    """将结果保存为JSON文件。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成！共处理 {len(results)} 个疾病档案，已保存至: {output_path}")


# ========== 主函数 ==========
async def main():
    print("🚀 开始构建犬病知识库...")

    try:
        disease_names = load_disease_list(DISEASE_LIST_FILE)
        print(f"📄 从 '{DISEASE_LIST_FILE}' 加载了 {len(disease_names)} 个疾病。")
    except FileNotFoundError:
        print(f"错误：未找到疾病列表文件 '{DISEASE_LIST_FILE}'，请检查路径。")
        return

    processor = VeterinaryDiseaseProcessor(
        api_url=DEEPSEEK_API_URL,
        api_key=API_KEY,
        model=MODEL_NAME
    )

    all_results = await processor.process_all_diseases(disease_names, max_concurrent=1, delay_seconds=5.0)

    save_results(all_results, OUTPUT_JSON_FILE)


if __name__ == "__main__":
    asyncio.run(main())