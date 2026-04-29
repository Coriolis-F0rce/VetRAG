# ==================== 犬品种完整列表 ====================
dog_breeds_list = [
    # 工作犬组
    "德国牧羊犬", "拉布拉多寻回犬", "金毛寻回犬", "杜宾犬", "罗威纳犬",
    "比利时马林诺斯犬", "边境牧羊犬", "澳大利亚牧羊犬", "柯利犬", "喜乐蒂牧羊犬",
    "伯恩山犬", "大瑞士山地犬", "圣伯纳犬", "纽芬兰犬", "大白熊犬",
    "阿拉斯加雪橇犬", "西伯利亚雪橇犬（哈士奇）", "萨摩耶犬", "拳师犬", "杜高犬",

    # 运动犬组
    "英国可卡犬", "美国可卡犬", "史宾格犬", "波音达犬", "威玛猎犬",
    "维兹拉犬", "切萨皮克湾寻回犬", "平毛寻回犬", "卷毛寻回犬", "爱尔兰雪达犬",
    "英国雪达犬", "戈登雪达犬", "布列塔尼犬", "德国短毛指示犬", "德国刚毛指示犬",

    # 梗犬组
    "约克夏梗", "杰克罗素梗", "西高地白梗", "苏格兰梗", "凯恩梗",
    "边境梗", "万能梗", "牛头梗", "斯塔福郡斗牛梗", "贝灵顿梗",
    "爱尔兰梗", "威尔士梗", "澳大利亚梗", "诺福克梗", "诺维奇梗",

    # 玩具犬组
    "贵宾犬（玩具型）", "博美犬", "吉娃娃", "马尔济斯犬", "西施犬",
    "北京犬", "巴哥犬", "法国斗牛犬", "意大利灵缇", "蝴蝶犬",
    "布鲁塞尔格里芬犬", "日本狆", "中国冠毛犬", "曼彻斯特梗（玩具型）", "骑士查理王小猎犬",

    # 非运动犬组
    "柴犬", "松狮犬", "斗牛犬", "沙皮犬", "比熊犬",
    "拉萨犬", "西藏梗", "西藏猎犬", "芬兰狐狸犬", "挪威猎鹿犬",
    "秋田犬", "美国爱斯基摩犬", "荷兰毛狮犬", "葡萄牙水犬", "标准贵宾犬",

    # 嗅觉猎犬组
    "比格犬", "巴吉度猎犬", "寻血猎犬", "腊肠犬", "罗德西亚背脊犬",
    "奥达猎犬", "英国猎狐犬", "美国猎狐犬", "哈利犬", "挪威猎麋犬",

    # 视觉猎犬组
    "阿富汗猎犬", "灵缇", "萨路基猎犬", "惠比特犬", "爱尔兰猎狼犬",
    "苏格兰猎鹿犬", "法老王猎犬", "伊维萨猎犬", "阿沙瓦犬", "苏俄猎狼犬"
]

# ==================== 简化的API调用脚本 ====================
import json
import requests
import time


def query_dog_breed_info(breed_name):
    """生成犬品种信息的API请求"""

    prompt_template = """你是一位专业的犬类行为学家、繁殖专家和兽医。请根据以下犬品种名称，提供详细、准确、专业的品种特性分析。

犬品种：{breed_name}

请按照以下JSON格式输出信息：

{{
  "breed": {{
    "chinese_name": "",
    "english_name": "",
    "akc_group": "",
    "origin": "",
    "original_purpose": "",
    "size_category": "",
    "average_weight_kg": {{"male": "", "female": ""}},
    "average_height_cm": {{"male": "", "female": ""}},
    "life_expectancy": "",
    "coat_type": "",
    "coat_colors": [],
    "primary_traits": [],
    "energy_level": 0,
    "intelligence_rank": 0,
    "trainability": 0,
    "with_family": "",
    "with_children": "",
    "with_other_dogs": "",
    "daily_exercise_minutes": 0,
    "apartment_friendly": false,
    "grooming_needs": "",
    "shedding_level": 0,
    "common_health_issues": [],
    "suitability_assessment": "",
    "knowledge_sources": []
  }}
}}

要求：
1. 提供准确、专业的犬品种信息
2. 所有评分均为1-10的整数
3. 不确定的信息注明"信息不足"或留空
4. 仅输出JSON格式

现在请为"{breed_name}"提供完整信息。"""

    prompt = prompt_template.format(breed_name=breed_name)

    api_request = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是犬类专家，请严格按照要求输出JSON格式，不添加任何额外文本。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    return api_request


def simple_api_call(api_key, breed_name):
    """简单的API调用函数"""

    base_url = "https://api.deepseek.com"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # 生成API请求
        api_request = query_dog_breed_info(breed_name)

        # 发送请求
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=api_request,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                return {"error": "API响应格式异常", "raw": str(result)[:200]}
        else:
            return {"error": f"HTTP {response.status_code}", "raw": response.text[:200]}

    except Exception as e:
        return {"error": str(e)}


def batch_process_breeds(api_key, breeds_list, output_file="dog_breeds_data.json", delay=1):
    """批量处理犬品种"""

    results = []

    print(f"开始处理 {len(breeds_list)} 个犬品种...")

    for i, breed_name in enumerate(breeds_list):
        print(f"处理第 {i + 1}/{len(breeds_list)} 个: {breed_name}")

        # 调用API
        result = simple_api_call(api_key, breed_name)

        # 添加品种名称
        if "error" not in result:
            result["breed_name"] = breed_name
            result["processed_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"  ✓ 成功")
        else:
            print(f"  ✗ 失败: {result.get('error', '未知错误')}")

        results.append(result)

        # 延迟避免速率限制
        time.sleep(delay)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_breeds": len(breeds_list),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成！结果已保存到: {output_file}")
    return results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"

    full_breeds_list = dog_breeds_list  #

    breeds_to_process = full_breeds_list

    print("=" * 50)
    print(f"准备处理 {len(breeds_to_process)} 个犬品种")
    print("=" * 50)

    results = batch_process_breeds(
        api_key=API_KEY,
        breeds_list=breeds_to_process,
        output_file="dog_breeds_results.json",
        delay=1  # 请求间隔1秒
    )

    # 显示统计信息
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])

    print(f"\n统计:")
    print(f"  成功: {successful}")
    print(f"  失败: {failed}")

    if successful > 0:
        print(f"\n前3个成功处理的结果预览:")
        for i, result in enumerate([r for r in results if "error" not in r][:3]):
            print(f"{i + 1}. {result.get('breed_name', '未知')}")
            if "breed" in result and "chinese_name" in result["breed"]:
                print(f"   中文名: {result['breed']['chinese_name']}")
                print(f"   英文名: {result['breed'].get('english_name', 'N/A')}")
                print(f"   体型: {result['breed'].get('size_category', 'N/A')}")
                print()