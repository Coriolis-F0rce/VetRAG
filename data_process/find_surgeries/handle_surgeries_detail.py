import json
import requests


def query_surgery_info(surgery_name):
    """调用DeepSeek API获取手术信息"""

    prompt_template = """你是一位专业的兽医外科专家。请根据以下手术名称，提供详细、准确、专业的犬类手术信息。

手术名称：{surgery_name}

请按照以下JSON格式输出信息，确保内容准确、专业，不虚构不明确的信息：

{{
  "surgery": {{
    "chinese_name": "",
    "english_name": "",
    "category": "",
    "sub_category": "",
    "indications": [],
    "alternative_therapies": [],
    "surgical_overview": "",
    "preoperative_preparation": [],
    "surgical_technique_brief": "",
    "postoperative_care": [],
    "common_complications": [],
    "prognosis": "",
    "cost_estimation": {{
      "range_cny": "",
      "notes": ""
    }},
    "breed_age_considerations": [],
    "knowledge_sources": []
  }}
}}

具体要求：

## 1. 手术名称
- 中文名称：提供准确的中文手术名称
- 英文名称：提供标准的英文手术名称（可选，但尽量提供）

## 2. 分类
- 主类别：从以下选择 → 生殖系统/骨科/软组织外科/眼科/牙科/神经外科/急诊外科/微创手术/诊断性手术/姑息性手术
- 子类别：根据手术性质细分（如：关节手术、肿瘤手术等）

## 3. 适应症（indications）
- 列出需要进行此手术的具体医学指征
- 每条适应症应简洁明确
- 按紧急程度或常见程度排序

## 4. 替代疗法简介（alternative_therapies）
- 列出可行的非手术治疗方案
- 说明每种替代疗法的适用情况和限制
- 注明"保守治疗"或"其他手术方案"

## 5. 手术简介
- 简要描述手术目的和基本原理
- 使用专业但易懂的语言
- 包含手术的基本步骤概述

## 6. 术前准备
- 列出必要的术前检查和准备
- 格式：项目 + 简要说明
- 包括：实验室检查、影像学检查、特殊准备等

## 7. 术后护理
- 分阶段列出术后护理要点
- 包括：疼痛管理、活动限制、伤口护理、复查时间点等
- 提供具体的护理指导

## 8. 常见并发症
- 列出可能发生的并发症（按发生率排序）
- 分为：术中并发症、术后早期并发症、术后晚期并发症
- 注明大概的发生率（如已知）

## 9. 预后
- 描述预期的治疗结果
- 包括：成功率、恢复时间、长期效果
- 注明影响预后的主要因素

## 10. 费用估算
- 提供人民币的费用范围参考
- 注明费用的影响因素（地区、医院等级、手术复杂程度等）
- 明确标注"仅供参考"

## 11. 品种/年龄特殊考虑
- 列出特定品种或年龄段的特殊注意事项
- 包括：品种易感性、年龄相关风险、体型影响等

## 12. 信息来源
- 列出参考的专业资料来源
- 格式：[作者/机构]《文献/书籍名称》
- 或：专业指南/共识名称

关键要求：
1. 所有信息必须基于兽医医学共识和现有知识
2. 不确定的信息请注明"信息不足"或留空
3. 使用专业术语，但保持可读性
4. 避免主观评价，保持客观描述
5. 中文输出，专业名词可附带英文

现在请为"{surgery_name}"提供完整信息。"""

    # 填充prompt
    prompt = prompt_template.format(surgery_name=surgery_name)

    # 构建API请求
    api_request = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一位兽医外科专家，精通犬类手术。请严格按照要求输出JSON格式，不添加任何额外文本。对于不确定的信息，可以留空或注明'信息不足'。"
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


def call_deepseek_api(api_key, surgery_names, output_file="surgery_database.json"):
    """批量调用API获取手术信息"""

    api_key =  api_key
    base_url = "https://api.deepseek.com"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    results = []

    print(f"开始处理 {len(surgery_names)} 个手术...")

    for i, surgery_name in enumerate(surgery_names):
        print(f"处理第 {i + 1}/{len(surgery_names)} 个: {surgery_name}")

        try:
            # 生成API请求
            api_request = query_surgery_info(surgery_name)

            # 发送请求
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=api_request,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                # 提取实际的JSON内容
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    try:
                        surgery_data = json.loads(content)
                        surgery_data["surgery_name"] = surgery_name
                        results.append(surgery_data)
                        print(f"  ✓ 成功")
                    except json.JSONDecodeError:
                        print(f"  ✗ JSON解析错误")
                        results.append({
                            "surgery_name": surgery_name,
                            "error": "JSON解析错误",
                            "raw_response": content[:200]
                        })
                else:
                    print(f"  ✗ API响应格式异常")
                    results.append({
                        "surgery_name": surgery_name,
                        "error": "API响应格式异常",
                        "raw_response": str(result)[:200]
                    })
            else:
                print(f"  ✗ HTTP错误 {response.status_code}")
                results.append({
                    "surgery_name": surgery_name,
                    "error": f"HTTP {response.status_code}",
                    "raw_response": response.text[:200]
                })

        except Exception as e:
            print(f"  ✗ 异常: {str(e)[:50]}")
            results.append({
                "surgery_name": surgery_name,
                "error": str(e)
            })

        # 避免速率限制
        import time
        time.sleep(1)

    # 保存结果
    output_data = {
        "total_surgeries": len(surgery_names),
        "successful": len([r for r in results if "surgery" in r]),
        "failed": len([r for r in results if "error" in r]),
        "data": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成！")
    print(f"成功: {output_data['successful']}")
    print(f"失败: {output_data['failed']}")
    print(f"结果已保存到: {output_file}")

    return output_data


# 使用示例
if __name__ == "__main__":
    # 手术列表（示例）
    surgery_list = [
        "犬绝育手术（Neutering/Spaying）",
        "公犬去势术（睾丸切除术）",
        "母犬卵巢子宫切除术",
        "剖腹产（犬分娩难产）",
        "人工授精辅助",
        "子宫蓄脓手术",
        "睾丸固定术（隐睾手术）",
        "前列腺手术（增生、脓肿）",
        "包皮过长矫正术",
        "胃内异物取出术",
        "胃扩张-扭转（GDV）矫正术",
        "胃溃疡修复术",
        "肠梗阻手术",
        "肠套叠复位术",
        "肠切除吻合术",
        "肛门腺手术（肛门囊炎、肿瘤）",
        "肝叶切除术",
        "胆道手术",
        "胰腺炎相关手术",
        "肾结石取出术",
        "肾切除术",
        "膀胱结石取出术",
        "膀胱肿瘤切除术",
        "尿道造口术（公犬尿道梗阻）",
        "尿道结石取出术",
        "闭合复位内固定",
        "开放复位内固定",
        "外固定支架",
        "髋关节发育不良手术（THR、FHO）",
        "十字韧带修复术（TPLO、TTA）",
        "髌骨脱位矫正术",
        "肘关节发育不良手术",
        "椎间盘突出手术",
        "颈椎不稳手术",
        "脊椎骨折修复",
        "皮肤肿瘤切除术",
        "乳腺肿瘤切除术",
        "脂肪瘤切除术",
        "恶性软组织肉瘤切除术",
        "耳血肿手术",
        "全耳道切除术（TECA）",
        "垂直耳道切除术",
        "白内障手术（超声乳化）",
        "青光眼手术",
        "眼睑内翻/外翻矫正术",
        "第三眼睑腺脱出（樱桃眼）修复",
        "眼球摘除术",
        "软腭过长矫正术",
        "鼻孔狭窄矫正术",
        "喉麻痹手术",
        "气管塌陷支架植入",
        "严重撕裂伤缝合",
        "咬伤清创修复",
        "腹部穿透伤修复",
        "食道异物取出",
        "气管异物取出",
        "消化道异物急诊手术",
        "脾破裂切除术",
        "肝破裂修复术",
        "大血管结扎术",
        "腹腔探查术",
        "活检取样",
        "腹腔镜绝育术",
        "关节腔探查清理",
        "关节镜辅助韧带修复",
        "胃镜异物取出",
        "结肠镜息肉切除",
        "支气管镜探查",
        "皮瓣移植术",
        "皮肤扩张术",
        "烧伤创面处理",
        "脑肿瘤切除术",
        "脑积水引流术",
        "外周神经修复",
        "动脉导管未闭结扎",
        "心包切除术",
        "牙齿拔除术",
        "龈下刮治术",
        "根管治疗",
        "颌骨骨折修复",
        "口腔肿瘤切除术",
        "腭裂修复术",
        "人道安乐死",
        "姑息性引流术（如胸水、腹水）",
        "镇痛泵植入术",
        "造瘘术（如食道造瘘、胃造瘘）",
        "导尿管植入",
        "压迫性肿瘤减积术",
        "脐疝/腹股沟疝修复",
        "腭裂修复",
        "肛门闭锁修复",
        "皮肤缺损修复",
        "肌肉功能重建",
        "美容性修复",
        "切开活检",
        "穿刺活检",
        "切除活检",
        "开腹探查术",
        "开胸探查术"
    ]
    call_deepseek_api("sk-2eedab5b21954b6bb26f7461706642f2",surgery_list)