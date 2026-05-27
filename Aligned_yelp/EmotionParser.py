import json
import time
from openai import OpenAI

# ================= 配置区 =================
DEEPSEEK_API_KEY = "sk-3d03fb0db4a94d92948ac373ac172690"  # 替换为你的真实 API Key
BASE_URL = "https://api.deepseek.com"

INPUT_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_Data\rl_cross_domain_data.jsonl"
OUTPUT_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_Data\llm_extracted_preferences.jsonl"
POI_META_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_Data\rl_poi_metadata.json"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# 提前加载 POI 元数据，辅助大模型进行 Schema 对齐
with open(POI_META_FILE, 'r', encoding='utf-8') as f:
    poi_metadata = json.load(f)

def build_prompt(records):
    """构建极致压缩、强制对齐 Schema 且要求细粒度分类的 Prompt"""
    
    simplified_records = []
    for r in records:
        poi_id = r["business_id"]
        meta = poi_metadata.get(poi_id, {})
        simplified_records.append({
            "poi_id": poi_id,
            "stars": r["stars"],
            "review": r["text"],
            "poi_categories": meta.get("categories", ""),
            "poi_attributes": meta.get("attributes", {})
        })
    
    system_prompt = """你是一个专业的推荐系统数据分析专家。
请阅读用户对多个地点的评价及其对应的地点元数据，并总结出该用户的【全局兴趣画像】以及对【每个地点的真实情绪得分】。
请严格输出以下 JSON 结构：
{
  "global_liked": {
    "categories": ["Sushi Bars", "Boutique Hotels"], // 提取用户喜欢的具体细分领域
    "attributes": {"WiFi": "u'free'", "OutdoorSeating": "True"} // 严格使用元数据中的属性键值对格式
  },
  "global_disliked": {
    "categories": ["Fast Food", "Dive Bars"], // 提取用户厌恶的具体细分领域
    "attributes": {"RestaurantsPriceRange2": "4", "NoiseLevel": "u'loud'"} // 厌恶的属性特点
  },
  "poi_scores": {
    "地点poi_id": 4.5 // 根据评论真实情绪给出的 1.0~5.0 评分
  }
}

【极其重要的分类（Categories）提取规则】：
绝对禁止使用过于宽泛的顶层类别（如 "Restaurants", "Food", "Hotels", "Shopping", "Nightlife" 等无意义的泛词）。
你必须从原始文本或元数据中，提取或总结出具体、细粒度的子类别！
例如：用 "Family-style Restaurant" 代替 "Restaurants"；用 "Japanese Sushi" 代替 "Food"；用 "Luxury Boutique Hotel" 代替 "Hotels"。

要求：
1. global_liked 和 global_disliked 必须是对该用户所有评论的全局大总结。
2. categories 必须遵守上述的细粒度规则，attributes 必须尽量与输入的地点元数据格式保持一致。
3. 直接输出合法 JSON，不要 Markdown 标记。"""

    user_prompt = f"这是该用户的打卡记录及 POI 元数据：\n{json.dumps(simplified_records, ensure_ascii=False)}"
    
    return system_prompt, user_prompt

def process_user_records(user_data):
    # 将源域和目标域的记录一起喂给模型，提取全局画像
    all_records = user_data.get("source_records", []) + user_data.get("target_records", [])
    if not all_records:
        return None
        
    system_prompt, user_prompt = build_prompt(all_records)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 
            )
            
            result_text = response.choices[0].message.content.strip()
            # 清洗 markdown 符号
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith("```"):
                result_text = result_text[3:-3].strip()
                
            parsed_json = json.loads(result_text)
            
            # 【核心修改】：只组装你要求的极简 JSON 格式，挂载 user_id
            final_user_profile = {
                "user_id": user_data["user_id"],
                "global_liked": parsed_json.get("global_liked", {"categories": [], "attributes": {}}),
                "global_disliked": parsed_json.get("global_disliked", {"categories": [], "attributes": {}}),
                "poi_scores": parsed_json.get("poi_scores", {})
            }
                
            return final_user_profile
            
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] API Error: {e}")
            time.sleep(2)
            
    return None

def main():
    print(f"🚀 开始使用 DeepSeek 提取用户极致纯净双极画像...")
    processed_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        lines = fin.readlines() 
        for i, line in enumerate(lines):
            user_data = json.loads(line)
            print(f"[{i+1}/{len(lines)}] 正在提取用户: {user_data['user_id']} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:
                # 严格按行输出 JSONL 格式
                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                processed_count += 1
                
            time.sleep(0.5) # 防止 API 并发超限

    print(f"\n✅ 处理完成！已成功提取并保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()