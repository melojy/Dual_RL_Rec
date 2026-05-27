import json
import time
import os
import re
from openai import OpenAI

# ================= 配置区 =================
DEEPSEEK_API_KEY = "sk-3d03fb0db4a94d92948ac373ac172690"  # 填入你的 API Key
BASE_URL = "https://api.deepseek.com"

# 原始全量数据
INPUT_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_Data\rl_cross_domain_data.jsonl"
# 你目前跑出来的 1216 个人的文件（作为基座，继续往里追加）
OUTPUT_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_Data\llm_extracted_preferences.jsonl"
POI_META_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_Data\rl_poi_metadata.json"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

with open(POI_META_FILE, 'r', encoding='utf-8') as f:
    poi_metadata = json.load(f)

def build_prompt(records):
    simplified_records = []
    for r in records:
        poi_id = r["business_id"]
        meta = poi_metadata.get(poi_id, {})
        
        text = r["text"]
        truncated_text = text[:100] + " ... " + text[-100:] if len(text) > 200 else text
            
        raw_attrs = meta.get("attributes", {})
        cleaned_attrs = {}
        if raw_attrs:
            for k in list(raw_attrs.keys())[:3]:
                val_str = str(raw_attrs[k])
                clean_val = val_str.replace("u'", "").replace("'", "").replace("{", "").replace("}", "").replace('"', '')
                cleaned_attrs[k] = clean_val.strip()

        simplified_records.append({
            "poi_id": poi_id,
            "stars": r["stars"],
            "review": truncated_text, 
            "poi_categories": meta.get("categories", ""),
            "poi_attributes": cleaned_attrs
        })
    
    system_prompt = """你是一个专业的推荐系统数据分析专家。
请严格输出以下 JSON 结构（务必保持 attributes 为键值对字典格式）：
{
  "global_liked": {
    "categories": ["Japanese Sushi", "Luxury Boutique Hotels"], 
    "attributes": {"WiFi": "free", "OutdoorSeating": "True"} 
  },
  "global_disliked": {
    "categories": ["Fast Food", "Dive Bars"], 
    "attributes": {"NoiseLevel": "loud"} 
  },
  "poi_scores": {
    "地点poi_id": 4.5 
  }
}
要求：提取细粒度 categories；attributes 必须是纯文本无嵌套字典；直接输出合法 JSON。"""

    user_prompt = f"这是用户的打卡记录：\n{json.dumps(simplified_records, ensure_ascii=False)}"
    return system_prompt, user_prompt

def process_user_records(user_data):
    # 【核心绝杀】：只取源域最近 20 条 + 目标域最近 20 条！杜绝 Token 爆炸和 JSON 截断！
    src_records = user_data.get("source_records", [])[-20:]
    tgt_records = user_data.get("target_records", [])[-20:]
    all_records = src_records + tgt_records
    
    if not all_records: return None
        
    system_prompt, user_prompt = build_prompt(all_records)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"} 
            )
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            else:
                raise ValueError("未找到合法 JSON")
                
            parsed_json = json.loads(result_text)
            return {
                "user_id": user_data["user_id"],
                "global_liked": parsed_json.get("global_liked", {"categories": [], "attributes": {}}),
                "global_disliked": parsed_json.get("global_disliked", {"categories": [], "attributes": {}}),
                "poi_scores": parsed_json.get("poi_scores", {})
            }
        except Exception as e:
            print(f"  [Attempt {attempt+1}/3] Error: {e}")
            time.sleep(2)
            
    # 彻底失败兜底
    return {
        "user_id": user_data["user_id"],
        "global_liked": {"categories": [], "attributes": {}},
        "global_disliked": {"categories": [], "attributes": {}},
        "poi_scores": {}
    }

def main():
    completed_users = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    completed_users.add(json.loads(line)['user_id'])
    
    print(f"🚀 已检测到 {len(completed_users)} 个完成记录，即将处理剩余怪兽级用户...")
    
    processed_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        for line in fin:
            user_data = json.loads(line)
            uid = user_data["user_id"]
            
            if uid in completed_users:
                continue
                
            processed_count += 1
            print(f"[{processed_count + len(completed_users)}/1256] 正在镇压用户: {uid} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:
                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                fout.flush() 
                
            time.sleep(0.5)

    print(f"\n✅ 全部 1256 名用户收官，完美落幕！")

if __name__ == "__main__":
    main()