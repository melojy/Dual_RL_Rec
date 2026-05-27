import json
import time
import re
from openai import OpenAI

# ================= 配置区 =================
DEEPSEEK_API_KEY = "sk-3d03fb0db4a94d92948ac373ac172690"  # 替换为你的真实 API Key
BASE_URL = "https://api.deepseek.com"

# ⚠️ 请替换为你的 Google 数据集真实路径
INPUT_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_google\rl_cross_domain_data_12k.jsonl"
OUTPUT_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_google\google_extracted_preferences.jsonl"
POI_META_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_google\rl_poi_metadata_12k.json"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

print("正在加载 POI 元数据...")
with open(POI_META_FILE, 'r', encoding='utf-8') as f:
    poi_metadata = json.load(f)

def clean_and_compress_records(records):
    """
    极简压缩与 ID 映射：将真实的长 ID 映射为 1, 2, 3 等短 ID
    """
    lines = []
    id_map = {}  # 用于保存 短ID -> 真实长ID 的映射字典
    
    # 限制最多处理 60 条，防止极端活跃用户撑爆 Token
    records = records[-60:] if len(records) > 60 else records
    
    for i, r in enumerate(records):
        poi_id = r.get("business_id", r.get("gmap_id", r.get("poi_id", "")))
        if not poi_id: continue
        
        # 🚀 ID 映射：为当前 POI 生成一个短数字 ID
        short_id = str(i + 1)
        id_map[short_id] = poi_id
        
        rating = r.get("stars", r.get("rating", 3.0))
        text = str(r.get("text", r.get("review", "")))
        text = re.sub(r'[^\w\s.,!?，。！？\'-]', '', text).replace('\n', ' ').replace('\r', '').strip()
        
        meta = poi_metadata.get(poi_id, {})
        # 新增提取 Name，并限制长度防止过长
        name = str(meta.get("name", "Unknown"))[:30]
        
        # 🚀 将全量画像样本的数量从 10 扩大到 20
        if i < 20:
            if len(text) > 60:
                text = text[:40] + "..." + text[-17:]
            if not text: text = "N/A"
                
            cats = meta.get("categories", meta.get("category", ""))
            if isinstance(cats, list): cats = ",".join(cats)
            
            attrs = meta.get("attributes", {})
            attr_str = ",".join([f"{k}:{v}" for k, v in list(attrs.items())[:3]]) if attrs else "N/A"
            
            # 使用 short_id，并加入 Name
            lines.append(f"ID:{short_id}|Name:{name}|★:{rating}|Cat:{cats[:40]}|Attr:{attr_str[:40]}|Rev:{text}")
        else:
            if len(text) > 25:
                text = text[:15] + "..." + text[-7:]
            if not text: text = "N/A"
            
            # 仅打分样本也带上 Name 辅助大模型判断
            lines.append(f"ID:{short_id}|Name:{name}|★:{rating}|Rev:{text}")
            
    return "\n".join(lines), id_map


def build_prompt(records):

    # 接收返回的文本和映射字典
    compressed_text, id_map = clean_and_compress_records(records)
    
    system_prompt = """You are a rigorous data analysis expert for recommendation systems. Please extract the user profile and score each POI based on the user's minimalist check-in records. Note: Records containing 'Cat' and 'Attr' represent POI samples, while those with 'Rev' represent user review samples.

    Please strictly output the following JSON format (do not output Markdown markers like ```json, just output the curly braces):

{
"global_liked": {"categories": ["sub-category 1"], "attributes": {"attribute_key": "value"}},
"global_disliked": {"categories": ["sub-category 2"], "attributes": {"attribute_key": "value"}},
"poi_scores": {"1": 4.5, "2": 3.0}
}

[Ensure the output adheres strictly to the following principles]:
1. Principle of Category Mutually Exclusive: The categories in global_liked and global_disliked must NEVER overlap. If similar categories exist, they must be subdivided with specific modifiers (e.g., liked: ["Luxury Hotel"], disliked: ["Noisy Motel"]).
2. Principle of Objective Environment: The attributes must be specific, objective physical or facility characteristics (e.g., "WiFi": "free").
3. Principle of Short-ID Scoring: The keys in poi_scores MUST be the minimalist pure numeric IDs read from the input (e.g., "1", "2", "3"). You must provide a comprehensive score ranging from 1.0 to 5.0 for all numeric IDs present in the input, based on the user's sentiment explicitly expressed in the reviews."""

    user_prompt = f"User Records: \n{compressed_text}"
    return system_prompt, user_prompt, id_map


def process_user_records(user_data):
    all_records = user_data.get("source_records", []) + user_data.get("target_records", [])
    if not all_records:
        return None
        
    # 接收映射字典 id_map
    system_prompt, user_prompt, id_map = build_prompt(all_records)
    
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
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'^```\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
                
            parsed_json = json.loads(result_text)
            
            # 🚀 核心还原：将大模型输出的短数字 ID 重新映射回真实的超长 Google ID
            raw_scores = parsed_json.get("poi_scores", {})
            mapped_scores = {}
            for short_id, score in raw_scores.items():
                short_id_str = str(short_id)
                if short_id_str in id_map:
                    # 获取真实的 Google POI ID
                    real_poi_id = id_map[short_id_str]
                    mapped_scores[real_poi_id] = score
            
            # 严格组装最终格式
            final_user_profile = {
                "user_id": user_data["user_id"],
                "global_liked": parsed_json.get("global_liked", {"categories": [], "attributes": {}}),
                "global_disliked": parsed_json.get("global_disliked", {"categories": [], "attributes": {}}),
                "poi_scores": mapped_scores # 使用映射还原后的真实得分字典
            }
                
            return final_user_profile
            
        except json.JSONDecodeError:
            print(f"  [Attempt {attempt+1}] JSON 解析失败，模型输出格式异常。")
            time.sleep(1)
        except Exception as e:
            print(f"  [Attempt {attempt+1}] API Error: {e}")
            time.sleep(2)
            
    return None

def main():
    print(f"🚀 开始使用 DeepSeek 提取 Google 用户画像 (极速省钱版)...")
    processed_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        lines = fin.readlines() 
        for i, line in enumerate(lines):
            user_data = json.loads(line)
            print(f"[{i+1}/{len(lines)}] 正在提取用户: {user_data.get('user_id', 'Unknown')} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:
                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                processed_count += 1
                
            # 降低 Sleep 时间提升效率，DeepSeek 并发允许的话可以调得更低
            time.sleep(0.1) 

    print(f"\n✅ 处理完成！成功提取 {processed_count} 个用户，已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()