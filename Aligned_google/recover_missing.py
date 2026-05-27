import json
import time
import re
from openai import OpenAI
import os

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
    """构建自带强约束规则的极简 Prompt"""
    # 接收返回的文本和映射字典
    compressed_text, id_map = clean_and_compress_records(records)
    
    system_prompt = """你是一个严谨的推荐系统数据分析专家。请根据用户的极简打卡记录提取画像和打分。
注意：记录中带有 'Cat' 和 'Attr' 的是画像样本，只有 'Rev' 的是打分样本。

请严格输出如下 JSON（不要输出 ```json 等标记，直接输出大括号）：
{
  "global_liked": {"categories": ["细分品类1"], "attributes": {"客观属性键": "值"}},
  "global_disliked": {"categories": ["细分品类2"], "attributes": {"客观属性键": "值"}},
  "poi_scores": {"1": 4.5, "2": 3.0}
}

【三大铁律 - 违反必究】：
1. 类别互斥原则：liked 和 disliked 的 categories 绝对不能有任何交叉！必须使用定语细分（如 liked: ["Luxury Hotel"], disliked: ["Noisy Motel"]）。
2. 客观环境原则：attributes 必须是具体的客观物理或设施特征（如 "WiFi": "free"）。绝对禁止使用主观废话。
3. 短 ID 打分原则：poi_scores 的键必须是你读取到的极简纯数字 ID（如 "1", "2", "3"）。你必须为输入中出现的所有数字 ID 提供 1.0~5.0 的综合评分！"""

    user_prompt = f"用户记录：\n{compressed_text}"
    return system_prompt, user_prompt, id_map



def process_user_records(user_data):
    all_records = user_data.get("source_records", []) + user_data.get("target_records", [])
    if not all_records:
        return None
        
    system_prompt, user_prompt, id_map = build_prompt(all_records)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 🚀 修复 1：增加 timeout 参数！
            # 强制规定如果大模型 30 秒内不吐出数据，直接掐断重试，绝对不允许无限期卡死！
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                timeout=30.0  
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'^```\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
                
            parsed_json = json.loads(result_text)
            
            raw_scores = parsed_json.get("poi_scores", {})
            mapped_scores = {}
            for short_id, score in raw_scores.items():
                short_id_str = str(short_id)
                if short_id_str in id_map:
                    mapped_scores[id_map[short_id_str]] = score
            
            final_user_profile = {
                "user_id": user_data["user_id"],
                "global_liked": parsed_json.get("global_liked", {"categories": [], "attributes": {}}),
                "global_disliked": parsed_json.get("global_disliked", {"categories": [], "attributes": {}}),
                "poi_scores": mapped_scores 
            }
                
            return final_user_profile
            
        except json.JSONDecodeError:
            print(f"  [Attempt {attempt+1}] JSON 解析失败，模型输出格式异常。")
            time.sleep(2)
        except Exception as e:
            # 这里会捕获 Timeout 超时错误和网络异常，然后安全进入下一次重试
            print(f"  [Attempt {attempt+1}] API 或网络异常: {e}")
            time.sleep(3) # 遇到网络抖动，多睡一会儿再重试
            
    print(f"  ❌ 放弃该用户: {user_data.get('user_id')} (已达最大重试次数)")
    return None


def main():
    print(f"🚀 开始使用 DeepSeek 提取 Google 用户画像 (断点续传加强版)...")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 🚀 修复 2：加载已处理的用户 ID，实现断点续传
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["user_id"])
                    except:
                        pass
    print(f"📦 检测到本地已存在 {len(processed_ids)} 个处理成功的用户，将自动跳过他们！")
    
    processed_count = 0
    
    # 🚀 修复 3：将写入模式从 'w' (覆盖) 改为 'a' (追加 Append)
    # 这样就算脚本断开100次，你的数据也会稳稳地接在文件末尾，一滴都不漏
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        lines = fin.readlines() 
        for i, line in enumerate(lines):
            user_data = json.loads(line)
            uid = user_data.get('user_id', 'Unknown')
            
            # 断点跳过逻辑
            if uid in processed_ids:
                continue
                
            print(f"[{i+1}/{len(lines)}] 正在提取用户: {uid} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:
                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                fout.flush() # 强制每次都立即写入硬盘，防止意外断电丢失
                processed_count += 1
                
            time.sleep(0.1) 

    print(f"\n✅ 本次新增处理完成！成功提取 {processed_count} 个用户，已追加保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
