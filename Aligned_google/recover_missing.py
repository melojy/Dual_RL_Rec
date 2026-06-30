import json
import time
import re
from openai import OpenAI
import os

# ================= settings =================
DEEPSEEK_API_KEY = "API KEY"  #  API Key
BASE_URL = "https://api.deepseek.com"

#  DATA PATH
INPUT_FILE = r"DUAL_RL_REC/data/google/cross_domain_data.jsonl"
OUTPUT_FILE = r""
POI_META_FILE = r"DUAL_RL_REC/data/google/POI_mata_data.json"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

print("loading POI metadata...")
with open(POI_META_FILE, 'r', encoding='utf-8') as f:
    poi_metadata = json.load(f)

def clean_and_compress_records(records):
    lines = []
    id_map = {} 
    
    records = records[-60:] if len(records) > 60 else records
    
    for i, r in enumerate(records):
        poi_id = r.get("business_id", r.get("gmap_id", r.get("poi_id", "")))
        if not poi_id: continue
        
        short_id = str(i + 1)
        id_map[short_id] = poi_id
        
        rating = r.get("stars", r.get("rating", 3.0))
        text = str(r.get("text", r.get("review", "")))
        text = re.sub(r'[^\w\s.,!?，。！？\'-]', '', text).replace('\n', ' ').replace('\r', '').strip()
        
        meta = poi_metadata.get(poi_id, {})
        name = str(meta.get("name", "Unknown"))[:30]
        
        if i < 20:
            if len(text) > 60:
                text = text[:40] + "..." + text[-17:]
            if not text: text = "N/A"
                
            cats = meta.get("categories", meta.get("category", ""))
            if isinstance(cats, list): cats = ",".join(cats)
            
            attrs = meta.get("attributes", {})
            attr_str = ",".join([f"{k}:{v}" for k, v in list(attrs.items())[:3]]) if attrs else "N/A"
            
            lines.append(f"ID:{short_id}|Name:{name}|★:{rating}|Cat:{cats[:40]}|Attr:{attr_str[:40]}|Rev:{text}")
        else:
            if len(text) > 25:
                text = text[:15] + "..." + text[-7:]
            if not text: text = "N/A"
            
            lines.append(f"ID:{short_id}|Name:{name}|★:{rating}|Rev:{text}")
            
    return "\n".join(lines), id_map


def build_prompt(records):

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
            print(f"  [Attempt {attempt+1}] JSON error")
            time.sleep(2)
        except Exception as e:

            print(f"  [Attempt {attempt+1}] API or network error: {e}")
            time.sleep(3) 
            
    print(f" give up user: {user_data.get('user_id')} (reached maximum retry times)")
    return None


def main():
    print(f"extracting Google user profile...")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["user_id"])
                    except:
                        pass
    print(f"detected {len(processed_ids)} users that have been processed, will skip them!")
    
    processed_count = 0
    

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        lines = fin.readlines() 
        for i, line in enumerate(lines):
            user_data = json.loads(line)
            uid = user_data.get('user_id', 'Unknown')
            
            if uid in processed_ids:
                continue
                
            print(f"[{i+1}/{len(lines)}] extracting user: {uid} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:
                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                fout.flush() 
                processed_count += 1
                
            time.sleep(0.1) 

    print(f"\n✅ this time added {processed_count} users, saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
