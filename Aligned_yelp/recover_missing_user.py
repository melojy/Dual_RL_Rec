import json
import time
import os
import re
from openai import OpenAI

# ================= settings =================
DEEPSEEK_API_KEY = "API KEY"  # true API Key
BASE_URL = "https://api.deepseek.com"


INPUT_FILE = r"DUAL_RL_REC/data/yelp/cross_domain_data.jsonl"

OUTPUT_FILE = r""
POI_META_FILE = r"DUAL_RL_REC/data/yelp/POI_mata_data.json"

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
    
    system_prompt = """You are a professional recommendation system data analysis expert.
Please strictly output the following JSON structure (ensure that attributes remain a key-value pair dictionary format):
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
    "place_poi_id": 4.5 
  }
}
Requirements: Extract fine-grained categories; attributes must be plain text with no nested dictionaries; output valid JSON directly."""

    user_prompt = f"This is the user's check-in records:\n{json.dumps(simplified_records, ensure_ascii=False)}"
    return system_prompt, user_prompt

def process_user_records(user_data):

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
                raise ValueError("No valid JSON found")
                
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
    
    print(f"Detected {len(completed_users)} completed records, processing remaining users...")
    
    processed_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        for line in fin:
            user_data = json.loads(line)
            uid = user_data["user_id"]
            
            if uid in completed_users:
                continue
                
            processed_count += 1
            print(f"[{processed_count + len(completed_users)}/1256] Processing user: {uid} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:
                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                fout.flush() 
                
            time.sleep(0.5)

    print(f"All users completed")

if __name__ == "__main__":
    main()