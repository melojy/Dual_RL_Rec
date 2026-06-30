import json
import time
from openai import OpenAI

# ================= SETTING =================
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
        simplified_records.append({
            "poi_id": poi_id,
            "stars": r["stars"],
            "review": r["text"],
            "poi_categories": meta.get("categories", ""),
            "poi_attributes": meta.get("attributes", {})
        })
    
    system_prompt = """You are a professional recommendation system data analysis expert.
Please read the user's reviews of multiple places and their corresponding place metadata, and summarize the user's 【Global Interest Profile】 as well as the 【Real Sentiment Score for Each Place】.
You must strictly output the following JSON structure:
{
  "global_liked": {
    "categories": ["Sushi Bars", "Boutique Hotels"], // Extract the specific subcategories the user likes
    "attributes": {"WiFi": "u'free'", "OutdoorSeating": "True"} // Strictly use the attribute key-value pair format from the metadata
  },
  "global_disliked": {
    "categories": ["Fast Food", "Dive Bars"], // Extract the specific subcategories the user dislikes
    "attributes": {"RestaurantsPriceRange2": "4", "NoiseLevel": "u'loud'"} // Disliked attribute characteristics
  },
  "poi_scores": {
    "place_poi_id": 4.5 // A 1.0~5.0 rating based on the real sentiment expressed in the reviews
  }
}

【Extremely Important Category Extraction Rules】:
It is absolutely forbidden to use overly broad top-level categories (such as meaningless general terms like "Restaurants", "Food", "Hotels", "Shopping", "Nightlife").
You MUST extract or summarize specific, fine-grained subcategories from the original text or metadata!
For example: use "Family-style Restaurant" instead of "Restaurants"; use "Japanese Sushi" instead of "Food"; use "Luxury Boutique Hotel" instead of "Hotels".

Requirements:
1. global_liked and global_disliked must be an overall global summary based on all of the user's reviews.
2. categories must strictly adhere to the fine-grained rule above, and attributes should match the format of the input place metadata as closely as possible.
3. Output valid JSON directly, without Markdown formatting."""

    user_prompt = f"This is the user's check-in records and POI metadata:\n{json.dumps(simplified_records, ensure_ascii=False)}"
    
    return system_prompt, user_prompt

def process_user_records(user_data):

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

            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith("```"):
                result_text = result_text[3:-3].strip()
                
            parsed_json = json.loads(result_text)
            
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
    print(f"Start to extract the user's Global Interest Profile...")
    processed_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        lines = fin.readlines() 
        for i, line in enumerate(lines):
            user_data = json.loads(line)
            print(f"[{i+1}/{len(lines)}] Extracting user: {user_data['user_id']} ...")
            
            final_profile = process_user_records(user_data)
            
            if final_profile:

                fout.write(json.dumps(final_profile, ensure_ascii=False) + '\n')
                processed_count += 1
                
            time.sleep(0.5)

    print(f"Successfully extracted and saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()