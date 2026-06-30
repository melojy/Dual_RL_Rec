import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ================= settings =================
BUSINESS_PATH = 'DUAL_RL_REC/data/yelp/yelp_academic_dataset_business.json'
REVIEW_PATH = 'DUAL_RL_REC/data/yelp/yelp_academic_dataset_review.json'
SOURCE_CITY = 'Philadelphia'
TARGET_CITY = 'New Orleans'
OUTPUT_FILE = 'DUAL_RL_REC/data/yelp/cross_domain_data.jsonl'
POI_META_FILE = 'DUAL_RL_REC/data/yelp/poi_metadata.json'
# ==========================================

def extract_and_save():

    print("Step 1: Loading Businesses...")
    biz_city_map = {}
    poi_details = {}
    with open(BUSINESS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['city'] in [SOURCE_CITY, TARGET_CITY]:
                biz_city_map[data['business_id']] = data['city']
                poi_details[data['business_id']] = {
                    'name': data['name'],
                    'city': data['city'],
                    'categories': data['categories'],
                    'stars': data['stars'],
                    'latitude': data['latitude'],
                    'longitude': data['longitude']
                }

    print("Step 2: Identifying common users...")
    user_cities = defaultdict(set)
    with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning for overlap"):
            data = json.loads(line)
            bid = data['business_id']
            if bid in biz_city_map:
                user_cities[data['user_id']].add(biz_city_map[bid])
    
    common_users = {uid for uid, cities in user_cities.items() if len(cities) >= 2}
    print(f"Confirmed {len(common_users)} common users.")

    print("Step 3: Extracting full sequences...")
    user_data = defaultdict(lambda: {'source_history': [], 'target_history': []})
    
    with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Collecting interactions"):
            data = json.loads(line)
            uid = data['user_id']
            bid = data['business_id']
            
            if uid in common_users and bid in biz_city_map:
                city = biz_city_map[bid]
                record = {
                    'business_id': bid,
                    'rating': data['stars'],
                    'date': data['date'],
                    'text': data['text']#[:300] 
                }
                
                if city == SOURCE_CITY:
                    user_data[uid]['source_history'].append(record)
                else:
                    user_data[uid]['target_history'].append(record)

    print("Step 4: Sorting and Saving...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for uid, domains in tqdm(user_data.items(), desc="Writing JSONL"):

            domains['source_history'].sort(key=lambda x: x['date'])
            domains['target_history'].sort(key=lambda x: x['date'])
            
            if domains['source_history'] and domains['target_history']:
                entry = {
                    'user_id': uid,
                    'source_city': SOURCE_CITY,
                    'target_city': TARGET_CITY,
                    'source_records': domains['source_history'],
                    'target_records': domains['target_history']
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(POI_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(poi_details, f, ensure_ascii=False, indent=2)

    print(f"Finished! Data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    extract_and_save()