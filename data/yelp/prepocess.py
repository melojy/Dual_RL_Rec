import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区 =================
BUSINESS_PATH = 'D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_business.json'
REVIEW_PATH = 'D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_review.json'
SOURCE_CITY = 'Philadelphia'
TARGET_CITY = 'New Orleans'
OUTPUT_FILE = 'D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\Yelp-JSON\Yelp JSON\yelp_dataset\cross_domain_data.jsonl'
POI_META_FILE = 'D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\Yelp-JSON\Yelp JSON\yelp_dataset\poi_metadata.json'
# ==========================================

def extract_and_save():
    # 1. 加载 Business 信息并筛选城市
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

    # 2. 第一次遍历 Review 找出共同用户 ID (你之前已经做过，这里快速复现)
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

    # 3. 第二次遍历 Review 提取这些用户的详细交互
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
                    'text': data['text']#[:300]  # 可以只保留一部分文本用于 EncoderRL，这里全部保存了
                }
                
                if city == SOURCE_CITY:
                    user_data[uid]['source_history'].append(record)
                else:
                    user_data[uid]['target_history'].append(record)

    # 4. 排序并保存为 JSONL
    print("Step 4: Sorting and Saving...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for uid, domains in tqdm(user_data.items(), desc="Writing JSONL"):
            # 按日期排序
            domains['source_history'].sort(key=lambda x: x['date'])
            domains['target_history'].sort(key=lambda x: x['date'])
            
            # 只有当两个城市都有真实数据时才保存
            if domains['source_history'] and domains['target_history']:
                entry = {
                    'user_id': uid,
                    'source_city': SOURCE_CITY,
                    'target_city': TARGET_CITY,
                    'source_records': domains['source_history'],
                    'target_records': domains['target_history']
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 保存 POI 字典以便后续环境快速检索
    with open(POI_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(poi_details, f, ensure_ascii=False, indent=2)

    print(f"Finished! Data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    extract_and_save()