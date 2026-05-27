import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ================= 配置路径 =================
BUSINESS_PATH = 'D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_business.json'
REVIEW_PATH = 'D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_review.json'
# ===========================================

def count_interactions_by_city():
    print("Step 1: Building Business-City Map...")
    
    # 字典结构: {business_id: city_name}
    business_city_map = {}
    
    # 读取 Business 文件
    with open(BUSINESS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Mapping Businesses"):
            data = json.loads(line)
            # 这里可以加个简单的过滤，比如只保留 'Restaurants' 相关的
            # if data['categories'] and 'Restaurants' in data['categories']:
            business_city_map[data['business_id']] = data['city']

    print(f"Mapped {len(business_city_map)} businesses.")

    print("\nStep 2: Counting Interactions (Reviews) per City...")
    city_counter = Counter()
    
    # 读取 Review 文件 (流式读取，不占内存)
    with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting Reviews"):
            data = json.loads(line)
            bid = data['business_id']
            
            # 如果这个商家在我们之前的Map里 (有时候review里的商家可能已经被删除了)
            if bid in business_city_map:
                city = business_city_map[bid]
                city_counter[city] += 1

    print("\nStep 3: Sorting and Displaying Top 20 Cities...")
    # 获取交互量最大的前 20 个城市
    top_cities = city_counter.most_common(20)
    
    print(f"{'Rank':<5} {'City':<25} {'Interactions':<15}")
    print("-" * 50)
    for rank, (city, count) in enumerate(top_cities, 1):
        print(f"{rank:<5} {city:<25} {count:<15,}")

    return top_cities, business_city_map

def check_user_overlap(city_a, city_b, business_map):
    """
    检查两个城市有多少共同用户
    """
    users_in_a = set()
    users_in_b = set()
    
    print(f"\nChecking overlap between {city_a} and {city_b}...")
    
    with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning Users"):
            data = json.loads(line)
            bid = data['business_id']
            uid = data['user_id']
            
            if bid in business_map:
                city = business_map[bid]
                if city == city_a:
                    users_in_a.add(uid)
                elif city == city_b:
                    users_in_b.add(uid)
    
    common_users = users_in_a.intersection(users_in_b)
    
    print(f"Users in {city_a}: {len(users_in_a)}")
    print(f"Users in {city_b}: {len(users_in_b)}")
    print(f"Common Users (Cross-Domain Data): {len(common_users)}")
    
    if len(common_users) < 1000:
        print("⚠️ 警告：共同用户太少，训练容易过拟合！建议更换城市组合。")
    else:
        print("✅ 数据合格：适合做跨域实验。")

if __name__ == '__main__':
    top_list, business_city_map = count_interactions_by_city()
    check_user_overlap('Philadelphia', 'New Orleans', business_city_map)