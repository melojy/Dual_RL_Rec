import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ================= settings =================
BUSINESS_PATH = 'DUAL_RL_REC/data/yelp/yelp_academic_dataset_business.json'
REVIEW_PATH = 'DUAL_RL_REC/data/yelp/yelp_academic_dataset_review.json'
# ===========================================

def count_interactions_by_city():
    print("Step 1: Building Business-City Map...")
    
    business_city_map = {}
    
    with open(BUSINESS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Mapping Businesses"):
            data = json.loads(line)

            business_city_map[data['business_id']] = data['city']

    print(f"Mapped {len(business_city_map)} businesses.")

    print("\nStep 2: Counting Interactions (Reviews) per City...")
    city_counter = Counter()
    
    with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting Reviews"):
            data = json.loads(line)
            bid = data['business_id']
            
            if bid in business_city_map:
                city = business_city_map[bid]
                city_counter[city] += 1

    print("\nStep 3: Sorting and Displaying Top 20 Cities...")

    top_cities = city_counter.most_common(20)
    
    print(f"{'Rank':<5} {'City':<25} {'Interactions':<15}")
    print("-" * 50)
    for rank, (city, count) in enumerate(top_cities, 1):
        print(f"{rank:<5} {city:<25} {count:<15,}")

    return top_cities, business_city_map

def check_user_overlap(city_a, city_b, business_map):

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
        print("warning:Insufficient overlapping users")
    else:
        print("suitable for cross-domain experiments.")

if __name__ == '__main__':
    top_list, business_city_map = count_interactions_by_city()
    check_user_overlap('Philadelphia', 'New Orleans', business_city_map)