import os
import json
from collections import defaultdict

# ================= settings =================
BASE_DIR = r"DUAL_RL_REC/data/google"

CA_REVIEW = os.path.join(BASE_DIR, "review-California.json")
FL_REVIEW = os.path.join(BASE_DIR, "review-Florida.json")
CA_META = os.path.join(BASE_DIR, "meta-California.json")
FL_META = os.path.join(BASE_DIR, "meta-Florida.json")

OUT_CROSS_DOMAIN = os.path.join(BASE_DIR, "google_cross_domain_data.jsonl")
OUT_POI_META = os.path.join(BASE_DIR, "google_poi_metadata.json")

MIN_INTERACTIONS = 10

def count_user_interactions(filepath):

    user_counts = defaultdict(int)
    print(f"Counting user interactions in {os.path.basename(filepath)}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                user_counts[data['user_id']] += 1
            except:
                continue
            if (i + 1) % 2000000 == 0:
                print(f"Read {i + 1} reviews...")
    return user_counts

def extract_reviews_for_users(filepath, valid_users):

    records = defaultdict(list)
    poi_set = set()
    print(f"Extracting trajectories for valid users in {os.path.basename(filepath)}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                uid = data['user_id']
                if uid in valid_users:
                    pid = data['gmap_id']

                    timestamp = int(data.get('time', 0)) // 1000 
                    
                    record = {
                        "business_id": pid,
                        "timestamp": timestamp,
                        "stars": float(data.get('rating', 0.0)),
                        "text": data.get('text') or ""
                    }
                    records[uid].append(record)
                    poi_set.add(pid)
            except:
                continue
    return records, poi_set

def extract_meta_for_pois(filepath, valid_pois, city_name):

    poi_meta = {}
    print(f"Extracting metadata for valid POIs in {os.path.basename(filepath)}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                pid = data['gmap_id']
                if pid in valid_pois:

                    raw_cats = data.get('category')
                    if raw_cats and isinstance(raw_cats, list):
                        category_str = ", ".join(raw_cats)
                    else:
                        category_str = "Unknown"

                    attributes = {}
                    misc = data.get('MISC')
                    if misc and isinstance(misc, dict):
                        for k, v in misc.items():
                            attributes[k] = str(v)

                    poi_meta[pid] = {
                        "name": data.get('name') or "Unknown",
                        "city": city_name,
                        "latitude": float(data.get('latitude', 0.0)),
                        "longitude": float(data.get('longitude', 0.0)),
                        "stars": float(data.get('avg_rating', 0.0)),
                        "categories": category_str,
                        "attributes": attributes
                    }
            except:
                continue
    return poi_meta

def main():

    ca_counts = count_user_interactions(CA_REVIEW)
    fl_counts = count_user_interactions(FL_REVIEW)

    valid_users = set()
    for uid, ca_cnt in ca_counts.items():
        if ca_cnt >= MIN_INTERACTIONS and fl_counts.get(uid, 0) >= MIN_INTERACTIONS:
            valid_users.add(uid)
            
    print(f"Found {len(valid_users)} users that have >= {MIN_INTERACTIONS} interactions in both domains")
    if len(valid_users) == 0:
        print("No users found that have >= {MIN_INTERACTIONS} interactions in both domains. Please consider lowering MIN_INTERACTIONS.")
        return

    ca_records, ca_pois = extract_reviews_for_users(CA_REVIEW, valid_users)
    fl_records, fl_pois = extract_reviews_for_users(FL_REVIEW, valid_users)

    print(f"Generating {os.path.basename(OUT_CROSS_DOMAIN)} ...")
    with open(OUT_CROSS_DOMAIN, 'w', encoding='utf-8') as f:
        for uid in valid_users:

            src_recs = sorted(ca_records.get(uid, []), key=lambda x: x['timestamp'])
            tgt_recs = sorted(fl_records.get(uid, []), key=lambda x: x['timestamp'])
            
            user_data = {
                "user_id": uid,
                "source_records": src_recs,
                "target_records": tgt_recs
            }
            f.write(json.dumps(user_data, ensure_ascii=False) + '\n')

    ca_poi_meta = extract_meta_for_pois(CA_META, ca_pois, "California")
    fl_poi_meta = extract_meta_for_pois(FL_META, fl_pois, "Florida")

    final_poi_meta = {**ca_poi_meta, **fl_poi_meta}

    print(f"Generating {os.path.basename(OUT_POI_META)} ...")
    with open(OUT_POI_META, 'w', encoding='utf-8') as f:
        json.dump(final_poi_meta, f, ensure_ascii=False, indent=2)

    print("Data processing completed successfully.")

if __name__ == '__main__':
    main()