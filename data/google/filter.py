import os
import json
import random

# ================= settings =================

BASE_DIR = r"DUAL_RL_REC/data/google"

IN_CROSS_DOMAIN = os.path.join(BASE_DIR, "google_cross_domain_data.jsonl")
IN_POI_META = os.path.join(BASE_DIR, "google_poi_metadata.json")

OUT_CROSS_DOMAIN = os.path.join(BASE_DIR, "cross_domain_data.jsonl")
OUT_POI_META = os.path.join(BASE_DIR, "poi_metadata.json")

TARGET_USER_COUNT = 12000

random.seed(42) 

def downsample_data():

    all_users = []
    print(f"Scanning the original file to get the user pool...")
    with open(IN_CROSS_DOMAIN, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_users.append(data['user_id'])

    print(f"Total number of users in the original file: {len(all_users)}")

    if len(all_users) < TARGET_USER_COUNT:
        print("The number of users to be extracted is greater than the actual number!")
        return

    sampled_users = set(random.sample(all_users, TARGET_USER_COUNT))
    print(f"Successfully sampled {len(sampled_users)} cross-domain users.")

    valid_pois = set()
    print(f"Generating the trajectory file: {os.path.basename(OUT_CROSS_DOMAIN)} ...")
    with open(IN_CROSS_DOMAIN, 'r', encoding='utf-8') as fin, \
         open(OUT_CROSS_DOMAIN, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())
            if data['user_id'] in sampled_users:

                fout.write(line)
                
                for r in data.get('source_records', []):
                    valid_pois.add(r['business_id'])
                for r in data.get('target_records', []):
                    valid_pois.add(r['business_id'])

    print(f"Loading and filtering the original metadata file...")
    with open(IN_POI_META, 'r', encoding='utf-8') as f:
        old_meta = json.load(f)

    new_meta = {pid: meta for pid, meta in old_meta.items() if pid in valid_pois}

    print(f"Generating the metadata file: {os.path.basename(OUT_POI_META)} ...")
    with open(OUT_POI_META, 'w', encoding='utf-8') as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    print("Data downsampling and pure filtering completed.")
    print(f" Final number of cross-domain users retained: {TARGET_USER_COUNT}")
    print(f" Final number of valid POIs retained: {len(new_meta)} (from {len(old_meta)} after downsampling)")

if __name__ == '__main__':
    downsample_data()