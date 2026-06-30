import json
import os

def get_unique_users(file_path):
    users = set()
    print(f"Reading {file_path} ...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return users

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())

                users.add(data.get('user_id'))
            except Exception as e:
                continue
            
            if (i + 1) % 1000000 == 0:
                print(f"  Scanned {i + 1} reviews...")

    print(f" {file_path} completed, {len(users)} unique users.")
    return users

if __name__ == '__main__':
    #true path
    ca_file = r"DUAL_RL_REC/data/google/review-California.json"
    fl_file = r"DUAL_RL_REC/data/google/review-Florida.json"
    ny_file = r"DUAL_RL_REC/data/google/review-New_York.json"

    ca_users = get_unique_users(ca_file)
    fl_users = get_unique_users(fl_file)
    ny_users = get_unique_users(ny_file)

    ca_fl_overlap = ca_users.intersection(fl_users)
    ca_ny_overlap = ca_users.intersection(ny_users)
    fl_ny_overlap = fl_users.intersection(ny_users)
    all_overlap = ca_users.intersection(fl_users).intersection(ny_users)

    print("\n================ Cross-domain user statistics ================")
    print(f"California (CA) unique users: {len(ca_users)}")
    print(f"Florida    (FL) unique users: {len(fl_users)}")
    print(f"New York   (NY) unique users: {len(ny_users)}")
    print("-" * 50)
    print(f"CA and FL cross-domain users: {len(ca_fl_overlap)}")
    print(f"CA and NY cross-domain users: {len(ca_ny_overlap)}")
    print(f"FL and NY cross-domain users: {len(fl_ny_overlap)}")
    print(f"Three states cross-domain users: {len(all_overlap)}")
    print("==================================================")

