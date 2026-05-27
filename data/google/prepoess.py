import os
import json
from collections import defaultdict

# ================= 配置区 =================
# 替换为你的真实文件夹路径
BASE_DIR = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local"

# 输入文件路径
CA_REVIEW = os.path.join(BASE_DIR, "review-California.json")
FL_REVIEW = os.path.join(BASE_DIR, "review-Florida.json")
CA_META = os.path.join(BASE_DIR, "meta-California.json")
FL_META = os.path.join(BASE_DIR, "meta-Florida.json")

# 输出文件路径
OUT_CROSS_DOMAIN = os.path.join(BASE_DIR, "google_cross_domain_data.jsonl")
OUT_POI_META = os.path.join(BASE_DIR, "google_poi_metadata.json")

MIN_INTERACTIONS = 10

def count_user_interactions(filepath):
    """统计单个文件中每个用户的交互次数"""
    user_counts = defaultdict(int)
    print(f"⏳ 正在统计 {os.path.basename(filepath)} 的用户频次...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                user_counts[data['user_id']] += 1
            except:
                continue
            if (i + 1) % 2000000 == 0:
                print(f"   已读取 {i + 1} 条评论...")
    return user_counts

def extract_reviews_for_users(filepath, valid_users):
    """提取有效用户的 review 数据，并转为 Yelp 格式"""
    records = defaultdict(list)
    poi_set = set()
    print(f"⏳ 正在为有效用户提取 {os.path.basename(filepath)} 的轨迹...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                uid = data['user_id']
                if uid in valid_users:
                    pid = data['gmap_id']
                    # Google Local 的 time 是毫秒，转换为 Yelp 的秒
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
    """提取被访问过的 POI 元数据，转为 Yelp 格式"""
    poi_meta = {}
    print(f"⏳ 正在提取 {os.path.basename(filepath)} 的有效 POI 元数据...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                pid = data['gmap_id']
                if pid in valid_pois:
                    # 将 List 形式的 category 转为 Yelp 风格的逗号字符串
                    raw_cats = data.get('category')
                    if raw_cats and isinstance(raw_cats, list):
                        category_str = ", ".join(raw_cats)
                    else:
                        category_str = "Unknown"

                    # 格式化 attributes
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
    # 1. 频次统计与严格 5-core 过滤
    ca_counts = count_user_interactions(CA_REVIEW)
    fl_counts = count_user_interactions(FL_REVIEW)

    # 找出在 CA 和 FL 都有 >= 5 次交互的用户
    valid_users = set()
    for uid, ca_cnt in ca_counts.items():
        if ca_cnt >= MIN_INTERACTIONS and fl_counts.get(uid, 0) >= MIN_INTERACTIONS:
            valid_users.add(uid)
            
    print(f"\n🎯 严格双边过滤完毕！共有 {len(valid_users)} 个用户满足跨域且两地均 >= {MIN_INTERACTIONS} 次交互。")
    if len(valid_users) == 0:
        print("❌ 满足条件的用户数为 0，程序终止。请考虑降低 MIN_INTERACTIONS。")
        return

    # 2. 提取交互轨迹并收集被访问的 POI
    ca_records, ca_pois = extract_reviews_for_users(CA_REVIEW, valid_users)
    fl_records, fl_pois = extract_reviews_for_users(FL_REVIEW, valid_users)

    # 3. 写入跨域 JSONL 文件并按时间排序
    print(f"\n📝 正在生成 {os.path.basename(OUT_CROSS_DOMAIN)} ...")
    with open(OUT_CROSS_DOMAIN, 'w', encoding='utf-8') as f:
        for uid in valid_users:
            # 必须按照时间戳对历史记录进行升序排序，保证序列推荐的时序正确性
            src_recs = sorted(ca_records.get(uid, []), key=lambda x: x['timestamp'])
            tgt_recs = sorted(fl_records.get(uid, []), key=lambda x: x['timestamp'])
            
            user_data = {
                "user_id": uid,
                "source_records": src_recs,
                "target_records": tgt_recs
            }
            f.write(json.dumps(user_data, ensure_ascii=False) + '\n')

    # 4. 提取元数据 (仅提取刚才收集到的有效 POI)
    ca_poi_meta = extract_meta_for_pois(CA_META, ca_pois, "California")
    fl_poi_meta = extract_meta_for_pois(FL_META, fl_pois, "Florida")

    # 合并元数据
    final_poi_meta = {**ca_poi_meta, **fl_poi_meta}

    # 5. 写入元数据 JSON 文件
    print(f"\n📝 正在生成 {os.path.basename(OUT_POI_META)} ...")
    with open(OUT_POI_META, 'w', encoding='utf-8') as f:
        json.dump(final_poi_meta, f, ensure_ascii=False, indent=2)

    print("\n✅ 数据处理全流程完美结束！你现在可以直接把这两个文件无缝喂给之前的模型了。")

if __name__ == '__main__':
    main()