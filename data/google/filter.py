import os
import json
import random

# ================= 配置区 =================
# 替换为你的真实文件夹路径
BASE_DIR = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local"

# 你上一轮生成的 2.1w 用户文件
IN_CROSS_DOMAIN = os.path.join(BASE_DIR, "google_cross_domain_data.jsonl")
IN_POI_META = os.path.join(BASE_DIR, "google_poi_metadata.json")

# 我们即将生成的 1.2w 用户精简版文件
OUT_CROSS_DOMAIN = os.path.join(BASE_DIR, "rl_cross_domain_data_12k.jsonl")
OUT_POI_META = os.path.join(BASE_DIR, "rl_poi_metadata_12k.json")

TARGET_USER_COUNT = 12000

# 固定随机种子，保证以后如果需要重新生成，抽到的还是这批人
random.seed(42) 

def downsample_data():
    # 1. 第一遍扫描：只拿所有用户的 ID
    all_users = []
    print(f"⏳ 正在扫描原始文件以获取用户池...")
    with open(IN_CROSS_DOMAIN, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            all_users.append(data['user_id'])

    print(f"📊 原始文件总用户数: {len(all_users)}")

    # 2. 随机抽样 12000 人
    if len(all_users) < TARGET_USER_COUNT:
        print("❌ 目标抽取人数大于实际人数，请检查文件！")
        return

    sampled_users = set(random.sample(all_users, TARGET_USER_COUNT))
    print(f"🎯 成功随机抽取 {len(sampled_users)} 个核心跨域用户。")

    # 3. 第二遍扫描：提取这 1.2w 人的完整轨迹，并收集他们去过的 POI 集合
    valid_pois = set()
    print(f"\n📝 正在生成精简版轨迹文件: {os.path.basename(OUT_CROSS_DOMAIN)} ...")
    with open(IN_CROSS_DOMAIN, 'r', encoding='utf-8') as fin, \
         open(OUT_CROSS_DOMAIN, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())
            if data['user_id'] in sampled_users:
                # 写入这 1.2w 人的序列
                fout.write(line)
                
                # 把他们去过的 POI 记录下来，作为清理元数据的“白名单”
                for r in data.get('source_records', []):
                    valid_pois.add(r['business_id'])
                for r in data.get('target_records', []):
                    valid_pois.add(r['business_id'])

    # 4. 加载庞大的原始元数据，依据“白名单”进行瘦身
    print(f"\n⏳ 正在加载并过滤原始元数据文件...")
    with open(IN_POI_META, 'r', encoding='utf-8') as f:
        old_meta = json.load(f)

    # 字典推导式：只保留白名单里的 POI
    new_meta = {pid: meta for pid, meta in old_meta.items() if pid in valid_pois}

    print(f"📝 正在生成精简版元数据文件: {os.path.basename(OUT_POI_META)} ...")
    with open(OUT_POI_META, 'w', encoding='utf-8') as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    # 5. 打印最终成果验收
    print("\n✅ 数据降采样与纯净过滤完美完成！")
    print(f"   => 最终保留跨域用户数: {TARGET_USER_COUNT}")
    print(f"   => 最终保留有效 POI 数: {len(new_meta)} (从原 {len(old_meta)} 中大幅瘦身)")

if __name__ == '__main__':
    downsample_data()