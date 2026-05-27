import json
import os

def get_unique_users(file_path):
    users = set()
    print(f"🚀 正在读取 {file_path} ...")
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        return users

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                # 我们只关心 user_id，不存其他占用内存的数据
                users.add(data.get('user_id'))
            except Exception as e:
                continue
            
            # 每处理 100 万条打印一次进度，防止你以为电脑死机了
            if (i + 1) % 1000000 == 0:
                print(f"   ⏳ 已扫描 {i + 1} 条评论...")

    print(f"✅ {file_path} 读取完成，共有 {len(users)} 个独立用户。")
    return users

if __name__ == '__main__':
    # 🚨 替换为你电脑上文件的真实绝对路径
    ca_file = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local\review-California.json"
    fl_file = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local\review-Florida.json"
    ny_file = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local\review-New_York.json"

    # 获取各自的用户集合
    ca_users = get_unique_users(ca_file)
    fl_users = get_unique_users(fl_file)
    ny_users = get_unique_users(ny_file)

    # 统计跨域交集
    ca_fl_overlap = ca_users.intersection(fl_users)
    ca_ny_overlap = ca_users.intersection(ny_users)
    fl_ny_overlap = fl_users.intersection(ny_users)
    all_overlap = ca_users.intersection(fl_users).intersection(ny_users)

    print("\n================ 📊 跨域用户统计结果 ================")
    print(f"California (CA) 独立用户数: {len(ca_users)}")
    print(f"Florida    (FL) 独立用户数: {len(fl_users)}")
    print(f"New York   (NY) 独立用户数: {len(ny_users)}")
    print("-" * 50)
    print(f"🌉 CA 和 FL 的跨域用户数: {len(ca_fl_overlap)}")
    print(f"🌉 CA 和 NY 的跨域用户数: {len(ca_ny_overlap)}")
    print(f"🌉 FL 和 NY 的跨域用户数: {len(fl_ny_overlap)}")
    print(f"🏆 三个州都有的跨域用户数: {len(all_overlap)}")
    print("==================================================")


#####统计结果
'''
✅ D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local\review-New_York.json 读取完成，共有 8086413 个独立用户。

================ 📊 跨域用户统计结果 ================
California (CA) 独立用户数: 14098916
Florida    (FL) 独立用户数: 13832725
New York   (NY) 独立用户数: 8086413
--------------------------------------------------
🌉 CA 和 FL 的跨域用户数: 549086
🌉 CA 和 NY 的跨域用户数: 459907
🌉 FL 和 NY 的跨域用户数: 689353
🏆 三个州都有的跨域用户数: 119221
==================================================
'''