import os

# 🚨 请将这里替换为你电脑上 review 文件的真实绝对路径
file_path = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\data\google local\review-California.json" 

print("🚀 脚本已启动，正在寻找文件...")

if not os.path.exists(file_path):
    print(f"❌ 找不到文件，请检查路径是否正确: {file_path}")
else:
    print(f"✅ 找到文件！文件大小: {os.path.getsize(file_path) / (1024*1024*1024):.2f} GB")
    print("⏳ 正在抽取最前面的 2000 个字符...\n")
    
    try:
        # 只读取最前面的 2000 个字符看看长什么样，绝对不会卡死
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = f.read(2000)
            
            print("================ 📦 文件开头的数据 ==================")
            print(raw_data)
            print("=====================================================")
            print("\n✅ 抽取完毕！请观察上面的数据结构。")
            
    except Exception as e:
        print(f"❌ 读取过程中发生错误: {e}")