import os
# 🚀 绝杀 1：在导入 torch 之前，注入底层环境变量，命令 PyTorch 动态合并显存碎片！彻底免疫长时间运行的碎片化 OOM！
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import json
import random
import numpy as np
import math
import datetime
import gc

from AgentRL import GlobalConfig, PPOAgent
from EncoderRL import DualPrototypeAlignmentModel, SemanticTokenizer
from EnvironmentRL import EnhancedTravelPlanningEnv

# ==========================================
# 📂 路径配置区 (请确保路径与你本地一致)
# ==========================================
RAW_DATA_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_yelp\rl_cross_domain_data.jsonl"
LLM_PROFILE_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_yelp\llm_extracted_preferences.jsonl"
POI_META_FILE = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\Aligned_yelp\rl_poi_metadata.json"
KDDC_TEST_CANDIDATES = r"D:\VsCodeProjects\RL（完美时间＋路程个性化，尝试多城市，2025.9.13 - 副本\yelp_candidate_N_P.json"

def build_training_candidates(raw_data_file, poi_meta, config):
    train_candidates = {}
    all_source_pois = [pid for pid, meta in poi_meta.items() if meta.get('city') == 'New Orleans']
    with open(raw_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            user_data = json.loads(line)
            uid = user_data["user_id"]
            gt_pois = list(set([r['business_id'] for r in user_data.get('target_records', [])]))[:config.MAX_POS_POIS]
            noise_needed = config.TOTAL_CANDIDATES - len(gt_pois)
            available_noise = list(set(all_source_pois) - set(gt_pois))
            noise_pois = random.sample(available_noise, min(noise_needed, len(available_noise)))
            final_100 = gt_pois + noise_pois
            while len(final_100) < config.TOTAL_CANDIDATES: final_100.append(all_source_pois[0])
            random.shuffle(final_100)
            train_candidates[uid] = final_100
    return train_candidates

def load_data(config):
    with open(POI_META_FILE, 'r', encoding='utf-8') as f: poi_meta = json.load(f)
    
    def truncate_text(val, max_chars=150, max_items=4):
        if isinstance(val, str): return val[:max_chars]
        elif isinstance(val, dict): return {k: truncate_text(v, max_chars, max_items) for k, v in val.items()}
        elif isinstance(val, list): return [truncate_text(v, max_chars, max_items) for v in val[:max_items]]
        return val

    llm_profiles = {}
    with open(LLM_PROFILE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d = truncate_text(d, max_chars=150, max_items=4)
                llm_profiles[d['user_id']] = d

    with open(KDDC_TEST_CANDIDATES, 'r', encoding='utf-8') as f: test_candidates = json.load(f)

    target_ground_truth = {}
    with open(RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            target_ground_truth[d['user_id']] = [r['business_id'] for r in d.get('source_records', [])]   

    train_candidates = build_training_candidates(RAW_DATA_FILE, poi_meta, config)
    return poi_meta, llm_profiles, train_candidates, test_candidates, target_ground_truth

def p_rec(topk, gt): return len(set(topk) & set(gt)) / len(gt) if gt else 0.0
def p_pre(topk, gt): return len(set(topk) & set(gt)) / len(topk) if topk else 0.0
def p_ndcg(topk, gt):
    dcg = sum([1.0 / math.log2(i + 2) for i, p in enumerate(topk) if p in gt])
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(gt), len(topk)))])
    return dcg / idcg if idcg > 0 else 0.0
def p_mrr(topk, gt):
    for i, p in enumerate(topk):
        if p in gt: return 1.0 / (i + 1)
    return 0.0
def p_hr(topk, gt): return 1.0 if set(topk) & set(gt) else 0.0

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.linear")

# ==========================================
# 🌟 论文级双域独立映射可视化（隐藏坐标轴 + 隐藏图例 + 居中对称版）
# ==========================================
def visualize_embeddings(encoder, train_env, test_env, tokenizer, episode, config, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True) 
    encoder.eval()
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        return

    # ==========================================
    # 🎨 论文作图核心参数调整区 (请在这里自行微调)
    # ==========================================
    # 1. 留白比例 (Padding)
    BASE_PAD = 0.05          # 基础四周留白，0.08 表示 8% 
    
    # 🚀 去掉图例后，不再需要大幅向右扩充，恢复居中对称的留白
    RIGHT_EXTEND = 1.05      # 右侧稍微留一点呼吸空间即可
    TOP_EXTEND = 1.05        # 上侧给标题留出空间
    
    # 2. 文本字号 (Font Size)
    TITLE_SIZE = 25         # 图片标题字号 (稍微再放大一点，更显大气)
    
    # 3. 散点大小 (Scatter Size)
    SIZE_POI = 30            # 背景普通商铺圆点的大小
    SIZE_LIKE = 350          # 红色五角星 (P_like) 的大小
    SIZE_DISLIKE = 160       # 蓝色叉号 (P_dislike) 的大小
    # ==========================================

    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            all_src_poi_embs, all_tgt_poi_embs = [], []
            all_like_embs, all_dislike_embs = [], []
            
            test_users = [uid for uid in test_env.candidates_dict.keys() if uid in test_env.llm_profiles][:100]
            if not test_users: return
            
            for i, uid in enumerate(test_users):
                _, src_candidate_embs, _ = train_env.reset(uid, tokenizer)
                all_src_poi_embs.append(src_candidate_embs.squeeze(0).cpu().numpy())

                state, tgt_candidate_embs, _ = test_env.reset(uid, tokenizer)
                all_tgt_poi_embs.append(tgt_candidate_embs.squeeze(0).cpu().numpy())
                
                all_like_embs.append(state[:config.EMBED_DIM].cpu().numpy())
                all_dislike_embs.append(state[config.EMBED_DIM:2*config.EMBED_DIM].cpu().numpy())
                
                del src_candidate_embs, tgt_candidate_embs, state
                if (i + 1) % 10 == 0: torch.cuda.empty_cache()
            
            all_src_poi_embs = np.unique(np.concatenate(all_src_poi_embs, axis=0), axis=0) 
            all_tgt_poi_embs = np.unique(np.concatenate(all_tgt_poi_embs, axis=0), axis=0) 
            all_like_embs, all_dislike_embs = np.array(all_like_embs), np.array(all_dislike_embs)
            
            combined = np.vstack([all_src_poi_embs, all_tgt_poi_embs, all_like_embs, all_dislike_embs])
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            reduced = tsne.fit_transform(combined)
            
            num_src = len(all_src_poi_embs)
            num_tgt = len(all_tgt_poi_embs)
            num_likes = len(all_like_embs)
            
            src_2d = reduced[:num_src]
            tgt_2d = reduced[num_src : num_src+num_tgt]
            like_2d = reduced[num_src+num_tgt : num_src+num_tgt+num_likes]
            dislike_2d = reduced[num_src+num_tgt+num_likes :]
            
            # 使用提取出的参数计算留白
            x_min, x_max = reduced[:, 0].min(), reduced[:, 0].max()
            y_min, y_max = reduced[:, 1].min(), reduced[:, 1].max()
            x_pad = (x_max - x_min) * BASE_PAD
            y_pad = (y_max - y_min) * BASE_PAD
            
            # 动态计算最终坐标轴界限
            final_x_min = x_min - x_pad
            final_x_max = x_max + x_pad * RIGHT_EXTEND
            final_y_min = y_min - y_pad
            final_y_max = y_max + y_pad * TOP_EXTEND
            
            # ==========================================
            # 🎨 绘图 1: 源域 (Source Domain) 分布图
            # ==========================================
            plt.figure(figsize=(10, 8))
            plt.scatter(src_2d[:, 0], src_2d[:, 1], c='#4DB6AC', alpha=0.45, s=SIZE_POI, edgecolors='none')
            plt.scatter(like_2d[:, 0], like_2d[:, 1], c='#E53935', marker='*', s=SIZE_LIKE, edgecolors='white', linewidths=1.2, zorder=5)
            plt.scatter(dislike_2d[:, 0], dislike_2d[:, 1], c='#1E88E5', marker='X', s=SIZE_DISLIKE, edgecolors='white', linewidths=1.5, zorder=5)
            
            plt.xlim(final_x_min, final_x_max)
            plt.ylim(final_y_min, final_y_max)
            
            plt.title(f'Feature Space: Source Domain (Epoch {episode})', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
            
            # 🚀 彻底隐藏坐标轴刻度及数字
            plt.xticks([])
            plt.yticks([])
            
            plt.grid(True, linestyle='--', alpha=0.4, color='#B0BEC5')
            
            ax = plt.gca()
            for spine in ax.spines.values(): 
                spine.set_color('#546E7A')
                spine.set_linewidth(1.5)
            
            src_filename = os.path.join(save_dir, f'embeddings_source_epoch_{episode}.png')
            plt.savefig(src_filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            # ==========================================
            # 🎨 绘图 2: 目标域 (Target Domain) 分布图
            # ==========================================
            plt.figure(figsize=(10, 8))
            plt.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c='#66BB6A', alpha=0.45, s=SIZE_POI, edgecolors='none')
            plt.scatter(like_2d[:, 0], like_2d[:, 1], c='#E53935', marker='*', s=SIZE_LIKE, edgecolors='white', linewidths=1.2, zorder=5)
            plt.scatter(dislike_2d[:, 0], dislike_2d[:, 1], c='#1E88E5', marker='X', s=SIZE_DISLIKE, edgecolors='white', linewidths=1.5, zorder=5)
            
            plt.xlim(final_x_min, final_x_max)
            plt.ylim(final_y_min, final_y_max)
            
            plt.title(f'Feature Space: Target Domain (Epoch {episode})', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
            
            # 🚀 彻底隐藏坐标轴刻度及数字
            plt.xticks([])
            plt.yticks([])
            
            plt.grid(True, linestyle='--', alpha=0.4, color='#B0BEC5')
            
            ax = plt.gca()
            for spine in ax.spines.values(): 
                spine.set_color('#546E7A')
                spine.set_linewidth(1.5)
            
            tgt_filename = os.path.join(save_dir, f'embeddings_target_epoch_{episode}.png')
            plt.savefig(tgt_filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close('all') 
            
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n📸 [可视化] 论文级双域特征图已分别保存至:\n -> {src_filename}\n -> {tgt_filename}\n")

def evaluate_zero_shot(agent, encoder, test_env, tokenizer, target_ground_truth, config):
    agent.eval()
    encoder.eval()
    
    test_users = [uid for uid in test_env.candidates_dict.keys() if uid in test_env.llm_profiles]
    if not test_users: return
    
    k_list = [1, 5, 10, 15, 20]
    metrics = {k: {'Recall': [], 'Pre': [], 'NDCG': [], 'MRR': [], 'HR_ACC': []} for k in k_list}
    
    def _eval_single_user(uid):
        gt = target_ground_truth.get(uid, [])
        if not gt: return None
        
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                state, candidate_embs, mask = test_env.reset(uid, tokenizer)
                recommended_pois = []
                
                for _ in range(20): 
                    dist, _ = agent(state.unsqueeze(0), candidate_embs.unsqueeze(0), mask.unsqueeze(0))
                    action_idx = torch.argmax(dist.probs, dim=-1).item()
                    next_state, next_candidate_embs, next_mask, _, _, done = test_env.step(action_idx)
                    recommended_pois.append(test_env.candidate_ids[action_idx])
                    
                    del state, candidate_embs, mask, dist
                    state, candidate_embs, mask = next_state, next_candidate_embs, next_mask
                    if done: break
                    
                del state, candidate_embs, mask
                return recommended_pois, gt

    for i, uid in enumerate(test_users):
        res = _eval_single_user(uid)
        if res is not None:
            recommended_pois, gt = res
            for k in k_list:
                topk = recommended_pois[:k]
                metrics[k]['Recall'].append(p_rec(topk, gt))
                metrics[k]['Pre'].append(p_pre(topk, gt))
                metrics[k]['NDCG'].append(p_ndcg(topk, gt))
                metrics[k]['MRR'].append(p_mrr(topk, gt))
                metrics[k]['HR_ACC'].append(p_hr(topk, gt))
        
        if (i + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
                
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{curr_time}] =================== 测试集最终评估结果 ===================")
    for k in k_list:
        rec = np.mean(metrics[k]['Recall']) if metrics[k]['Recall'] else 0.0
        pre = np.mean(metrics[k]['Pre']) if metrics[k]['Pre'] else 0.0
        ndcg = np.mean(metrics[k]['NDCG']) if metrics[k]['NDCG'] else 0.0
        mrr = np.mean(metrics[k]['MRR']) if metrics[k]['MRR'] else 0.0
        hr = np.mean(metrics[k]['HR_ACC']) if metrics[k]['HR_ACC'] else 0.0
        print(f"[{curr_time}] [K={k:<2}] Recall: {rec:.4f} | Pre: {pre:.4f} | NDCG: {ndcg:.4f} | MRR: {mrr:.4f} | HR/ACC: {hr:.4f}")
    print(f"[{curr_time}] ==========================================================\n")
    
    agent.train()
    encoder.train()
    gc.collect()
    torch.cuda.empty_cache()

    return np.mean(metrics[20]['Recall'])

def main():
    config = GlobalConfig()
    
    if hasattr(config, 'TEXT_MAX_LEN'): config.TEXT_MAX_LEN = min(config.TEXT_MAX_LEN, 256)
        
    tokenizer = SemanticTokenizer(config.VOCAB_SIZE, config.TEXT_MAX_LEN)
    poi_meta, llm_profiles, train_candidates, test_candidates, target_ground_truth = load_data(config)
    
    encoder = DualPrototypeAlignmentModel(config).to(config.DEVICE)
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=config.LR_RL_FINETUNE)
    agent = PPOAgent(config).to(config.DEVICE)
    
    train_env = EnhancedTravelPlanningEnv(config, encoder, train_candidates, poi_meta, llm_profiles)
    test_env = EnhancedTravelPlanningEnv(config, encoder, test_candidates, poi_meta, llm_profiles)
    
    valid_train_users = [uid for uid in train_candidates.keys() if uid in llm_profiles]
    train_users = valid_train_users 
    
    history_logs = {"episode": [], "encoder_loss": [], "actor_loss": [], "critic_loss": [], "avg_reward": []}
    encoder_buffer = {'like': [], 'dislike': [], 'poi': [], 'reward': []}
    
    ENCODER_UPDATE_FREQ = 50  
    MAX_EPISODES = 6000     
    
    episode_count = 0
    total_reward = 0
    
    print("\n🚀 全系统端到端联合训练启动...")
    print("📸 [初始化] 正在绘制未经强化的初始特征分布 (Epoch 0)...")
    
    # 🌟 修改调用传参，注入 train_env
    visualize_embeddings(encoder, train_env, test_env, tokenizer, 0, config)
    
    random.shuffle(train_users)
    user_idx = 0
    best_test_recall = 0.0
    
    while episode_count < MAX_EPISODES:
        uid = train_users[user_idx]
        user_idx += 1
        if user_idx >= len(train_users):
            random.shuffle(train_users)  
            user_idx = 0
            
        with torch.no_grad():
            state, candidate_embs, mask = train_env.reset(uid, tokenizer)
            
        done = False
        ep_reward = 0
        
        while not done:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    action_idx = agent.select_action(state, candidate_embs, mask)
                    next_state, next_candidate_embs, next_mask, rl_reward, semantic_reward, done = train_env.step(action_idx)
            
            ep_reward += rl_reward
            agent.memory.rewards.append(rl_reward)
            agent.memory.is_terminals.append(done)
            
            safe_reward = max(min(semantic_reward / 10.0, 1.0), -1.0)
            
            encoder_buffer['like'].append(train_env.like_tensor.squeeze(0).detach().cpu())
            encoder_buffer['dislike'].append(train_env.dislike_tensor.squeeze(0).detach().cpu())
            encoder_buffer['poi'].append(train_env.poi_tensor[action_idx].detach().cpu())
            encoder_buffer['reward'].append(safe_reward)
            
            del state, candidate_embs, mask
            state, candidate_embs, mask = next_state, next_candidate_embs, next_mask
            
        episode_count += 1
        total_reward += ep_reward
        
        del state, candidate_embs, mask
        
        agent.anneal_exploration(episode_count, MAX_EPISODES)
        
        if episode_count % ENCODER_UPDATE_FREQ == 0:
            if len(encoder_buffer['reward']) > 0:
                l_t = torch.stack(encoder_buffer['like']).to(config.DEVICE)
                d_t = torch.stack(encoder_buffer['dislike']).to(config.DEVICE)
                p_t = torch.stack(encoder_buffer['poi']).to(config.DEVICE)
                r_t = torch.tensor(encoder_buffer['reward'], dtype=torch.float32).to(config.DEVICE)
                
                current_progress = min(episode_count / MAX_EPISODES, 1.0)
                enc_loss = encoder.rl_feedback_step_batch(l_t, d_t, p_t, r_t, encoder_optimizer, progress_ratio=current_progress)
                
                del l_t, d_t, p_t, r_t
                for k in encoder_buffer: encoder_buffer[k].clear()
            else:
                enc_loss = 0.0
                
            a_loss, c_loss = agent.update()
            if torch.is_tensor(a_loss): a_loss = a_loss.item()
            if torch.is_tensor(c_loss): c_loss = c_loss.item()
            
            avg_r = total_reward / ENCODER_UPDATE_FREQ
            
            history_logs["episode"].append(episode_count)
            history_logs["encoder_loss"].append(enc_loss)
            history_logs["actor_loss"].append(a_loss)
            history_logs["critic_loss"].append(c_loss)
            history_logs["avg_reward"].append(avg_r)
            
            print(f"[{episode_count}/{MAX_EPISODES}] Avg Reward: {avg_r:.2f} | EncLoss: {enc_loss:.4f} | ActLoss: {a_loss:.4f}")
            total_reward = 0
            
            gc.collect()
            torch.cuda.empty_cache()
            
        if episode_count % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
            # 🌟 修改调用传参，注入 train_env
            visualize_embeddings(encoder, train_env, test_env, tokenizer, episode_count, config)
            current_recall_20 = evaluate_zero_shot(agent, encoder, test_env, tokenizer, target_ground_truth, config)
            
            if current_recall_20 and current_recall_20 > best_test_recall:
                best_test_recall = current_recall_20
                print(f"🏆 [新 SOTA 诞生] 发现更高 Recall@20: {best_test_recall:.4f}！正在保存模型权重...")
                
                torch.save(encoder.state_dict(), "best_encoder_weights.pth")
                torch.save(agent.actor.state_dict(), "best_agent_actor_weights.pth")
                torch.save(agent.critic.state_dict(), "best_agent_critic_weights.pth")

    with open('training_logs.json', 'w') as f:
        json.dump(history_logs, f)
    print("✅ 训练完成，日志文件已保存为 training_logs.json")

if __name__ == "__main__":
    main()