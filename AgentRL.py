import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class GlobalConfig:
    def __init__(self):

        '''
        LR_ACTOR (Actor 学习率) 和 LR_CRITIC (Critic 学习率)：
        作用： 控制网络更新的步伐。
        怎么调： 在 RL 中，Critic 的学习率必须比 Actor 大（通常是 3 到 10 倍）。因为裁判（Critic）必须先学得快、看得准，才能去指导演员（Actor）怎么演。如果在后期（Finetune 阶段）曲线震荡，就需要把它们双双调低（比如分别降到 1e-4 和 3e-4）。

        PPO_EPOCHS 和 PPO_CLIP：
        作用： 防止“走火入魔”。PPO_CLIP = 0.2 意味着每次更新时，Actor 的新策略最多只能比老策略改变 20%。
        怎么调： 如果模型一更新就崩溃，就把 CLIP 调小（如 0.1）；如果学得太慢，可以调到 0.25。目前 0.2 是最稳的甜点位。

        R_SUPER_LIKE (10.0), R_HATE (-10.0), R_EXPLORE (-1.0)：
        作用： 设定小白鼠迷宫里的奶酪和陷阱。
        为什么这么设： R_EXPLORE = -1.0 是点睛之笔。它是一个“时间税”，逼迫 Agent 不要瞎逛，必须用最少的步数去命中 +10.0 的好店。如果没有这 -1.0，Agent 就会在迷宫里原地转圈。

        RESIDUAL_START 和 RESIDUAL_END (动态残差半径)：
        作用： 决定 Agent 有多大的权利去偏离静态用户画像（$P_{like}$）。
        怎么调： 你现在的瓶颈就在这！如果 RESIDUAL_END 太大（如 0.5），Agent 就会“自作聪明”，为了追求花里胡哨的路线而放弃最稳的选项；必须将其降到 0.05，把主导权还给完美的 Encoder，Recall 必将瞬间破 40%。TEMP_START 和 
        
        TEMP_END (Softmax 温度)：
        作用： 控制动作概率的“尖锐度”。温度越低，Agent 越贪婪（只选概率最高的）；温度越高，Agent 越随机。
        怎么调： 如果曲线剧烈震荡（像你现在的图一样），说明后期温度太低（如 0.1），导致微小的数值变化引发动作翻转。调高底线（如 TEMP_END = 0.5）可以让动作分布变柔和，消除锯齿震荡。

        ENTROPY_COEF_START 和 ENTROPY_COEF_END：
        作用： 强行在 Actor 的损失函数中加入一项，奖励它“多尝试不同选择”。
        怎么调： 前期给 0.1，逼迫它多看世界；后期必须降到 0.001 甚至 0，让它专心拿高分。

        '''

        self.TOTAL_CANDIDATES = 100    
        self.MAX_POS_POIS = 20         
        self.EMBED_DIM = 128           
        self.STATE_DIM = 384           
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.VOCAB_SIZE = 20000        
        self.TEXT_MAX_LEN = 30         
        
        self.MAX_STEPS = 20            
        self.GAMMA = 0.99  
                    
        self.LR_ACTOR = 1e-4           
        self.LR_CRITIC = 3e-4          
        self.LR_PRETRAIN = 1e-4        
        self.LR_RL_FINETUNE = 1e-5     
        self.PPO_EPOCHS = 4            
        self.PPO_CLIP = 0.2            
        
        self.R_SUPER_LIKE = 10.0       
        self.R_HATE = -10.0            
        self.R_NORMAL_LIKE = 1.0       
        self.R_EXPLORE = -1.0           
        
        self.SIMILARITY_PENALTY = 0.0  
        self.SIMILARITY_THRESH = 1.0  
        
        self.ENTROPY_LOW_THRESH = 1.0  
        self.ENTROPY_PENALTY = 5.0    
        self.ENTROPY_HIGH_THRESH = 2.5 
        self.ENTROPY_BONUS = 2.0       

        # ==========================================
        # 🌟 探索与利用动态退火 (Exploration vs Exploitation)
        # ==========================================
        self.TEMP_START = 2.0          
        self.TEMP_END = 0.5            
        self.ENTROPY_COEF_START = 0.1 
        self.ENTROPY_COEF_END = 0.001  
        
        # 🚑 核心修复：解除后期的物理封印！
        self.RESIDUAL_START = 1.0      # 初期：允许偏离原型 100%
        
        # 将原来的 0.05 改为 0.5 (甚至可以直接设为 1.0 取消衰减)
        # 这意味着：即使在最末期，Agent 依然拥有 50% 的决策权，
        # 它可以根据当前路线灵活地偏离静态偏好，打出完美的组合拳！
        self.RESIDUAL_END = 0.1

class Memory:
    def __init__(self):
        self.states, self.candidate_embs, self.masks, self.actions = [], [], [], []
        self.logprobs, self.rewards, self.is_terminals, self.values = [], [], [], []

    def clear(self):
        del self.states[:], self.candidate_embs[:], self.masks[:], self.actions[:]
        del self.logprobs[:], self.rewards[:], self.is_terminals[:], self.values[:]

class PPOAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config.TEMP_START
        self.entropy_coef = config.ENTROPY_COEF_START
        self.residual_weight = config.RESIDUAL_START # 初始化探索半径
        
        self.actor = nn.Sequential(
            nn.Linear(config.STATE_DIM, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, config.EMBED_DIM)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(config.STATE_DIM, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': config.LR_ACTOR},
            {'params': self.critic.parameters(), 'lr': config.LR_CRITIC}
        ])
        
        self.memory = Memory()
        self.MseLoss = nn.MSELoss()

    def anneal_exploration(self, current_episode, max_episodes):
        fraction = min(current_episode / max_episodes, 1.0)
        self.temperature = self.config.TEMP_START - fraction * (self.config.TEMP_START - self.config.TEMP_END)
        self.entropy_coef = self.config.ENTROPY_COEF_START - fraction * (self.config.ENTROPY_COEF_START - self.config.ENTROPY_COEF_END)
        
        # 【核心新增】：动态衰减探索半径
        self.residual_weight = self.config.RESIDUAL_START - fraction * (self.config.RESIDUAL_START - self.config.RESIDUAL_END)

    def forward(self, state, candidate_embs, mask):
        p_like = state[:, :self.config.EMBED_DIM] if state.dim() == 2 else state[:self.config.EMBED_DIM]
        residual = self.actor(state)
        
        # 使用动态探索半径控制自由度
        query = p_like + residual * self.residual_weight 
        
        query = F.normalize(query, p=2, dim=-1, eps=1e-8)
        
        candidate_embs_t = candidate_embs.transpose(1, 2).contiguous()
        action_logits = torch.bmm(query.unsqueeze(1), candidate_embs_t).squeeze(1)
        
        action_logits = (action_logits * 10.0) / self.temperature
        
        huge_negative = torch.tensor(-1e8, dtype=torch.float32, device=self.config.DEVICE)
        action_logits = torch.where(mask > 0.5, action_logits, huge_negative)
        
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        value = self.critic(state)
        return dist, value

    def select_action(self, state, candidate_embs, mask):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            candidate_embs = candidate_embs.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        with torch.no_grad():
            dist, value = self.forward(state, candidate_embs, mask)
            action = dist.sample()
            
        self.memory.states.append(state.squeeze(0))
        self.memory.candidate_embs.append(candidate_embs.squeeze(0))
        self.memory.masks.append(mask.squeeze(0))
        self.memory.actions.append(action.squeeze(0))
        self.memory.logprobs.append(dist.log_prob(action).squeeze(0))
        self.memory.values.append(value.squeeze(0))
        
        return action.item()

    def update(self):
        if len(self.memory.rewards) == 0: return 0.0, 0.0
        
        # 🌟 核心升级 1：使用 GAE (Generalized Advantage Estimation) 替代纯蒙特卡洛回报
        gae_lambda = 0.95  # GAE 的平滑衰减因子 (学术界标准值)
        advantages = []
        gae = 0.0
        
        # 提前把预估价值提取出来，方便计算 TD Error
        values = [v.item() for v in self.memory.values]
        values.append(0.0) # 为最后一步的 next_value 兜底
        
        for i in reversed(range(len(self.memory.rewards))):
            if self.memory.is_terminals[i]:
                next_value = 0.0
                gae = 0.0  # 回合结束，未来的预期红利清零
            else:
                next_value = values[i + 1]
                
            # TD Error: 真实单步奖励 + 折扣后的未来预估 - 当前身价
            delta = self.memory.rewards[i] + self.config.GAMMA * next_value - values[i]
            # GAE 累加：将每一步的短期优势平滑地传导到过去
            gae = delta + self.config.GAMMA * gae_lambda * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.config.DEVICE)
        
        # 计算 Critic 需要拟合的真实平滑回报 (Returns = Advantages + Values)
        old_values = torch.stack(self.memory.values).squeeze(-1).to(self.config.DEVICE).detach()
        returns = advantages + old_values
        
        # 标准化优势函数：极大收缩 Actor 的震荡幅度，保证梯度的健康度
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        old_states = torch.stack(self.memory.states).to(self.config.DEVICE).detach()
        old_candidate_embs = torch.stack(self.memory.candidate_embs).to(self.config.DEVICE).detach()
        old_masks = torch.stack(self.memory.masks).to(self.config.DEVICE).detach()
        old_actions = torch.stack(self.memory.actions).to(self.config.DEVICE).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.config.DEVICE).detach()

        total_a_loss, total_c_loss = 0, 0
        for _ in range(self.config.PPO_EPOCHS):
            dist, state_values = self.forward(old_states, old_candidate_embs, old_masks)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.PPO_CLIP, 1 + self.config.PPO_CLIP) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 🌟 核心升级 2：对 Critic 进行截断防爆 (Value Clipping)
            # 防止 Critic 因为偶然的极端分数而导致网络参数崩溃
            value_pred_clipped = old_values + torch.clamp(
                state_values.squeeze(-1) - old_values, 
                -self.config.PPO_CLIP, 
                self.config.PPO_CLIP
            )
            v_loss1 = (state_values.squeeze(-1) - returns) ** 2
            v_loss2 = (value_pred_clipped - returns) ** 2
            critic_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            
            entropy_bonus = self.entropy_coef * dist_entropy.mean()
            
            loss = actor_loss + critic_loss - entropy_bonus
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_a_loss += actor_loss.item()
            total_c_loss += critic_loss.item()
            
        self.memory.clear()
        return total_a_loss / self.config.PPO_EPOCHS, total_c_loss / self.config.PPO_EPOCHS