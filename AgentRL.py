import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class GlobalConfig:
    def __init__(self):

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

        self.TEMP_START = 2.0          
        self.TEMP_END = 0.5            
        self.ENTROPY_COEF_START = 0.1 
        self.ENTROPY_COEF_END = 0.001  
        
        self.RESIDUAL_START = 1.0   
        
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
        self.residual_weight = config.RESIDUAL_START 
        
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
        
        self.residual_weight = self.config.RESIDUAL_START - fraction * (self.config.RESIDUAL_START - self.config.RESIDUAL_END)

    def forward(self, state, candidate_embs, mask):
        p_like = state[:, :self.config.EMBED_DIM] if state.dim() == 2 else state[:self.config.EMBED_DIM]
        residual = self.actor(state)
        
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
        
        gae_lambda = 0.95
        advantages = []
        gae = 0.0
        
        values = [v.item() for v in self.memory.values]
        values.append(0.0) 
        
        for i in reversed(range(len(self.memory.rewards))):
            if self.memory.is_terminals[i]:
                next_value = 0.0
                gae = 0.0  
            else:
                next_value = values[i + 1]
                
            delta = self.memory.rewards[i] + self.config.GAMMA * next_value - values[i]

            gae = delta + self.config.GAMMA * gae_lambda * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.config.DEVICE)
        
        old_values = torch.stack(self.memory.values).squeeze(-1).to(self.config.DEVICE).detach()
        returns = advantages + old_values
        
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