import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SemanticTokenizer:
    def __init__(self, vocab_size, max_len):
        self.vocab_size = vocab_size
        self.max_len = max_len
        
    def encode(self, text_list) -> list:
        if not text_list: return [0] * self.max_len
        text = " ".join(text_list) if isinstance(text_list, list) else str(text_list)
        words = text.lower().split()
        
        tokens = [(hash(w) % (self.vocab_size - 2)) + 2 for w in words]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [0] * (self.max_len - len(tokens))
        return tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerSemanticEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, emb_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.emb_dim = emb_dim
        
        # padding_idx=0 已经足以让填充位归零，无需使用沉重的底层掩码
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=0.2, max_len=max_len + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 4, 
            dropout=0.2, activation='gelu', batch_first=True
        )
        
        # 🚀 删除了坑人的 enable_nested_tensor=False，重新启用 PyTorch 底层原生加速！
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.LayerNorm(emb_dim * 2),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, text_ids):
        B = text_ids.size(0)
        
        x = self.word_emb(text_ids) * math.sqrt(self.emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)         
        x = self.pos_encoder(x)
        
        # 🚑 终极物理防御：强制在 GPU 显存中重新分配一块连续的物理空间！
        # 没有这一行，底层的 C++ 极速矩阵乘法在反向传播时必崩。
        x = x.contiguous()
        
        # 🚀 满血原生加速的 Transformer
        out = self.transformer(x) 
        
        cls_out = out[:, 0, :]
        return F.normalize(self.projector(cls_out), p=2, dim=-1, eps=1e-8)

# ==========================================
# [完美底座] 基于能量的对比对齐模型
# ==========================================
class DualPrototypeAlignmentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerSemanticEncoder(
            config.VOCAB_SIZE, config.TEXT_MAX_LEN, config.EMBED_DIM,
            num_heads=8, num_layers=4
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM * 2),
            nn.LayerNorm(config.EMBED_DIM * 2),
            nn.GELU(),
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        )
        self.tau = 0.1  

    def encode_dual_stream(self, dual_ids):
        cat_ids = dual_ids[:, 0, :]
        env_ids = dual_ids[:, 1, :]
        
        cat_emb = self.encoder(cat_ids)
        env_emb = self.encoder(env_ids)
        
        fused_emb = self.fusion_layer(torch.cat([cat_emb, env_emb], dim=-1))
        return F.normalize(fused_emb, p=2, dim=-1, eps=1e-8)

    def compute_core_loss(self, p_like, p_dislike, poi_emb, reward, progress_ratio=0.0, mask_like=None, mask_dislike=None):
        tau = self.tau
        
        # 1. 核心对齐
        sim_like = F.cosine_similarity(p_like, poi_emb, dim=-1) / tau
        sim_dislike = F.cosine_similarity(p_dislike, poi_emb, dim=-1) / tau
        logits = torch.stack([sim_like, sim_dislike], dim=1)
        probs = F.softmax(logits, dim=1)
        loss_align = -torch.mean(reward * (probs[:, 0] - probs[:, 1]))
        
        # 2. 原型解耦
        sim_prototypes = F.cosine_similarity(p_like, p_dislike, dim=-1) / tau
        loss_sep = torch.mean(F.softplus(sim_prototypes))
        
        # 3. 全局均匀性
        batch_size = p_like.size(0)
        if batch_size > 1 and mask_like is not None:
            sim_like_matrix = torch.matmul(p_like, p_like.T) / tau
            sim_dislike_matrix = torch.matmul(p_dislike, p_dislike.T) / tau
            
            sim_like_matrix = sim_like_matrix.masked_fill(mask_like, -1e9)
            sim_dislike_matrix = sim_dislike_matrix.masked_fill(mask_dislike, -1e9)
            
            valid_like = (~mask_like).any(dim=-1)
            valid_dislike = (~mask_dislike).any(dim=-1)
            
            loss_uni_like = torch.logsumexp(sim_like_matrix[valid_like], dim=-1).mean() if valid_like.any() else torch.tensor(0.0, device=p_like.device)
            loss_uni_dislike = torch.logsumexp(sim_dislike_matrix[valid_dislike], dim=-1).mean() if valid_dislike.any() else torch.tensor(0.0, device=p_like.device)
            
            loss_uniformity = loss_uni_like + loss_uni_dislike
        else:
            loss_uniformity = torch.tensor(0.0, device=p_like.device)

    # ---------------------------------------------------------
        # 🌟 极限区分度课程学习 (严密的数学边界)
        # ---------------------------------------------------------
        # 确保 decay 严格在 [1.0, 0.0] 之间平滑衰减
        decay = 0.5 * (1.0 + math.cos(math.pi * progress_ratio))
        
        # 前期 decay=1.0 时，w_sep = 0.9 (绝对主导)
        # 后期 decay=0.0 时，w_sep = 0.3 (保底防线)
        w_sep = 0.6 + 0.3 * decay    
        
        # 前期 decay=1.0 时，w_align = 0.3 (被死死压制)
        # 后期 decay=0.0 时，w_align = 0.9 (作用增大)
        w_align = 0.6 - 0.3 * decay  
        
        # 均匀性反坍缩权重保持常量在线
        w_uni = 0.1 
        
        total_loss = w_align * loss_align + w_sep * loss_sep + w_uni * loss_uniformity
        return total_loss

    def rl_feedback_step_batch(self, like_ids, dislike_ids, poi_ids, rewards, optimizer, progress_ratio=0.0):
        self.train()
        dataset_size = like_ids.size(0)
        
        # 🚀 既然解除了掩码诅咒，显存彻底解放，直接恢复 128 大 Batch，享受极致提速！
        batch_size = 64  
        epochs = 3       
        
        total_loss = 0.0
        num_updates = 0
        
        for _ in range(epochs):
            indices = torch.randperm(dataset_size, device=like_ids.device)
            
            for start_idx in range(0, dataset_size, batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                batch_like = like_ids[idx]
                batch_dislike = dislike_ids[idx]
                batch_poi = poi_ids[idx]
                batch_rewards = rewards[idx]
                
                optimizer.zero_grad()
                
                flat_like = batch_like.view(batch_like.size(0), -1)
                mask_like = (flat_like.unsqueeze(1) == flat_like.unsqueeze(0)).all(dim=-1)
                
                flat_dislike = batch_dislike.view(batch_dislike.size(0), -1)
                mask_dislike = (flat_dislike.unsqueeze(1) == flat_dislike.unsqueeze(0)).all(dim=-1)
                
                p_like = self.encode_dual_stream(batch_like)
                p_dislike = self.encode_dual_stream(batch_dislike)
                poi_emb = self.encode_dual_stream(batch_poi)
                
                loss = self.compute_core_loss(p_like, p_dislike, poi_emb, batch_rewards, progress_ratio, mask_like, mask_dislike)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
                # 🚑 斩断计算图，防止显存碎片累积引发 OOM
                del p_like, p_dislike, poi_emb, loss
                
        return total_loss / num_updates if num_updates > 0 else 0.0