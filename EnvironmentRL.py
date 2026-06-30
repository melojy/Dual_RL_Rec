import torch
import torch.nn.functional as F
import math
from collections import Counter

class EnhancedTravelPlanningEnv:
    def __init__(self, config, encoder, candidates_dict, poi_meta, llm_profiles):
        self.config = config
        self.encoder = encoder
        self.candidates_dict = candidates_dict
        self.poi_meta = poi_meta
        self.llm_profiles = llm_profiles
        
        self.current_user = None
        self.candidate_ids = []
        self.valid_action_len = 100
        self.max_steps = 20
        self.current_step = 0
        self.selected_indices = set()
        
        self.poi_tensor = None
        self.like_tensor = None
        self.dislike_tensor = None
        self.candidate_embs = None
        self.p_like = None
        self.p_dislike = None
        
        self.history_embs = []       
        self.history_categories = [] 
        self.history_poi_ids = []    

    def reset(self, user_id, tokenizer):
        self.current_user = user_id
        self.current_step = 0
        self.selected_indices = set()
        self.history_embs = []
        self.history_categories = []
        self.history_poi_ids = []    
        
        self.candidate_ids = self.candidates_dict.get(user_id, []).copy()
        self.valid_action_len = len(self.candidate_ids)
        
        if self.valid_action_len < self.config.TOTAL_CANDIDATES:
            pad_len = self.config.TOTAL_CANDIDATES - self.valid_action_len
            pad_poi = self.candidate_ids[0] if self.valid_action_len > 0 else list(self.poi_meta.keys())[0]
            self.candidate_ids.extend([pad_poi] * pad_len)
        elif self.valid_action_len > self.config.TOTAL_CANDIDATES:
            self.candidate_ids = self.candidate_ids[:self.config.TOTAL_CANDIDATES]
            self.valid_action_len = self.config.TOTAL_CANDIDATES
            
        poi_tokens = []
        for pid in self.candidate_ids:
            meta = self.poi_meta.get(pid, {})

            cats = meta.get('categories') or "Unknown"
            raw_attrs = meta.get('attributes')
            attrs = " ".join(raw_attrs.keys()) if isinstance(raw_attrs, dict) else "None"
            
            cat_ids = tokenizer.encode(cats)
            attr_ids = tokenizer.encode(attrs)
            poi_tokens.append([cat_ids, attr_ids])
            
        self.poi_tensor = torch.tensor(poi_tokens, dtype=torch.long).to(self.config.DEVICE)
        
        profile = self.llm_profiles[user_id]
        
        like_cat_text = " ".join(profile['global_liked'].get('categories') or []) or "None"
        like_raw_attrs = profile['global_liked'].get('attributes')
        like_attr_text = " ".join(like_raw_attrs.keys()) if isinstance(like_raw_attrs, dict) else "None"
        self.like_tensor = torch.tensor([[tokenizer.encode(like_cat_text), tokenizer.encode(like_attr_text)]], dtype=torch.long).to(self.config.DEVICE)
        
        dislike_cat_text = " ".join(profile['global_disliked'].get('categories') or []) or "None"
        dislike_raw_attrs = profile['global_disliked'].get('attributes')
        dislike_attr_text = " ".join(dislike_raw_attrs.keys()) if isinstance(dislike_raw_attrs, dict) else "None"
        self.dislike_tensor = torch.tensor([[tokenizer.encode(dislike_cat_text), tokenizer.encode(dislike_attr_text)]], dtype=torch.long).to(self.config.DEVICE)
        
        with torch.no_grad():
            self.candidate_embs = self.encoder.encode_dual_stream(self.poi_tensor) 
            self.p_like = self.encoder.encode_dual_stream(self.like_tensor).squeeze(0)
            self.p_dislike = self.encoder.encode_dual_stream(self.dislike_tensor).squeeze(0)

        return self._get_state()

    def _get_state(self):

        h_t = torch.zeros(self.config.EMBED_DIM).to(self.config.DEVICE) if not self.history_embs else torch.stack(self.history_embs).mean(dim=0)
        state = torch.cat([self.p_like, self.p_dislike, h_t], dim=-1)
        mask = torch.ones(self.config.TOTAL_CANDIDATES).to(self.config.DEVICE)
        
        for idx in self.selected_indices: mask[idx] = 0.0
        if self.valid_action_len < self.config.TOTAL_CANDIDATES: mask[self.valid_action_len:] = 0.0
            
        return state, self.candidate_embs, mask

    def step(self, action_idx):
        self.selected_indices.add(action_idx)
        chosen_poi_id = self.candidate_ids[action_idx]
        chosen_emb = self.candidate_embs[action_idx]
        
        self.history_embs.append(chosen_emb)
        self.history_poi_ids.append(chosen_poi_id) 
        
        meta_cats = self.poi_meta.get(chosen_poi_id, {}).get('categories') or 'Unknown'
        cat = meta_cats.split(',')[0].strip()
        self.history_categories.append(cat)
        
        user_scores = self.llm_profiles[self.current_user].get('poi_scores', {})
        
        if chosen_poi_id in user_scores:
            llm_score = user_scores[chosen_poi_id]
            if llm_score >= 4.0: semantic_reward = self.config.R_SUPER_LIKE
            elif llm_score <= 2.5: semantic_reward = self.config.R_HATE
            else: semantic_reward = self.config.R_NORMAL_LIKE
        else:
            semantic_reward = self.config.R_EXPLORE
            
        step_reward = semantic_reward 
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if done:
            cat_counts = Counter(self.history_categories)
            entropy = sum([- (c/len(self.history_categories)) * math.log2(c/len(self.history_categories)) for c in cat_counts.values()])
            
            valid_hits = sum(1 for pid in self.history_poi_ids if user_scores.get(pid, 0.0) >= 3.0)

            if entropy < self.config.ENTROPY_LOW_THRESH: 
                step_reward -= self.config.ENTROPY_PENALTY 
            elif entropy > self.config.ENTROPY_HIGH_THRESH and valid_hits >= 2: 
                step_reward += self.config.ENTROPY_BONUS
                
        next_state, next_candidate_embs, next_mask = self._get_state()
        return next_state, next_candidate_embs, next_mask, step_reward, semantic_reward, done