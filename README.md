# Dual_RL_Rec

This is the repository for the paper: **"Bridging the Domain Gap: POI Recommendation via Dual-Prototype Semantic Alignment and Reinforcement Learning"**.

---

## 1. Introduction
Here we provide the overall intuition and performance comparison of our proposed method.

<p align="center">
  <img src="results/visualization/Introduction.png" alt="Introduction" width="70%">
</p>

---

## 2. Methodology & Framework

<p align="center">
  <img src="results/visualization/FrameWork.png" alt="Framework" width="70%">
</p>

---

## 3. User Profiling Prompts
As specified in the paper, we leverage Large Language Models (LLMs) to perform semantic data augmentation and extract fine-grained user profiles from review data. 

Below is the complete **System Prompt** and **User Prompt** structure used for user preference extraction:

### System Prompt
"""You are a rigorous data analysis expert for recommendation systems. Please extract the user profile and score each POI based on the user's minimalist check-in records. Note: Records containing 'Cat' and 'Attr' represent POI samples, while those with 'Rev' represent user review samples.

    Please strictly output the following JSON format (do not output Markdown markers like ```json, just output the curly braces):

{
"global_liked": {"categories": ["sub-category 1"], "attributes": {"attribute_key": "value"}},
"global_disliked": {"categories": ["sub-category 2"], "attributes": {"attribute_key": "value"}},
"poi_scores": {"1": 4.5, "2": 3.0}
}

[Ensure the output adheres strictly to the following principles]:
1. Principle of Category Mutually Exclusive: The categories in global_liked and global_disliked must NEVER overlap. If similar categories exist, they must be subdivided with specific modifiers (e.g., liked: ["Luxury Hotel"], disliked: ["Noisy Motel"]).
2. Principle of Objective Environment: The attributes must be specific, objective physical or facility characteristics (e.g., "WiFi": "free").
3. Principle of Short-ID Scoring: The keys in poi_scores MUST be the minimalist pure numeric IDs read from the input (e.g., "1", "2", "3"). You must provide a comprehensive score ranging from 1.0 to 5.0 for all numeric IDs present in the input, based on the user's sentiment explicitly expressed in the reviews."""

### User Prompt
   user_prompt = f"User Records: \n{compressed_text}"



## 4. Other Implementation Details

### Model Architecture
* **Feature Extraction Module:** Adopts a 4-layer Transformer encoder structure. The multi-head attention mechanism consists of 8 attention heads. 
* **Reinforcement Learning (RL) Agent:** Both the Actor and Critic networks adopt a multi-layer perceptron (MLP) architecture. The input dimension is 384 (formed by concatenating the positive prototype, negative prototype, and historical trajectory mean). The Actor network outputs a 128-dimensional continuous residual vector, while the Critic network outputs a scalar state value.
* **Semantic Parsing:** Utilizes the **DeepSeek-V3** large language model. The sampling temperature is strictly set to 0.1 to ensure the determinism and consistency of entity-attribute extraction.

### Hyperparameters & Training Settings
The joint training process ensures each user is updated at least once, with the agent performing a maximum of $T=20$ exploration steps per epoch. 

| Component / Parameter | Setting / Value |
| :--- | :--- |
| **Transformer Encoder** | Embedding dim $d=128$, Max seq length $J=60$ |
| **Encoder Regularization** | GELU activation, Dropout rate = 0.2 |
| **Max Training Epochs** | 6000 (Yelp), 24000 (Google Local) |
| **Optimizers** | AdamW (Encoder: lr=$10^{-5}$)<br>Adam (Actor: lr=$10^{-4}$, Critic: lr=$3 \times 10^{-4}$) |
| **PPO Algorithm** | Discount factor $\gamma=0.99$<br>GAE smoothing $\lambda=0.95$<br>Clipping threshold $\varepsilon=0.2$<br>Update iterations $F=4$ |

### Zero-Shot Cross-Domain Evaluation
In this stage, the parameters of both the underlying semantic encoder and the Actor network are **frozen**. 
1.  **Offline Processing:** The system first performs a one-time offline processing of the candidate POI metadata and the target user's global profile in the target domain. 
2.  **Inference:** The Actor network receives the initial state of the target domain and completes action output solely through lightweight forward propagation and matrix inner-product operations. 
3.  **Recommendation Standard:** We adopt the Top-K standard, selecting the top-$k$ uninteracted entities with the highest scores from the environment candidate set.
