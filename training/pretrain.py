import torch
import torch.nn.functional as F
from tqdm import tqdm

def run_pretraining(env, model, steps=1000, max_episode_length=100):
    torch.autograd.set_detect_anomaly(True)
    beta = 0.00005  # strength of intrinsic curiosity reward
    gamma = 0.99  # discount factor
    # Pretraining loop
    print("Starting pretraining...")
    # Initialize variables
    episode_action_indices = []
    episode_states = []
    episode_rewards = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    done=False
    state = env.reset()
    episode_step_count = 0
    for step in tqdm(range(steps)):
        state = torch.tensor(state.flatten(), dtype=torch.float32)
        episode_states.append(state.detach())  # detach to avoid storing computation graph
        output = model(state)
        behavior_logits = output["behavior"]
        prediction_logits = output["prediction"]
        # Sample action from behavior logits
        beh_prob = F.softmax(behavior_logits, dim=-1)
        action = torch.multinomial(beh_prob, num_samples=1).item()
        next_state, reward, done, _ = env.step(action)
        next_state_tensor = torch.tensor(next_state.flatten(), dtype=torch.float32)
        target_index = torch.argmax(next_state_tensor).unsqueeze(0)  # shape [1]
        
        # 1. Prediction loss (always computed)
        pred_logits = output["prediction"].unsqueeze(0)  # shape [1, 25]
        pred_loss = F.cross_entropy(pred_logits, target_index)
        optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        optimizer.step()
        
        # 2. Behavior loss
        log_prob = torch.log(beh_prob[action])
        intrinsic_reward = pred_loss.item()
        total_reward = reward + beta * intrinsic_reward

        # In the loop where you store step data
        episode_action_indices.append(action)
        episode_rewards.append(total_reward)

        # Warning: 'retain_graph=True' in line 29 may cause memory issues here
        episode_step_count += 1
        if done or episode_step_count >= max_episode_length:
            # Compute discounted return with intrinsic + extrinsic
            G = 0
            returns = []
            for r in reversed(episode_rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            beh_loss = 0
            for state_step, action_idx, Gt in zip(episode_states, episode_action_indices, returns):
                output_step = model(state_step)
                behavior_logits = output_step["behavior"]
                beh_prob = F.softmax(behavior_logits, dim=-1)
                log_prob = torch.log(beh_prob[action_idx])
                beh_loss += -log_prob * Gt

            optimizer.zero_grad()
            beh_loss.backward()
            optimizer.step()

            # Clear buffers and reset episode
            episode_action_indices = []
            episode_rewards = []
            episode_states = []
            done = False
            # Reset the environment
            episode_step_count = 0
            state = env.reset()
        else:
            # Lightweight per-step intrinsic update only
            # TODO: use a different curiosity learning rule that actually works – this one tanks learning
            # intrinsic_loss = -log_prob * intrinsic_reward
            # optimizer.zero_grad()
            # intrinsic_loss.backward()
            # optimizer.step()
            state = next_state

            
    print("Pretraining completed.")