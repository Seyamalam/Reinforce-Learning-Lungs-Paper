import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from collections import defaultdict
import random
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- Configuration & Setup ---
DATA_FILE = 'Lung Cancer Dataset.csv'
RESULTS_DIR = 'research_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# RL Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON_START = 1.0 # Exploration rate start
EPSILON_END = 0.01 # Exploration rate end
EPSILON_DECAY = 0.995 # Exploration decay rate
NUM_EPISODES = 10000 # Number of training episodes
MAX_STEPS_PER_EPISODE = 10 # Max steps within an episode before reset (to avoid infinite loops if no risk reduction is possible)
ACTION_COST = 0.01 # Small penalty for taking any intervention action

# --- 1. Load Data ---
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- 2. Exploratory Data Analysis ---")

print("\nDataset Info:")
df.info()

print("\nDataset Head:")
print(df.head())

print("\nDescriptive Statistics (Numerical Features):")
print(df.describe())

print("\nValue Counts (Categorical/Binary Features):")
for col in df.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'object': # Assuming binary are int64 for now
         print(f"\nFeature: {col}")
         print(df[col].value_counts())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
# No missing values found in this dataset based on typical structure.

# Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='PULMONARY_DISEASE', data=df)
plt.title('Distribution of Pulmonary Disease (Target Variable)')
plt.savefig(os.path.join(RESULTS_DIR, 'target_distribution.png'))
plt.close()
print(f"\nTarget Distribution:\n{df['PULMONARY_DISEASE'].value_counts(normalize=True)}")
# Note: The dataset seems heavily biased towards 'YES'. This needs careful handling (e.g., stratification, appropriate metrics).

# Convert target to numerical if it's not already
le = LabelEncoder()
df['PULMONARY_DISEASE'] = le.fit_transform(df['PULMONARY_DISEASE']) # YES=1, NO=0
print("Target variable 'PULMONARY_DISEASE' encoded (YES=1, NO=0).")

# Visualizing Numerical Features
num_features = ['AGE', 'ENERGY_LEVEL', 'OXYGEN_SATURATION']
print("\nVisualizing Numerical Features...")
for col in num_features:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x='PULMONARY_DISEASE', y=col, data=df)
    plt.title(f'{col} vs Pulmonary Disease')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{col}_distribution_boxplot.png'))
    plt.close()

# Visualizing Key Categorical Features vs Target
cat_features = ['SMOKING', 'MENTAL_STRESS', 'ALCOHOL_CONSUMPTION', 'EXPOSURE_TO_POLLUTION', 'IMMUNE_WEAKNESS']
print("Visualizing Key Categorical Features vs Target...")
for col in cat_features:
    plt.figure(figsize=(7, 5))
    sns.countplot(x=col, hue='PULMONARY_DISEASE', data=df, palette='viridis')
    plt.title(f'{col} vs Pulmonary Disease')
    plt.legend(title='Disease', labels=['No', 'Yes'])
    plt.savefig(os.path.join(RESULTS_DIR, f'{col}_vs_target.png'))
    plt.close()

# Correlation Analysis (only numerical/binary features)
# Ensure all binary features are numeric (already done mostly)
plt.figure(figsize=(18, 15))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False) # Annot=True can be too crowded
plt.title('Correlation Matrix of Features')
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
plt.close()

print("\nCorrelation with Target Variable (PULMONARY_DISEASE):")
print(correlation_matrix['PULMONARY_DISEASE'].sort_values(ascending=False))

# Analyze potentially redundant features like 'STRESS_IMMUNE'
# If it's simply stress AND immune_weakness, it might be redundant for modeling, though potentially useful for state definition later.
# Let's assume for now we drop it to avoid perfect multicollinearity if it's derived.
if 'STRESS_IMMUNE' in df.columns:
    print("\nDropping 'STRESS_IMMUNE' feature assuming it might be redundant.")
    df = df.drop('STRESS_IMMUNE', axis=1)

# --- 3. Supervised Model Training (Risk Oracle) ---
print("\n--- 3. Training Supervised Risk Model (XGBoost) ---")

# Define features (X) and target (y)
target = 'PULMONARY_DISEASE'
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

# Train-Test Split (Stratified due to potential imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Initialize and Train XGBoost Model
# Handle potential imbalance using scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                              scale_pos_weight=scale_pos_weight, random_state=42)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate Model
print("\nEvaluating Supervised Model...")
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # Probability of class 1 (YES)

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - XGBoost Model')
plt.savefig(os.path.join(RESULTS_DIR, 'xgboost_confusion_matrix.png'))
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'xgboost_roc_curve.png'))
plt.close()

# Visualize Feature Importance to understand risk factors
print("\nAnalyzing Feature Importance from XGBoost Model...")
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
plt.close()

# Display feature importance values
importances_dict = {features[i]: importance for i, importance in enumerate(feature_importance)}
importances_sorted = {k: v for k, v in sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)}
print("\nFeature Importance Ranking:")
for feature, importance in importances_sorted.items():
    print(f"{feature:<30} : {importance:.4f}")

# Define modifiable and non-modifiable features for importance analysis
modifiable_features = ['SMOKING', 'MENTAL_STRESS', 'ALCOHOL_CONSUMPTION', 'EXPOSURE_TO_POLLUTION']
non_modifiable_features = [col for col in features if col not in modifiable_features]

# Highlight modifiable vs non-modifiable in importance ranking
print("\nImportance of Modifiable vs Non-Modifiable Features:")
# Calculate importance sums for modifiable and non-modifiable features
modifiable_importance_sum = sum(importances_dict[f] for f in modifiable_features)
non_modifiable_importance_sum = sum(importances_dict[f] for f in non_modifiable_features)
total_importance = modifiable_importance_sum + non_modifiable_importance_sum

print(f"Modifiable Features: {modifiable_importance_sum:.4f} ({(modifiable_importance_sum/total_importance)*100:.1f}% of total importance)")
print(f"Non-Modifiable Features: {non_modifiable_importance_sum:.4f} ({(non_modifiable_importance_sum/total_importance)*100:.1f}% of total importance)")

print("Supervised model evaluation complete. Results saved.")

# Define a function to get risk prediction (needed for the RL environment)
def get_predicted_risk(state_features_dict):
    """
    Predicts the risk of pulmonary disease for a given state using the trained XGBoost model.
    
    Args:
        state_features_dict (dict): Dictionary representing the features of a state.
            Must contain all features expected by the XGBoost model.
            
    Returns:
        float: Predicted probability of having pulmonary disease (between 0 and 1).
    """
    # Create a DataFrame with the correct feature order
    feature_df = pd.DataFrame([state_features_dict], columns=features) # Use global 'features' list
    # Predict probability for the positive class (Disease=YES)
    risk_prob = xgb_model.predict_proba(feature_df)[0, 1]
    return risk_prob

# --- 4. RL Environment Setup ---
print("\n--- 4. Setting up Reinforcement Learning Environment ---")

# Identify modifiable features for the RL state and actions
# Let's assume: SMOKING, MENTAL_STRESS, ALCOHOL_CONSUMPTION, EXPOSURE_TO_POLLUTION
modifiable_features = ['SMOKING', 'MENTAL_STRESS', 'ALCOHOL_CONSUMPTION', 'EXPOSURE_TO_POLLUTION']
non_modifiable_features = [col for col in features if col not in modifiable_features]

print(f"Modifiable Features: {modifiable_features}")
print(f"Non-Modifiable Features: {len(non_modifiable_features)}")

# State Space Definition
# A state is defined by the values of the modifiable features.
# Since these are binary (0/1), we can represent a state as a tuple.
# Example state: (smoking_status, stress_status, alcohol_status, pollution_status) -> (1, 1, 0, 1)

# Action Space Definition
# Actions correspond to attempting to change *one* modifiable feature from 1 to 0.
# We also include a 'NO_ACTION'.
actions = ['NO_ACTION']
action_map = {'NO_ACTION': -1} # Map action name to index for easier handling
idx_counter = 0
for i, feature in enumerate(modifiable_features):
    action_name = f"REDUCE_{feature}"
    actions.append(action_name)
    action_map[action_name] = i # Index corresponds to the feature in modifiable_features
    idx_counter += 1

num_actions = len(actions)
print(f"\nAction Space ({num_actions} actions): {actions}")
print(f"Action Map: {action_map}")

# Get a sample non_modifiable profile from the dataset (use the first row for simplicity)
# In a real scenario, you might want to average or sample different profiles
initial_patient_profile = df.iloc[0].to_dict()
non_modifiable_profile = {k: initial_patient_profile[k] for k in non_modifiable_features}
print("\nUsing a sample non-modifiable profile for simulation.")


# RL Environment Class (Simplified)
class LungCancerEnv:
    """
    Reinforcement Learning environment for simulating intervention policies for lung cancer prevention.
    
    This environment simulates the effect of various interventions on modifiable risk factors
    to reduce the predicted risk of pulmonary disease. It uses a pre-trained supervised model
    to predict the risk based on the current state.
    
    Attributes:
        modifiable_features (list): List of feature names that can be modified by actions.
        non_modifiable_profile (dict): Dictionary of non-modifiable features and their values.
        actions (list): List of possible action names.
        action_map (dict): Mapping from action names to indices.
        num_actions (int): Total number of possible actions.
        get_risk (function): Function to predict risk given state features.
        action_cost (float): Cost of taking an action (for reward calculation).
        state (tuple): Current state as a tuple of modifiable feature values.
    """
    def __init__(self, modifiable_features, non_modifiable_profile, actions, action_map, risk_predictor_func, action_cost):
        """
        Initialize the lung cancer intervention environment.
        
        Args:
            modifiable_features (list): Feature names that can be modified by actions.
            non_modifiable_profile (dict): Non-modifiable features and their values.
            actions (list): Possible action names.
            action_map (dict): Mapping from action names to indices.
            risk_predictor_func (function): Function to predict risk given state features.
            action_cost (float): Cost of taking an action.
        """
        self.modifiable_features = modifiable_features
        self.non_modifiable_profile = non_modifiable_profile
        self.actions = actions
        self.action_map = action_map
        self.num_actions = len(actions)
        self.get_risk = risk_predictor_func
        self.action_cost = action_cost
        self.state = None # Will be tuple representing modifiable features' values
        self.reset()

    def _get_full_features(self, state_tuple):
        """
        Combines modifiable state tuple and non-modifiable profile.
        
        Args:
            state_tuple (tuple): Tuple representing modifiable feature values.
            
        Returns:
            dict: Complete feature dictionary with both modifiable and non-modifiable features.
        """
        full_features = self.non_modifiable_profile.copy()
        for i, feature_name in enumerate(self.modifiable_features):
            full_features[feature_name] = state_tuple[i]
        return full_features

    def reset(self):
        """
        Resets the environment to a random initial state from the dataset.
        
        Returns:
            tuple: Initial state as a tuple of modifiable feature values.
        """
        # Sample a random row to get an initial state
        random_row = df.sample(1).iloc[0]
        self.non_modifiable_profile = {k: random_row[k] for k in non_modifiable_features} # Update non-modifiable based on sample
        initial_modifiable_state = tuple(random_row[mf] for mf in self.modifiable_features)
        self.state = initial_modifiable_state
        return self.state

    def step(self, action_index):
        """
        Applies an action, calculates reward, and transitions state.
        
        Args:
            action_index (int): Index of the action to take.
            
        Returns:
            tuple: (next_state, reward, done)
                next_state (tuple): New state after action.
                reward (float): Reward for taking the action.
                done (bool): Whether the episode is done.
        """
        current_risk = self.get_risk(self._get_full_features(self.state))
        current_modifiable_list = list(self.state)
        new_modifiable_list = list(self.state) # Start with current state

        action_taken = False
        action_name = self.actions[action_index]

        if action_name == 'NO_ACTION':
            pass # State remains the same
        else:
            feature_idx_to_modify = self.action_map[action_name]
            feature_to_modify = self.modifiable_features[feature_idx_to_modify]

            # Only apply action if the feature is currently '1' (modifiable state)
            if current_modifiable_list[feature_idx_to_modify] == 1:
                new_modifiable_list[feature_idx_to_modify] = 0 # Intervention successful
                action_taken = True
            # If feature is already '0', the action has no effect on state
            # but might still incur cost depending on reward definition

        next_state_tuple = tuple(new_modifiable_list)
        next_risk = self.get_risk(self._get_full_features(next_state_tuple))

        # Reward Calculation: Reduction in risk minus action cost
        # Higher reward for larger risk reduction
        reward = (current_risk - next_risk)
        if action_name != 'NO_ACTION' and action_taken: # Apply cost only if an intervention was attempted and successful
             reward -= self.action_cost
        elif action_name != 'NO_ACTION' and not action_taken: # Optionally penalize trying to change an already good state
             reward -= self.action_cost / 2 # Smaller penalty maybe?

        self.state = next_state_tuple # Update the environment's state

        # Define 'done' condition - e.g., if risk is below a threshold or no improvement possible
        # For simplicity here, we rely on max_steps_per_episode in the training loop
        done = False # Let the training loop handle termination by steps

        return self.state, reward, done

# Instantiate the environment
env = LungCancerEnv(modifiable_features, non_modifiable_profile, actions, action_map, get_predicted_risk, ACTION_COST)
print("RL Environment instantiated.")

# --- 5. Q-Learning Agent Training ---
print("\n--- 5. Training Q-Learning Agent ---")

# Initialize Q-table using defaultdict for convenience
# Q[state][action] = value
q_table = defaultdict(lambda: np.zeros(num_actions))

# Track rewards per episode for plotting
episode_rewards = []
epsilon_values = []

epsilon = EPSILON_START

# Training Loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(num_actions)) # Explore
        else:
            action = np.argmax(q_table[state]) # Exploit

        next_state, reward, done_env = step_result = env.step(action) # done_env is currently always False

        # Q-table update (Bellman equation)
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state]) # Estimate of future optimal value

        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state][action] = new_value

        total_reward += reward
        state = next_state
        steps += 1
        # done condition is handled by max_steps

    episode_rewards.append(total_reward)
    epsilon_values.append(epsilon)

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES} completed. Average Reward (last 100): {np.mean(episode_rewards[-100:]):.4f}, Epsilon: {epsilon:.4f}")

print("\nQ-Learning training finished.")

# Plotting training progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Plot moving average of rewards
rewards_smoothed = pd.Series(episode_rewards).rolling(100, min_periods=100).mean()
plt.plot(rewards_smoothed)
plt.xlabel('Episode')
plt.ylabel('Average Reward (Smoothed)')
plt.title('Episode Rewards over Time (Moving Average)')

plt.subplot(1, 2, 2)
plt.plot(epsilon_values)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay over Time')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rl_training_progress.png'))
plt.close()
print("RL training progress plots saved.")


# --- 6. Policy Evaluation & Visualization ---
print("\n--- 6. Evaluating and Visualizing the Learned Policy ---")

# Extract the optimal policy (greedy action for each state)
policy = {}
states_encountered = list(q_table.keys())
print(f"Number of states encountered during training: {len(states_encountered)}")

# Limit the number of states to display for clarity, focusing on common/risky ones
# Let's calculate risk for each encountered state to prioritize
state_risks = {}
for state in states_encountered:
    state_features = env._get_full_features(state)
    state_risks[state] = get_predicted_risk(state_features)

# Sort states by risk (descending) to see high-risk policies first
sorted_states = sorted(states_encountered, key=lambda s: state_risks[s], reverse=True)

# Evaluate risk reduction potential for top high-risk states
print("\nRisk Reduction Potential for Top 10 Highest Risk States:")
print("-" * 80)
print(f"{'State (S, M, A, P)':<20} | {'Initial Risk':<12} | {'Best Action':<20} | {'Reduced Risk':<12} | {'Reduction %':<12}")
print("-" * 80)

for i, state in enumerate(sorted_states):
    if i >= 10:  # Limit to top 10 states
        break
    
    best_action_index = np.argmax(q_table[state])
    best_action_name = actions[best_action_index]
    initial_risk = state_risks[state]
    
    # Simulate taking the best action to compute risk reduction
    current_modifiable_list = list(state)
    new_modifiable_list = list(state)
    
    if best_action_name != 'NO_ACTION':
        feature_idx = action_map[best_action_name]
        if current_modifiable_list[feature_idx] == 1:
            new_modifiable_list[feature_idx] = 0
    
    next_state = tuple(new_modifiable_list)
    reduced_risk = get_predicted_risk(env._get_full_features(next_state))
    risk_reduction = initial_risk - reduced_risk
    risk_reduction_pct = (risk_reduction / initial_risk) * 100 if initial_risk > 0 else 0
    
    print(f"{str(state):<20} | {initial_risk:.4f}      | {best_action_name:<20} | {reduced_risk:.4f}      | {risk_reduction_pct:.2f}%")

print("-" * 80)
print("(S=SMOKING, M=MENTAL_STRESS, A=ALCOHOL_CONSUMPTION, P=EXPOSURE_TO_POLLUTION)")

print("\nOptimal Policy (Best Action) for Top 20 Highest Risk Encountered States:")
print("-" * 60)
print(f"{'State (S, M, A, P)':<25} | {'Risk':<10} | {'Best Action':<20}")
print("-" * 60)
# Header for state: S=Smoking, M=Mental Stress, A=Alcohol, P=Pollution
header_state_tuple = tuple(f[0] for f in modifiable_features) # (S, M, A, P)

for i, state in enumerate(sorted_states):
    if i >= 20 and len(states_encountered) > 20: # Limit output for brevity
        break
    best_action_index = np.argmax(q_table[state])
    best_action_name = actions[best_action_index]
    risk = state_risks[state]
    print(f"{str(state):<25} | {risk:.4f}     | {best_action_name:<20}")

print("-" * 60)
print("(S=SMOKING, M=MENTAL_STRESS, A=ALCOHOL_CONSUMPTION, P=EXPOSURE_TO_POLLUTION)")

# Optional: Create a heatmap visualization if the state space is small enough or for a subset
# This is tricky with 4 binary features (16 states total). We can show all.
all_possible_states = []
from itertools import product
for state_vals in product([0, 1], repeat=len(modifiable_features)):
    all_possible_states.append(state_vals)

policy_data = []
for state in all_possible_states:
    if state in q_table:
         best_action_idx = np.argmax(q_table[state])
         best_action_name = actions[best_action_idx]
         q_values = q_table[state]
    else:
         # If state not seen, assume no action is best, Q-values are zero
         best_action_idx = action_map['NO_ACTION']
         best_action_name = 'NO_ACTION'
         q_values = np.zeros(num_actions) # Or handle differently

    risk = get_predicted_risk(env._get_full_features(state)) # Need to recalculate risk for all states
    policy_data.append({
        'state': str(state),
        'risk': risk,
        'best_action': best_action_name,
        **{f'Q({act})': qv for act, qv in zip(actions, q_values)} # Include Q-values for context
    })

policy_df = pd.DataFrame(policy_data)
policy_df = policy_df.sort_values('risk', ascending=False).reset_index(drop=True)

print("\nLearned Policy and Q-Values for All Possible Modifiable States:")
print(policy_df[['state', 'risk', 'best_action'] + [f'Q({act})' for act in actions]])

# Save the policy table
policy_df.to_csv(os.path.join(RESULTS_DIR, 'learned_policy.csv'), index=False)
print("\nFull learned policy saved to 'learned_policy.csv'")

# Generate policy heatmap visualization to help understand intervention patterns
def visualize_policy_heatmap():
    """
    Create a more detailed heatmap visualization of the policy for better understanding
    of how intervention recommendations change with different combinations of risk factors.
    """
    # Create a matrix of best actions for each state combination
    action_indices = {action: i for i, action in enumerate(actions)}
    
    # Prepare the data for the heatmap
    # We'll use (Smoking, Mental Stress) on one axis and (Alcohol, Pollution) on the other
    heatmap_data = np.zeros((4, 4))  # 2x2 for each axis = 4x4 grid
    heatmap_labels = []
    
    # Define positions on the heatmap
    positions = {
        (0, 0): 0,  # (S=0, M=0)
        (0, 1): 1,  # (S=0, M=1)
        (1, 0): 2,  # (S=1, M=0)
        (1, 1): 3,  # (S=1, M=1)
    }
    
    # Fill the heatmap with the index of the best action for each state
    for state in all_possible_states:
        smoking, mental_stress, alcohol, pollution = state
        row = positions[(smoking, mental_stress)]
        col = positions[(alcohol, pollution)]
        
        if state in q_table:
            best_action_idx = np.argmax(q_table[state])
        else:
            best_action_idx = 0  # Default to NO_ACTION if state not seen
            
        heatmap_data[row, col] = best_action_idx
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create action labels for the colorbar
    action_labels = [action for action in actions]
    
    # Create custom colormap with distinct colors for each action
    # Use pyplot.colormaps instead of get_cmap (fixes deprecation warning)
    cmap = plt.colormaps['viridis'].resampled(len(actions))
    
    # Create the heatmap
    ax = sns.heatmap(heatmap_data, cmap=cmap, linewidths=0.5, annot=True, 
                     fmt='.0f', cbar=False, ax=ax)
    
    # Add custom colorbar with action labels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(actions)-1))
    sm.set_array([])
    
    # Fix colorbar by providing the 'ax' argument to specify which axes to use
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(actions)))
    cbar.set_ticklabels(action_labels)
    
    # Label axes
    row_labels = ['S=0,M=0', 'S=0,M=1', 'S=1,M=0', 'S=1,M=1']
    col_labels = ['A=0,P=0', 'A=0,P=1', 'A=1,P=0', 'A=1,P=1']
    
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5)
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    plt.title('Optimal Policy Heatmap\nS=Smoking, M=Mental Stress, A=Alcohol, P=Pollution')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'policy_heatmap.png'))
    plt.close()
    print("Policy heatmap visualization saved to 'policy_heatmap.png'")

# Generate the policy heatmap
visualize_policy_heatmap()

# --- 7. Prepare Content for Research Paper ---
print("\n--- 7. Notes for Research Paper ---")
print("""
Research Paper Structure Suggestions:

1.  **Introduction:**
    *   Problem: Lung cancer burden, importance of prevention and early risk assessment.
    *   Gap: Limitations of static risk models, potential for personalized intervention strategies.
    *   Proposed Solution: Using RL on a simulated environment derived from observational data to suggest interventions.
    *   Contribution: Novel application of RL in this context, methodology outline, highlighting the *simulated* nature and caveats.

2.  **Related Work:**
    *   Traditional lung cancer risk models (e.g., PLCOm2012).
    *   Machine learning applications in lung cancer prediction/diagnosis.
    *   Explainable AI in healthcare.
    *   Reinforcement learning applications in healthcare (e.g., treatment planning, but likely not much in preventative policy suggestion from static data).

3.  **Dataset and EDA:**
    *   Describe the dataset source and features (Table 1).
    *   Present EDA findings: Feature distributions, correlations, target imbalance (include figures generated: target_distribution.png, feature plots, correlation_heatmap.png).
    *   Discuss data preprocessing steps.

4.  **Methodology:**
    *   **Supervised Risk Model ("Oracle"):**
        *   Detail the choice of XGBoost, hyperparameters (mention tuning if performed, though not in this basic script), handling class imbalance (scale_pos_weight).
        *   Present evaluation metrics (Accuracy, ROC AUC, Precision, Recall, F1) and figures (xgboost_confusion_matrix.png, xgboost_roc_curve.png). Justify model reliability.
    *   **Reinforcement Learning Framework:**
        *   Define the MDP: State space (modifiable features, how represented), Action space (interventions), Transition function (simulation logic), Reward function (risk reduction minus cost). Clearly state assumptions.
        *   Explain the choice of Q-Learning algorithm.
        *   Detail RL training parameters (alpha, gamma, epsilon decay, episodes).

5.  **Results:**
    *   Present RL training convergence plots (rl_training_progress.png).
    *   Show the extracted policy table (learned_policy.csv or a formatted version). Highlight key findings for high-risk states.
    *   Analyze the policy: Does it prioritize certain interventions? Are the suggestions intuitive (e.g., suggesting smoking cessation for smokers)? Discuss states where 'NO_ACTION' is optimal.

6.  **Discussion:**
    *   Interpret the results: What does the learned policy imply about intervention strategies *within the simulation*?
    *   **Crucial Limitations:**
        *   **Correlation vs. Causation:** Emphasize that the model learns based on correlations in the *observational data*. The XGBoost model doesn't know *why* smoking cessation reduces risk, only that the predicted risk is lower for non-smokers in the data. Real-world causal effects might differ.
        *   **Simulation Fidelity:** The environment is a simplified simulation based solely on the risk model's predictions. It doesn't capture real-world complexities, time delays, patient adherence, or other factors influencing intervention success.
        *   **Static Data:** The policy is derived from a snapshot dataset, not longitudinal data showing actual intervention effects.
        *   **Choice of Non-Modifiable Profile:** The policy might change depending on the fixed characteristics (age, gender, etc.). The current implementation uses random samples per episode reset, averaging effects somewhat, but could be explored further (e.g., training separate agents for different demographic groups).
        *   **Action Cost:** The chosen action cost is arbitrary and influences the policy.
    *   **Future Work:** Using causal inference methods, incorporating longitudinal data, validating with real intervention studies, exploring more complex RL algorithms or state representations.

7.  **Conclusion:**
    *   Summarize the findings and the potential utility (and significant limitations) of using RL in this simulated context for generating hypotheses about personalized prevention strategies. Reiterate the need for caution and further validation.

8.  **References**
9.  **Appendix (Optional):** Full Q-table, detailed hyperparameter tuning process.
""")

print(f"\nAll results and plots saved in the '{RESULTS_DIR}' directory.")
print("Code execution complete.")