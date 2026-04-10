import pandas as pd
import matplotlib.pyplot as plt

# 1. Define file paths and their corresponding labels
experiments = [
    ('exp/LunarLander-v2_dqn_sd1_20260306_213230/log.csv', 'LR 1e-3'),
    ('exp/LunarLander-v2_dqn_lr_2e-3_sd1_20260314_181523/log.csv', 'LR 2e-3'),
    ('exp/LunarLander-v2_dqn_lr_2e-4_sd1_20260314_181620/log.csv', 'LR 2e-4'),
    ('exp/LunarLander-v2_dqn_lr_5e-4_sd1_20260314_181537/log.csv', 'LR 5e-4'),
]

# 2. Setup the plot
plt.figure(figsize=(10, 6))
x_col = 'step'
y_col = 'Eval_AverageReturn'

# 3. Iterate through experiments, load, and plot
for path, label in experiments:
    try:
        df = pd.read_csv(path)
        # Drop rows where the evaluation metric is missing
        df = df.dropna(subset=[y_col])
        
        plt.plot(df[x_col], df[y_col], label=label, linewidth=1.5)
    except FileNotFoundError:
        print(f"Warning: File not found at {path}")

# 4. Add formatting
plt.title('Comparison of Average Return over Steps (Learning Rate Study)')
plt.xlabel('Number of Steps')
plt.ylabel('Average Return')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 5. Show the plot
plt.tight_layout()
plt.show()