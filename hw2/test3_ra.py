import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV files
df1 = pd.read_csv('processed_eval_data_lambda0_high_n.csv')
df2 = pd.read_csv('processed_eval_data_lambda0.95_high_n.csv')
df3 = pd.read_csv('processed_eval_data_lambda0.98_high_n.csv')
df4 = pd.read_csv('processed_eval_data_lambda0.99_high_n.csv')
df5 = pd.read_csv('processed_eval_data_lambda1_high_n.csv')
# 2. Create the plot
plt.figure(figsize=(10, 6))

x_col = 'Train_EnvstepsSoFar'
y_col = 'Eval_AverageReturn_RunningAvg'

# 3. Plot all 5 dataframes with correct Lambda labels
plt.plot(df1[x_col], df1[y_col], label='lambda = 0', linewidth=1.5)
plt.plot(df2[x_col], df2[y_col], label='lambda = 0.95', linewidth=1.5)
plt.plot(df3[x_col], df3[y_col], label='lambda = 0.98', linewidth=1.5)
plt.plot(df4[x_col], df4[y_col], label='lambda = 0.99', linewidth=1.5)
plt.plot(df5[x_col], df5[y_col], label='lambda = 1.0', linewidth=1.5)

# 4. Add formatting
plt.title('GAE Performance Comparison: LunarLander-v2')
plt.xlabel('Number of Steps')
plt.ylabel('Average Return')
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Show Legend
plt.legend()

# 6. Show
plt.show()