import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV files
df1 = pd.read_csv('exp/CartPole-v0_cartpole_lb_na_sd1/log.csv')
df2 = pd.read_csv('exp/CartPole-v0_cartpole_lb_rtg_na_sd1/log.csv')
df3 = pd.read_csv('exp/CartPole-v0_cartpole_lb_rtg_sd1/log.csv')
df4 = pd.read_csv('exp/CartPole-v0_cartpole_lb_sd1/log.csv')

# 2. Create the plot
plt.figure(figsize=(10, 6))

# Define the shared column names
x_col = 'Train_EnvstepsSoFar'
y_col = 'Eval_AverageReturn'

# 3. Plot each dataframe
# Note: I've given them labels based on the folder names to help you distinguish them
plt.plot(df1[x_col], df1[y_col], label='lb_na', linewidth=1.5)
plt.plot(df2[x_col], df2[y_col], label='lb_rtg_na', linewidth=1.5)
plt.plot(df3[x_col], df3[y_col], label='lb_rtg', linewidth=1.5)
plt.plot(df4[x_col], df4[y_col], label='lb', linewidth=1.5)

# 4. Add formatting
plt.title('Comparison of Average Return over Steps')
plt.xlabel('Number of Steps')
plt.ylabel('Average Return')
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Show Legend (Crucial for identifying which line is which)
plt.legend()

# 6. Show or save the plot
plt.show()
# plt.savefig('comparison_graph.png')