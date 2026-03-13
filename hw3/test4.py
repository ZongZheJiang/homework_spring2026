import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV files
df1 = pd.read_csv('exp/HalfCheetah-v4_sac_sd1_20260308_192322/log.csv')

# 2. Create the plot
plt.figure(figsize=(10, 6))

# Define the shared column names
x_col = 'step'
y_col1 = 'Eval_AverageReturn'

df1 = df1.dropna(subset=[y_col1])

# 3. Plot each dataframe
# Note: I've given them labels based on the folder names to help you distinguish them
plt.plot(df1[x_col], df1[y_col1], label='Eval_AverageReturn', linewidth=1.5)
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