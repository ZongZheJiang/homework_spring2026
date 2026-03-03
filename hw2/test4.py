import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file
# Replace 'your_file.csv' with the actual path to your file
df = pd.read_csv('exp/InvertedPendulum-v4_pendulum_default_sd1_2/log.csv')
df2 = pd.read_csv('exp/InvertedPendulum-v4_pendulum_longer_sd1_2/log.csv')
df3 = pd.read_csv('exp/InvertedPendulum-v4_pendulum_sd1_2/log.csv')

# 2. Extract the columns
# Ensure these names match your CSV headers exactly (case-sensitive)
steps = df['Train_EnvstepsSoFar']
loss = df['Eval_AverageReturn']  
steps2 = df2['Train_EnvstepsSoFar']
loss2 = df2['Eval_AverageReturn']
steps3 = df3['Train_EnvstepsSoFar']
loss3 = df3['Eval_AverageReturn']
# 3. Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, loss, label='Average Return', color='red', linewidth=1)
plt.plot(steps2, loss2, label='Average Return Longer', color='blue', linewidth=1)
plt.plot(steps3, loss3, label='Average Return Default', color='green', linewidth=1)

# 4. Add formatting
plt.title('Average Return over Steps')
plt.xlabel('Number of Steps')
plt.ylabel('Average Return')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 5. Show or save the plot
plt.show()
# plt.savefig('CS 185 HW 2 Qn 4 Plot 1.png') # Uncomment to save as an image
