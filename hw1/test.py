import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file
# Replace 'your_file.csv' with the actual path to your file
df = pd.read_csv('flow_loss.csv')

# 2. Extract the columns
# Ensure these names match your CSV headers exactly (case-sensitive)
steps = df['Step']
loss = df['seed_42_20260203_072716 - train/loss']

# 3. Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, loss, label='Training Loss', color='blue', linewidth=1)

# 4. Add formatting
plt.title('Training Loss over Steps')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 5. Show or save the plot
plt.show()
# plt.savefig('loss_graph.png') # Uncomment to save as an image