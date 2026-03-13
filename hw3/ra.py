import pandas as pd

# Mapping of input directory suffix to output filename suffix
input_path = f'exp/MsPacman_dqn_sd1_20260307_024000/log.csv'
output_path = f'processed_eval_data_mspacman.csv'

# Load
df = pd.read_csv(input_path)

# Process
df['Train_EpisodeReturn_RunningAvg'] = (
    df['Train_EpisodeReturn']
    .rolling(window=20, min_periods=1)
    .mean()
)

# Save
df.to_csv(output_path, index=False)
print(f"Processed {input_path} -> {output_path}")