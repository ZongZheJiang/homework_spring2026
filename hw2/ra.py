import pandas as pd

# Mapping of input directory suffix to output filename suffix
experiments = ["0", "0.95", "0.98", "0.99", "1"]

for lamb in experiments:
    input_path = f'exp/LunarLander-v2_lunar_lander_lambda{lamb}_high_n_sd1_2/log.csv'
    output_path = f'processed_eval_data_lambda{lamb}_high_n.csv'
    
    # Load
    df = pd.read_csv(input_path)
    
    # Process
    df['Eval_AverageReturn_RunningAvg'] = (
        df['Eval_AverageReturn']
        .rolling(window=80, min_periods=1)
        .mean()
    )
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Processed lambda {lamb} -> {output_path}")