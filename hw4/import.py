import wandb

api = wandb.Api()
run = api.run("/zjiang026-university-of-california-berkeley/llm-rl-hw4/runs/3ojxx7qg")

run.history(pandas=True).to_csv("math_hard_grpo.csv")