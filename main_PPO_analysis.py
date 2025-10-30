from PPO_Library import ModelReport

def model_analysis(task_path):
    task = ModelReport(task_path)
    task.plot(show=True, save_path="./analysis")

if __name__ == "__main__":

    task_path = "multitask_PPO/task_1"
    task = model_analysis(task_path)