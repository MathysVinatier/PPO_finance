from utils import ModelReport

def model_analysis(task_path):
    model1 = ModelReport(task_path)
    print(model1._db)

if __name__ == "__main__":

    task_path = "multitask_PPO/task_4"
    model_analysis(task_path)