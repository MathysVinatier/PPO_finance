
__all__ = [
    "TradingEnv",
    "DataLoader",
    "QLearning",
    "DeepQLearning",
    "PPOAgent",
    "ACAgent",
    "ModelReport",
]

def __getattr__(name):
    if name == "TradingEnv" or name == "DataLoader":
        from .Environment import TradingEnv, DataLoader
        return locals()[name]

    if name in {"QLearning", "DeepQLearning", "PPOAgent"}:
        from .RLMethod import QLearning, DeepQLearning, PPOAgent
        return locals()[name]

    if name == "ACAgent":
        from .PPO import ACAgent
        return ACAgent

    if name == "ModelReport":
        from .PPO_analysis import ModelReport
        return ModelReport

    raise AttributeError(f"module {__name__} has no attribute {name}")