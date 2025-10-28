import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns

import numpy as np
import io

def plot_training(df, title="Training Plot"):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    vmin = np.percentile(df["mean_actor_loss"], 1)
    vmax = np.percentile(df["mean_actor_loss"], 99)

    scatter = sns.scatterplot(
        data=df,
        x="episode_number",
        y="final_reward",
        hue="mean_actor_loss",
        palette="coolwarm",
        hue_norm=(vmin, vmax),
        s=50,
        ax=ax1,
        legend=False
    )

    ax1.set_ylabel("Reward")
    ax1.grid(True)

    ax2 = ax1.twinx()
    sns.histplot(
        data=df,
        x="episode_number",
        weights="actions_buy",
        bins=df["episode_number"].nunique(),
        alpha=0.3,
        color="gray",
        multiple="layer",
        shrink=0.8,
        ax=ax2
    )
    ax2.set_ylabel("Number of Trades")

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Mean Actor Loss")

    plt.xlabel("Episode")
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
