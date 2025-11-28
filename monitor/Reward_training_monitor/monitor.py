from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import sqlite3
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import psutil
import subprocess
import sys

# --- INIT APP ---
app = FastAPI(title="Database Monitor API")

# --- UTILS ---
def list_databases(directory="."):
    return [f for f in os.listdir(directory) if f.endswith(".db")]

def compute_drawdown(equity_curve):
    peak = np.maximum.accumulate(np.nan_to_num(equity_curve, nan=-np.inf))
    dd = (equity_curve - peak) / peak
    return dd


def optimization2_plot(df, fname, num_trials_to_show=None):
    if num_trials_to_show is None:
        num_trials_to_show = len(df)
    base_color = 'steelblue'
    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    axes = axes.flatten()

    max_len = max(len(r) for r in df["equity"])
    padded_equity = [r + [np.nan]*(max_len-len(r)) for r in df["equity"]]
    equity_matrix = np.array(padded_equity, dtype=float)

    max_len = max(len(r) for r in df["reward"])
    padded_reward = [r + [np.nan]*(max_len-len(r)) for r in df["reward"]]
    reward_matrix = np.array(padded_reward, dtype=float)
    mean_reward = np.nanmean(reward_matrix, axis=0)
    std_reward = np.nanstd(reward_matrix, axis=0)

    mean_equity = np.nanmean(equity_matrix, axis=0)
    drawdown_matrix = np.array([compute_drawdown(curve) for curve in equity_matrix])
    mean_drawdown = np.nanmean(drawdown_matrix, axis=0)
    arg_mmd = np.argmax(np.abs(mean_drawdown))
    mmd = mean_drawdown[arg_mmd]
    final_drawdowns = drawdown_matrix[:, -1]
    std_equity = np.nanstd(equity_matrix, axis=0)

    axes[0].plot(mean_equity, color="steelblue", lw=2)
    axes[0].fill_between(range(len(mean_equity)), mean_equity-std_equity, mean_equity+std_equity, color="blue", alpha=0.3)
    axes[0].set_title(f"Mean Equity ± Std (Max Drawdown={mmd:.2%})")
    axes[0].grid()

    for row in equity_matrix[:num_trials_to_show]:
        axes[1].plot(row, color=base_color, alpha=0.2)
    axes[1].plot(mean_equity, color="blue", lw=2, alpha=0.5)
    axes[1].set_title(f"Equity Curves ({num_trials_to_show} trials)"); axes[1].grid()

    sns.histplot(df["score"]-100, bins=40, kde=True, ax=axes[2], color="steelblue")
    axes[2].set_title("Equity Profit Distribution"); axes[2].grid()

    axes[3].plot(mean_reward, color="darkgreen", lw=2)
    axes[3].fill_between(range(len(mean_reward)), mean_reward-std_reward, mean_reward+std_reward, color="green", alpha=0.3)
    axes[3].set_title("Mean Reward ± Std"); axes[3].grid()

    for row in drawdown_matrix[:num_trials_to_show]:
        axes[4].plot(row, color="red", alpha=0.15)
    axes[4].plot(mean_drawdown, color="black", lw=2)
    axes[4].set_title(f"Drawdown Curves ({num_trials_to_show} trials)"); axes[4].grid()

    sns.histplot(final_drawdowns, bins=40, kde=True, ax=axes[5], color="red")
    axes[5].set_title("Final Drawdowns Distribution"); axes[5].grid()

    fig.suptitle(fname)
    plt.tight_layout()
    return fig

def kill_all_main_processes():
    """Kill all main_PPO processes"""
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any("main" in part for part in cmdline):
                proc.kill()
        except Exception:
            continue

# --- ROUTES ---
@app.get("/system-status")
def system_status():
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent

    # Main process status
    main_status = "Running" if any(
        "main_PPO.py" in " ".join(proc.info['cmdline'] or [])
        for proc in psutil.process_iter(['cmdline'])
    ) else "Not running"

    # Collect all temperature values
    temps = psutil.sensors_temperatures()
    temp_values = []
    if temps:
        for entries in temps.values():
            for entry in entries:
                if entry.current is not None:
                    temp_values.append(entry.current)
    temp_str = "°C / ".join(f"{t:.1f}" for t in temp_values) if temp_values else "N/A"

    return {
        "cpu": cpu,
        "ram": ram,
        "main_status": main_status,
        "temps": temp_str
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(db: str = Query(default=None), folder: str = Query(default=None)):
    try:
        dbs_root = list_databases(".")
        dbs_appl = list_databases("./databases/AAPL")
        dbs_vix = list_databases("./databases/VIX")
        all_dbs = [(d, "./") for d in dbs_root] + [(d, "./databases/AAPL") for d in dbs_appl] + [(d, "./databases/VIX") for d in dbs_vix]

        if db is None:
            if not all_dbs:
                return "<h3>No databases found</h3>"
            links = "<br>".join([f'<a href="/dashboard?db={d}&folder={f}">{d} ({f})</a>' for d,f in all_dbs])
            return f"<h3>Select a database : <br>{links}</h3>"

        if folder is None:
            folder = "./" if db in dbs_root else "./databases"
        db_path = os.path.join(folder, db)
        if not os.path.exists(db_path):
            return f"<h3>Database {db} not found</h3>"

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT trial_id, result FROM results LIMIT 100", conn)
        conn.close()
        html_table = df.to_html(classes="table table-striped", border=0)

        html = f"""
                <html>
                <head>
                    <title>DB Dashboard</title>
                    <script>
                    async function killMain() {{
                        let resp = await fetch('/kill-main', {{method:'POST'}});
                        let data = await resp.json();
                    }}

                    async function restartMain(reward_type, reward_evolution) {{
                        let resp = await fetch('/restart-main', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{
                                reward_type: reward_type,
                                reward_evolution: reward_evolution
                            }})
                        }});
                        let data = await resp.json();
                        console.log(data);
                    }}

                    function updateSystemStatus() {{
                        fetch("/system-status")
                            .then(resp => resp.json())
                            .then(data => {{
                                document.getElementById("cpu_status").innerText = "CPU: " + data.cpu.toFixed(1) + "%";
                                document.getElementById("ram_status").innerText = "RAM: " + data.ram.toFixed(1) + "%";
                                document.getElementById("main_status_inline").innerText = "main_PPO: " + data.main_status;
                                document.getElementById("temp_status").innerText = "Temp: " + data.temps;
                            }});
                    }}

                    setInterval(updateSystemStatus, 1000);
                    updateSystemStatus();
                    </script>
                </head>
                <body>
                    <h1>📊 Dashboard - {db}</h1>
                    <!-- System status in line -->
                    <div style="display:flex; gap:20px; align-items:center;">
                        <span id="cpu_status">CPU: --%</span>
                        <span id="ram_status">RAM: --%</span>
                        <span id="temp_status">Temp: --</span>
                        <span id="main_status_inline">main_PPO: --</span>
                        <button onclick="killMain()">Kill main_PPO</button>
                    </div>

                    <div style="margin-top:20px;">
                        <h2>🚀 Launch Tests</h2>
                        <button onclick="restartMain('portfolio_diff', 'value')"> Train Optimal agent</button>
                    </div>

                    <h2>📈 Analysis Plot</h2>
                    <img src="/plot?db={db}&folder={folder}" style="max-width:100%; border:1px solid #ccc;"/>

                    <h2>📋 Table Preview</h2>
                    {html_table}
                </body>
                </html>
                """
        return html
    except Exception as e:
        return f"<h2>Error: {e}</h2>"

@app.get("/plot")
def plot(db: str = Query(...), folder: str = Query(default=None)):
    if folder is None:
        folder = "./" if os.path.exists(db) else "./databases"
    db_path = os.path.join(folder, db)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"{db} not found")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT trial_id, result, equity, reward FROM results LIMIT 100").fetchall()
    conn.close()
    records = [{"id": tid, "score": res, "equity": json.loads(eq), "reward": json.loads(rw)} for tid,res,eq,rw in rows]
    df = pd.DataFrame(records).set_index("id")
    fig = optimization2_plot(df, db)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/kill-main")
def kill_main():
    kill_all_main_processes()
    return JSONResponse({"status": "All main_PPO.py processes killed"})

@app.post("/restart-main")
async def restart_main(request: Request):
    data = await request.json()
    reward_type = data.get("reward_type")
    reward_evolution = data.get("reward_evolution")

    kill_all_main_processes()

    if os.name == 'nt':
        subprocess.Popen(f'start cmd /k python main_PPO.py', shell=True)
    else:
        subprocess.Popen([
            'gnome-terminal', '--',
            'python3', 'main_PPO.py',
            '--reward_type', reward_type,
            '--reward_evolution', reward_evolution
        ])

    return JSONResponse({"status": f"main_PPO restarted with {reward_type}/{reward_evolution}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
