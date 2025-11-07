import os, psutil, threading, os, signal

from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ansi2html import Ansi2HTMLConverter

from typing import Optional

from utils import get_df_training, get_db_path, plot_training, get_system_stats, list_trial, list_task, OptunaTask, TrainingTask, get_all_main_processes, kill_all_main_processes

app = FastAPI(title="PPO Training Monitor")
app.mount("/static", StaticFiles(directory="/home/mathys/Documents/PPO_finance/monitor/PPO_training_monitor/static"), name="static")

templates = Jinja2Templates(directory="/home/mathys/Documents/PPO_finance/monitor/PPO_training_monitor/templates")

# --------------------------
# Process / Log management
# --------------------------
task   = TrainingTask()
optuna = OptunaTask()

# --------------------------
# ROUTES
# --------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/logs")
async def logs(task_name: Optional[str] = Query(None)):
    if not task_name:
        return HTMLResponse(f"<p>{task_name} task</p>")
    # current_task check (you may want to ignore that if you want to read logs even if nothing is running)
    log_path = os.path.join("logs", f"training_{task_name}.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        conv = Ansi2HTMLConverter()
        return HTMLResponse(conv.convert(content, full=False))
    else:
        return HTMLResponse("<p>No logs yet.</p>")
    
@app.get("/logs_optuna", response_class=HTMLResponse)
async def logs_optuna():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return HTMLResponse("<p>No logs directory found.</p>")

    logs_html = ""
    conv = Ansi2HTMLConverter()

    for filename in sorted(os.listdir(log_dir)):
        if filename.startswith("optuna_"):
            file_path = os.path.join(log_dir, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            logs_html += f"<h3>{filename}</h3>"
            logs_html += conv.convert(content, full=False)
            logs_html += "<hr>"

    if not logs_html:
        return HTMLResponse("<p>No Optuna logs found.</p>")

    return HTMLResponse(logs_html)

@app.post("/launch_task")
async def launch_task(
    episode: int = Form(...),
    epoch: int = Form(...),
    batch_size: int = Form(...),
    gamma: float = Form(...),
    lr: float = Form(...),
    gae: float = Form(...),
    policy_clip: float = Form(...),
    trial: int = Form(...)
):
    if task.running_process:
        return JSONResponse({"status": "A process is already running"})

    threading.Thread(
        target=task.run,
        args=("/home/mathys/Documents/PPO_finance/multitask_PPO/launch_task.sh", episode, epoch, batch_size, gamma, lr, gae, policy_clip, trial),
        daemon=True
    ).start()

    return JSONResponse({"status": "Task launched!"})


@app.post("/launch_optuna")
async def launch_optuna(
    n_trial: int = Form(...),
    n_agent: int = Form(...)
):
    """
    Launch the Optuna process using the OptunaTask class.
    """

    if optuna.running_process != True:
        optuna.run(n_trial, n_agent)

        return JSONResponse({
            "status": f"🚀 Optuna launched (trial={n_trial}, worker={n_agent})",
            "log_file": optuna.log_file or "pending initialization..."
        })
    else:
        return JSONResponse({
            "status": f"Optuna already launched (trial={n_trial}, worker={n_agent})",
            "log_file": optuna.log_file
        })


@app.get("/optuna_logs")
async def optuna_logs():
    """Stream the Optuna optimization log file to the dashboard."""
    log_path = optuna.log_file
    if not log_path or not os.path.exists(log_path):
        return PlainTextResponse("[No Optuna log file found yet]", status_code=200)
    try:
        with open(log_path, "r") as f:
            content = f.read()
    except Exception as e:
        content = f"[Error reading log file: {e}]"
    return PlainTextResponse(content)

@app.get("/tasks")
async def get_tasks():
    sorted_list = sorted(list_task(), key=lambda x: int(x.split("_")[1]), reverse=True)
    return JSONResponse(sorted_list)

@app.get("/trials/{task_name}")
async def get_trials(task_name: str):
    sorted_list = sorted(list_trial(task_name), key=lambda x: int(x.split("_")[1]), reverse=True)
    return JSONResponse(list_trial(task_name))

@app.get("/trial_data")
async def trial_data(task_name: str, trial_name: str):
    df = get_df_training(get_db_path(task_name), trial_name)
    table_html = df.to_html(classes="table table-striped", index=False)
    return JSONResponse({"table_html": table_html})

@app.get("/system_stats")
async def system_stats():
    """Return real-time CPU, RAM, temps, and active task info."""
    stats = get_system_stats()
    procs_alive = get_all_main_processes()
    if procs_alive != None:
        state_proc = " / ".join(list(procs_alive.values()))
    else :
        state_proc = None

    return JSONResponse({
        "cpu": stats["cpu_percent"],
        "ram": psutil.virtual_memory().percent,
        "temps": stats.get("temps", {})["k10temp"],
        "active_task": state_proc,
        "is_running": (task.running_process is not None) or (optuna.running_process is not None)
    })

@app.post("/kill_task")
async def kill_task():
    procs = get_all_main_processes()
    if not procs:
        return JSONResponse({"status": "❌ No active task or Optuna process found"})

    try:
        kill_all_main_processes()

        # Update both task states
        task.process_log.append("[USER ACTION] Task manually terminated")
        task.running_process = None
        task.current_task = None
        optuna.running_process = None

        return JSONResponse({"status": "🛑 All processes (training + optuna) killed successfully"})
    except Exception as e:
        return JSONResponse({"status": f"Error killing processes: {e}"})


@app.post("/task", response_class=HTMLResponse)
async def show_task(task_name: str = Form(...), trial_name: str = Form(...)):
    db_path = get_db_path(task_name)
    if not os.path.exists(db_path):
        return HTMLResponse(f"<h1>DB not found for task {task_name}</h1>", status_code=404)
    try:
        df = get_df_training(db_path, trial_name)
    except Exception as e:
        return HTMLResponse(f"<h1>Error reading trial '{trial_name}': {e}</h1>")

    html = f"<h1>Selection {task_name} {trial_name}</h1>"
    html += "<a href='/'>Back to Dashboard</a><br><br>"
    html += df.to_html(classes="table table-striped")
    html += f"<br><img src='/task/{task_name}/{trial_name}/plot.png'>"
    return html

@app.get("/task/{task_name}/{trial_name}/plot.png")
async def trial_plot(task_name: str, trial_name: str):
    db_path = get_db_path(task_name)
    df      = get_df_training(db_path, trial_name)
    buf     = plot_training(df, f"{task_name} - {trial_name}")
    return StreamingResponse(buf, media_type="image/png")

@app.get("/task/{task_name}/{trial_name}/analysis_plot.png")
async def analysis_plot(task_name: str, trial_name: str):
    analysis_path = f"/home/mathys/Documents/PPO_finance/multitask_PPO/{task_name}/data_training/plot/trial_{int(trial_name.split("_")[-1]):03d}/analysis.png"
    if not os.path.exists(analysis_path):
        return JSONResponse({"error": f"Analysis plot not found ({analysis_path})"}, status_code=404)
    return StreamingResponse(open(analysis_path, "rb"), media_type="image/png")


@app.get("/task/{task_name}/{trial_name}/test.png")
async def test_plot(task_name: str, trial_name: str):
    test_path = f"/home/mathys/Documents/PPO_finance/multitask_PPO/{task_name}/data_training/plot/trial_{int(trial_name.split("_")[-1]):03d}/test.png"
    if not os.path.exists(test_path):
        return JSONResponse({"error": f"Test plot not found ({test_path})"}, status_code=404)
    return StreamingResponse(open(test_path, "rb"), media_type="image/png")

@app.get("/task/{task_name}/{trial_name}/train.png")
async def train_plot(task_name: str, trial_name: str):
    test_path = f"/home/mathys/Documents/PPO_finance/multitask_PPO/{task_name}/data_training/plot/trial_{int(trial_name.split("_")[-1]):03d}/train.png"
    if not os.path.exists(test_path):
        return JSONResponse({"error": f"Train plot not found ({test_path})"}, status_code=404)
    return StreamingResponse(open(test_path, "rb"), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

