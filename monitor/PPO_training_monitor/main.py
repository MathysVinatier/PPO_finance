import os, psutil, threading, os, signal

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ansi2html import Ansi2HTMLConverter

from utils import get_df_training, get_db_path, plot_training, get_system_stats, list_trial, list_task, TrainingTask

app = FastAPI(title="PPO Training Monitor")
app.mount("/static", StaticFiles(directory="/home/mathys/Documents/PPO_finance/monitor/PPO_training_monitor/static"), name="static")

templates = Jinja2Templates(directory="/home/mathys/Documents/PPO_finance/monitor/PPO_training_monitor/templates")

# --------------------------
# Process / Log management
# --------------------------
task = TrainingTask()

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
    task_state = task.current_task if task.current_task else "None"
    return JSONResponse({
        "cpu": stats["cpu_percent"],
        "ram": psutil.virtual_memory().percent,
        "temps": stats.get("temps", {})["k10temp"],
        "active_task": task_state,
        "is_running": task.running_process is not None
    })

@app.post("/kill_task")
async def kill_task():
    if task.running_process:
        try:
            pid = task.running_process.pid
            # UNIX: kill the whole process group
            if os.name == "posix":
                try:
                    os.killpg(pid, signal.SIGTERM)
                except PermissionError:
                    # Try stronger kill
                    os.killpg(pid, signal.SIGKILL)
            else:
                # Windows: send CTRL_BREAK_EVENT to the group if possible
                try:
                    task.running_process.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    task.running_process.terminate()

            # wait for it to die (best-effort)
            try:
                task.running_process.wait(timeout=5)
            except Exception:
                # fallback: terminate forcefully
                try:
                    task.running_process.kill()
                except Exception:
                    pass

            task.process_log.append("[USER ACTION] Task manually terminated.")
            task.running_process = None
            task.current_task = None
            return JSONResponse({"status": "Task killed successfully."})
        except Exception as e:
            return JSONResponse({"status": f"Error killing task: {e}"})
    return JSONResponse({"status": "No active task to kill."})


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
        return JSONResponse({"error": f"Test plot not found ({test_path})"}, status_code=404)
    return StreamingResponse(open(test_path, "rb"), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

