import os
import psutil
import time
import platform
from subprocess import Popen, PIPE
import signal

def get_system_stats():
    """
    Return detailed system stats:
    - CPU usage (total and per core)
    - RAM usage
    - Temperatures (if available)
    - System uptime
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    ram = psutil.virtual_memory().percent

    temps = {}
    try:
        if hasattr(psutil, "sensors_temperatures"):
            raw_temps = psutil.sensors_temperatures()
            for name, entries in raw_temps.items():
                temps[name] = [round(t.current, 1) for t in entries if t.current is not None]
    except Exception:
        temps = {}

    # Uptime
    boot_time = psutil.boot_time()
    uptime_seconds = int(time.time() - boot_time)
    uptime_hours = uptime_seconds // 3600
    uptime_minutes = (uptime_seconds % 3600) // 60

    return {
        "cpu_percent": cpu_percent,
        "per_cpu": per_cpu,
        "ram_percent": ram,
        "temps": temps,
        "uptime": f"{uptime_hours}h {uptime_minutes}m",
        "os": platform.system()
    }

def get_all_task_processes():
    proc_dict = dict()

    ps = Popen(['ps', '-eo', "pid,args"], stdout=PIPE, text=True)
    grep = Popen(['grep', 'multitask'], stdin=ps.stdout, stdout=PIPE, text=True)
    ps.stdout.close()
    stdout, _ = grep.communicate()

    for line in stdout.splitlines():
        pid, cmdline = line.split(' ', 1)
        if "--episode" in cmdline:
            proc_dict[pid] = cmdline.split(" ")[2].split("/")[-2]
    if len(proc_dict.keys()) == 0:
        return {}
    else:
        return proc_dict
    
def get_all_optuna_processes():
    proc_dict = {}

    ps = Popen(['ps', '-eo', "pid,args"], stdout=PIPE, text=True)
    grep = Popen(['grep', 'optuna_optimization/main.py'], stdin=ps.stdout, stdout=PIPE, text=True)
    ps.stdout.close()
    stdout, _ = grep.communicate()

    for line in stdout.splitlines():
        if "grep" in line:
            continue  # skip our own grep line
        pid, cmdline = line.strip().split(" ", 1)
        if "--trial" in cmdline:
            # Try to get a nice name for the process
            name = f"optuna_{cmdline.split('--trial')[1].strip().split(' ')[0]}"
            proc_dict[pid] = name

    return proc_dict

def get_all_main_processes():
    tasks_procs = get_all_task_processes() or {}
    optunas_procs = get_all_optuna_processes() or {}
    return tasks_procs | optunas_procs

def kill_all_main_processes():
    procs_alive = get_all_main_processes()
    errors = []

    for pid_str, name in procs_alive.items():
        try:
            pid = int(pid_str)
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            errors.append((pid_str, str(e)))

    if errors:
        raise RuntimeError(f"Failed to kill some processes: {errors}")