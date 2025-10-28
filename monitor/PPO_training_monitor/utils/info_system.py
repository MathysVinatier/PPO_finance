import psutil
import time
import platform

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


def kill_all_main_processes():
    """Kill all main_PPO processes"""
    for proc in psutil.process_iter(attrs=['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any("main" in part for part in cmdline):
                proc.kill()
        except Exception:
            continue