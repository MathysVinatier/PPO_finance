import os, sys, subprocess, threading, time

from .config import TASK_FOLDER

class TrainingTask:
    def __init__(self):
        self.process_log = []
        self.running_process = None   # Popen object for the training Python process
        self.current_task = None
        self.log_file = None

    def run(self, script_path, episode, epoch, batch_size,
            gamma, lr, gae, policy_clip, trial):
        """
        Launch the training Python process directly (in its own process group),
        redirect stdout/stderr to a log file, and also open a terminal that tails that log.
        """

        self.process_log = []

        # 1) Create the task folder using your existing script
        task_create = subprocess.run(["bash", script_path], capture_output=True, text=True)
        self.process_log.append(f"[INIT] {task_create.stdout.strip() or task_create.stderr.strip()}")
        time.sleep(0.2)

        # 2) Extract task name
        lines = task_create.stdout.strip().split()
        task_name = next((w for w in lines if "task_" in w), None)
        if not task_name:
            self.process_log.append("[ERROR] Could not detect new task folder")
            return

        self.current_task = task_name
        training_file = os.path.join(TASK_FOLDER, task_name, "main_PPO.py")

        # 3) Prepare log file
        os.makedirs("logs", exist_ok=True)
        log_file = os.path.abspath(os.path.join("logs", f"training_{task_name}.log"))
        self.log_file = log_file

        # 4) Build python command
        py_cmd = [
            sys.executable, "-u", training_file,
            "--episode", str(episode),
            "--epoch", str(epoch),
            "--batch_size", str(batch_size),
            "--gamma", str(gamma),
            "--lr", str(lr),
            "--gae", str(gae),
            "--policy_clip", str(policy_clip),
            "--trial", str(trial)
        ]

        # 5) Launch Python process directly, set new process group (Unix)
        #    So we can kill the whole group later with os.killpg(...)
        try:
            # open the logfile for writing (append)
            logfile_handle = open(log_file, "a", buffering=1)  # line-buffered
        except Exception as e:
            self.process_log.append(f"[ERROR] Could not open log file {log_file}: {e}")
            return

        # Unix: use preexec_fn to create new session (process group leader)
        preexec_fn = None
        creationflags = 0
        if os.name == "posix":
            preexec_fn = os.setsid
        else:
            # Windows: we'll set CREATE_NEW_PROCESS_GROUP
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self.running_process = subprocess.Popen(
                py_cmd,
                stdout=logfile_handle,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=preexec_fn,
                creationflags=creationflags
            )
        except Exception as e:
            logfile_handle.close()
            self.process_log.append(f"[ERROR] Failed to start training process: {e}")
            return

        self.process_log.append(f"[RUNNING] {' '.join(py_cmd)} (pid={self.running_process.pid})")
        self.process_log.append(f"[LOG] Writing to {log_file}")

        # 6) Open a terminal that tails the log so you get a real terminal view
        #    Using tail -f and keeping shell open (exec bash)
        tail_cmd = f'tail -f "{log_file}"; exec bash'
        try:
            # try gnome-terminal first
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", tail_cmd])
        except FileNotFoundError:
            # fallback to x-terminal-emulator if available
            try:
                subprocess.Popen(["x-terminal-emulator", "-e", f"bash -c \"{tail_cmd}\""])
            except Exception as e:
                # not fatal — we still have the logfile and the process
                self.process_log.append(f"[WARN] Could not open terminal to tail logs: {e}")

        # 7) Spin a watcher thread to detect process exit and append status to log file + process_log
        def watcher():
            rc = self.running_process.wait()
            status = "[DONE]" if rc == 0 else f"[FAILED rc={rc}]"
            try:
                with open(log_file, "a") as fh:
                    fh.write(status + "\n")
            except Exception:
                pass
            self.process_log.append(status)
            # cleanup
            self.running_process = None
            self.current_task = None
            # don't close logfile_handle here — Popen already has it; but close our handle reference
            try:
                logfile_handle.close()
            except Exception:
                pass

        threading.Thread(target=watcher, daemon=True).start()
