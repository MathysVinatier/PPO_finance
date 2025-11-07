import multiprocessing
import optuna
from config import *

class OptunaAPI:
    def __init__(self, objective):
        """
        objective: callable(trial) -> float
        """
        if not callable(objective):
            raise ValueError("objective must be a callable(trial) -> float")
        self.objective = objective

    def _run_worker(self, study, n_trials_per_worker):
        """
        Each process runs this. Because the study uses a shared storage (SQLite/RDB),
        parallel optimization is coordinated by Optuna.
        """
        try:
            study.optimize(self.objective, n_trials=n_trials_per_worker, show_progress_bar=True, n_jobs=N_JOBS_PER_WORKER)
        except Exception as e:
            # print to help debugging worker crashes
            print(f"[Optuna worker] Exception: {e}")

    def optimization(self, n_workers=N_WORKERS, n_trials=N_TRIALS, storage_url=OPTUNA_DB_PATH):
        """
        Start optimization using multiple processes.
        n_workers: number of parallel processes
        n_trials: total number of trials (will be divided among workers)
        storage_url: optuna storage string (sqlite or other)
        """
        if n_workers < 1:
            n_workers = 1

        # create or load study
        study = optuna.create_study(
            direction="maximize",
            study_name="ppo_study",
            storage=storage_url,
            load_if_exists=True
        )

        # distribute trials across workers as evenly as possible
        base = n_trials // n_workers
        remainder = n_trials % n_workers
        trials_per_worker = [base + (1 if i < remainder else 0) for i in range(n_workers)]

        processes = []
        for i in range(n_workers):
            p = multiprocessing.Process(target=self._run_worker, args=(study, trials_per_worker[i]))
            p.start()
            processes.append(p)

        # wait for all workers to finish
        for p in processes:
            p.join()
