
import subprocess
import sys
import os
import time
import pandas as pd

scripts_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../project_codes/SubGraphX_main_codes"))

scripts_to_run = [
    "main_SubGraphX.py", 
    "main_deploying_SubGraphX.py",
    "main_SubGraphX_plot_saved_graphs.py",
]

log_dir = "logs"
done_dir = "done_flags"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(done_dir, exist_ok=True)

script_results = []

def run_script(script_name):
    script_path = os.path.join(scripts_directory, script_name)
    log_file_path = os.path.join(log_dir, f"{script_name}.log")
    done_file_path = os.path.join(done_dir, f"{script_name}.done")

    if os.path.exists(done_file_path):
        print(f"⏭️  Skipping {script_name} (already done)")
        return {
            "script_name": script_name,
            "execution_time_sec": 0,
            "exit_status": "Skipped",
            "log_file": log_file_path
        }

    try:
        start_time = time.perf_counter()
        with open(log_file_path, "w") as log_file:
            result = subprocess.run(
                [sys.executable, script_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False
            )
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 2)

        if result.returncode == 0:
            print(f"✅ {script_name} completed in {execution_time}s")
            with open(done_file_path, "w") as f:
                f.write("done\n")
        else:
            print(f"❌ {script_name} failed with exit code {result.returncode}")

        return {
            "script_name": script_name,
            "execution_time_sec": execution_time,
            "exit_status": result.returncode,
            "log_file": log_file_path
        }

    except Exception as e:
        print(f"🚨 Error running {script_name}: {e}")
        return {
            "script_name": script_name,
            "execution_time_sec": None,
            "exit_status": "Error",
            "log_file": log_file_path
        }

if __name__ == "__main__":
    for script in scripts_to_run:
        script_results.append(run_script(script))

    os.makedirs("execution_results_csv", exist_ok=True)
    results_file = "execution_results_csv/6_script_execution_results.csv"
    pd.DataFrame(script_results).to_csv(results_file, index=False)
    print(f"📄 Execution results saved to {results_file}")
