
import subprocess
import sys
import os
import time
import pandas as pd

scripts_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../project_codes/graphs_main_codes"))

scripts_to_run = [
    'main_building_graphs.py',
]

script_results = []
log_dir = "logs"
done_dir = "done_flags"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(done_dir, exist_ok=True)

def run_script(script_name):
    script_path = os.path.join(scripts_directory, script_name)
    log_file_path = os.path.join(log_dir, f"{script_name}.log")
    done_file_path = os.path.join(done_dir, f"{script_name}.done")

    if os.path.exists(done_file_path):
        print(f"⏩ Skipping {script_name} (already done)")
        return

    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            [sys.executable, script_path],
            stdout=open(log_file_path, "w"),
            stderr=subprocess.STDOUT,
            check=False,
        )
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 2)

        script_results.append({
            "script_name": script_name,
            "execution_time_sec": execution_time,
            "exit_status": result.returncode,
            "log_file": log_file_path
        })

        if result.returncode == 0:
            open(done_file_path, "w").write("done\n")
            print(f"✅ {script_name} finished in {execution_time}s")
        else:
            print(f"❌ {script_name} failed — see {log_file_path}")

    except Exception as e:
        print(f"🔥 Error running {script_name}: {e}")
        script_results.append({
            "script_name": script_name,
            "execution_time_sec": None,
            "exit_status": "Failed",
            "log_file": log_file_path
        })

if __name__ == "__main__":
    for script in scripts_to_run:
        run_script(script)

    os.makedirs("execution_results_csv", exist_ok=True)
    results_file = "execution_results_csv/2_2_script_execution_results.csv"
    pd.DataFrame(script_results).to_csv(results_file, index=False)
    print(f"📝 Execution results saved to {results_file}")
