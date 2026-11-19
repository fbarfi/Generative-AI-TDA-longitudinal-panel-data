import os
import subprocess
import glob

# Directory containing the exported YAML files
yml_dir = os.path.abspath("conda_env_ymls")

if not os.path.isdir(yml_dir):
    print(f"Directory {yml_dir} does not exist.")
    exit(1)

# Find all .yml files in the directory
yml_files = sorted(glob.glob(os.path.join(yml_dir, "*.yml")))

if not yml_files:
    print(f"No .yml files found in {yml_dir}.")
    exit(1)

for yml_file in yml_files:
    # Environment name is the base filename (without extension)
    env_name = os.path.splitext(os.path.basename(yml_file))[0]
    print(f"Creating conda environment '{env_name}' from {yml_file}...")
    try:
        subprocess.run(
            ["conda", "env", "create", "-f", yml_file, "-n", env_name],
            check=True
        )
        print(f"✅ Environment '{env_name}' created successfully.\n")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to create environment '{env_name}' from {yml_file}.\n")

print("All done.")