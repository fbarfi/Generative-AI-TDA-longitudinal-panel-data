
# 🐳 Complete Docker Project

This project contains a structured set of Dockerfiles and Python-based build scripts to create Docker images for various components (e.g., SubgraphX, PyG, Gudhi, Bayesian models, etc.).

---

## 📁 Directory Structure

```
complete_docker_project/
│
├── docker_execution/
│   ├── scripts_master/             # Main execution scripts
│   ├── run_all.sh                  # Shell script to run everything
│   ├── project_data/               # Data folder
│   ├── project_codes/              # Python codes
│   └── docker_images/              # Output Docker images (optional output directory)
│
└── docker_building_scripts/
    ├── Dockerfile<component>       # One Dockerfile per component
    ├── build_dockers_<component>.py  # One build script per component
```

---

## 🛠 How to Use the Build Scripts

Each component (like `subgraphx`, `gudhi`, `pyg`, etc.) has a corresponding Dockerfile and a Python script to build it. All builds should be run from the `docker_building_scripts` directory to maintain path consistency.

### 1. Move to the build script directory

```bash
cd complete_docker_project/docker_building_scripts
```

### 2. Build a Specific Component

Run any of the provided Python scripts to build the corresponding Docker image. Example:

```bash
python build_dockers_subgraphx.py
```

This script will:
- Load `Dockerfilesubgraphx`
- Build the Docker image
- Tag the image appropriately

---

## 🔧 Available Components and Scripts

| Component         | Dockerfile                   | Build Script                          |
|------------------|------------------------------|----------------------------------------|
| SubgraphX        | `Dockerfilesubgraphx`         | `build_dockers_subgraphx.py`          |
| Gudhi            | `Dockerfilegudhi`             | `build_dockers_gudhi.py`              |
| PyTorch Geometric| `Dockerfilepyg`               | *(Script not shown; optional)*        |
| DGL              | `Dockerfiledgl`               | `build_dockers_dgl.py`                |
| Figures Extraction | `Dockerfilefiguresextraction` | `build_dockers_figures_extraction.py`|
| Bayesian         | `Dockerfilebayesian`          | `build_dockers_bayesian.py`           |
| Bayesian+SubgraphX| `Dockerfilebayesiansubgraphx`| `build_dockers_bayesian_subgraphx.py` |
| ARGVA            | `Dockerfileargva`             | `build_dockers_argva.py`              |
| Data Preprocessing | `Dockerfiledata`            | `build_dockers_data.py`               |
| Data+Graph Setup | *(uses `Dockerfiledata`) *    | `build_dockers_data_graph.py`         |

---

## ▶️ Example Build Script: `build_dockers_subgraphx.py`

```python
import subprocess
from pathlib import Path

# Constants
ROOT_DIR = Path(__file__).parent
DOCKERFILE = "Dockerfilesubgraphx"
IMAGE_TAG = "6_run_all_subgraphx_scripts"
OUTPUT_DIR = ROOT_DIR / "docker_images"
LOGS_DIR = ROOT_DIR / "logs"
PYTHON_VERSION = "3.11"
TARGETPLATFORMS = ["linux/amd64", "linux/arm64"]

def build_image():
    stdout_path = LOGS_DIR / f"{IMAGE_TAG}_stdout.log"
    stderr_path = LOGS_DIR / f"{IMAGE_TAG}_stderr.log"

    build_cmd = [
        "docker", "buildx", "build",
        "--platform", ",".join(TARGETPLATFORMS),
        "--build-arg", f"PYTHON_VERSION={PYTHON_VERSION}",
        "-t", IMAGE_TAG,
        "--load",
        "-f", DOCKERFILE,
        "."
    ]

    print(f"\n🔧 Building {IMAGE_TAG} for platforms: {', '.join(TARGETPLATFORMS)}...")
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        subprocess.run(build_cmd, check=True, stdout=out, stderr=err)

def save_image():
    tar_path = OUTPUT_DIR / f"{IMAGE_TAG}.tar"
    print(f"💾 Saving image to {tar_path}")
    subprocess.run(["docker", "save", "-o", str(tar_path), IMAGE_TAG], check=True)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    build_image()
    save_image()
    print(f"✅ Done: {IMAGE_TAG}")

if __name__ == "__main__":
    main()

```

---

## 📤 Output

Once the Docker images are built, they are saved into the `docker_images/` folder.

---

## ✅ Requirements

- Docker installed and running
- Python 3.7+
- `docker` Python SDK (install via `pip install docker` if needed)
- the docker images are created simultaneoously for both amd64 and arm64 machines and are to be used as is.

---

## 🧩 Notes

- Make sure each `build_dockers_*.py` script reads the correct Dockerfile (e.g., using `open("Dockerfilesubgraphx", "r")`)
- Keep all builds reproducible and modular by avoiding hard-coded paths outside `complete_docker_project/`

---
### Author: Badredine Arfi
### Date: July 14, 2025