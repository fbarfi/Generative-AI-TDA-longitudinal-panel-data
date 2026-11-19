import subprocess
from pathlib import Path

# Constants
ROOT_DIR = Path(__file__).parent
DOCKERFILE = "Dockerfilebayesiansubgraphx"
IMAGE_TAG = "7_run_all_bayesian_subgraphx_scripts"
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
