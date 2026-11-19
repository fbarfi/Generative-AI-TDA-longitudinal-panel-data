import subprocess
from pathlib import Path

# Constants
ROOT_DIR = Path(__file__).parent
DOCKERFILE = "Dockerfiledgl"
IMAGE_TAG = "3_run_all_dgl_classification_scripts"
OUTPUT_DIR = ROOT_DIR / "docker_images"
LOGS_DIR = ROOT_DIR / "logs"
PYTHON_VERSION = "3.11"
TARGETPLATFORMS = ["linux/amd64", "linux/arm64"]

def build_image():
    """
    Build the multi-architecture Docker image using Docker Buildx.
    Logs stdout and stderr to separate log files.
    """
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
    """
    Save the Docker image locally as a tarball.
    This is optional if the image is already loaded into the local Docker daemon.
    """
    tar_path = OUTPUT_DIR / f"{IMAGE_TAG}.tar"
    print(f"💾 Saving image to {tar_path}")
    subprocess.run(["docker", "save", "-o", str(tar_path), IMAGE_TAG], check=True)

def main():
    """
    Main function to create required directories, build the image, and optionally save it.
    """
   
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

  
    build_image()

    save_image()

    print(f"✅ Done: {IMAGE_TAG}")

if __name__ == "__main__":
    main()