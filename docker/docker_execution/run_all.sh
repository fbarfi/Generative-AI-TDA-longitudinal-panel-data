#!/bin/bash
set -e

echo ""
echo "==============================================================="
echo "📝 Before executing this file, make sure you do:"
echo "    chmod +x run_all.sh"
echo "    ./run_all.sh"
echo "    This works for both macOS and Linux systems."
echo "==============================================================="
echo ""

read -p "Type 'yes' to confirm you are ready to proceed: " answer
if [[ "$answer" != "yes" ]]; then
    echo "Aborting. Please review the instructions above and try again."
    exit 1
fi

mkdir -p logs

# Auto-detect platform
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  PLATFORM="linux/arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
  PLATFORM="linux/amd64"
else
  echo "❌ Unsupported architecture: $ARCH"
  exit 1
fi

echo "🧠 Detected platform: $PLATFORM"

# Define images and scripts
declare -a images_and_scripts=(
    "1_run_all_data_scripts scripts_master/1_run_all_data_scripts.py"
    "2_1_run_all_gudhi_scripts scripts_master/2_1_run_all_gudhi_scripts.py"
    "2_2_run_graph_data_scripts scripts_master/2_2_run_graph_data_scripts.py"
    "3_run_all_dgl_classification_scripts scripts_master/3_run_all_dgl_classification_scripts.py"
    "4_run_all_argva_scripts scripts_master/4_run_all_argva_scripts.py"
    "5_run_all_bayesian_scripts scripts_master/5_run_all_bayesian_scripts.py"
    "6_run_all_subgraphx_scripts scripts_master/6_run_all_subgraphx_scripts.py"
    "7_run_all_bayesian_subgraphx_scripts scripts_master/7_run_all_bayesian_subgraphx_scripts.py"
    "8_extract_paper_figures_and_tables_to_new_folder scripts_master/8_extract_paper_figures_and_tables_to_new_folder.py"
)

summary_file="logs/run_summary.log"
echo "Script, Status" > "$summary_file"

for entry in "${images_and_scripts[@]}"
do
    IMAGE=$(echo "$entry" | awk '{print $1}')
    SCRIPT=$(echo "$entry" | awk '{print $2}')
    BASE_NAME=$(echo "$IMAGE" | sed 's/-image//')

    echo "Running script $SCRIPT with image $IMAGE..."

  START=$(date +%s)

  docker run --rm -i --platform="$PLATFORM" \
    -v "$PWD/project_data":/app/project_data \
    -v "$PWD/scripts_master":/app/scripts_master \
    -v "$PWD/project_codes":/app/project_codes \
    -v "$PWD/logs":/app/logs \
    "$IMAGE" python -u "$SCRIPT" \
    > "logs/${BASE_NAME}_stdout.log" \
    2> "logs/${BASE_NAME}_stderr.log"

  EXIT_CODE=$?
  END=$(date +%s)
  DURATION=$((END - START))

  if [ $EXIT_CODE -eq 0 ]; then
    echo "$SCRIPT, ✅ SUCCESS, ${DURATION}s" >> "$summary_file"
    echo "✅ Finished $SCRIPT in ${DURATION}s"
  else
    echo "$SCRIPT, ❌ FAILED, ${DURATION}s" >> "$summary_file"
    echo "❌ FAILED $SCRIPT in ${DURATION}s"
  fi

  echo "-------------------------------"
done

# Archive logs
echo "📦 Zipping logs..."
zip -r logs.zip logs/ >/dev/null

echo "✅ All scripts attempted. Check logs/ folder, logs.zip archive, and run_summary.log for details."
