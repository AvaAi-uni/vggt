#!/bin/bash
# VGGT Quantization Experiment - One-Click Runner
# VGGT量化实验 - 一键运行脚本

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   VGGT Quantization Experiment - RunPod       ║${NC}"
echo -e "${GREEN}║   无需数据集的完整量化实验                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration with defaults
MODEL_NAME="${MODEL_NAME:-facebook/VGGT-1B}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/quantization_results}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_FRAMES="${NUM_FRAMES:-5}"
IMG_SIZE="${IMG_SIZE:-518}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Model Name    : $MODEL_NAME"
echo "  Output Dir    : $OUTPUT_DIR"
echo "  Device        : $DEVICE"
echo "  Batch Size    : $BATCH_SIZE"
echo "  Num Frames    : $NUM_FRAMES"
echo "  Image Size    : $IMG_SIZE"
echo ""

# Check GPU availability
if [ "$DEVICE" == "cuda" ]; then
    echo -e "${YELLOW}[1/6] Checking GPU...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo ""
    else
        echo -e "${RED}Warning: nvidia-smi not found. Switching to CPU mode.${NC}"
        DEVICE="cpu"
        echo ""
    fi
else
    echo -e "${YELLOW}[1/6] Running in CPU mode (this will be slow)${NC}"
    echo ""
fi

# Check Python environment
echo -e "${YELLOW}[2/6] Checking Python environment...${NC}"
python --version
echo ""

# Check dependencies
echo -e "${YELLOW}[3/6] Checking dependencies...${NC}"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || {
    echo -e "${RED}Installing PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" || {
    echo -e "${RED}Installing Matplotlib...${NC}"
    pip install matplotlib
}

python -c "import PIL; print(f'✓ Pillow {PIL.__version__}')" || {
    echo -e "${RED}Installing Pillow...${NC}"
    pip install pillow
}

python -c "from huggingface_hub import HfApi; print('✓ HuggingFace Hub')" || {
    echo -e "${RED}Installing HuggingFace Hub...${NC}"
    pip install huggingface-hub
}

echo ""

# Set environment variables
echo -e "${YELLOW}[4/6] Setting environment variables...${NC}"
export HF_HOME="${HF_HOME:-/workspace/huggingface_cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export PYTHONWARNINGS="ignore"

mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"
mkdir -p "$OUTPUT_DIR"

echo "  HF_HOME    : $HF_HOME"
echo "  TORCH_HOME : $TORCH_HOME"
echo ""

# Check if script exists
echo -e "${YELLOW}[5/6] Locating experiment script...${NC}"
SCRIPT_PATH="scripts/run_quantization_experiment_no_data.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Script not found at $SCRIPT_PATH${NC}"
    echo "Please make sure you're in the vggt directory."
    exit 1
fi

echo "  ✓ Found: $SCRIPT_PATH"
echo ""

# Run experiment
echo -e "${GREEN}[6/6] Starting quantization experiment...${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

python "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_frames "$NUM_FRAMES" \
    --img_size "$IMG_SIZE" \
    2>&1 | tee "$OUTPUT_DIR/experiment.log"

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Experiment Completed Successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo ""

    # Show results
    echo -e "${BLUE}Results saved to: ${OUTPUT_DIR}${NC}"
    echo ""

    if [ -d "$OUTPUT_DIR" ]; then
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{printf "  %-40s %10s\n", $9, $5}'
        echo ""
    fi

    # Show summary
    if [ -f "$OUTPUT_DIR/quantization_report.txt" ]; then
        echo -e "${YELLOW}Quick Summary (first 35 lines):${NC}"
        echo -e "${BLUE}───────────────────────────────────────────────────${NC}"
        head -n 35 "$OUTPUT_DIR/quantization_report.txt"
        echo -e "${BLUE}───────────────────────────────────────────────────${NC}"
        echo ""
        echo "Full report: $OUTPUT_DIR/quantization_report.txt"
        echo ""
    fi

    # Instructions for viewing results
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. View visualization: $OUTPUT_DIR/quantization_plots.png"
    echo "  2. Read full report: $OUTPUT_DIR/quantization_report.txt"
    echo "  3. Analyze metrics: $OUTPUT_DIR/quantization_metrics.csv"
    echo "  4. Check raw data: $OUTPUT_DIR/quantization_results.json"
    echo ""

    # Download instructions
    echo -e "${YELLOW}Download Results:${NC}"
    echo "  # Via SCP (from your local machine):"
    echo "  scp -r runpod@<pod-ip>:$OUTPUT_DIR ./local_results/"
    echo ""
    echo "  # Or use RunPod web file browser"
    echo ""

else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ Experiment Failed!${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo "Check the log file for details: $OUTPUT_DIR/experiment.log"
    echo ""
    exit 1
fi
