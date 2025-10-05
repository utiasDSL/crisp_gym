#!/bin/bash

# Default values
CAMERA1_NAME="camera1"
CAMERA2_NAME="camera2"
WIDTH=640
HEIGHT=640
FPS=15.0
NAMESPACE1=""
NAMESPACE2=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera1-name)
            CAMERA1_NAME="$2"
            shift 2
            ;;
        --camera2-name)
            CAMERA2_NAME="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --namespace1)
            NAMESPACE1="$2"
            shift 2
            ;;
        --namespace2)
            NAMESPACE2="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --camera1-name NAME    Name of the first camera (default: camera1)"
            echo "  --camera2-name NAME    Name of the second camera (default: camera2)"
            echo "  --width WIDTH          Image width for both cameras (default: 640)"
            echo "  --height HEIGHT        Image height for both cameras (default: 640)"
            echo "  --fps FPS              Frames per second for both cameras (default: 30.0)"
            echo "  --namespace1 NS        ROS namespace for first camera (default: none)"
            echo "  --namespace2 NS        ROS namespace for second camera (default: none)"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build command arguments
CMD1_ARGS="--name $CAMERA1_NAME --width $WIDTH --height $HEIGHT --fps $FPS"
CMD2_ARGS="--name $CAMERA2_NAME --width $WIDTH --height $HEIGHT --fps $FPS"

if [[ -n "$NAMESPACE1" ]]; then
    CMD1_ARGS="$CMD1_ARGS --namespace $NAMESPACE1"
fi

if [[ -n "$NAMESPACE2" ]]; then
    CMD2_ARGS="$CMD2_ARGS --namespace $NAMESPACE2"
fi

echo "Starting dual fake cameras..."
echo "Camera 1: $CAMERA1_NAME (namespace: ${NAMESPACE1:-none})"
echo "Camera 2: $CAMERA2_NAME (namespace: ${NAMESPACE2:-none})"
echo "Resolution: ${WIDTH}x${HEIGHT} @ ${FPS} FPS"
echo "Press Ctrl+C to stop both cameras"
echo

# Start both cameras in background
python3 "$SCRIPT_DIR/fake_camera.py" $CMD1_ARGS &
PID1=$!

python3 "$SCRIPT_DIR/fake_camera.py" $CMD2_ARGS &
PID2=$!

# Function to handle cleanup on exit
cleanup() {
    echo
    echo "Shutting down dual fake cameras..."
    kill $PID1 $PID2 2>/dev/null
    wait $PID1 $PID2 2>/dev/null
    echo "Cameras stopped."
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Wait for both processes
wait $PID1 $PID2
