#!/bin/bash
# Simple bash script for lazy profiling. 
cd "$(dirname "$0")"

TARGET_DIR="../openacc"
#TARGET_NAMELIST="namelist.128.profile"
TARGET_NAMELIST="namelist.512.profile"
#TARGET_NAMELIST="namelist.4096.profile"

NAMELIST="namelist"
EXECUTABLE="nemolite2d.exe"
WORKING_DIR="."

# Get the date time this was initiated.
DATETIME=$(date '+%Y-%m-%d-%H%M%S')
TIMELINE_FILE="$DATETIME-timeline.nvvp"
METRICS_FILE="$DATETIME-metrics.nvvp"

# help message fn.
usage(){
    echo "usage: $0 [-h] [-m|--metrics]]"
}


profile(){
    metrics=$1
    # Log a messsage.
    echo "Profiling $TARGET_NAMELIST using $TARGET_DIR/$EXECUTABLE"

    # Update the namelist file (symlink)
    rm -f "$WORKING_DIR/$NAMELIST"
    ln -s "$TARGET_DIR/$TARGET_NAMELIST" "$WORKING_DIR/$NAMELIST"

    # Capture a timeline
    nvprof -f -o "$TARGET_DIR/$TIMELINE_FILE" "$TARGET_DIR/$EXECUTABLE"

    # optionally capture full details
    if [ "$metrics" = "1" ]; then
        nvprof -f -o "$TARGET_DIR/$METRICS_FILE" --analysis-metrics "$TARGET_DIR/$EXECUTABLE"
    fi

    # Revert the namelist file.
    rm "$WORKING_DIR/$NAMELIST"
}

# Check some usage and run the sript. 
metrics=0

while [ "$1" != "" ]; do
    case $1 in
        -m | --metrics )        metrics=1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# Run the profile fn
profile $metrics
