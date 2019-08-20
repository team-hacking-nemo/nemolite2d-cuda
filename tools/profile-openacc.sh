#!/bin/bash
# Simple bash script for lazy profiling. 
cd "$(dirname "$0")"

TARGET_DIR="../openacc"
TARGET_NAMELIST="namelist.128.profile"

NAMELIST="namelist"
EXECUTABLE="nemolite2d.exe"
TMP_NAMELIST="namelist.tmpprofile"

TIMELINE_FILE="timeline.nvvp"
METRICS_FILE="metrics.nvvp"


# help message fn. 
usage(){
    echo "usage: $0 [-h] [-m|--metrics]]"
}


profile(){
    metrics=$1
    # Log a messsage.
    echo "Profiling $TARGET_NAMELIST usign $TARGET_DIR/$EXECUTABLE"

    # Update the namelist file (symlink)
    mv "$TARGET_DIR/$NAMELIST" "$TARGET_DIR/$TMP_NAMELIST"
    ln -s "$TARGET_DIR/$TARGET_NAMELIST" "$TARGET_DIR/$NAMELIST"

    # Capture a timeline
    nvprof -o "$TARGET_DIR/$TIMELINE_FILE" ./"$EXECUTABLE"

    # optionally capture full details
    if [ "$metrics" = "1" ]; then
        nvprof -o "$TARGET_DIR/$METRICS_FILE" --analysis-metrics ./"$EXECUTABLE"
    fi

    # Revert the namelist file.
    mv "$TARGET_DIR/$TMP_NAMELIST" "$TARGET_DIR/$NAMELIST"
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
echo $metrics
# Run the profile fn
profile $metrics
