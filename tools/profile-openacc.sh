#!/bin/bash
# Simple bash script for lazy profiling. 
cd "$(dirname "$0")"

# Configuration params
DEFAULT_SCALE=512
TARGET_DIR="../openacc"
WORKING_DIR="."

# Constants
NAMELIST="namelist"
EXECUTABLE="nemolite2d.exe"

# Get the date time this was initiated.
DATETIME=$(date '+%Y%m%d-%H%M%S')

# help message fn.
usage(){
    echo "usage: $0 [-h] [-s|--scale SCALE] [-m|--metrics] [--dry-run]]"
}

# function to run th eprofile.
profile(){
    SCALE=$1
    METRICS=$2
    DRYRUN=$3

    TARGET_NAMELIST="namelist.$SCALE.profile"
    TIMELINE_FILE="$SCALE-$DATETIME-timeline.nvvp"
    METRICS_FILE="$SCALE-$DATETIME-metrics.nvvp"
    # Log a messsage.
    echo "Profiling $SCALE using $TARGET_DIR/$EXECUTABLE"

    # Check the namelist file exists.
    if [ ! -f "$TARGET_DIR/$TARGET_NAMELIST" ]; then
        echo "Error: $TARGET_DIR/$TARGET_NAMELIST does not exist"
        return
    fi

    # Update the namelist file (symlink)
    rm -f "$WORKING_DIR/$NAMELIST"
    ln -s "$TARGET_DIR/$TARGET_NAMELIST" "$WORKING_DIR/$NAMELIST"

    # Capture a timeline
    timeline_command="nvprof -s -f -o $TARGET_DIR/$TIMELINE_FILE $TARGET_DIR/$EXECUTABLE"
    if [ "$DRYRUN" = "0" ]; then
        $timeline_command
    else
        echo "$timeline_command"
    fi 

    # optionally capture full details
    metrics_command="nvprof -f -o $TARGET_DIR/$METRICS_FILE --analysis-metrics $TARGET_DIR/$EXECUTABLE"
    if [ "$METRICS" = "1" ]; then
        if [ "$DRYRUN" = "0" ]; then
            $metrics_command
        else
            echo "$metrics_command"
        fi 
    fi

    # Revert the namelist file.
    rm "$WORKING_DIR/$NAMELIST"
}

# Check some usage and run the sript. 
scale=$DEFAULT_SCALE
metrics=0
dryrun=0

while [ "$1" != "" ]; do
    case $1 in
        -s | --scale )          shift
                                scale=$1
                                ;;
        -m | --metrics )        metrics=1
                                ;;
             --dry-run )        dryrun=1
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
profile $scale $metrics $dryrun
