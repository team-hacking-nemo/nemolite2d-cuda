#!/bin/bash
# Simple bash script for lazy profiling. 
cd "$(dirname "$0")"

# Configuration params
SCALES=(
    32
    64
    128
    256
    512
    1024
    2048
    4096
)

REPS=3

# Get the date time this was initiated.
DATETIME=$(date '+%Y%m%d-%H%M%S')

TARGET_DIR="../original"
WORKING_DIR="../benchmark/original"
LOGDIR="../benchmark/original/logs-$DATETIME"
ARCHIVEDIR="../benchmark/original/archive-$DATETIME"

# Constants
NAMELIST="namelist"
EXECUTABLE="nemolite2d.exe"
DATFILES="*.dat"


# help message fn.
usage(){
    echo "usage: $0 [-h] [--dry-run]]"
}



  
single_benchmark(){
    # Pass args. Scale, rep
    SCALE=$1
    REP=$2
    DRYRUN=$3

    # Calculate paths / compound variables
    TARGET_NAMELIST="namelist.$SCALE"
    ABS_LOGDIR=$(realpath "$LOGDIR")
    ABS_LOGFILE="$ABS_LOGDIR/$SCALE"_"$REP".log
    ARCHIVE="$ARCHIVEDIR/$SCALE"_"$REP"
    ABS_TARGET_NAMELIST=$(realpath $NAMELIST)

    mkdir -p $ARCHIVE

    # Check the namelist file exists.
    if [ ! -f "$TARGET_DIR/$TARGET_NAMELIST" ]; then
        echo "Error: $TARGET_DIR/$TARGET_NAMELIST does not exist."
        return
    fi
    # Update the namelist file (symlink)
    rm -f "$WORKING_DIR/$NAMELIST"

    cp "$TARGET_DIR/$TARGET_NAMELIST" "$WORKING_DIR/$NAMELIST"

    # Log a message
    echo "Benchmark: $SCALE repeat $REP/$REPS using $TARGET_DIR/$EXECUTABLE"

    # Run the command
    cmd="./$EXECUTABLE"
    if [ "$DRYRUN" = "0" ]; then
        pushd "$WORKING_DIR" > /dev/null
        $cmd > "$ABS_LOGFILE"
        popd > /dev/null
    else
        echo "$cmd"
    fi 


    # move outputs.
    find "$WORKING_DIR" -name "$DATFILES" -exec mv {} "$ARCHIVE" 2>/dev/null \; 
    # clean up 
    rm "$WORKING_DIR/$NAMELIST"
}

batch_benchmark(){
    DRYRUN=$1
    # Make directories

    mkdir -p $WORKING_DIR
    mkdir -p $LOGDIR

    # Copy the executable to the working dif
    cp "$TARGET_DIR/$EXECUTABLE" $WORKING_DIR

    # For each scale
    for scale in ${SCALES[@]}; do
        let END=$REPS 
        let rep=1
        while ((rep<=END)); do
            # Run the single benchmark
            single_benchmark $scale $rep $DRYRUN
            let rep++
        done
    done

    # clean up.

    return
}

# Check some usage and run the sript. 
dryrun=0

while [ "$1" != "" ]; do
    case $1 in
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

# Run the benchmark
batch_benchmark $dryrun
