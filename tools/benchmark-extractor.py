#! /usr/bin/python3

"""
Python3 script for parsing benchmark data from directories of log files, to produce a csv containing raw info.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

EXIT_FAILURE = 1
EXIT_SUCCESS = 0

LOG_EXT=".log"

# Define command line arguments for this application
def cli_args():
    # Get arguments using argparse
    parser = argparse.ArgumentParser(
        description="Extract benchmark data from nemo2d stdout logs"
        )
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="paths to files or directories of log files to parse."
        )

    parser.add_argument(
        "-b",
        "--build",
        type=str,
        help="Hardcoded build flag. @temp.",
        default="Unknown"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="location for output file"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force overwriting of --output file."
      )


    args = parser.parse_args()
    return args

# Validate command line arguments for pre-empative abortion.
def validate_args(args):
    # Build the list of log files to be parsed.
    valid_files = []
    if args.files is not None and len(args.files) > 0:
        for file_or_path in args.files:
            if os.path.isfile(file_or_path) and file_or_path.endswith(LOG_EXT):
                valid_files.append(file_or_path)
            elif os.path.isdir(file_or_path):
                for root, dirs, files in os.walk(file_or_path):
                    for file in files:
                        if file.endswith(LOG_EXT):
                            valid_files.append(os.path.join(root, file))
            else:
                print("Warning: Provided file {:} does not exist".format(file_or_path))
    # Remove duplicates from valid files and sort
    valid_files = sorted(list(set(valid_files)))

    if valid_files is None or len(valid_files) == 0:
        print("Error: No valid files provided.")
        return False, None
    # Check use of --output and --force
    if args.output is not None:
        # @todo - potential disk race by checking this upfront. Should use try catch later too.
        if os.path.exists(args.output):
            if os.path.isdir(args.output):
                print("Error: {:} is a directory".format(args.sample))
                return False, None
            elif os.path.isfile(args.output):
                if not args.force:
                    print("Error: Output file {:} already exists. Please use -f/--force")
                    return False, None

    return True, valid_files

def parse_files(files, build):
    cols = [
        "build",
        "scale",
        "repeat",
        "jpiglo",
        "jpjglo",
        "ua_checksum",
        "va_checksum",
        "time_stepping_counts",
        "time_stepping_total",
        "time_stepping_average",
        "time_stepping_avg_per_repeat",
        "time_stepping_stderr",
        "continuity_counts",
        "continuity_total",
        "continuity_average",
        "continuity_avg_per_repeat",
        "continuity_stderr",
        "momentum_counts",
        "momentum_total",
        "momentum_average",
        "momentum_avg_per_repeat",
        "momentum_stderr",
        "bcs_counts",
        "bcs_total",
        "bcs_average",
        "bcs_avg_per_repeat",
        "bcs_stderr",
        "next_counts",
        "next_total",
        "next_average",
        "next_avg_per_repeat",
        "next_stderr",
        # "filename", 
    ]
    df = pd.DataFrame(columns=cols, index=range(len(files)), dtype=np.float64)
    if files is not None:
        for index, file in enumerate(files):
            # df["filename"][index] = file
            splitpath = os.path.split(file)
            filename = os.path.splitext(splitpath[-1])[0]
            scale = 0
            repeat = 0
            if "_" in filename:
                splitname = filename.split("_")
                scale = int(splitname[0])
                repeat = int(splitname[1])
            df.loc[index, "build"] = build
            df.loc[index, "scale"] = scale
            df.loc[index, "repeat"] = repeat

            with open(file, "r") as fp:
                for line in fp:
                    line = line.strip()

                    if line.startswith("JPIGLO"):
                        df.loc[index, "jpiglo"] = int(line.split("=")[-1].replace(",", "").strip()) 
                    if line.startswith("JPJGLO"):
                        df.loc[index, "jpjglo"] = int(line.split("=")[-1].replace(",", "").strip()) 
                    if line.startswith("ua checksum"):
                        df.loc[index, "ua_checksum"] = np.float64(line.split("=")[-1].replace(",", "").strip()) 
                    if line.startswith("va checksum"):
                        df.loc[index, "va_checksum"] = np.float64(line.split("=")[-1].replace(",", "").strip()) 

                    if line.startswith("Time-stepping"):
                        splitline = line.split()
                        df.loc[index, "time_stepping_counts"] = np.float64(splitline[1])
                        df.loc[index, "time_stepping_total"] = np.float64(splitline[2])
                        df.loc[index, "time_stepping_average"] = np.float64(splitline[3])
                        df.loc[index, "time_stepping_avg_per_repeat"] = np.float64(splitline[4])
                        df.loc[index, "time_stepping_stderr"] = np.float64(splitline[5])

                    if line.startswith("Continuity"):
                        splitline = line.split()
                        df.loc[index, "continuity_counts"] = np.float64(splitline[1])
                        df.loc[index, "continuity_total"] = np.float64(splitline[2])
                        df.loc[index, "continuity_average"] = np.float64(splitline[3])
                        df.loc[index, "continuity_avg_per_repeat"] = np.float64(splitline[4])
                        df.loc[index, "continuity_stderr"] = np.float64(splitline[5])

                    if line.startswith("Momentum"):
                        splitline = line.split()
                        df.loc[index, "momentum_counts"] = np.float64(splitline[1])
                        df.loc[index, "momentum_total"] = np.float64(splitline[2])
                        df.loc[index, "momentum_average"] = np.float64(splitline[3])
                        df.loc[index, "momentum_avg_per_repeat"] = np.float64(splitline[4])
                        df.loc[index, "momentum_stderr"] = np.float64(splitline[5])

                    if line.startswith("BCs"):
                        splitline = line.split()
                        df.loc[index, "bcs_counts"] = np.float64(splitline[1])
                        df.loc[index, "bcs_total"] = np.float64(splitline[2])
                        df.loc[index, "bcs_average"] = np.float64(splitline[3])
                        df.loc[index, "bcs_avg_per_repeat"] = np.float64(splitline[4])
                        df.loc[index, "bcs_stderr"] = np.float64(splitline[5])

                    if line.startswith("Next"):
                        splitline = line.split()
                        df.loc[index, "next_counts"] = np.float64(splitline[1])
                        df.loc[index, "next_total"] = np.float64(splitline[2])
                        df.loc[index, "next_average"] = np.float64(splitline[3])
                        df.loc[index, "next_avg_per_repeat"] = np.float64(splitline[4])
                        df.loc[index, "next_stderr"] = np.float64(splitline[5])

    return df

def processed_data(data):
    if data is not None and data.shape[0] > 0:
        KEY_COLS = [
            "build", 
            "scale"
        ]
        MEAN_COLS = [
            "time_stepping_total",
            "time_stepping_average",
            "continuity_total",
            "continuity_average",
            "momentum_total",
            "momentum_average",
            "bcs_total",
            "bcs_average",
            "next_total",
            "next_average",
            "ua_checksum",
            "va_checksum",
        ]
        processed = data.groupby(KEY_COLS)

        df_means = processed[MEAN_COLS].agg("mean")

        # print(df_means[:,])
        return df_means

    else:
        return None

def save_data(processed, output, force):
    index_enabled=True # Enable the index to include the group key.
    if output:
        with open(output, "w") as fp:
            processed.to_csv(fp, index=index_enabled)  
    else:
        processed.to_csv(sys.stdout, index=index_enabled)
    return True

def main():
    # parse command line args
    args = cli_args()
    # Validate input, building the list of files to be parsed. 
    valid, files = validate_args(args)
    if not valid:
        return False

    # Parse the files
    data = parse_files(files, args.build)

    # Process the data
    processed = processed_data(data)

    # Save the data to disk
    saved = save_data(processed, args.output, args.force)

    # Return the success of the function. True == success. 
    return saved


if __name__ == "__main__":
    # Exit with an appropriate errorcode.
    success = main()
    if success:
        exit(EXIT_SUCCESS)
    else:
        exit(EXIT_FAILURE)
