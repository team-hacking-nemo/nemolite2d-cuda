#! /usr/bin/python3

"""
Python3 script for comparing the checksums from two csv files. 

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
import tabulate

EXIT_FAILURE = 1
EXIT_SUCCESS = 0

CSV_EXT=".csv"

# Define command line arguments for this application
def cli_args():
    # Get arguments using argparse
    parser = argparse.ArgumentParser(
        description="compare extracted checksums for two files"
    )
    parser.add_argument(
        "left",
        type=str,
        help="left csv file"
    )

    parser.add_argument(
        "right",
        type=str,
        help="right csv file"
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=np.float64,
        help="epsilon value for float comparison",
        default=0.0
    )

    args = parser.parse_args()
    return args

# Validate command line arguments for pre-empative abortion.
def validate_args(args):
    # Build the list of log files to be parsed.
  
    if not os.path.isfile(args.left):
        print("Error: left file {:} does not exist.\n".format(args.left))

    if not os.path.isfile(args.right):
        print("Error: right file {:} does not exist.\n".format(args.right))

    return True

def parse_files(left, right):
    # Load each file into a data frame. 
    left_dataframe = pd.read_csv(left)
    right_dataframe = pd.read_csv(right)

    return left_dataframe, right_dataframe

def checksum_deltas(left_dataframe, right_dataframe, epsilon):
    # Calculate the differences butween the checksums
    COLS=[
        "scale",
        "left_ua_checksum", "right_ua_checksum", "delta_ua_checksum",
        "left_va_checksum", "right_va_checksum", "delta_va_checksum",
    ]

    dataframe = pd.DataFrame(columns=COLS)
    dataframe["scale"] = left_dataframe["scale"]
    dataframe["left_ua_checksum"] = left_dataframe["ua_checksum"]
    dataframe["right_ua_checksum"] = right_dataframe["ua_checksum"]
    dataframe["left_va_checksum"] = left_dataframe["va_checksum"]
    dataframe["right_va_checksum"] = right_dataframe["va_checksum"]
    dataframe["delta_ua_checksum"] = (dataframe["left_ua_checksum"] - dataframe["left_ua_checksum"]).abs() > epsilon
    dataframe["delta_va_checksum"] = (dataframe["left_va_checksum"] - dataframe["left_va_checksum"]).abs()
    dataframe["ua_correct"] = (dataframe["delta_ua_checksum"] <= epsilon).astype(int)
    dataframe["va_correct"] = (dataframe["delta_va_checksum"] <= epsilon).astype(int)

    # print(dataframe)
    # Collect incorrect rows.
    incorrect_rows = dataframe.query("ua_correct == 0 | ua_correct == 0")
    return dataframe, incorrect_rows

def print_output(deltas, incorrect_rows, args):
    # print correctness to console

    print("Comparing checksums using epsilong {:}\n".format(args.epsilon))

    incorrect_count = incorrect_rows.shape[0]

    if incorrect_count:
        incorrect_count_ua = incorrect_rows.query("ua_correct == 0").shape[0]
        incorrect_count_va = incorrect_rows.query("va_correct == 0").shape[0]
        print("{:} Incorrect rows (ua: {:}, v: {:})".format(incorrect_count, incorrect_count_ua, incorrect_count_va))
        output_cols = ["scale", "left_ua_checksum", "right_ua_checksum", "left_va_checksum", "right_va_checksum"]
        output_df  = incorrect_rows[output_cols]
        print(tabulate.tabulate(output_df.values, output_df.columns, tablefmt="pipe"))

    else: 
        print("Correct!")

    # Get the number of ua errors

    # get the number of uv errors

    return

def main():
    # parse command line args
    args = cli_args()
    # Validate input, building the list of files to be parsed. 
    valid = validate_args(args)
    if not valid:
        return False

    # Parse the files
    left_dataframe, right_dataframe = parse_files(args.left, args.right)

    dataframe, incorrect = checksum_deltas(left_dataframe, right_dataframe, args.epsilon)

    print_output(dataframe, incorrect, args)

    # Return the success of the function. True == success. 
    return True


if __name__ == "__main__":
    # Exit with an appropriate errorcode.
    success = main()
    if success:
        exit(EXIT_SUCCESS)
    else:
        exit(EXIT_FAILURE)
