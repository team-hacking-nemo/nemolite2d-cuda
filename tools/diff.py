#! /usr/bin/python3

"""
Python3 script for comparing tow .dat files. 
@todo - compare directory of files? for quicker vlaidation of full runs? (i.e. find .dat files in directories and compare if the name is the same?)
"""

import os
import argparse
import numpy as np
import pandas as pd

# Constants
COL_X = "x"
COL_Y = "y"
COL_DEPTH = "depth"
COL_SSH = "ssh"
COL_UV = "u_velocity"
COL_VV = "v_velocity"

COLUMNS=[COL_X, COL_Y, COL_DEPTH, COL_SSH, COL_UV, COL_VV]

EXIT_FAILURE = 1
EXIT_SUCCESS = 0

# Define command line arguments for this application
def cli_args():
    # Get arguments using argparse
    parser = argparse.ArgumentParser(
        description="Generate a dot file from an xml model file."
        )
    parser.add_argument(
        "reference",
        type=str,
        help="Path to refrence datfile(s)"
        )
    parser.add_argument(
        "sample",
        type=str,
        help="Path to sample datfile(s)"
        )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="location for output error data."
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
    # Check reference file exist
    if not os.path.isfile(args.reference):
        print("Error: {:} is not a file".format(args.reference))
        return False
    # Check new file exists
    if not os.path.isfile(args.sample):
        print("Error: {:} is not a file".format(args.sample))
        return False

    # Check use of --output and --force
    if args.output is not None:
        # @todo - potential disk race by checking this upfront. Should use try catch later too.
        if os.path.exists(args.output):
            if os.path.isdir(args.output):
                print("Error: {:} is a directory".format(args.sample))
                return False
            elif os.path.isfile(args.output):
                if not args.force:
                    print("Error: Output file {:} already exists. Please use -f/--force")
                    return False
    return True

# Parse an output .dat file into a pandas dataframe. 
def parse_file_df(filepath, columns=None):
    df = pd.read_csv(filepath, header=None, names=columns, delim_whitespace=True)
    # print(df)
    for c in df.columns:
        # print(df.dtypes[c])
        df[c] = df[c].astype(np.float64)
    return df

# Calculate the difference between two dataframes
def delta_df(a, b):
    delta = a.subtract(b).abs()
    return delta

# Find rows with non-zero values in relevant columns.
def nonzero_rows(df):
    q = "{:} != 0 | {:} != 0 | {:} != 0".format(COL_SSH, COL_UV, COL_VV)
    nnz = df.query(q)
    return nnz

# Accumulate values in columns for erorr summary
def aggregate_columns(df):
    sum_df = df.sum()
    return sum_df

# Print a summary to stdout
def summary(df, nnz, output):
    error_count = nnz.shape[0]
    if error_count == 0:
        print("Success")
        return True
    else:
        print("Failure. {:} incorrect rows.".format(error_count))
        print("Summary:\n\t{:} = {:}\n\t{:} = {:}\n\t{:} = {:}".format(
            COL_SSH, df[COL_SSH],
            COL_UV, df[COL_UV],
            COL_VV, df[COL_VV]
        ))
        if output:
            print("See '{:}' for details.".format(output))
        return False

# Write incorrect rows to disk if output is valid. 
def store_error_rows(output, df_reference, df_sample, nnz):

    COL_SSH_A = "{:}_a".format(COL_SSH)
    COL_SSH_B = "{:}_b".format(COL_SSH)
    COL_UV_A = "{:}_a".format(COL_UV)
    COL_UV_B = "{:}_b".format(COL_UV)
    COL_VV_A = "{:}_a".format(COL_VV)
    COL_VV_B = "{:}_b".format(COL_VV)

    cols = [
        COL_X, 
        COL_Y, 
        COL_DEPTH, 
        COL_SSH_A, 
        COL_SSH_B, 
        COL_UV_A, 
        COL_UV_B, 
        COL_VV_A, 
        COL_VV_B
    ]

    error_count = nnz.shape[0]
    if error_count != 0:
        # @todo - this is probably super inefficient
        df = pd.DataFrame(columns=cols, index=list(nnz.index) )
        for index, row in nnz.iterrows():
            df[COL_X][index] =  df_reference[COL_X][index]
            df[COL_Y][index] =  df_reference[COL_Y][index]
            df[COL_DEPTH][index] =  df_reference[COL_DEPTH][index]
            df[COL_SSH_A][index] =  df_reference[COL_SSH][index]
            df[COL_SSH_B][index] =  df_sample[COL_SSH][index]
            df[COL_UV_A][index] =  df_reference[COL_UV][index]
            df[COL_UV_B][index] =  df_sample[COL_UV][index]
            df[COL_VV_A][index] =  df_reference[COL_VV][index]
            df[COL_VV_B][index] =  df_sample[COL_VV][index]
        
        # Write out to disk
        with open(output, "w") as fp:
            df.to_csv(fp)

def main():
    # parse command line args
    args = cli_args()
    valid = validate_args(args)
    if not valid:
        return False

    df_reference = parse_file_df(args.reference, COLUMNS)
    df_sample = parse_file_df(args.sample, COLUMNS)
    
    #get the difference between the files. 
    df_delta = delta_df(df_reference, df_sample)
    
    df_sum = aggregate_columns(df_delta)
    nnz_rows = nonzero_rows(df_delta)

    if args.output is not None:
        store_error_rows(args.output, df_reference, df_sample, nnz_rows)

    status = summary(df_sum, nnz_rows, args.output)

    # Return the success of the function. True == success. 
    return status


if __name__ == "__main__":
    # Exit with an appropriate errorcode.
    success = main()
    if success:
        exit(EXIT_SUCCESS)
    else:
        exit(EXIT_FAILURE)
