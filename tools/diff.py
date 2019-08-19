#! /usr/bin/python3

import argparse
import pandas as pd
import os
import numpy as np


COL_X = "x"
COL_Y = "y"
COL_DEPTH = "depth"
COL_SSH = "ssh"
COL_UV = "u_velocity"
COL_VV = "v_velocity"

COLUMNS=[COL_X, COL_Y, COL_DEPTH, COL_SSH, COL_UV, COL_VV]

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

    args = parser.parse_args()
    return args



def validate_args(args):
    # Ensure validation 
    # @todo support directory of files
    if not os.path.isfile(args.reference):
        print("Error: {:} is not a file".format(args.reference))
        return False
    # @todo support directory of files
    if not os.path.isfile(args.sample):
        print("Error: {:} is not a file".format(args.sample))
        return False

    return True

def parse_file_df(filepath, columns=None):
    df = pd.read_csv(filepath, header=None, names=columns, delim_whitespace=True)
    # print(df)
    for c in df.columns:
        # print(df.dtypes[c])
        df[c] = df[c].astype(np.float64)
    return df

def delta_df(a, b):
    # Find the difference between two dataframes
    delta = a.subtract(b).abs()
    return delta

def nonzer_rows(df):
    q = "{:} != 0 | {:} != 0 | {:} != 0".format(COL_SSH, COL_UV, COL_VV)
    nnz = df.query(q)
    return nnz


def aggregate_columns(df):
    sum_df = df.sum()
    return sum_df

def summary(df, nnz):
    error_count = nnz.shape[0]
    if error_count == 0:
        print("Match")
        return True
    else:
        print("Errors in {:} rows".format(error_count))
        print("Aggregate Errors:\n\t{:} = {:}\n\t{:} = {:}\n\t{:} = {:}\n".format(
            COL_SSH, df[COL_SSH],
            COL_UV, df[COL_UV],
            COL_VV, df[COL_VV]
        ))
        return False

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
        
        # @todo - validate input file doesnt exist etc. 
        with open(output, "w") as fp:
            df.to_csv(output)
        print("{:} rows written to {:}".format(error_count, output))


def main():
    # parse command line args
    args = cli_args()
    valid = validate_args(args)
    if not valid:
        print("abort")
        return EXIT_ERROR

    df_reference = parse_file_df(args.reference, COLUMNS)
    df_sample = parse_file_df(args.sample, COLUMNS)
    
    #get the difference between the files. 
    df_delta = delta_df(df_reference, df_sample)
    
    df_sum = aggregate_columns(df_delta)
    nnz_rows = nonzer_rows(df_delta)

    if args.output is not None:
        store_error_rows(args.output, df_reference, df_sample, nnz_rows)

    status = summary(df_sum, nnz_rows)

    return status


if __name__ == "__main__":
    exit(main())
