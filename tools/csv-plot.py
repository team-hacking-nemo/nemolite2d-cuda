#! /usr/bin/python3

"""
Python3 script for plotting (one or more) CSVs using matplotlib. 

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
import tabulate

EXIT_FAILURE = 1
EXIT_SUCCESS = 0

CSV_EXT=".csv"

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
        help="csv files to parse."
        )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="location for output file"
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="show the graph."
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="render a stacked bar chart rather than line figure."
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="speedup lineplot"
    )
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="render time per kernel rather than total."
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="log scale for y axis."
    )
    parser.add_argument(
        "--sharey",
        action="store_true",
        help="share y axis."
    )
    parser.add_argument(
        "--scale-minmax",
        type=int,
        nargs=2,
        help="Apply lower/upper bounds to the scales shown"
    )
    parser.add_argument(
        "-a",
        "--average-runtimes",
        action="store_true",
        help="Use average kernel times to avoid iteration count impact."
    )
    parser.add_argument(
        "--print-speedup",
        action="store_true",
        help="print speedup table to console"
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
            if os.path.isfile(file_or_path) and file_or_path.endswith(CSV_EXT):
                valid_files.append(file_or_path)
            elif os.path.isdir(file_or_path):
                for root, dirs, files in os.walk(file_or_path):
                    for file in files:
                        if file.endswith(CSV_EXT):
                            valid_files.append(os.path.join(root, file))
            else:
                print("Warning: Provided file {:} does not exist".format(file_or_path))
    # Remove duplicates from valid files and sort
    valid_files = list(set(valid_files))

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

def parse_files(files, scalelim):
    # Load each file into a data frame. 
    dfs = OrderedDict()
    for index, file in enumerate(files):
        df = pd.read_csv(file)


        if scalelim is not None:
            scalemin = scalelim[0]
            scalemax = scalelim[1]
            if scalemin <= scalemax:
                df = df.query("scale >= {:} and scale <= {:}".format(scalemin, scalemax))
            else:
                print("Warning, scalemin greater than scale max.")
        dfs[file] = df

    # Calculate additional values
    dfs = post_process_dataframes(dfs)

    # Combine dataframes
    combined = pd.concat(dfs.values(), sort=False)


    return dfs, combined


def print_speedup(individuals):
    print("Speedup:")

    for k in list(individuals.keys())[1:]:
        speedup_cols = []
        df = individuals[k]
        for col in df.columns:
            tot_or_av = "average" if args.average_runtimes else "total"
            if "speedup" in col and tot_or_av in col and col not in speedup_cols:
                speedup_cols.append(col)

        # Construct a new data frame with just the speedup cols and the index. 
        build = df["build"].unique()
        cols = ["scale"] + speedup_cols
        speedupdf = df[cols]

        print("Speedup for {:}".format(build[0]))
        # for index, row in speedupdf.iterrows():
            # print(index, int(row["scale"]), row["speedup_time_stepping_total"])#, row["speedup_+"])
        print(tabulate.tabulate(speedupdf.values,speedupdf.columns, tablefmt="pipe"))


    # print(speedup_cols)


def speedup_plot(data, args):
    # Get the columns 
    columns = list(data.columns)

    # Get the builds.
    if "build" not in columns:
        print("Error: build column missing!")
        return False 
    builds = list(data["build"].unique())

    # Get the Scales 
    if "scale" not in columns:
        print("Error: scale column missing!")
        return False
    scales = list(data["scale"].unique())

    # Select a column to plot

    ycols = ["speedup_time_stepping_total"]    
    if args.kernels:
        ycols = [
            "speedup_continuity_total",
            "speedup_momentum_total",
            "speedup_bcs_total",
            "speedup_next_total",
        ]
        if args.average_runtimes:
            ycols = [
            "speedup_continuity_average",
            "speedup_momentum_average",
            "speedup_bcs_average",
            "speedup_next_average",
        ]

    valid_ycols = []
    for ycol in ycols:
        if ycol not in columns:
            print("Warning, selected column {:} not present".format(ycol))
        else:
            valid_ycols.append(ycol)

    ycols = valid_ycols
    if len(ycol) == 0:    
        print("Warning, setting default ycol")
        ycols = [columns[2]]

    # Plot the data
    sns.set(style="darkgrid")

    markers = ["o", "s", "P", "^", "h", "X" ] 
    linestyles=["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(16,9))
    for bindex, build in enumerate(builds[1:]):
        palette = sns.color_palette("husl", len(ycols))
        # palette = sns.cubehelix_palette(len(ycols), start=bindex, dark=0.5, light=0.8, reverse=True)

        linestyle = linestyles[bindex % len(linestyles)]
        qdata = data.query("build == '{:}'".format(build))

        for yindex, ycol in enumerate(ycols):
            sindex = bindex * len(builds) + yindex
            marker = markers[sindex % len(markers)]
            colour = palette[yindex % len(ycols)]
            label="{:} {:}".format(build, ycol)
            ax.plot(qdata["scale"], qdata[ycol], marker=marker, linestyle=linestyle, color=colour, label=label)

    plt.legend(loc='upper left')
    plt.xlabel("scale")
    plt.ylabel("Speedup")
    plt.xscale("linear")
    ax.set_ylim(bottom=0, top=None)
    if args.logy:
        plt.yscale("log", basey=10)

    # Save to disk?
    if args.output is not None and (not os.path.exists(args.output) or args.force):
        plt.savefig(args.output, dpi=150) 
        print("Figure saved to {:}".format(args.output))

    # Show on the screen?
    if args.show == True:
        plt.show()

    return True



def line_plot(data, args):
    # Get the columns 
    columns = list(data.columns)

    # Get the builds.
    if "build" not in columns:
        print("Error: build column missing!")
        return False 
    builds = list(data["build"].unique())

    # Get the Scales 
    if "scale" not in columns:
        print("Error: scale column missing!")
        return False
    scales = list(data["scale"].unique())

    # Select a column to plot

    ycols = ["time_stepping_total"]    
    # "time_stepping_average",
    if args.kernels:
        ycols = [
            "continuity_total",
            "momentum_total",
            "bcs_total",
            "next_total",
        ]
        if args.average_runtimes:
            ycols = [
                "continuity_average",
                "momentum_average",
                "bcs_average",
                "next_average",
            ]

    valid_ycols = []
    for ycol in ycols:
        if ycol not in columns:
            print("Warning, selected column {:} not present".format(ycol))
        else:
            valid_ycols.append(ycol)

    ycols = valid_ycols
    if len(ycol) == 0:    
        print("Warning, setting default ycol")
        ycols = [columns[2]]

    # Plot the data
    sns.set(style="darkgrid")

    markers = ["o", "s", "P", "^", "h", "X" ] 
    linestyles=["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(16,9))
    for bindex, build in enumerate(builds):
        palette = sns.color_palette("husl", len(ycols))
        # palette = sns.cubehelix_palette(len(ycols), start=bindex, dark=0.5, light=0.8, reverse=True)

        linestyle = linestyles[bindex % len(linestyles)]
        qdata = data.query("build == '{:}'".format(build))

        for yindex, ycol in enumerate(ycols):
            sindex = bindex * len(builds) + yindex
            marker = markers[sindex % len(markers)]
            colour = palette[yindex % len(ycols)]
            label="{:} {:}".format(build, ycol)
            ax.plot(qdata["scale"], qdata[ycol], marker=marker, linestyle=linestyle, color=colour, label=label)

    plt.legend(loc='upper left')
    plt.xlabel("scale")
    plt.ylabel("time(seconds)")
    plt.xscale("linear")
    if args.logy:
        plt.yscale("log", basey=10)

    # Save to disk?
    if args.output is not None and (not os.path.exists(args.output) or args.force):
        plt.savefig(args.output, dpi=150) 
        print("Figure saved to {:}".format(args.output))

    # Show on the screen?
    if args.show == True:
        plt.show()

    return True


def post_process_dataframes(dataframes):
    COL_BLACKLIST = ["build", "scale"]

    # Calculate speedup relative to the first file.
    reference_df = list(dataframes.values())[0]

    # For each other individual one, calcualte speedup per column

    # process the first data frame
    for k in list(dataframes.keys())[0:1]:
        df = dataframes[k]
        dfcopy = df.copy()
        for col in df.columns:
            if col not in COL_BLACKLIST:
                speedupcol="speedup_{:}".format(col)
                dfcopy[speedupcol] = 1.0
            dataframes[k] = dfcopy

    for k in list(dataframes.keys())[1:]:
        df = dataframes[k]
        dfcopy = df.copy()
        for col in df.columns:
            if col not in COL_BLACKLIST:
                speedupcol="speedup_{:}".format(col)
                dfcopy[speedupcol] = reference_df[col] / df[col]
        dataframes[k] = dfcopy



    return dataframes


def plot(individuals, combined, args):
    if args.bar:
        return stackedbar_plot(combined, args)
    elif args.speedup:
        return speedup_plot(combined, args)
    else:
        return line_plot(combined, args)

def stackedbar_plot(data, args):
    # Get the columns 
    columns = list(data.columns)

    # Get the builds.
    if "build" not in columns:
        print("Error: build column missing!")
        return False 
    builds = list(data["build"].unique())

    # Get the Scales 
    if "scale" not in columns:
        print("Error: scale column missing!")
        return False
    scales = sorted(list(data["scale"].unique()))

    # Select a column to plot
    

    ycols = ["time_stepping_total"]    
    # "time_stepping_average",
    if args.kernels:
        ycols = [
            "continuity_total",
            "momentum_total",
            "bcs_total",
            "next_total",
        ]
        if args.average_runtimes:
            ycols = [
                "continuity_average",
                "momentum_average",
                "bcs_average",
                "next_average",
            ]

    valid_ycols = []
    for ycol in ycols:
        if ycol not in columns:
            print("Warning, selected column {:} not present".format(ycol))
        else:
            valid_ycols.append(ycol)

    ycols = valid_ycols
    if len(ycol) == 0:    
        print("Warning, setting default ycol")
        ycols = [columns[2]]

    # Plot the data
    sns.set(style="darkgrid")

    markers = ["o", "s", "P", "^", "6", "X" ] 
    linestyles=["-", "--", ":"]

    xlocs = np.arange(len(scales))
    barwidth = 0.5

    fig, axes = plt.subplots(ncols=len(builds), figsize=(16,9), sharey=args.sharey)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    maximums = []
    for bindex, build in enumerate(builds):
        ax = axes.flat[bindex] if len(builds) > 1 else axes
        palette = sns.color_palette("husl", len(ycols))
        # palette = sns.cubehelix_palette(len(ycols), start=bindex, dark=0.5, light=0.8, reverse=True)

        linestyle = linestyles[bindex % len(linestyles)]
        qdata = data.query("build == '{:}'".format(build))
        # print(qdata["scale"])

        ymaxes = []
        ax.set_title(build)
        prev_data = np.zeros(len(xlocs))
        for yindex, ycol in enumerate(ycols):
            coldata = list(qdata[ycol])
            sindex = bindex * len(builds) + yindex
            marker = markers[sindex % len(markers)]
            colour = palette[yindex % len(ycols)]
            label="{:} {:}".format(build, ycol)
            ax.bar(xlocs,coldata, color=colour, label=label, bottom=prev_data)
            prev_data = prev_data + coldata
            ymaxes.append(max(coldata))
        ax.legend(loc='upper left')
        ax.set_xlabel("scale")
        ax.set_ylabel("time(seconds)")
        ax.set_xticks(xlocs)
        ax.set_xticklabels(scales)
        maximums.append(sum(ymaxes))
    
    # Don't use logy for stacked bar, makes it look like all of the runtime is in the lowest block.
    # if args.logy:
        # plt.yscale("log", basey=10)

    # Save to disk?
    if args.output is not None and (not os.path.exists(args.output) or args.force):
        plt.savefig(args.output, dpi=150) 
        print("Figure saved to {:}".format(args.output))

    # Show on the screen?
    if args.show == True:
        plt.show()

    return True

def main():
    # parse command line args
    args = cli_args()
    # Validate input, building the list of files to be parsed. 
    valid, files = validate_args(args)
    if not valid:
        return False

    # Parse the files
    individuals, combined, = parse_files(files, args.scale_minmax)

    # Plot the data
    plotted = plot(individuals, combined, args)

    # Print the speedup
    if args.print_speedup:
        print_speedup(individuals)

    # Return the success of the function. True == success. 
    return plotted


if __name__ == "__main__":
    # Exit with an appropriate errorcode.
    success = main()
    if success:
        exit(EXIT_SUCCESS)
    else:
        exit(EXIT_FAILURE)
