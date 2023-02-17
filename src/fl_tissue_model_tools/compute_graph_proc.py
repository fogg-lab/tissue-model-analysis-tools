"""
Compute graph with DMTGraph module and pickle the resultant (verts, edges) tuple.
cmdline args:
    - img
    - threshold1
    - threshold2
"""

import argparse
import os
import pickle
import sys
import cv2
import numpy as np
from pydmtgraph.dmtgraph import DMTGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Path to image to compute graph on")
    parser.add_argument("threshold1", help="Threshold 1")
    parser.add_argument("threshold2", help="Threshold 2")
    parser.add_argument("output", help="Path to output pickle file")
    return parser.parse_args()

def main():
    args = parse_args()
    img = cv2.imread(args.img, cv2.IMREAD_ANYDEPTH)

    if img is None:
        print("Error: Could not read image")
        sys.exit(1)

    threshold1 = float(args.threshold1)
    threshold2 = float(args.threshold2)
    output = args.output
    if os.path.exists(output):
        print("Error: Output file already exists")
        sys.exit(1)

    img = img.astype(np.double)
    dmtG = DMTGraph(img)
    verts, edges = dmtG.computeGraph(threshold1, threshold2)

    with open(output, "wb") as f:
        pickle.dump((verts, edges), f)

if __name__ == "__main__":
    main()
