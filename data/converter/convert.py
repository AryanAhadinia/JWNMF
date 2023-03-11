import argparse

import numpy as np
import pandas as pd


def main():
    # get path arg
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset name")
    args = parser.parse_args()

    dataset = args.dataset

    A_GT = pd.read_csv(
        f"{dataset}/MatrixCASL.txt", header=None, index_col=False
    ).to_numpy()
    A_GT = np.where(A_GT != 0, 1, 0)
    pd.DataFrame(A_GT).to_csv(f"{dataset}/A_GT.csv", header=False, index=False)

    A_Ob = pd.read_csv(
        f"{dataset}/MatrixC.txt", header=None, index_col=False
    ).to_numpy()
    A_Ob = np.where(A_Ob != 0, 1, 0)
    pd.DataFrame(A_Ob).to_csv(f"{dataset}/A_Ob.csv", header=False, index=False)

    S_GT = pd.read_csv(f"{dataset}/MatrixGASL.txt", header=None, index_col=False)
    S_GT = S_GT.to_numpy().astype(int)
    pd.DataFrame(S_GT).to_csv(f"{dataset}/S_GT.csv", header=False, index=False)

    S_Ob = pd.read_csv(f"{dataset}/MatrixG.txt", header=None, index_col=False)
    S_Ob = S_Ob.to_numpy().astype(int)
    pd.DataFrame(S_Ob).to_csv(f"{dataset}/S_Ob.csv", header=False, index=False)


if __name__ == "__main__":
    main()
