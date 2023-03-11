import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tabulate import tabulate

from evaluate import evaluate
from jwnmf import jwnmf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset Path; Must Contain A.csv and S.csv.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-k",
        "--latent_dim",
        help="Dimension of latent factor",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-l",
        "--lambda_coefficient",
        help="Lambda",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-i",
        "--max_iterations",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output Directory",
        type=Path,
    )
    return parser.parse_args()


def read_matrix(path):
    return pd.read_csv(path, header=None, index_col=False).to_numpy()


def write_matrix(matrix, path):
    pd.DataFrame(matrix).to_csv(path, header=None, index=False)


def plot_loss(losses, path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.savefig(path)


def main():
    args = parse_args()

    dataset = args.dataset

    A_Ob = read_matrix(dataset / "A_Ob.csv")
    S_Ob = read_matrix(dataset / "S_Ob.csv")

    S_height, S_width = S_Ob.shape
    n, m = A_Ob.shape
    assert S_height == S_width == n

    k = args.latent_dim
    lambda_coefficient = args.lambda_coefficient
    max_iters = args.max_iterations
    epsilon = args.epsilon

    output_dir = args.output_dir if args.output_dir else dataset / "JWNMF_results"
    os.mkdir(output_dir)

    table_data = [
        ("dataset", dataset),
        ("output", output_dir),
        ("num users", n),
        ("num attributes", m),
        ("latent dim", k),
        ("lambda", lambda_coefficient),
        ("max iterations", max_iters),
        ("epsilon", epsilon),
    ]

    table = tabulate(
        table_data,
        headers=["Parameter", "Value"],
        tablefmt="orgtbl",
    )

    print(table)

    V, U, W, losses, i = jwnmf.train(
        S_Ob, A_Ob, m, n, k, lambda_coefficient, max_iters, epsilon
    )
    print("Terminated")

    table_data.append(("iterations", i))
    table = tabulate(
        table_data,
        headers=["Parameter", "Value"],
        tablefmt="orgtbl",
    )
    print(table)

    with open(output_dir / "info.txt", "w") as f:
        f.write(table)

    write_matrix(V, output_dir / "V.csv")
    write_matrix(U, output_dir / "U.csv")
    write_matrix(W, output_dir / "W.csv")

    S_Pr = V @ V.T
    A_Pr = V @ U.T

    S_Pr[S_Pr > 1] = 1
    S_Pr[S_Pr < 0] = 0

    A_Pr[A_Pr > 1] = 1
    A_Pr[A_Pr < 0] = 0

    write_matrix(S_Pr, output_dir / "S_Pr.csv")
    write_matrix(A_Pr, output_dir / "A_Pr.csv")

    plot_loss(losses, output_dir / "losses.png")

    try:
        A_Gt = read_matrix(dataset / "A_GT.csv")
        S_Gt = read_matrix(dataset / "S_GT.csv")
    except:
        print("Ground Truth Not Found")
        return

    assert A_Gt.shape == A_Ob.shape
    assert S_Gt.shape == S_Ob.shape

    predicted_for_eval_S = evaluate.predicted_scores_for_eval(S_Pr, S_Ob)
    ground_truth_for_eval_S = evaluate.ground_truth_for_eval(S_Gt, S_Ob)

    with open(output_dir / "unobserved_inferred.txt", "w") as f:
        f.write(",".join(map(str, list(predicted_for_eval_S))))

    with open(output_dir / "unobserved_gt.txt", "w") as f:
        f.write(",".join(map(str, list(ground_truth_for_eval_S))))

    metrics = evaluate.evaluate(ground_truth_for_eval_S, predicted_for_eval_S)
    table = tabulate(
        {(k, v) for k, v in metrics.items()},
        headers=["Parameter", "Value"],
        tablefmt="orgtbl",
    )
    print(table)
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(table)

    predicted_for_eval_A = evaluate.predicted_scores_for_eval(A_Pr, A_Ob)
    ground_truth_for_eval_A = evaluate.ground_truth_for_eval(A_Gt, A_Ob)

    metrics = evaluate.evaluate_cascade(ground_truth_for_eval_A, predicted_for_eval_A)
    table = tabulate(
        {(k, v) for k, v in metrics.items()},
        headers=["Parameter", "Value"],
        tablefmt="orgtbl",
    )
    print(table)
    with open(output_dir / "metrics_A.txt", "a") as f:
        f.write(table)


if __name__ == "__main__":
    main()
