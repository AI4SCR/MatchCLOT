import os
import anndata as ad
import torch
import numpy
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def evaluate(prediction_test, sol_test, scores_path="scores/scores.txt"):
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, 'a+') as f:
        if type(prediction_test.X) != numpy.ndarray:
            X = prediction_test.X.toarray()
        else:
            X = prediction_test.X
        X = torch.tensor(X)

        Xsol = torch.tensor(sol_test.X.toarray())
        Xsol.argmax(1)
        # Order the columns of the prediction matrix so that the perfect prediction is the identity matrix
        X = X[:, Xsol.argmax(1)]

        labels = torch.arange(X.shape[0])
        forward_accuracy = (torch.argmax(X, dim=1) == labels).float().mean().item()
        backward_accuracy = (
            (torch.argmax(X, dim=0) == labels).float().mean().item()
        )
        avg_accuracy = 0.5 * (forward_accuracy + backward_accuracy)
        print("top1 forward acc:", forward_accuracy)
        print("top1 forward acc:", forward_accuracy, file=f)
        print("top1 backward acc:", backward_accuracy)
        print("top1 backward acc:", backward_accuracy, file=f)
        print("top1 avg acc:", avg_accuracy)
        print("top1 avg acc:", avg_accuracy, file=f)

        _, top_indexes_forward = X.topk(5, dim=1)
        _, top_indexes_backward = X.topk(5, dim=0)
        l_forward = labels.expand(5, X.shape[0]).T
        l_backward = l_forward.T
        top5_forward_accuracy = (
            torch.any(top_indexes_forward == l_forward, 1).float().mean().item()
        )
        top5_backward_accuracy = (
            torch.any(top_indexes_backward == l_backward, 0).float().mean().item()
        )
        top5_avg_accuracy = 0.5 * (top5_forward_accuracy + top5_backward_accuracy)

        print("top5 forward acc:", top5_forward_accuracy)
        print("top5 forward acc:", top5_forward_accuracy, file=f)
        print("top5 backward acc:", top5_backward_accuracy)
        print("top5 backward acc:", top5_backward_accuracy, file=f)
        print("top5 avg acc:", top5_avg_accuracy)
        print("top5 avg acc:", top5_avg_accuracy, file=f)

        logits_row_sums = X.clip(min=0).sum(dim=1)
        top1_competition_metric = X.clip(min=0).diagonal().div(logits_row_sums).mean().item()
        print("top1 competition metric:", top1_competition_metric)
        print("top1 competition metric:", top1_competition_metric, file=f)

        # For soft predictions, the competition score can be made equal to the forward accuracy (or backward accuracy) by
        # putting 1 at the max of each row (or each column) and 0 elsewhere
        mx = torch.max(X, dim=1, keepdim=True).values
        hard_X = (mx == X).float()
        logits_row_sums = hard_X.clip(min=0).sum(dim=1)
        top1_competition_metric = hard_X.clip(min=0).diagonal().div(logits_row_sums).mean().item()
        print("top1 competition metric for soft predictions:", top1_competition_metric)
        print("top1 competition metric for soft predictions:", top1_competition_metric, file=f)

        # FOSCTTM
        foscttm = (X > torch.diag(X)).float().mean().item()
        foscttm_x = (X >= torch.diag(X)).float().mean(axis=1).mean().item()
        foscttm_y = (X >= torch.diag(X)).float().mean(axis=0).mean().item()
        print("foscttm:", foscttm, "foscttm_x:", foscttm_x, "foscttm_y:", foscttm_y)
        print("foscttm:", foscttm, "foscttm_x:", foscttm_x, "foscttm_y:", foscttm_y, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='TASK')
    parser_GEX2ADT = subparsers.add_parser('GEX2ADT', help='GEX2ADT model')
    parser_GEX2ATAC = subparsers.add_parser('GEX2ATAC', help='GEX2ATAC model')
    parser.add_argument("--prediction_path", "-p", type=str, default="GEX2ADT.h5ad")
    parser.add_argument("--dataset_path", "-d", type=str, default="datasets")
    args, unknown_args = parser.parse_known_args()

    if args.TASK == 'GEX2ADT':
        test_path = os.path.join(args.dataset_path, "openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna"
                                                    ".censor_dataset.output_")
    elif args.TASK == 'GEX2ATAC':
        test_path = os.path.join(args.dataset_path, "openproblems_bmmc_multiome_phase2_rna"
                                                    "/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_")
    else:
        raise ValueError('Unknown task: ' + args.TASK)

    par = {
        "input_test_prediction": args.prediction_path,
        "input_test_sol": f"{test_path}test_sol.h5ad",
    }

    prediction_test = ad.read_h5ad(par["input_test_prediction"])
    sol_test = ad.read_h5ad(par["input_test_sol"])

    scores_path = os.path.join("scores", args.TASK + ".txt")
    evaluate(prediction_test, sol_test, scores_path)
