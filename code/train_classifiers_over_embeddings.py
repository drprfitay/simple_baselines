import argparse
import pandas as pd
import numpy as np
import os
import re
import torch
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, RidgeClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from utils_for_analysis import *

# Use xgboost from scikit-learn interface (since 1.3.0: sklearn.ensemble)
try:
    from sklearn.ensemble import HistGradientBoostingClassifier as SkXGBClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor as SkXGBRegressor
    sklearn_xgb_available = True
except ImportError:
    sklearn_xgb_available = False


def load_tensor(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def mutation_orders(embedding_dir):
    orders = []
    pattern = re.compile(r"^embeddings_of_nmut_(\d+)\.pt$")
    for name in os.listdir(embedding_dir):
        match = pattern.match(name)
        if match:
            orders.append(int(match.group(1)))
    return sorted(orders)


def load_mutation_chunk(embedding_dir, nmut):
    paths = {
        "labels": os.path.join(embedding_dir, f"y_values_of_nmut_{nmut}.pt"),
        "indices": os.path.join(embedding_dir, f"indices_of_nmut_{nmut}.pt"),
        "embeddings": os.path.join(embedding_dir, f"embeddings_of_nmut_{nmut}.pt"),
    }
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Incomplete nmut={nmut} embedding chunk; missing {missing}")

    labels = load_tensor(paths["labels"]).view(-1)
    indices = load_tensor(paths["indices"]).long().view(-1)
    embeddings = load_tensor(paths["embeddings"])
    if embeddings.shape[0] != indices.numel() or indices.numel() != labels.numel():
        raise ValueError(
            f"Mismatched nmut={nmut} rows: embeddings={embeddings.shape[0]}, "
            f"indices={indices.numel()}, labels={labels.numel()}"
        )
    return labels, indices, embeddings


def embedding_features(embeddings, mean_embeddings):
    if mean_embeddings:
        return embeddings.mean(dim=1)
    return embeddings.reshape(embeddings.shape[0], -1)


def training_normalization(features):
    train_mean = features.mean(dim=0, keepdim=True)
    train_std = features.std(dim=0, keepdim=True)
    train_std = torch.where(
        torch.isfinite(train_std) & (train_std > 0),
        train_std,
        torch.ones_like(train_std),
    )
    normalized = (features - train_mean) / train_std
    if not torch.isfinite(normalized).all():
        raise ValueError("Training embeddings contain non-finite normalized values")
    return normalized, train_mean, train_std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', nargs='+', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default="gfp", help='Dataset name')
    parser.add_argument('--delta_embeddings', action='store_true', default=False, help='Flag to overwrite existing results file')
    parser.add_argument('--mean_embeddings', action='store_true', default=False, help='Flag to use mean embeddings')
    parser.add_argument('--regression', action='store_true', default=False, help='Flag to use regression instead of classification')
    parser.add_argument('--external_labels_column', type=str, default=None,
                        help='If set, use the specified column from the original dataset as labels')
    parser.add_argument('--hl', type=int, nargs='+', default=[100], 
                        help='List of hidden layer sizes for the MLP (e.g. --hidden_layers 100 50)')
    parser.add_argument('--op', type=str, default='adam', choices=['adam', 'lbfgs'],
                        help='Optimizer to use in training the classifier (adam or sgd)')
    parser.add_argument('--niters', type=int, default=50,
                        help='Number of iterations (epochs) for training the classifier')
    parser.add_argument('--n_start', type=int, default=2,
                        help='Start value for the number of training mutations (inclusive)')
    parser.add_argument('--n_end', type=int, default=3,
                        help='End value for the number of training mutations (exclusive)')
    parser.add_argument("--pca", type=int, default=None, help="Number of PCA components to use")
    parser.add_argument("--min_n", type=int, default=1, help="Minimum number of training mutations")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional result directory. Defaults to the historical dataset result path.")
    parser.add_argument(
        '--classifier_model', 
        type=str, 
        default='mlp', 
        choices=['mlp', 'ridgeregression', 'logisticregression', 'xgboost', 'lasso'],
        help="Choose classifier/regressor: mlp, ridgeregression, logisticregression, xgboost, lasso"
    )

    args = parser.parse_args()
    if args.n_start >= args.n_end:
        raise ValueError("--n_start must be smaller than the exclusive --n_end")
    if args.min_n < 0:
        raise ValueError("--min_n must be non-negative")

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embedding_root = os.path.join(base_path, "data", args.dataset_name, "embeddings")

    external_labels = None

    if args.external_labels_column is not None:
        csv_path = "%s/data/%s/%s.csv" % (base_path, args.dataset_name, args.dataset_name)

        df = pd.read_csv(csv_path)
        external_labels = df[args.external_labels_column].values

        if not args.regression:
            external_labels = (external_labels > np.median(external_labels)).astype(int)
            external_labels = 1 - external_labels # evaluate_classifier in this implementation expects inverted labels
            
        print(
            f"Using external labels from {csv_path}:{args.external_labels_column} "
            f"(n={len(external_labels)}, min={np.min(external_labels):.4g}, max={np.max(external_labels):.4g})"
        )

    for model_name in args.model_name:
        print(f"Training classifiers over {model_name} embeddings")

        classifier_embeddings_path = os.path.join(embedding_root, model_name)
        if not os.path.isdir(classifier_embeddings_path):
            available = sorted(
                name for name in os.listdir(embedding_root)
                if os.path.isdir(os.path.join(embedding_root, name))
            ) if os.path.isdir(embedding_root) else []
            raise FileNotFoundError(
                f"Embedding directory does not exist: {classifier_embeddings_path}. "
                f"Available models for {args.dataset_name}: {available}"
            )
        print(f"Classifier embeddings path: {classifier_embeddings_path}")

        available_orders = mutation_orders(classifier_embeddings_path)
        if not available_orders:
            raise FileNotFoundError(
                f"No embeddings_of_nmut_*.pt files found in {classifier_embeddings_path}"
            )
        print(f"Available mutation orders: {available_orders}")

        if args.delta_embeddings:
            if 0 not in available_orders:
                raise ValueError("--delta_embeddings requires embeddings_of_nmut_0.pt")
            wt_embedding = load_tensor(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_0.pt"))


        for n_train in range(args.n_start, args.n_end):
            train_data = {}
            n_train_data = n_train
            train_orders = list(range(args.min_n, n_train_data + 1))
            missing_train_orders = [order for order in train_orders if order not in available_orders]
            if missing_train_orders:
                raise ValueError(
                    f"Missing training mutation orders {missing_train_orders} in {classifier_embeddings_path}"
                )
            n_test_data = [order for order in available_orders if order > n_train_data]
            if not n_test_data:
                raise ValueError(
                    f"No held-out mutation orders above n_train={n_train_data}; "
                    f"available orders are {available_orders}"
                )

            for i in train_orders:
                labels, indices, embeddings = load_mutation_chunk(classifier_embeddings_path, i)

                if external_labels is not None:
                    labels = torch.from_numpy(external_labels[indices.numpy()].reshape(-1))
                
                if args.delta_embeddings:
                    print(f"Subtracting WT embedding from {model_name} embeddings")
                    embeddings = embeddings - wt_embedding

                train_data["nmuts_%d" % i] = {
                    "labels": labels,
                    "indices": indices,
                    "embeddings": embeddings
                }
                
            indices_all = []
            labels_all = []
            embeddings_all = []

            for k, v in train_data.items():
                indices_all.append(v["indices"])
                labels_all.append(v["labels"])
                embeddings_all.append(v["embeddings"])

            indices_all = torch.cat(indices_all)
            embeddings_all = torch.cat(embeddings_all)
            labels_all = torch.cat(labels_all)

            mlp_base_parameters = {
                "activation" : 'relu',           
                "solver" : args.op, 
                "batch_size": 64,   
                "alpha" : 1,                
                "learning_rate_init" : 1e-4,    
                "max_iter" :args.niters,
                "random_state" : 4321,                
                "n_iter_no_change" : 10,         
                "verbose": False
            }

            flat_embeddings = embedding_features(embeddings_all, args.mean_embeddings)
            normalized_embeddings, train_mean, train_std = training_normalization(flat_embeddings)
            train_features = normalized_embeddings.numpy()

            if args.pca is not None:
                train_pca = PCA(n_components=args.pca)
                train_features = train_pca.fit_transform(train_features)

            print("Fitting model (%s) over %dx%d embeddings to %d labels [%d mutations]" % 
                        (args.classifier_model,
                         train_features.shape[0],
                         train_features.shape[1],
                         len((labels_all)),
                         n_train_data))

            if isinstance(labels_all, torch.Tensor):                
                print(labels_all.shape)
                print(type(labels_all))
            else:
                print(labels_all.shape)
                print(type(labels_all))

            # Select classifier/regressor according to flag
            model = None
            if args.classifier_model == 'mlp':
                if args.regression:
                    model = MLPRegressor(hidden_layer_sizes=args.hl, **mlp_base_parameters)
                    model.fit(train_features, labels_all.numpy().astype(float))
                else:
                    model = MLPClassifier(hidden_layer_sizes=args.hl, **mlp_base_parameters)
                    model.fit(train_features, labels_all)
            elif args.classifier_model == 'ridgeregression':
                if args.regression:
                    model = Ridge()
                    model.fit(train_features, labels_all.numpy().astype(float))
                else:
                    model = RidgeClassifier(alpha=1.0, solver="auto", max_iter=args.niters)
                    model.fit(train_features, labels_all.numpy())
            elif args.classifier_model == 'lasso':
                if not args.regression:
                    raise ValueError("Lasso is only supported for regression tasks. Use --regression.")
                model = Lasso()
                model.fit(train_features, labels_all.numpy().astype(float))
            elif args.classifier_model == 'logisticregression':
                if args.regression:
                    raise ValueError("LogisticRegression is only supported for classification tasks. Do not use --regression.")
                model = LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=args.niters,
                    n_jobs=-1
                )
                model.fit(train_features, labels_all.numpy())
            elif args.classifier_model == 'xgboost':
                if not sklearn_xgb_available:
                    raise ImportError("scikit-learn's HistGradientBoosting (recommended xgboost-alternative) is not available. Please ensure scikit-learn >=0.21 is installed.")
                if args.regression:
                    model = GradientBoostingRegressor(n_estimators=args.niters)
                    model.fit(train_features, labels_all.numpy().astype(float))
                else:
                    model = GradientBoostingClassifier(n_estimators=args.niters)
                    model.fit(train_features, labels_all.numpy())
            else:
                raise ValueError("Unknown classifier_model selected: %s" % args.classifier_model)

            test_data = []
            for i in n_test_data:
                labels, indices, embeddings = load_mutation_chunk(classifier_embeddings_path, i)

                if external_labels is not None:
                    labels = torch.from_numpy(external_labels[indices.numpy()].reshape(-1))

                if len(labels) == 0:
                    print(f"Skipping test n_muts={i}, no samples.")
                    continue

                if args.delta_embeddings:
                    embeddings = embeddings - wt_embedding

                flat_embeddings = embedding_features(embeddings, args.mean_embeddings)
                normalized_embeddings = (flat_embeddings - train_mean) / train_std
                if not torch.isfinite(normalized_embeddings).all():
                    raise ValueError(f"Test nmut={i} contains non-finite normalized values")
                test_features = normalized_embeddings.numpy()

                if args.pca is not None:
                    test_features = train_pca.transform(test_features)

                # Predict and evaluate
                if args.regression:
                    preds = model.predict(test_features)
                    preds_np = np.asarray(preds)
                    if not np.isfinite(preds_np).all():
                        raise ValueError(f"Model produced non-finite predictions for nmut={i}")
                    correlation = spearmanr(preds, labels.numpy())
                    evaluation = {
                        "correlation": correlation[0],
                        "p_value": correlation[1],
                    }
                else:
                    # For classifiers, always use probabilities and 0/1 threshold at 0.5, if possible
                    try:
                        predictions_proba = model.predict_proba(test_features)
                        predictions = (predictions_proba[:, 1] > 0.5).astype(int)
                        evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, labels.numpy())
                    except AttributeError:
                        # Some classifiers might not have predict_proba, e.g., some linear models or HistGradientBoostingClassifier returns both predict_proba (for binary) or decision_function (for multiclass)
                        try:
                            predictions_proba = model.predict_proba(test_features)
                            predictions = (predictions_proba[:, 1] > 0.5).astype(int)
                            evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, labels.numpy())
                        except Exception:
                            predictions = model.predict(test_features)
                            evaluation = evaluate_classifier(predictions, predictions, labels.numpy())

                print("Evaluation for %d mutations: %s" % (i, evaluation))

                evaluation["test_mutations"] = i
                evaluation["train_mutations"] = n_train_data
                evaluation["classifier"] = classifier_embeddings_path
                evaluation["model_name"] = model_name
                evaluation["clf_type"] = args.classifier_model
                test_data.append(evaluation)

            folder_name = "delta_grid_searched" if args.delta_embeddings else "new_grid_searched"
            folder_name = folder_name + "_mean" if args.mean_embeddings else folder_name + "_flat"

            if args.output_dir is not None:
                save_dir = os.path.abspath(args.output_dir)
            elif args.regression:
                result_folder = "controlled_extrapolation"
                if args.delta_embeddings:
                    result_folder += "_delta"
                if not args.mean_embeddings:
                    result_folder += "_flat"
                save_dir = os.path.join(
                    base_path,
                    "data",
                    args.dataset_name,
                    result_folder,
                    model_name,
                )
            else:
                save_dir = os.path.join(
                    base_path,
                    "data",
                    f"{args.dataset_name}_classification_results",
                    folder_name,
                    model_name,
                )
            if args.pca is not None and args.output_dir is None:
                save_dir = save_dir + f"_pca_{args.pca}"
            print(f"Saving results to {save_dir}")

            os.makedirs(save_dir, exist_ok=True)

            test_data = pd.DataFrame(test_data)
            print(test_data)

            test_data.to_csv(os.path.join(save_dir, f"{model_name}_evaluation_train_on_{n_train_data}_{args.classifier_model}.csv"), index=False)

if __name__ == '__main__':
    main()
