import numpy as np
import argparse
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset
import pickle


# Define dataset class
class MoltoxpredDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define MLP model
class MoltoxMLP(nn.Module):
    def __init__(self, input_size):
        super(MoltoxMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.elu1(self.fc1(x))
        return self.fc2(x)


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(1024, dtype=int)
    fpt = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    return np.array(fpt)


def run_rf(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=64)
    rf.fit(X_train, y_train)
    rf_preds_proba = rf.predict_proba(X_test)
    return rf_preds_proba[:, 1]


def run_mlp(X_train, y_train, X_test, regression=True):
    # Prepare data for MLP
    train_dataset = MoltoxpredDataset(X_train, y_train)
    test_dataset = MoltoxpredDataset(X_test, np.zeros(X_test.shape[0]))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train MLP
    model = MoltoxMLP(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    
    if regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    model.train()
    for epoch in tqdm(range(15)):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).reshape(-1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

    # Evaluate MLP
    model.eval()
    all_outputs = []
    for data, labels in test_loader:
        outputs = model(data).reshape(-1)
        if not regression:
            outputs = sigmoid(outputs)
        all_outputs.append(outputs.detach().numpy())
    all_outputs = np.concatenate(all_outputs)
    return all_outputs


def run_lr(X_train, y_train, X_test):
    # logistic = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear')
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_train, y_train)
    logistic_preds_proba = logistic.predict_proba(X_test)
    return logistic_preds_proba[:, 1]


def main(label_file, embedding_file, embedding_type, model, outfile):
    """
    Runs a 5-fold cross-validation experiment using the specified classifier model on the embedding data to predict labels.
    """
    # Load and process data
    df = pd.read_csv(label_file, header=0)

    task_cols = [col for col in df.columns if col.startswith("Label_")]
    morgan_embeddings = [get_fingerprint(smiles) for smiles in df["SMILES"] if smiles]
    conplex_data = np.load(embedding_file, allow_pickle=True)
    conplex_embeddings = [embedding for embedding in conplex_data]

    avg_results = []

    for task in task_cols:
        # task_name = task.split('_')[-1]
        task_name = task
        labels = [label for label in df[task]]

        assert len(labels) == len(morgan_embeddings) == len(conplex_embeddings)
        labels = np.array(labels)

        # Choose embedding type
        if embedding_type == "morgan":
            data_embeddings = np.array(morgan_embeddings)
        elif embedding_type == "conplex":
            data_embeddings = np.array(conplex_embeddings)
        elif embedding_type == "combined":
            data_embeddings = np.concatenate(
                [morgan_embeddings, conplex_embeddings], axis=1
            )
        else:
            raise ValueError("Invalid embedding type")

        # Choose classifier model
        if model == "rf":
            run_model = run_rf
        elif model == "mlp":
            run_model = run_mlp
        elif model == "lr":
            run_model = run_lr
        else:
            raise ValueError("Invalid model type")

        # Setup cross-validation

        # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)

        kf = KFold(n_splits=5, shuffle=True, random_state=64)
            
        results = []
        for train_index, test_index in tqdm(kf.split(data_embeddings, labels)):
            X_train, X_test = data_embeddings[train_index], data_embeddings[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Run model
            preds = run_model(X_train, y_train, X_test)

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            # Store results
            results.append(
                {
                    "fold": len(results) + 1,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "task": task_name,
                }
            )

        # Aggregate results
        avg_mse = np.mean([r["mse"] for r in results])
        std_mse = np.std([r["mse"] for r in results])
        avg_rmse = np.mean([r["rmse"] for r in results])
        std_rmse = np.std([r["rmse"] for r in results])
        avg_mae = np.mean([r["mae"] for r in results])
        std_mae = np.std([r["mae"] for r in results])
        avg_r2 = np.mean([r["r2"] for r in results])
        std_r2 = np.std([r["r2"] for r in results])

        avg_results = [{
            "mse": avg_mse,
            "std_mse": std_mse,
            "rmse": avg_rmse,
            "std_rmse": std_rmse,
            "mae": avg_mae,
            "std_mae": std_mae,
            "r2": avg_r2,
            "std_r2": std_r2,
            "task": task_name,
        }]

        # Print results
        print(f"Task: {task_name}")
        print(f"MSE:\t{avg_mse:.4f} ± {std_mse:.4f}")
        print(f"RMSE:\t{avg_rmse:.4f} ± {std_rmse:.4f}")
        print(f"MAE:\t{avg_mae:.4f} ± {std_mae:.4f}")
        print(f"R²:\t{avg_r2:.4f} ± {std_r2:.4f}")

    with open(
        outfile,
        "wb",
    ) as f:
        pickle.dump(avg_results, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=str,
        default="benchmark_data/moltoxpred_conplex_embeddings.npz",
    )
    parser.add_argument(
        "--labels", type=str, default="benchmark_data/moltoxpred_data_processed.tsv"
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="combined",
        help="conplex, morgan, or combined",
    )
    parser.add_argument("--model", type=str, default="rf", help="rf or mlp or lr")
    parser.add_argument("--outfile", type=str, help="path to write results pkl file")
    args = parser.parse_args()
    main(args.labels, args.embeddings, args.embedding_type, args.model, args.outfile)
