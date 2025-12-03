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


def run_mlp(X_train, y_train, X_test):
    # Prepare data for MLP
    train_dataset = MoltoxpredDataset(X_train, y_train)
    test_dataset = MoltoxpredDataset(X_test, np.zeros(X_test.shape[0]))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train MLP
    model = MoltoxMLP(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
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
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)
        results = []
        for train_index, test_index in tqdm(kf.split(data_embeddings, labels)):
            X_train, X_test = data_embeddings[train_index], data_embeddings[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Run model
            preds_proba = run_model(X_train, y_train, X_test)

            # Evaluate predictions
            preds_binary = preds_proba > 0.5
            auroc = roc_auc_score(y_test, preds_proba)
            accuracy = accuracy_score(y_test, preds_binary)
            f1 = f1_score(y_test, preds_binary)

            # Store results
            results.append(
                {
                    "fold": len(results) + 1,
                    "accuracy": accuracy,
                    "auroc": auroc,
                    "f1": f1,
                    "task": task_name,
                }
            )

        # Print average results
        avg_accuracy = np.mean([r["accuracy"] for r in results])
        std_accuracy = np.std([r["accuracy"] for r in results])
        avg_auroc = np.mean([r["auroc"] for r in results])
        std_auroc = np.std([r["auroc"] for r in results])
        avg_f1 = np.mean([r["f1"] for r in results])
        std_f1 = np.std([r["f1"] for r in results])

        avg_results.append(
            {
                "accuracy": avg_accuracy,
                "std_accuracy": std_accuracy,
                "auroc": avg_auroc,
                "std_auroc": std_auroc,
                "f1": avg_f1,
                "std_f1": std_f1,
                "task": task_name,
            }
        )
        print(f"Task: {task_name}")
        print(f"Accuracy:\t{avg_accuracy}\t({std_accuracy})")
        print(f"AUROC:\t\t{avg_auroc}\t({std_auroc})")
        print(f"F1:\t\t{avg_f1}\t({std_f1})")

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
    parser.add_argument("--outfile", type=str, help="path to results pkl file to output")
    args = parser.parse_args()
    main(args.labels, args.embeddings, args.embedding_type, args.model, args.outfile)
