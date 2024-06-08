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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset
    

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


def run_logistic(X_train, y_train, X_test):
    # logistic = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear')
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_train, y_train)
    logistic_preds_proba = logistic.predict_proba(X_test)
    return logistic_preds_proba[:, 1]


def main(label_file, embedding_file, embedding_type, model):
    # Load dataframe of moltoxpred data smiles and toxicity label
    df = pd.read_csv(label_file, sep='\t', header=None, names=['ID', 'SMILES', 'Class'])
    # smiles_labels = {smiles: label for smiles, label in zip(df['SMILES'], df['Class'])}
    labels = [label for label in df['Class']]

    # # Load the morgan fingerprint embeddings
    # if os.path.exists('benchmark_data/morgan_moltox_embeddings.npy'):
    #     morgan_embeddings = np.load('benchmark_data/morgan_moltox_embeddings.npy')
    # else:
    #     morgan_embeddings = np.array([get_fingerprint(smiles) for smiles in df['SMILES']])
    #     np.save('benchmark_data/morgan_moltox_embeddings.npy', morgan_embeddings)

    # smiles_morgan_embeddings = {smiles: get_fingerprint(smiles) for smiles in df['SMILES']}
    morgan_embeddings = [get_fingerprint(smiles) for smiles in df['SMILES']]

    # Load the ConPLeX embeddings
    data = np.load(embedding_file, allow_pickle=True)
    # smiles_conplex_embeddings = {smiles: embedding for smiles, embedding in zip(data['proteinID'], data['embedding'])}
    conplex_embeddings = [embedding for embedding in data['embedding']]

    # labels = []
    # morgan_embeddings = []
    # conplex_embeddings = []
    # for smiles in smiles_labels.keys():
    #     if smiles not in smiles_morgan_embeddings or smiles not in smiles_conplex_embeddings:
    #         continue
    #     labels.append(smiles_labels[smiles])
    #     morgan_embeddings.append(smiles_morgan_embeddings[smiles])
    #     conplex_embeddings.append(smiles_conplex_embeddings[smiles])
    assert len(labels) == len(morgan_embeddings) == len(conplex_embeddings)
    toxicity = np.array(labels)

    if embedding_type == 'morgan':
        data_embeddings = np.array(morgan_embeddings)
    elif embedding_type == 'conplex':
        data_embeddings = np.array(conplex_embeddings)
    elif embedding_type == 'combined':
        data_embeddings = np.concatenate([morgan_embeddings, conplex_embeddings], axis=1) 
    else:
        raise ValueError('Invalid embedding type')
    
    if model == 'rf':
        run_model = run_rf
    elif model == 'mlp':
        run_model = run_mlp
    elif model == 'logistic':
        run_model = run_logistic
    else:
        raise ValueError('Invalid model type')

    # Setup cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)
    results = []

    for train_index, test_index in tqdm(kf.split(data_embeddings, toxicity)):
        X_train, X_test = data_embeddings[train_index], data_embeddings[test_index]
        y_train, y_test = toxicity[train_index], toxicity[test_index]

        preds_proba = run_model(X_train, y_train, X_test)
        preds_binary = preds_proba > 0.5
        auroc = roc_auc_score(y_test, preds_proba)
        accuracy = accuracy_score(y_test, preds_binary)
        f1 = f1_score(y_test, preds_binary)

        # Store results
        results.append({
            'fold': len(results) + 1,
            'accuracy': accuracy,
            'auroc': auroc,
            'f1': f1,
        })

    # Print average results
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    std_accuracy = np.std([r['accuracy'] for r in results])
    avg_auroc = np.mean([r['auroc'] for r in results])
    std_auroc = np.std([r['auroc'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    std_f1 = np.std([r['f1'] for r in results])
    print(f'Accuracy:\t{avg_accuracy}\t({std_accuracy})')
    print(f'AUROC:\t\t{avg_auroc}\t({std_auroc})')
    print(f'F1:\t\t{avg_f1}\t({std_f1})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default='benchmark_data/moltoxpred_conplex_embeddings.npz')
    parser.add_argument('--labels', type=str, default='benchmark_data/moltoxpred_data_processed.tsv')
    parser.add_argument('--embedding_type', type=str, default='combined', help='conplex, morgan, or combined')
    parser.add_argument('--model', type=str, default='rf', help='rf or mlp or logistic')
    args = parser.parse_args()
    main(args.labels, args.embeddings, args.embedding_type, args.model)