import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fpt = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    return np.array(fpt)

# Load dataframe of moltoxpred data smiles and toxicity label
df = pd.read_csv('benchmark_data/moltoxpred_data_processed.tsv', sep='\t', header=None, names=['ID', 'SMILES', 'Toxicity'])
toxicity = df['Toxicity'].values

# Load the morgan fingerprint embeddings
if os.path.exists('benchmark_data/morgan_moltox_embeddings.npy'):
    morgan_embeddings = np.load('benchmark_data/morgan_moltox_embeddings.npy')
else:
    morgan_embeddings = np.array([get_fingerprint(smiles) for smiles in df['SMILES']])
    np.save('benchmark_data/morgan_moltox_embeddings.npy', morgan_embeddings)

# Load the ConPLeX embeddings
data = np.load('benchmark_data/moltoxpred_conplex_embeddings.npz', allow_pickle=True)
data_embeddings = data['embedding']
data_embeddings = np.concatenate((data_embeddings, morgan_embeddings), axis=1)

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

# Setup cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)
results = []

for train_index, test_index in kf.split(data_embeddings, toxicity):
    X_train, X_test = data_embeddings[train_index], data_embeddings[test_index]
    y_train, y_test = toxicity[train_index], toxicity[test_index]

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=64)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_auroc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    rf_accuracy = accuracy_score(y_test, rf_preds)

    # Prepare data for MLP
    train_dataset = MoltoxpredDataset(X_train, y_train)
    test_dataset = MoltoxpredDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train MLP
    model = MoltoxMLP(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(15):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).reshape(-1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

    # Evaluate MLP
    model.eval()
    all_labels, all_outputs = [], []
    for data, labels in test_loader:
        outputs = model(data).reshape(-1)
        outputs = sigmoid(outputs)
        all_labels.append(labels.numpy())
        all_outputs.append(outputs.detach().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    mlp_auroc = roc_auc_score(all_labels, all_outputs)
    mlp_accuracy = accuracy_score(all_labels, all_outputs > 0.5)

    # Store results
    results.append({
        'fold': len(results) + 1,
        'rf_accuracy': rf_accuracy,
        'rf_auroc': rf_auroc,
        'mlp_accuracy': mlp_accuracy,
        'mlp_auroc': mlp_auroc
    })

# Print average results
avg_rf_accuracy = np.mean([r['rf_accuracy'] for r in results])
avg_rf_auroc = np.mean([r['rf_auroc'] for r in results])
avg_mlp_accuracy = np.mean([r['mlp_accuracy'] for r in results])
avg_mlp_auroc = np.mean([r['mlp_auroc'] for r in results])
print(f'Average RF Accuracy: {avg_rf_accuracy}, Average RF AUROC: {avg_rf_auroc}')
print(f'Average MLP Accuracy: {avg_mlp_accuracy}, Average MLP AUROC: {avg_mlp_auroc}')

