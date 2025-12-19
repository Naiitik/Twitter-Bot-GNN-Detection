import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --- 1. LOAD DATA ---
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'bot_detection_data.csv')

df = pd.read_csv(file_path)
print("--- Step 1: File Loaded! Checking columns... ---")

# --- 2. EXACT COLUMN MAPPING (Aditya Goyal Dataset) ---
# We use these exact names based on the Kaggle dataset structure
feature_cols = ['Follower Count', 'Retweet Count', 'Mention Count']
label_col = 'Bot Label'

# Verify if these columns exist, otherwise print what's actually there
missing = [c for c in feature_cols + [label_col] if c not in df.columns]
if missing:
    print(f"ERROR: Still missing columns: {missing}")
    print(f"Available columns in your file are: {df.columns.tolist()}")
    exit()

# Clean and Prepare
df = df.dropna(subset=feature_cols + [label_col])
x_raw = df[feature_cols].values
y = torch.tensor(df[label_col].values, dtype=torch.long)

# Scale
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float)

# --- 3. GRAPH SETUP ---
num_nodes = min(len(x), 1000) 
edge_list = []
for i in range(num_nodes):
    for _ in range(2):
        target = (i + np.random.randint(1, 10)) % num_nodes
        edge_list.append([i, target])
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

data = Data(x=x[:num_nodes], edge_index=edge_index, y=y[:num_nodes])
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[:int(num_nodes*0.7)] = True
data.test_mask = ~data.train_mask

# --- 4. MODEL ---
class BotGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(len(feature_cols), 16)
        self.conv2 = GCNConv(16, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

# --- 5. TRAINING & RESULTS ---
print("--- Step 2: Training GNN... ---")
model = BotGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(1, 51):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

print("--- Step 3: Success! Saving final_results.png ---")
model.eval()
pred = model(data).argmax(dim=1)
y_true = data.y[data.test_mask].numpy()
y_pred = pred[data.test_mask].numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Oranges')
plt.title("Bot Detection Success Rate")

plt.subplot(1, 2, 2)
plt.text(0.1, 0.5, classification_report(y_true, y_pred), fontsize=10, family='monospace')
plt.axis('off')

plt.savefig('final_results.png')
print("FINISHED! Open 'final_results.png' to see your project output.")
plt.show()
import networkx as nx

# --- VISUALIZE THE ACTUAL SOCIAL GRAPH ---
plt.figure(figsize=(8, 8))
G = nx.Graph()

# Add nodes and edges from your GNN data
edge_list = data.edge_index.t().numpy()
G.add_edges_from(edge_list[:100]) # Limit to 100 edges so it's not a mess

# Color bots red and humans blue
colors = ['red' if data.y[i] == 1 else 'blue' for i in range(100)]

nx.draw(G, node_size=50, node_color=colors, with_labels=False)
plt.title("Visualizing the Social Network (Red=Bots, Blue=Humans)")
plt.show()