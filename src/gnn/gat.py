import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np

class GATRiskModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super(GATRiskModel, self).__init__()
        # input: node features (severity, stage, etc.)
        # edge_attr: confidence * strength
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        # Layer 1
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        return x

def run_gat_exposure(nodes, edges, risk_events, cfg_gnn):
    """
    Compute GAT-based exposure score as per '가이드.docx' Section 4.3.
    """
    # 1. Prepare Graph Data
    node_list = nodes['company_id'].tolist()
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # Node features: [Base Severity, Value Chain Stage (dummy)]
    # For now, let's use a simple 2D feature: [current_risk, stage_index]
    x = torch.zeros((num_nodes, 2), dtype=torch.float)
    
    # Edge index and attributes
    edge_index = []
    edge_attr = []
    for _, row in edges.iterrows():
        u = node_to_idx.get(str(row['src_company_id']))
        v = node_to_idx.get(str(row['dst_company_id']))
        if u is not None and v is not None:
            edge_index.append([u, v])
            # confidence * strength
            w = float(row.get('confidence_plink', 0.5)) * float(row.get('strength', 1.0))
            edge_attr.append([w])
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 2. Model Initialization
    model = GATRiskModel(in_channels=2, hidden_channels=16, out_channels=1)
    model.eval() # Inference mode for now (MVP)

    # 3. Process each event
    all_exposures = []
    for ev in risk_events.itertuples():
        # Reset node features for this event
        current_x = x.clone()
        sev = float(getattr(ev, 'severity', 1.0))
        ents = getattr(ev, 'entity_ids', [])
        
        # Injection of risk into mentioned entities
        for ent in ents:
            if ent in node_to_idx:
                current_x[node_to_idx[ent], 0] = sev
                
        # Forward pass (Propagation)
        with torch.no_grad():
            exposure_tensor = model(current_x, edge_index, edge_attr)
            
        # Temporal Decay (가이드 4.2 반영: e^-lambda*dist)
        # Since GNN does one-hop attention, we can treat it as dynamic weighted propagation
        # For simplicity, we normalize the output
        exp_values = F.relu(exposure_tensor).squeeze().numpy()
        
        for i, node_id in enumerate(node_list):
            all_exposures.append({
                "event_id": getattr(ev, "event_id"),
                "company_id": node_id,
                "exposure_gat": float(exp_values[i])
            })
            
    return pd.DataFrame(all_exposures)

if __name__ == "__main__":
    print("[*] GAT Risk Model Loaded Successfully.")
