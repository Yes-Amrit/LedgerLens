import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import torch
import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not found. LSTM Autoencoder will be disabled.")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, num_layers: int = 1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x: (batch, seq, features)
        _, (hidden, cell) = self.encoder(x)
        # Repeat hidden state for sequence length
        # hidden: (num_layers, batch, hidden_dim) -> (batch, 1, hidden_dim) for last layer
        hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(hidden)
        return out

def detect_sequential_anomalies(df: pd.DataFrame, sequence_length: int = 10) -> dict:
    """
    LSTM Autoencoder for detecting sequential anomalies.
    Returns normalized scores in [0, 1].
    Degrades gracefully if PyTorch is not available or data is insufficient.
    """
    empty_result = {
        "flagged_transactions": [],
        "anomaly_scores": {},
        "method_used": "lstm_autoencoder"
    }
    
    if df.empty or not TORCH_AVAILABLE:
        return empty_result

    # Ensure transaction_id exists
    if 'transaction_id' not in df.columns:
        df = df.copy()
        df['transaction_id'] = [f"tx_{i}" for i in range(len(df))]

    account_col = 'account_id' if 'account_id' in df.columns else ('Sender_account' if 'Sender_account' in df.columns else 'nameOrig')
    time_col = 'timestamp' if 'timestamp' in df.columns else ('Time' if 'Time' in df.columns else 'step')
    amount_col = 'amount' if 'amount' in df.columns else ('Amount' if 'Amount' in df.columns else None)
    
    if account_col not in df.columns or amount_col not in df.columns or time_col not in df.columns:
        return empty_result
        
    df_temp = df.copy().sort_values(by=[account_col, time_col])
    
    # We will use Amount as the primary feature for sequence. More features could be added.
    # For a real implementation, we'd scale this globally. MinMax for simplicity here.
    amount_min = df_temp[amount_col].min()
    amount_max = df_temp[amount_col].max()
    if amount_max > amount_min:
        df_temp['scaled_amount'] = (df_temp[amount_col] - amount_min) / (amount_max - amount_min)
    else:
        df_temp['scaled_amount'] = 0.0
        
    # Group by account and build sequences
    sequences = []
    tx_ids = []
    
    for account, group in df_temp.groupby(account_col):
        if len(group) >= sequence_length:
            vals = group['scaled_amount'].values
            ids = group['transaction_id'].values
            # Rolling window sequences
            for i in range(len(group) - sequence_length + 1):
                seq = vals[i:i+sequence_length]
                seq_ids = ids[i:i+sequence_length]
                sequences.append(seq)
                tx_ids.append(seq_ids)
                
    if not sequences:
        # Not enough sequence data
        return {
            "flagged_transactions": [],
            "anomaly_scores": {str(k): 0.0 for k in df['transaction_id']},
            "method_used": "lstm_autoencoder"
        }
        
    X = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(-1)
    
    # Instantiate and "train" (fast single pass or random weights for demo if untrainable)
    # Since this is to be used dynamically, training on the fly might be too slow for large df.
    # We will do a single quick pass optimization to make it functional.
    model = LSTMAutoencoder(input_dim=1, hidden_dim=4, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Quick 1-epoch training for demonstration
    model.train()
    dataset = TensorDataset(X, X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for _ in range(1): # 1 epoch
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        # MSE per sequence
        mse = torch.mean((reconstructed - X)**2, dim=[1, 2]).numpy()
        
    # Normalize MSE to [0, 1] using min-max on the batch
    mse_min, mse_max = mse.min(), mse.max()
    if mse_max > mse_min:
        norm_scores = (mse - mse_min) / (mse_max - mse_min)
    else:
        norm_scores = np.zeros_like(mse)
        
    anomaly_scores = {str(k): 0.0 for k in df['transaction_id']}
    flagged_txs = {}
    
    for i, seq_ids in enumerate(tx_ids):
        score = float(norm_scores[i])
        for tx_id in seq_ids:
            # A transaction can be part of multiple sequences; take the max score
            anomaly_scores[str(tx_id)] = max(anomaly_scores.get(str(tx_id), 0.0), score)
            
            if score > 0.5:
                if str(tx_id) not in flagged_txs:
                    row = df_temp[df_temp['transaction_id'] == tx_id].iloc[0]
                    account_id = row[account_col]
                    amount = row[amount_col]
                    timestamp = row[time_col]
                    flagged_txs[str(tx_id)] = {
                        "transaction_id": str(tx_id),
                        "account_id": account_id,
                        "amount": float(amount),
                        "timestamp": timestamp,
                        "reason_features": {"lstm_reconstruction_error": f"{score:.2f}"}
                    }
                    
    return {
        "flagged_transactions": list(flagged_txs.values()),
        "anomaly_scores": anomaly_scores,
        "method_used": "lstm_autoencoder"
    }
