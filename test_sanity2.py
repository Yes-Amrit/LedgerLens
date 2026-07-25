import pandas as pd
from data.loader import load_dataset
from agent.nodes.anomaly_node import run_anomaly_detection

def main():
    print('Loading data...')
    df = load_dataset('data/SAML-D.csv', nrows=None) 
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = df.index.astype(str)
    
    accounts_str = ['92172', '344654', '672794']
    df_acc = df[df['Sender_account'].astype(str).isin(accounts_str)].copy()
    
    print(f'Found {len(df_acc)} transactions for the target accounts.')
        
    res = run_anomaly_detection(df_acc, target_pattern='none')
    flagged = res.get('flagged_transactions', [])
    print(f'Method Used: {res.get("method_used")}')
    if len(flagged) == 0:
        print('SUCCESS: 0 transactions flagged.')
    else:
        print(f'FAILURE: {len(flagged)} transactions flagged!')
        for t in flagged: print(t)

if __name__ == '__main__': main()
