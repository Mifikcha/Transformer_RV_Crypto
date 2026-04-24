import pandas as pd
import sys

df = pd.read_parquet(sys.argv[1])
df.to_csv(sys.argv[2], index=False)

'''
python get_data/parquet_to_csv.py get_data/output/_main/_final/btcusdt_5m_final_cleaned.parquet get_data/output/_main/_final/btcusdt_5m_final_cleaned.csv  
'''
