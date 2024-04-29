import pandas as pd
import sys

def upload(file, extension):
    # Read the Parquet file into a Pandas DataFrame
    # df = pd.read_parquet(f'data/{file}.{extension}')
    df = pd.read_parquet(f'data/pubmedqa-labeled.parquet')
    # Convert the DataFrame to a CSV file
    df.to_csv(f'data/{file}.csv', index=False)
    df.to_excel(f'data/{file}.xlsx', index=False)
    df.to_json(f'data/{file}.jsonl', orient='records', lines=True)

upload('pubmedqa-labeled', 'parquet')
# upload('pubmedqa-unlabeled', 'parquet')
# upload('pubmedqa-artificial', 'parquet')