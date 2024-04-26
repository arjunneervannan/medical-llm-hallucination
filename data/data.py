import pandas as pd

def upload(file, extension):
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(f'data/{file}.{extension}')

    # Convert the DataFrame to a CSV file
    df.to_csv(f'data/{file}.csv', index=False)
    df.to_excel(f'data/{file}.xlsx', index=False)
    df.to_json(f'data/{file}.json', index=False)

upload('pubmedqa-labeled', 'parquet')
upload('pubmedqa-unlabeled', 'parquet')
upload('pubmedqa-artificial', 'parquet')