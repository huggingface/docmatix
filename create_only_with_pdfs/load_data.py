import os
import re

import pandas as pd
import datasets
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
tqdm.pandas(desc="Pandas apply progress")


DATA_PATH = '/fsx/andi/pdfa_data/'
TAR_FILE_PATTERN = 'pdfa-eng-train-{:06d}.tar'

# Function to determine if a string contains code-like structures
def is_valid_question_or_answer(text):
    if not text or text.strip() == "":
        return False

    # Define patterns that indicate code
    patterns = [
        r'\{.*?\}',  # Matches { ... }
        r'\[.*?\]',  # Matches [ ... ]
        r'<.*?>',    # Matches < ... >
        r'\b\d{1,3}(\.\d{1,3}){3}\b',  # Matches IP addresses
        r'\w+\.\w+',  # Matches word.word patterns
        r'\n\s*\n',  # Matches two consecutive newlines
        r'unanswerable',  # Matches 'unanswerable' regardless of case
        r'Q\d+: ',  # Contains other questions
        r'A\d+: ',  # Contains other answers
    ]
    return not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

# Function to process a single group
def process_group(key_group):
    try:
        key, group = key_group
        qa_pairs = []
        for _, row in group.iterrows():
            question = re.sub(r'^Q\d+: ', '', row['question'])
            answer = re.sub(r'^A\d+: ', '', row['answer'])
            if is_valid_question_or_answer(question) and is_valid_question_or_answer(answer):
                qa_pairs.append({
                    "user": question,
                    "assistant": answer,
                    "source": "PDFA key: " + str(row['__key__'])
                })
        if qa_pairs:
            return {
                "texts": qa_pairs,
                "pdf": group['pdf'].iloc[0]
            }    
    except Exception as e:
        print(f"Error processing group {key}: {e}")
        return None

def process_tar_index(tar_index, step_size, question_answer_df):
    shard_nr = tar_index//step_size
    loaded_datasets = []

    for inner_idx in range(step_size):
        tar_file = os.path.join(DATA_PATH, TAR_FILE_PATTERN.format(tar_index+inner_idx))
        try:
            print(f"Loading dataset from: {tar_file}")
            hf_dataset = datasets.load_dataset('webdataset', split='train', data_files=tar_file, cache_dir="/fsx/.cache").to_pandas()
            hf_dataset.__key__ = hf_dataset.__key__.apply(pd.to_numeric)
            loaded_datasets.append(hf_dataset)
        except Exception as e:
            print(f"Error loading dataset from: {tar_file}")
            print(e)

    hf_dataset = pd.concat(loaded_datasets, ignore_index=True)
    print(f"Concatenated datasets with {len(hf_dataset)} samples")

    hf_dataset = hf_dataset[hf_dataset['__key__'].isin(question_answer_df['__key__'].unique())] # Filter samples that are not present in question_answer_df

    # Merging dataframes on '__key__' column
    merged_df = pd.merge(hf_dataset, question_answer_df, on='__key__', how='inner')

    # Using ThreadPoolExecutor for parallel processing of groups
    data_extracted = []
    max_threads = 10  # Number of threads to use
    with ThreadPoolExecutor(max_threads) as executor:
        results = list(tqdm(executor.map(process_group, merged_df.groupby('__key__')), desc='Extracting data', total=len(merged_df['__key__'].unique())))

    data_extracted.extend(results)
    data_extracted = list(filter(lambda item: item is not None, data_extracted)) # Filter out None values

    FEATURES = datasets.Features(
        {
            "pdf": datasets.Value("binary"),
            "texts": [
                {
                    "user": datasets.Value("string"),
                    "assistant": datasets.Value("string"),
                    "source": datasets.Value("string"),
                }
            ],
        }
    )
    def data_generator():
        for data_dict in data_extracted:
            yield data_dict
    #
    ds_shard = datasets.Dataset.from_generator(data_generator, features=FEATURES, writer_batch_size=100, cache_dir="/fsx/.cache")
    ds_shard.save_to_disk(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}')

def load_and_concatenate_dataframes():
    if os.path.exists('/fsx/andi/llm-swarm/concatenated_synthetic_dataset.parquet.gzip'):
        return pd.read_parquet('/fsx/andi/llm-swarm/concatenated_synthetic_dataset.parquet.gzip')  

    # Directory where the .h5 files are stored
    directory = '.'

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out the .h5 files and sort them
    h5_files = sorted([f for f in all_files if re.match(r'synthetic_dataset_batch_\d+\.h5$', f)])

    # Initialize an empty list to hold the dataframes
    dataframes = []

    # Load each .h5 file and append the dataframe to the list
    for file in tqdm(h5_files, desc="Loading data"):
        file_path = os.path.join(directory, file)
        df = pd.read_hdf(file_path)
        if '__key__' not in df.columns:
            raise ValueError(f"Key column not found in {file_path}")
        df.__key__ = df.__key__.apply(pd.to_numeric)
        dataframes.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    concatenated_df.to_parquet('concatenated_synthetic_dataset.parquet.gzip', compression='gzip')  

    return concatenated_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process .h5 files and tar indices.")
    parser.add_argument('--start_index', type=int, default=0, help='The starting index for tar processing.')
    parser.add_argument('--step_size', type=int, default=1, help='The step size for tar processing.')
    args = parser.parse_args()

    question_answer_df = load_and_concatenate_dataframes()
    print(len(question_answer_df))

    process_tar_index(args.start_index, args.step_size, question_answer_df=question_answer_df)
