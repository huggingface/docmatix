import os
import re
import io
from io import BytesIO

import pandas as pd
import datasets
from pdf2image import convert_from_bytes
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import fitz
import PIL.Image
tqdm.pandas(desc="Pandas apply progress")

fitz.TOOLS.mupdf_display_errors(False)

DATA_PATH = '/fsx/andi/pdfa_data/'
TAR_FILE_PATTERN = 'pdfa-eng-train-{:06d}.tar'


def resize_large_images(image, max_image_size=2940):
    width, height = image.size
    aspect_ratio = width / height

    resized = False
    if width >= height and width > max_image_size:
        width = max_image_size
        height = int(width / aspect_ratio)
        resized = True
    elif height > width and height > max_image_size:
        height = max_image_size
        width = int(height * aspect_ratio)
        resized = True
    if resized:
        image = image.resize((width, height), PIL.Image.LANCZOS)

    return image

def _decode_pdf_pages(
    sample,
):
    try:
        image_fmt = "L"
        with io.BytesIO(sample) as b:
            doc = fitz.Document(stream=b)
            num_image_pages = doc.page_count
            decoded_image_pages = []
            for page_index in range(num_image_pages):
                page = doc.load_page(page_index)
                pixmap = page.get_pixmap(dpi=150)
                page_image = PIL.Image.frombuffer("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                page_image = resize_large_images(page_image.convert(image_fmt))

                decoded_image_pages += [page_image]

            return decoded_image_pages
    except Exception as e:
        print(f"Error decoding pdf pages: {e}")
        return None

def convert_img_to_png_bytes(img):
    with BytesIO() as buffer:
        img.save(buffer, format='PNG')
        return buffer.getvalue()

def process_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=150)
    return [convert_img_to_png_bytes(resize_large_images(img)) for img in images]

# Function to determine if a string contains code-like structures
def is_valid_question_or_answer(text):
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
                "images": group['pdf'].iloc[0]
            }    
    except Exception as e:
        print(f"Error processing group {key}: {e}")
        return None

def process_tar_index(tar_index, step_size):
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

    hf_dataset = hf_dataset[hf_dataset['__key__'].isin(concatenated_df['__key__'].unique())] # Filter samples that are not present in concatenated_df
    
    df_data = pd.DataFrame({'key': []})
    if os.path.exists(f"/fsx/m4/datasets/large_docvqa/shard_{shard_nr}"):
        print('using saved data')
        df_data = datasets.load_from_disk(f"/fsx/m4/datasets/large_docvqa/shard_{shard_nr}").to_pandas()
        df_data["__key__"] = df_data.texts.apply(lambda x: x[0]['source'].split('_')[1])
        df_data["__key__"] = df_data["__key__"].apply(pd.to_numeric)
        df_data.drop(columns=['texts'], inplace=True)
        hf_dataset = hf_dataset[hf_dataset['__key__'].isin(df_data['__key__'].unique())] # Filter out samples that failed conversion
        hf_dataset = pd.merge(hf_dataset, df_data, on='__key__', how='inner')
        hf_dataset['pdf'] = hf_dataset['images']
        hf_dataset.drop(columns=['images'], inplace=True)
        del df_data
    else:
        hf_dataset['pdf'] = hf_dataset['pdf'].progress_apply(lambda x: process_images(x)) # Decode pdf pages in place to save memory
        hf_dataset = hf_dataset[~hf_dataset['pdf'].isnull()] # Filter out images that failed

    # Merging dataframes on '__key__' column
    merged_df = pd.merge(hf_dataset, concatenated_df, on='__key__', how='inner')

    # Using ThreadPoolExecutor for parallel processing of groups
    data_extracted = []
    max_threads = 10  # Number of threads to use
    with ThreadPoolExecutor(max_threads) as executor:
        results = list(tqdm(executor.map(process_group, merged_df.groupby('__key__')), desc='Extracting data', total=len(merged_df['__key__'].unique())))

    data_extracted.extend(results)
    data_extracted = list(filter(lambda item: item is not None, data_extracted)) # Filter out None values

    FEATURES = datasets.Features(
        {
            "images": datasets.Sequence(datasets.Image(decode=True)),
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
    ds_shard.save_to_disk(f'/fsx/m4/datasets/docvqa_instruct/shard_{shard_nr}')

def load_and_concatenate_dataframes():
    if os.path.exists('concatenated_synthetic_dataset.parquet.gzip'):
        return pd.read_parquet('concatenated_synthetic_dataset.parquet.gzip')  

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

    concatenated_df = load_and_concatenate_dataframes()
    print(len(concatenated_df))

    process_tar_index(args.start_index, args.step_size)
