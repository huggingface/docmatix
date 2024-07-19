from functools import partial
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import re
import pandas as pd
import os
import datasets

IMAGE_FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "__key__": datasets.Value("int64"),
        }
)
TEXT_FEATURES = datasets.Features(
    {
            "question": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "__key__": datasets.Value("int64"),
    }
)

def text_generator(df_text):
    for i, row in df_text.iterrows():
        print(i, row['__key__'])
        yield {
            "question": row['question'],
            "answer": row['answer'],
            "__key__": row['__key__'],
        }

def img_generator(df_img):
    for i, row in df_img.iterrows():
        print(i, row['__key__'])
        yield {
            "image": row['images'][0],
            "__key__": row['__key__'],
        }

pre_key_len = len("PDFA key: ")

for shard_number in tqdm(range(0, 200)):
    try:
        if os.path.exists(f'/fsx/m4/datasets/florence_vqa_instruct/shard_{shard_number}') and os.path.exists(f'/fsx/m4/datasets/florence_vqa_instruct_images/shard_{shard_number}'):
            continue
        df_data = load_from_disk(f"/fsx/m4/datasets/docvqa_instruct/shard_{shard_number}").to_pandas()

        # Create the images DataFrame
        df_data['__key__'] = df_data.texts.apply(lambda x: x[0]['source'][pre_key_len:])
        df_data["__key__"] = df_data["__key__"].apply(pd.to_numeric)

        df_images = df_data[['images', '__key__']].copy()

        # Filter out rows with more than one image
        df_images = df_images[df_images['images'].apply(len) <= 1]

        # Explode the texts into separate rows and create a DataFrame
        df_texts = df_data[['texts']].explode('texts')
        
        # Extract 'user', 'assistant', and 'source' from the 'texts' column
        df_texts['question'] = df_texts['texts'].apply(lambda x: x.get('user'))
        df_texts['answer'] = df_texts['texts'].apply(lambda x: x.get('assistant'))
        df_texts['__key__'] = df_texts['texts'].apply(lambda x: x.get('source')[pre_key_len:])
        df_texts["__key__"] = df_texts["__key__"].apply(pd.to_numeric)

        df_texts = df_texts[df_texts['__key__'].isin(df_images['__key__'].unique())] # Filter out rows with more than one image

        # Drop the original 'texts' column
        df_texts.drop(columns=['texts'], inplace=True)

        # Filter out rows where the combined length of words in the questions is longer than self._max_length
        df_texts = df_texts[df_texts['question'].apply(lambda x: len(x.split()) <= 900)]
        df_texts = df_texts[df_texts['answer'].apply(lambda x: len(x.split()) <= 900)]
        df_images = df_images[df_images['__key__'].isin(df_texts['__key__'].unique())]
                
        ds_text = datasets.Dataset.from_generator(partial(text_generator, df_texts), features=TEXT_FEATURES, writer_batch_size=100, cache_dir="/fsx/.cache")
        ds_text.save_to_disk(f'/fsx/m4/datasets/florence_vqa_instruct/shard_{shard_number}')
        df_image = datasets.Dataset.from_generator(partial(img_generator, df_images), features=IMAGE_FEATURES, writer_batch_size=100, cache_dir="/fsx/.cache")
        df_image.save_to_disk(f'/fsx/m4/datasets/florence_vqa_instruct_images/shard_{shard_number}')

        print(f"Finished processing shard: {shard_number}")

    except:
        print(f"shard {shard_number} failed")


all_ds = []
for shard in tqdm(range(0, 200)):
    try:
        data = load_from_disk(f"/fsx/m4/datasets/florence_vqa_instruct/shard_{shard}")
        all_ds.append(data)
    except:
        print(f"shard {shard} failed")

all_ds = concatenate_datasets(all_ds)
all_ds.save_to_disk("/fsx/m4/datasets/complete_florence_vqa_instruct", num_proc=96)
