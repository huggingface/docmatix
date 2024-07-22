from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import shutil
import os

def get_datasets():
    hf_datasets = []
    for shard_nr in tqdm(range(200)):
        try:
            hf_datasets.append(load_from_disk(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}'))
        except Exception as e:
            if os.path.isdir(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}'):
                shutil.rmtree(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}')
            print(f"Error loading dataset from: {shard_nr}")
            print(e)
    return hf_datasets


data = get_datasets()

print(data)
print(data[0].features)
print(data[0]['texts'][0])
print(data[0]['pdf'][0][:10])

hf_data = concatenate_datasets(data)
hf_data.push_to_hub('HuggingFaceM4/Docmatix', 'pdf')