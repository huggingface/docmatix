from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import os

def get_datasets():
    if os.path.isdir('/fsx/m4/datasets/docmatix_pdf/concatenated'):
       return load_from_disk('/fsx/m4/datasets/docmatix_pdf/concatenated') 
    
    hf_datasets = []
    for shard_nr in tqdm(range(200)):
        try:
            hf_datasets.append(load_from_disk(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}'))
        except Exception as e:
            # if os.path.isdir(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}'):
            #     shutil.rmtree(f'/fsx/m4/datasets/docmatix_pdf/shard_{shard_nr}')
            print(f"Error loading dataset from: {shard_nr}")
            print(e)
    hf_data = concatenate_datasets(hf_datasets)
    hf_data.save_to_disk('/fsx/m4/datasets/docmatix_pdf/concatenated')
    return hf_data


data = get_datasets()

print(data.features)
print(data[0]['texts'])
print(data[0]['pdf'][:10])
print(len(data))
data.push_to_hub('HuggingFaceM4/Docmatix', 'pdf')
