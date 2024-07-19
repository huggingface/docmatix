import asyncio
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import IterableDataset, load_dataset
from huggingface_hub import AsyncInferenceClient
from tqdm import trange
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from examples.question_answer_pairs.phase_1.base_prompts import (BASE_PROMPT,
                                                         BASE_USER_CONTENT,
                                                         PROMPTS)
from llm_swarm import LLMSwarm, LLMSwarmConfig

CHECKPOINT_FILE = 'checkpoint.json'
DATA_PATH = '/fsx/andi/pdfa_data/'
TAR_FILE_PATTERN = 'pdfa-eng-train-{:06d}.tar'
NUM_TAR_FILES = 1800  # Total number of tar files
MAX_PAGES_PER_PDF = 4
STEP_SIZE = 10

model_id = "microsoft/Phi-3-small-8k-instruct"

def create_llm_prompt(prompt, text):
    system_content = BASE_PROMPT.format(
        role_description=prompt["role_description"],
        examples=prompt["examples"]
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": BASE_USER_CONTENT.format(text=text)}
    ]


def extract_text_per_page_from_sample(sample: Dict[str, Any]) -> List[str]:
    """
    Extracts text from each page of a given sample and returns it as a list of strings.

    Args:
        sample (Dict[str, Any]): The sample containing page data in JSON format.

    Returns:
        List[str]: A list of strings, where each string represents the text of a page.
    """
    texts = []
    for page in sample['json']['pages']:
        pages_text = ' \n '.join(page['lines']['text'])
        texts.append(pages_text)
    return texts


def extract_chunks(pages: List[Any], max_tokens_per_group: int, max_pages_per_group: int, n_overlap: int) -> List[str]:
    """
    Splits a list of pages into chunks with a specified maximum number of tokens per chunk,
    a maximum number of pages per chunk, and overlap between chunks.

    Args:
        pages (List[Any]): The list of pages to be chunked.
        max_tokens_per_group (int): The maximum number of tokens allowed per chunk.
        max_pages_per_group (int): The maximum number of pages allowed per chunk.
        n_overlap (int): The number of overlapping pages between consecutive chunks.

    Returns:
        List[str]: A list of chunked text, each chunk containing text from multiple pages.
    """
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    current_chunk_pages = 0
    page_token_counts = [len(tokenizer.encode(page, add_special_tokens=False)) for page in pages]
    
    for i, page in enumerate(pages):
        page_tokens = page_token_counts[i]
        if page_tokens > max_tokens_per_group:
            print(f"Skipping document where page nr {i} has {page_tokens} tokens.")
            return []
        
        if (current_chunk_tokens + page_tokens > max_tokens_per_group) or (current_chunk_pages + 1 > max_pages_per_group):
            if current_chunk:
                chunks.append('\nNEW PAGE\n'.join(current_chunk))
            current_chunk = current_chunk[-n_overlap:] if n_overlap > 0 else []
            current_chunk_tokens = sum(page_token_counts[max(0, i - n_overlap):i])
            current_chunk_pages = len(current_chunk)
        
        current_chunk.append(page)
        current_chunk_tokens += page_tokens
        current_chunk_pages += 1
    
    if current_chunk:
        chunks.append('\nNEW PAGE\n'.join(current_chunk))
    
    return chunks

def create_tasks(dataset: IterableDataset, prompt_id: Optional[int] = None, n_overlap: int = 2) -> List[Dict[str, Any]]:
    """
    Processes a dataset to generate question and answer pairs for each sample.

    Args:
        dataset (IterableDataset): The dataset containing samples.
        prompt_id (Optional[int]): The ID of the prompt template to use for generating questions. If set to None, prompt_id is random.
        n_overlap (int): The number of overlapping pages between consecutive chunks.
        num_samples (int): The number of samples to process.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the sample key, page count, generated Q/A pairs, and prompt ID.
    """
    if prompt_id is not None:
        selected_id_prompt = prompt_id

    tasks = []

    for index, sample in dataset.iterrows():
        text_per_page = extract_text_per_page_from_sample(sample)
        if len(text_per_page) > MAX_PAGES_PER_PDF:
            continue
        page_chunks = extract_chunks(text_per_page, max_tokens_per_group=5000, max_pages_per_group=5, n_overlap=n_overlap)

        for chunk in page_chunks:
            if prompt_id is None:
                selected_id_prompt = random.randint(0, 4)
            prompt = PROMPTS[selected_id_prompt]
            messages = create_llm_prompt(prompt, chunk)
            prompt = tokenizer.apply_chat_template(messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            tasks_dict = {
                "__key__": sample['__key__'],
                "Page count": len(text_per_page),
                "messages": prompt,
                "Prompt ID": selected_id_prompt
            }
            tasks.append(tasks_dict)
        
    return tasks

# Function to extract Q&A pairs from a string
def extract_qa_pairs(text):
    qa_pattern = re.compile(r'(Q\d+:\s*.*?)(A\d+:\s*.*?)(?=(Q\d+:)|$)', re.DOTALL)
    matches = qa_pattern.findall(text)
    qa_pairs = [(q.strip(), a.strip()) for match in matches for q, a in [match[:2]]]
    return qa_pairs

def process_outputs_to_df(df):
    all_data = []

    for index, row in df.iterrows():
        task = row['Task']
        completion = row['Completion']

        sample_key = task['__key__']
        page_count = task['Page count']
        prompt_id = task['Prompt ID']

        qa_pairs = extract_qa_pairs(completion)
        if len(qa_pairs) == 0:
            print('No Q&A pairs found for sample:', sample_key)

        for question, answer in qa_pairs:
            all_data.append({
                '__key__': sample_key,
                'Page count': page_count,
                'Prompt ID': prompt_id,
                'question': question,
                'answer': answer
            })

    qa_df = pd.DataFrame(all_data)
    return qa_df

def save_checkpoint(tar_index, total_examples):
    checkpoint_data = {
        'tar_index': tar_index,
        'total_examples': total_examples
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'tar_index': 0, 'total_examples': 0}

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def launch():
    with LLMSwarm(
        LLMSwarmConfig(
            instances=8,
            inference_engine="vllm",
            gpus=1,
            model=model_id,
            slurm_template_path="templates/vllm_h100.template.slurm",
            load_balancer_template_path="templates/nginx.template.conf",
            trust_remote_code=True,
            per_instance_max_parallel_requests=200,
        )
    ) as llm_swarm:
        semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
        client = AsyncInferenceClient(model=llm_swarm.endpoint)

        async def process_text(prompt):
            async with semaphore:
                response = await client.post(
                    json={
                        "prompt": prompt,
                        "max_tokens": 2000,
                    }
                )
                res = json.loads(response.decode("utf-8"))["text"][0][len(prompt):]
                return res

        def load_and_process_dataset(tar_file):
            try:
                print(f"Loading dataset from: {tar_file}")
                dataset = load_dataset('webdataset', split='train', data_files=tar_file).to_pandas()
                tasks = create_tasks(dataset, prompt_id=None, n_overlap=1)
                return tasks
            except Exception as e:
                print(f"Error loading dataset from: {tar_file}")
                print(e)
                return []

        def get_future_tasks(tar_index, executor):
            futures = []
            for inner_idx in range(STEP_SIZE):
                tar_file = os.path.join(DATA_PATH, TAR_FILE_PATTERN.format(tar_index + inner_idx))
                futures.append(executor.submit(load_and_process_dataset, tar_file))
            return futures

        async def process_dataset(tar_index, total_examples):
            next_future_tasks = get_future_tasks(tar_index, ThreadPoolExecutor(max_workers=STEP_SIZE))
            for idx in trange(tar_index, NUM_TAR_FILES + STEP_SIZE, STEP_SIZE, desc="Creating Dataset"):
                print(f"Processing tar file {idx}")
                tasks = []
                future_tasks = next_future_tasks

                results = [f.result() for f in future_tasks]
                for result in results:
                    tasks.extend(result)
                # Once you created the tasks for this batch, load the next batch in parallel
                # Otherwise, the tasks for this batch compete with the tasks from next batch for resources
                next_future_tasks = get_future_tasks(idx + STEP_SIZE, ThreadPoolExecutor(max_workers=1)) # Only one thread to avoid cpu clogging

                results = await tqdm_asyncio.gather(*(process_text(task['messages']) for task in tasks))
                df = pd.DataFrame({"Task": tasks, "Completion": results})
                df_new = process_outputs_to_df(df)

                # Save the batch to HDF5
                df_new.to_hdf(f'synthetic_dataset_batch_{idx}.h5', key='df', mode='w')

                unique_keys = df_new['__key__'].nunique()
                total_examples += unique_keys

                save_checkpoint(idx, total_examples)

        async def main():
            checkpoint = load_checkpoint()
            tar_index = checkpoint['tar_index']

            if tar_index != 0:
                tar_index += STEP_SIZE
                print(f"Resuming from tar file {tar_index}")

            total_examples = checkpoint['total_examples']
            processor = asyncio.create_task(process_dataset(tar_index, total_examples))
            await processor

            print("All batches processed.")

        asyncio.run(main())

launch()
