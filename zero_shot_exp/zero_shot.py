from datasets import Dataset, Features, Value, load_dataset, Image, Sequence

SUBSET_LEN = 200
FEATURES = Features(
    {
        "images": Sequence(Image(decode=True)),
        "texts": [
            {
                "user": Value("string"),
                "assistant": Value("string"),
                "source": Value("string"),
            }
        ],
    }
)

ds = load_dataset("HuggingFaceM4/Docmatix", "images", streaming=True)

subset = []
for idx, sample in enumerate(ds['train']):
    subset.append(sample)
    if idx >= SUBSET_LEN - 1:
        break

new_data = Dataset.from_list(subset, features=FEATURES)
new_data.push_to_hub('HuggingFaceM4/Docmatix', 'zero-shot-exp')