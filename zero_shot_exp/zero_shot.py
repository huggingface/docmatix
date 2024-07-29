from datasets import Dataset, Features, Value, load_dataset, Image, Sequence

TEST_SUBSET_LEN = 200
TRAIN_SUBSET_LEN = 1700
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

test_subset = []
train_subset = []
for idx, sample in enumerate(ds['train']):
    if idx < TEST_SUBSET_LEN:
        test_subset.append(sample)
    if idx >= TEST_SUBSET_LEN - 1:
        if idx >= TEST_SUBSET_LEN + TRAIN_SUBSET_LEN - 1:
            break
        train_subset.append(sample)

new_test_data = Dataset.from_list(test_subset, features=FEATURES)
new_train_data = Dataset.from_list(train_subset, features=FEATURES)
new_test_data.push_to_hub('HuggingFaceM4/Docmatix', 'zero-shot-exp', split='test')
new_train_data.push_to_hub('HuggingFaceM4/Docmatix', 'zero-shot-exp', split='train')
