import torch
from torch.utils.data import Dataset


def get_preprocessed_curious(tokenizer, dataset):
    prompt = f"Question:\n{{question}}\n\Answer:\n"

    def apply_template(sample):
        return {
            "question": prompt.format(question=sample["question"]),
            "answer": sample["answer"],
        }

    dataset = dataset.map(apply_template, remove_columns=list(dataset.features))

    def tokenize(sample):
        question = tokenizer.encode(sample["question"], bos=True, eos=False)
        answer = tokenizer.encode(sample["answer"], bos=False, eos=True)

        return {
            "input_ids": question + answer,
            "loss_mask": [0] * len(question) + [1] * len(answer),
        }

    dataset = dataset.map(tokenize, remove_columns=list(dataset.features))
    return dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, seq_length=8191):
        self.dataset = dataset
        self.seq_length = seq_length

        self.samples = []

        buffer = {
            "input_ids": [],
            "loss_mask": [],
        }

        for sample in self.dataset:
            buffer = {k: v + sample[k] for k, v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.seq_length:
                self.samples.append(
                    {k: v[: self.seq_length] for k, v in buffer.items()}
                )
                buffer = {k: v[self.seq_length :] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def collate_fn(batch, pad_id):
    input_ids = [sample["input_ids"] for sample in batch]
    loss_mask = [sample["loss_mask"] for sample in batch]

    max_len = max(len(ids) for ids in input_ids)
    input_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids]
    loss_mask = [mask + [0] * (max_len - len(mask)) for mask in loss_mask]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
    }
