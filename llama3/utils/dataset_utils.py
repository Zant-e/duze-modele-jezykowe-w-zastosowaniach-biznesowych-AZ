import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import pandas as pd


def get_preprocessed_curious(tokenizer, dataset):
    """
    Preprocesses the xiyuez/im-feeling-curious dataset for training.

    Args:
        tokenizer: The tokenizer object.
        dataset: The dataset object.
    """

    def apply_template(sample):
        return {
            "question": prompt.format(question=sample["question"]),
            "answer": sample["answer"],
        }

    def tokenize(sample):
        question = tokenizer.encode(sample["question"], bos=True, eos=False)
        answer = tokenizer.encode(sample["answer"], bos=False, eos=True)

        return {
            "input_ids": question + answer,
            "loss_mask": [0] * len(question) + [1] * len(answer),
        }

    prompt = f"Question:\n{{question}}\n\Answer:\n"
    dataset = dataset.map(apply_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize, remove_columns=list(dataset.features))
    return dataset


def encode_header(tokenizer, message: dict) -> List[int]:
    tokens = []
    tokens.append(tokenizer.special_tokens["<|start_header_id|>"])
    tokens.extend(tokenizer.encode(message["role"], bos=False, eos=False))
    tokens.append(tokenizer.special_tokens["<|end_header_id|>"])
    tokens.extend(tokenizer.encode("\n\n", bos=False, eos=False))
    return tokens


def encode_message(tokenizer, message: dict) -> Tuple[List[int], List[int]]:
    header = encode_header(tokenizer, message)
    message_content = tokenizer.encode(message["content"].strip(), bos=False, eos=False)
    eot_token = tokenizer.special_tokens["<|eot_id|>"]
    tokens = header + message_content + [eot_token]

    if message["role"] == "user":
        loss_mask = [0] * len(tokens)
    else:
        loss_mask = [0] * len(header) + [1] * (len(message_content) + 1)

    return tokens, loss_mask


def encode_dialog_prompt(
    tokenizer, dialog: List[dict]
) -> List[Tuple[List[int], List[int]]]:
    """
    Encodes a dialog prompt.
    Will return an empty list if the prompt exceeds 700 tokens.
    """
    tokens = []
    loss_mask = []
    chunks = []
    tokens.append(tokenizer.special_tokens["<|begin_of_text|>"])
    loss_mask.append(0)
    for message in dialog:
        message_tokens, mask = encode_message(tokenizer, message)
        tokens.extend(message_tokens)
        loss_mask.extend(mask)
    if len(tokens) <= 700:
        chunks.append((tokens, loss_mask))
    return chunks


def get_preprocessed_cust_support(tokenizer, dataset):
    """
    Preprocesses the bitext/Bitext-customer-support-llm-chatbot-training-dataset dataset for training.

    Args:
        tokenizer: The tokenizer object.
        dataset: The dataset object.
    """

    def convert_to_dialog_format(row) -> List[dict]:
        return [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]

    df = dataset.to_pandas()
    df["conversation"] = df.apply(convert_to_dialog_format, axis=1)
    df = df["conversation"]

    rows = []
    for dialog in df:
        encoded_chunks = encode_dialog_prompt(tokenizer, dialog)
        if encoded_chunks != []:
            for input_ids, loss_mask in encoded_chunks:
                rows.append({"input_ids": input_ids, "loss_mask": loss_mask})

    df_result = pd.DataFrame(rows)
    dataset = ConversationDataset(df_result)
    return dataset


def get_preprocessed_pure_dove(tokenizer, dataset):
    """
    Preprocesses the LDJnr/Pure-Dove dataset for training.

    Args:
        tokenizer: The tokenizer object.
        dataset: The dataset object.
    """

    def convert_to_dialog_format(messages: List[dict]) -> List[dict]:
        dialog = []
        for message in messages:
            dialog.append({"role": "user", "content": message["input"]})
            dialog.append({"role": "assistant", "content": message["output"]})
        return dialog

    df = dataset.to_pandas()
    df.drop(columns=["source"], inplace=True)
    df["conversation"] = df["conversation"].apply(convert_to_dialog_format)

    rows = []
    for _, row in df.iterrows():
        dialog = row["conversation"]
        encoded_chunks = encode_dialog_prompt(tokenizer, dialog)
        if encoded_chunks != []:
            for input_ids, loss_mask in encoded_chunks:
                rows.append({"input_ids": input_ids, "loss_mask": loss_mask})

    df_result = pd.DataFrame(rows)
    dataset = ConversationDataset(df_result)
    return dataset


class ConversationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset.reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return {"input_ids": row["input_ids"], "loss_mask": row["loss_mask"]}

    def __len__(self):
        return len(self.dataset)


class ConcatDataset(Dataset):
    def __init__(self, dataset, seq_length=700):
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
