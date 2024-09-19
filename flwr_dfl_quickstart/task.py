"""flwr-dfl-quickstart: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from datasets import Dataset
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from quickstart_docker.util import download_JSON_to_dataframe

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, model_name: str):
    """Load training data (training and eval)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        df = download_JSON_to_dataframe(
            "bs-llm-sandbox",
            "keenanh/training_data.jsonl",
            aws_access_key_id,
            aws_secret_access_key,
            aws_region,
        )
        raw_datasets = (
            Dataset()
            .from_pandas(df)
            .train_test_split(test_size=0.2, seed=42)  # Set seed for reproducibility
        )
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset
        fds = partitioner
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()  # Shifted labels for causal language modeling
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = net(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader, device):
    metric = load_metric("accuracy")  # Or another relevant metric for your task
    loss = 0
    net.eval()
    for batch in testloader:
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()  # Shifted labels for causal language modeling
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = net(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        loss += outputs.loss.item()
        # Predictions are made based on the last token of the output
        predictions = torch.argmax(logits[:, -1, :], dim=-1)
        metric.add_batch(
            predictions=predictions, references=labels[:, -1]
        )  # Compare with the actual next token
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
