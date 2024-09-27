import random
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AdamW
from evaluate import load as load_metric
import flwr as fl
from logging import INFO, DEBUG
from flwr.common.logger import log
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.2"

fl.common.logger.configure(identifier="flowerIngredMistral", filename="log.txt")


def load_data():
    log(INFO, "Loading JSON data...")
    """Load data (training and eval) and prepare data loaders."""
    # Load data from JSONL file and split into training and testing sets
    raw_datasets = (
        Dataset()
        .from_pandas(pd.read_json("data/training_data.jsonl", lines=True))
        .train_test_split(test_size=0.2, seed=42)  # Set seed for reproducibility
    )

    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        """Tokenize the prompt and completion in each example."""
        # Apply chat template and tokenize the prompt and completion
        examples["prompt"] = tokenizer.apply_chat_template(
            [examples["messages"][0]], tokenize=True
        )
        examples["completion"] = tokenizer.apply_chat_template(
            [examples["messages"][1]], tokenize=True
        )
        return examples

    # Tokenize the datasets
    log(INFO, "Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=False)

    # Create a data collator with padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create data loaders
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )
    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )
    log(INFO, "Data loaded.")
    return trainloader, testloader


def train(net, trainloader, epochs):
    log(INFO, "Training started for 1 Epoch...")
    optimizer = AdamW(net.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    net.train()
    for _ in tqdm(range(epochs)):
        total_loss = 0
        for batch in tqdm(trainloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch["completion"]  # Get the true labels (next tokens)
            inputs = batch["prompt"]  # Get the input sequence (context)
            outputs = net(inputs, labels=labels)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        log(INFO, f"Epoch 1, Loss: {total_loss/len(trainloader):.4f}")
    log(INFO, "Training finished.")


def test(net, testloader):
    log(INFO, "Testing started...")
    net.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            inputs = batch["prompt"]  # Get the input sequence (context)
            labels = batch["completion"]  # Get the true labels (next tokens)
            outputs = net(inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=-1)
            total_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_tokens += labels.size(0)
    avg_loss = total_loss / len(testloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    accuracy = total_correct / total_tokens
    log(INFO, f"Training Finished. Average Loss {avg_loss:.4f}")
    log(INFO, f"Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.4f}")
    return perplexity, accuracy


from transformers import AutoModelForCausalLM

net = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(DEVICE)
from collections import OrderedDict
import flwr as fl


class MistralClient(fl.client.NumPyClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainloader, self.testloader = load_data()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


fl.client.start_client(
    server_address=os.environ.get("FL_SERVER_ADDRESS", "localhost:8080"),
    client=MistralClient.to_client(),
    root_certificates=Path("./certificates/ca.crt").read_bytes(),
)


# Start with SuperNode
"""
flower-client-app client:app
    --root-certificates certificates/ca.crt
    --superlink 127.0.0.1:9092
"""
