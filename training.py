import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os
import re
import fcntl
import errno
import shutil
import tempfile
from convokit import Corpus, download
from datasets import load_dataset
from datetime import datetime

# -----------------------
# Config
# -----------------------
oversample_amount = 1
quotes_data_limit = 0
conversations_limit = 0
extra_data_limit = 0

# -----------------------
# Model configuration
# -----------------------
model_name = "microsoft/DialoGPT-medium"
models_path = "./models"
model_dirs = sorted(
    [d for d in os.listdir(models_path) if d.startswith("bob_")],
    key=lambda x: os.path.getmtime(os.path.join(models_path, x)),
    reverse=True
)

if model_dirs:
    latest_model_dir = os.path.join(models_path, model_dirs[0])
    print("Latest model:", latest_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(latest_model_dir)
    model = AutoModelForCausalLM.from_pretrained(latest_model_dir)
    print("Loaded local model")
else:
    print("Local model not found, using base model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

EXTERNAL_CORPORA = [
]

HF_DATASETS = [
]

# -----------------------
# Extra data collection
# -----------------------
def get_extra_data(limit=500):
    examples = []
    # ConvoKit
    for corpus_name in EXTERNAL_CORPORA:
        try:
            print(f"Loading convokit corpus: {corpus_name}")
            corpus = Corpus(filename=download(corpus_name))
            count = 0
            for convo in corpus.iter_conversations():
                utts = list(convo.iter_utterances())
                for i in range(len(utts) - 1):
                    context = str(utts[i].text or "").strip()
                    response = str(utts[i + 1].text or "").strip()
                    if context and response:
                        examples.append((context, response))
                        count += 1
                        if count >= limit:
                            break
                if count >= limit:
                    break
            print(f"  → Retrieved {count} examples from {corpus_name}")
        except Exception as e:
            print(f"  !! Failed to load {corpus_name}: {str(e)}")

    # Hugging Face
    for dataset_name in HF_DATASETS:
        print(f"Loading Hugging Face dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, split="train")
            count = 0
            for sample in dataset:
                if "context" in sample and "response" in sample:
                    context = str(sample["context"]).strip()
                    response = str(sample["response"]).strip()
                elif "instruction" in sample and "output" in sample:
                    context = sample["instruction"].strip()
                    if sample.get("input"):
                        context += "\n" + sample["input"].strip()
                    response = sample["output"].strip()
                else:
                    continue

                if context and response:
                    examples.append((context, response))
                    count += 1
                if count >= limit:
                    break

            print(f"  → Retrieved {count} examples from {dataset_name}")
        except Exception as e:
            print(f"  !! Failed to load {dataset_name}: {str(e)}")

    return examples

# -----
# Lock
# -----
def acquire_lock(lock_file):
    try:
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except IOError as e:
        if e.errno == errno.EAGAIN:
            print("Another training process is already running. Exiting.")
            return None
        raise

def release_lock(lock_fd):
    if lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

# -----------------------
# Database helpers
# -----------------------
def create_training_db_copy():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name
    shutil.copy2('conversations.db', temp_db_path)
    return temp_db_path

def get_conversation_stats(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM conversations")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
    users = c.fetchone()[0]
    conn.close()
    return {"total_conversations": total, "unique_users": users}

def get_convs_data(db_path, limit=1000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT message, response FROM conversations WHERE user_id != -1 LIMIT ?", (limit,))
    results = c.fetchall()
    conn.close()
    return results

def get_quotes_data(db_path, limit=1000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT message, response FROM custom_data LIMIT ?", (limit,))
    results = c.fetchall()
    conn.close()
    return results

# -----------------------
# Quality filter
# -----------------------
def is_quality(response):
    if len(response.strip()) < 3:
        return False
    if len(response.strip()) > 512:
        return False
    if not re.match(r"^[A-Za-zА-Яа-я0-9 ,.!?;:()'\-\n]{3,}$", response.strip()):
        return False
    low_quality = [
        r"^\s*(ok|okay|maybe|k)\s*[.!?]*\s*$",
    ]
    response_lower = response.lower().strip()
    for pattern in low_quality:
        if re.match(pattern, response_lower):
            return False
    return True

# -----------------------
# Dataset
# -----------------------
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, conversations, max_length=256):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        input_text, target_text = self.conversations[idx]

        combined_text = f"User: {input_text}\nBot: {target_text}{tokenizer.eos_token}"

        encoding = tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = encoding["input_ids"].clone()
        input_len = len(tokenizer(f"User: {input_text}\n")["input_ids"])
        labels[:, :input_len] = -100  # mask user part

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

# -----------------------
# Fine-tuning
# -----------------------
def fine_tune_model():
    lock_fd = acquire_lock('./training.lock')
    if not lock_fd:
        return False

    try:
        print("Starting manual fine-tuning process...")

        temp_db_path = create_training_db_copy()
        stats = get_conversation_stats(temp_db_path)
        print(f"Database stats: {stats['total_conversations']} conversations, "
              f"{stats['unique_users']} users")

        training_data = get_convs_data(temp_db_path, limit=conversations_limit)
        extra_data = get_extra_data(limit=extra_data_limit)
        quotes_data = get_quotes_data(temp_db_path, limit=quotes_data_limit)
        print(f"Retrieved {len(quotes_data)} quotes")
        training_data.extend(extra_data)
        training_data.extend(quotes_data)
        print(f"Retrieved {len(training_data)} training examples")

        quality_data = []
        for msg, resp in training_data:
            if msg and resp and is_quality(resp):
                quality_data.append((msg, resp))
        print(f"After quality filtering: {len(quality_data)} examples")

        if len(quality_data) < 5:
            print("Not enough quality training examples (need at least 5)")
            return False

        quality_data = quality_data * oversample_amount
        print(f"After oversampling: {len(quality_data)} examples")

        dataset = ConversationDataset(tokenizer, quality_data)

        training_args = TrainingArguments(
            output_dir="./bob_tmp",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            save_steps=200,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=5e-6,
            warmup_steps=50,
            logging_steps=20,
            logging_dir="./logs",
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=lambda data: {
                'input_ids': torch.stack([f['input_ids'] for f in data]),
                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                'labels': torch.stack([f['labels'] for f in data])
            },
            train_dataset=dataset,
        )

        print("Starting training...")
        train_result = trainer.train()

        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
        model_dir = f"./models/bob_{timestamp}"

        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Training completed! Loss: {train_result.training_loss:.4f}")
        print(f"Model saved at {model_dir}")

        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        return True

    except Exception as e:
        print(f"Error during training: {str(e)}")
        if os.path.exists("./bob_tmp"):
            shutil.rmtree("./bob_tmp")
        if 'temp_db_path' in locals() and os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        return False
    finally:
        release_lock(lock_fd)

# ------
# Start
# ------
if __name__ == "__main__":
    print("=" * 50)
    print("Bob Discord Bot - Manual Training Script")
    print("=" * 50)

    if not os.path.exists('conversations.db'):
        print("Error: conversations.db not found.")
        exit(1)

    success = fine_tune_model()
    if success:
        print("\nTraining completed successfully!")
        print("The bot will use the fine-tuned model in its next responses.")
    else:
        print("\nTraining failed or skipped due to insufficient data.")

    print("=" * 50)
