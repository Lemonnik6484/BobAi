import discord
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sqlite3
from datetime import datetime
import os
import re

# =============================
# Bot setup
# =============================
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# =============================
# Model loading
# =============================
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on device: {device}")

# =============================
# Database setup
# =============================
def init_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (user_id INTEGER, timestamp TEXT, message TEXT, response TEXT)''')
    conn.commit()

    timestamp = datetime.now().isoformat()
    separator = "=============================="
    c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
              (-1, timestamp, separator, separator))
    conn.commit()
    
    c.execute('''CREATE TABLE IF NOT EXISTS custom_data
                 (message TEXT, response TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_conversation(user_id, message, response):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
              (user_id, timestamp, message, response))
    conn.commit()
    conn.close()

# =============================
# Response generation
# =============================
def generate_response(user_message):
    prompt = f"""
    User: {user_message}
    Bob:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_len = input_ids.shape[1]
    max_length = min(input_len + 80, 1024)
    if input_len + 80 >= 1024:
        print("[WARN] " + "limited to 1024 tokens. (" + str(input_len) + " + 80)")
    input_length = input_ids.shape[1]

    with torch.no_grad():
        response_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # true = sample from top k, false = greedy
            top_k=50, # options (variants to choose from)
            top_p=0.90, # choose percent (bigger = closer)
            temperature=0.95, # bigger = more random
            repetition_penalty=1.0, # less = more repetitive
        )

    new_tokens = response_ids[0, input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    response = re.sub(r"User:.*", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"Bob:\s*", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"BobBot.*", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"Bot:\s*", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"<@925509619403616256>", "Bob", response).strip()
    response = re.sub(r"<[@#!&]?\d+>", "", response).strip()

    if not response:
        simple_prompt = f"User: {user_message}\nBob:"
        inputs = tokenizer(simple_prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            response_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=min(input_ids.shape[1] + 50, 1024),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=1.1,
                repetition_penalty=1.05,
            )
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        response = response[len(simple_prompt):].strip()
        response = response.split('\n')[0].split('User:')[0].strip()

    return response if response else ""

# =============================
# Events
# =============================
@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if "rate-limited" in message.content:
        return

    if client.user in message.mentions or "bob" in message.content.lower():
        bot_member = message.guild.me if message.guild else client.user
        perms = message.channel.permissions_for(bot_member)
        if not perms.send_messages:
            print(f"[INFO] Cannot reply in #{message.channel}")
            return

        response = generate_response(message.content)
        if not response:
            return

        save_conversation(message.author.id, message.content, response)

        await message.reply(response)

# =============================
# Run bot
# =============================
def load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()

load_env_file()

client.run(os.getenv('TOKEN'))