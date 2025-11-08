import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
import gc

df = pd.read_csv('training.1600000.processed.noemoticon.csv', usecols=[5], encoding='latin1')
df.columns = ['original_text_column']

df['clean_text'] = df['original_text_column']  # No cleaning ;p

train_data, val_data = train_test_split(df['clean_text'], test_size=0.2, random_state=42)

# Use pretrained because easier
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add a proper pad token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")


def process_batch_efficient(tweet_batch, tokenizer, max_length=128):
    # Tokenize all tweets in the batch
    encoded = tokenizer(
        tweet_batch,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # For causal language modeling, inputs and labels are the same
    # but labels are shifted by one position
    labels = input_ids.clone()

    # Set padding tokens to -100 so they're ignored in loss calculation
    labels[attention_mask == 0] = -100

    return input_ids, labels, attention_mask


def train():
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate

    n_epochs = 4
    batch_size = 16
    max_length = 128  # sequence length
    batches_per_epoch = len(train_data) // batch_size // 4

    print(f"Training on {batches_per_epoch} batches per epoch")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for i in range(batches_per_epoch):
            try:
                start = i * batch_size
                end = min(start + batch_size, len(train_data))
                tweet_batch = train_data.iloc[start:end].tolist()

                # Process batch efficiently
                input_ids, labels, attention_mask = process_batch_efficient(
                    tweet_batch, tokenizer, max_length
                )

                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)

                # Forward pass with attention mask
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Memory cleanup every 10 batches
                if i % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {i}, skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

        avg_loss = total_loss / batches_per_epoch
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    print("Training complete!")

    # Save model
    model.save_pretrained("./my_twitter_bot_model")
    tokenizer.save_pretrained("./my_twitter_bot_model")
    print("Model saved!")

    # Test model (optional)
    # print("\nTesting generation...")
    # model.eval()
    # test_input = "I think"
    # input_ids = tokenizer.encode(test_input, return_tensors='pt').to(device)
    #
    # with torch.no_grad():
    #     outputs = model.generate(
    #         input_ids,
    #         max_length=50,
    #         num_return_sequences=1,
    #         temperature=0.8,
    #         do_sample=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #         eos_token_id=tokenizer.eos_token_id
    #     )
    #
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(f"Generated: {generated_text}")