# OLID Classification with mBERT-lite-base-p1 (Token already saved)

## Import Libraries
import pandas as pd
import numpy as np
import time 
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from transformers import TFBertForSequenceClassification

## Check for GPU
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected. Training will use GPU.")
else:
    print("⚠️ GPU not found. Training will run on CPU.")

## Load Tokenized Data
train_df = pd.read_csv("dataset/Sushil/TTVWE/no_emoji_train_mbert.csv")
val_df = pd.read_csv("dataset/Sushil/TTVWE/no_emoji_val_mbert.csv")
test_df = pd.read_csv("dataset/Sushil/TTVWE/no_emoji_test_mbert.csv")

X_train = train_df.drop(columns=["label"]).values
y_train = train_df["label"].values

X_val = val_df.drop(columns=["label"]).values
y_val = val_df["label"].values

X_test = test_df.drop(columns=["label"]).values
y_test = test_df["label"].values

## Convert to Tensorflow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_train, "attention_mask": X_train}, tf.convert_to_tensor(y_train, dtype=tf.float32)
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_val, "attention_mask": X_val}, tf.convert_to_tensor(y_val, dtype=tf.float32)
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_test, "attention_mask": X_test}, tf.convert_to_tensor(y_test, dtype=tf.float32)
))

## Model Initialization
model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=1
)

## Hyperparameter Tuning
batch_sizes = [16, 32]
learning_rates = [5e-5, 3e-5, 2e-5]
epoch_list = [2, 3, 4]

best_params = None
best_val_loss = float('inf')
loss_history = {}

for batch_size, lr, epochs in itertools.product(batch_sizes, learning_rates, epoch_list):
    print(f"\nRunning experiment with batch_size={batch_size}, learning_rate={lr}, epochs={epochs}")
    
    train_data = train_dataset.shuffle(len(X_train)).batch(batch_size)
    val_data = val_dataset.batch(batch_size)
    
    temp_model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=1
    )
    
    optimizer = Adam(learning_rate=lr)
    loss_fn = BinaryCrossentropy(from_logits=True)
    
    best_loss = float('inf')
    wait = 0
    patience = 3
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        total_loss = 0
        steps = 0
        
        for x_batch, y_batch in train_data:
            with tf.GradientTape() as tape:
                logits = temp_model(x_batch, training=True).logits
                loss = loss_fn(y_batch, logits)
            
            grads = tape.gradient(loss, temp_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, temp_model.trainable_variables))
            
            total_loss += loss
            steps += 1
            
            avg_train_loss = total_loss/steps
            train_losses.append(avg_train_loss.numpy())
            
            total_val_loss = 0
            for x_batch, y_batch in val_data:
                val_logits = temp_model(x_batch, training=False).logits
                val_loss = loss_fn(y_batch, val_logits)
                total_val_loss += val_loss
            
            avg_val_loss = total_val_loss / len(val_data)
            val_losses.append(avg_val_loss.numpy())
            
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")
            
            if avg_val_loss < best_loss - 1e-4:
                best_loss = avg_val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping...")
                    break
            
            loss_history[(batch_size, lr, epochs)] = {
                "train_loss": train_losses,
                "val_loss" : val_losses
            }
            
            if best_loss < best_val_loss:
                best_val_loss = best_loss
                best_params = (batch_size, lr, epochs)

## Save Loss History for Visualization & Plot Loss Curves
loss_df = []
for (batch_size, lr, epochs), losses in loss_history.items():
    for i, (train_l, val_l) in enumerate(zip(losses['train_loss'], losses['val_loss'])):
        loss_df.append({
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs_config": epochs,
            "epoch": i+1,
            "train_loss": train_l,
            "val_loss": val_l
        })

loss_df = pd.DataFrame(loss_df)
loss_df.to_csv("model/sushil_mbert_loss_history.csv", index=False)

## Final Training with Best Parameters
batch_size, lr, epochs = best_params
print(f"\nBest Params: batch={batch_size}, lr={lr}, epochs={epochs}")

train_data = train_dataset.shuffle(len(X_train)).batch(batch_size)
val_data = val_dataset.batch(batch_size)
test_data = test_dataset.batch(batch_size)

model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=1
)
optimizer = Adam(learning_rate=lr)
loss_fn = BinaryCrossentropy(from_logits=True)

wait = 0
best_loss = float('inf')

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    total_loss = 0
    steps = 0

    for x_batch, y_batch in train_data:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True).logits
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += loss
        steps += 1

    val_loss = 0
    for x_batch, y_batch in val_data:
        val_logits = model(x_batch, training=False).logits
        val_loss += loss_fn(y_batch, val_logits)

    print(f"Train Loss: {total_loss / steps:.4f} | Val Loss: {val_loss / len(val_data):.4f}")

    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= 3:
            print("Early stopping")
            break

## Save Trained Model
model.save_pretrained("model/sushil_mbert_no_emoji")

## Evaluation on Test Set
from sklearn.metrics import f1_score
y_true = []
y_pred = []

for x_batch, y_batch in test_data:
    logits = model(x_batch, training=False).logits
    y_true.extend(y_batch.numpy())
    y_pred.extend(tf.nn.sigmoid(logits).numpy())

y_pred_bin = np.where(np.array(y_pred) >= 0.5, 1, 0)

accuracy = accuracy_score(y_true, y_pred_bin)
precision = precision_score(y_true, y_pred_bin)
recall = recall_score(y_true, y_pred_bin)
conf_matrix = confusion_matrix(y_true, y_pred_bin)

print("\nFinal Evaluation")
print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
f1 = f1_score(y_true, y_pred_bin)
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not HS", "HS"], yticklabels=["Not HS", "HS"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()