from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

train_dataset = load_dataset("snli", split="train").filter(lambda ex: ex['label'] != -1)
val_dataset = load_dataset("snli", split="validation").filter(lambda ex: ex['label'] != -1)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()