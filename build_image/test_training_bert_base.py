'''
This Python script is used for fine-tuning a pre-trained language model (bert-base-uncased) on the MRPC dataset.
Credits : This awesome HF tutorial
          https://huggingface.co/learn/nlp-course/chapter3/3
'''


# Importing necessary modules from the datasets and transformers libraries.
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          DataCollatorWithPadding,
                          TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification)
import evaluate
import numpy as np


# Load the dataset the MRPC dataset and the bert-base-uncased tokenizer.
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
                
# The map method is used to apply the tokenize_function to the entire dataset.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# The compute_metrics function 
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# The TrainingArguments class is used to specify the training configuration.
# Note : the output_dir is set to /volume/model_name in order
#        to save the model in the volume directory rather than the container.
training_args = TrainingArguments("/volume/bert-base-uncased", 
                                  evaluation_strategy="epoch")

# We load the Sequence Classification Model bert-base-uncased from the checkpoint name.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# We instantiate a Trainer class to perform training.
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    )


# Start the training process
trainer.train()
