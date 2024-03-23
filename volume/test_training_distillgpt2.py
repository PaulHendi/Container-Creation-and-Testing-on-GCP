'''
This Python script is used for fine-tuning a pre-trained language model (distilgpt2) on the wikitext-2-raw-v1 dataset.
Credits : This awesome tutorial by Sylvain Gugger 
          https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
'''

#Importing necessary modules from the datasets and transformers libraries. 
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          TrainingArguments,
                          Trainer,
                          AutoModelForCausalLM)


# Prepare the 'wikitext-2-raw-v1'dataset 
# The AutoTokenizer.from_pretrained method is used to load a pre-trained tokenizer from the 'distilgpt2' checkpoint. 
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"])

# The map method applies the tokenize_function to the entire dataset.
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])


block_size = 128

# The group_texts function is defined to group the tokenized texts into chunks of a specified size (block_size).
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# The map method applies the group_texts function to the tokenized dataset.
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# We load the Causal Language Model distillGPT2 from the checkpoint name.
model = AutoModelForCausalLM.from_pretrained(checkpoint)


# The TrainingArguments class specifies the training configuration.
# Note : the output_dir is set to /volume/model_name in order
#        to save the model in the volume directory rather than the container.
model_name = checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"/volume/{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01
)


# The Trainer class is then used to create a trainer with the specified model, training arguments, and datasets.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# Finally, the train method is used to start the training process.
trainer.train()

