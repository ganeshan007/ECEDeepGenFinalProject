from random import randrange
import evaluate
import nltk
import numpy as np
from transformers import DataCollatorForSeq2Seq
import json
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
from datasets import load_dataset
from datasets import concatenate_datasets
from utils import postprocess_text, preprocess_function
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from parrot import Parrot
import torch
import warnings



dataset = load_dataset('samsum')

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

sample = dataset['train'][randrange(len(dataset["train"]))]
print(f"dialogue: \n{sample['dialogue']}\n---------------")
print(f"summary: \n{sample['summary']}\n---------------")

# from transformers import AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration
# tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
# # model = BartForConditionalGeneration.from_pretrained("google/flan-t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
model = AutoModelForCausalLM.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")


tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")



tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")



# Metric
rouge = evaluate.load("rouge")
google_bleu = evaluate.load("google_bleu")



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    metric_dict = {}
    rogue_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    gleu_score = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rogue_score = {k: round(v * 100, 4) for k, v in rogue_score.items()}
    gleu_score = {k: round(v * 100, 4) for k, v in gleu_score.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    metric_dict["gen_len"] = np.mean(prediction_lens)
    for k,v in rogue_score.items():
        metric_dict[k] = v
    for k,v in gleu_score.items():
        metric_dict[k] = v
    return metric_dict

def save_outputs(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    with open(f'/content/drive/MyDrive/ECE285-Project/preds.json','w') as f:
        json.dump(decoded_preds,f)

    metric_dict = {}
    rogue_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    gleu_score = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rogue_score = {k: round(v * 100, 4) for k, v in rogue_score.items()}
    gleu_score = {k: round(v * 100, 4) for k, v in gleu_score.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    metric_dict["gen_len"] = np.mean(prediction_lens)
    for k,v in rogue_score.items():
        metric_dict[k] = v
    for k,v in gleu_score.items():
        metric_dict[k] = v
    return metric_dict


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=4
)



# Hugging Face repository id
repository_id = f"/content/drive/MyDrive/ECE285-Project/T5-SAMSUM"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="wandb",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.evaluate()

trainer.train()
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained('T5_SAMSUM')

model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/ECE285-Project/T5-SAMSUM/checkpoint-921", device_map="auto")
test_args = Seq2SeqTrainingArguments(
    output_dir='T5-SAMSUM',
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=8,
    predict_with_generate=True)


trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics = save_outputs)

trainer.evaluate()

"""## Testing and Inference"""

import json
preds_file = '/content/drive/MyDrive/ECE285-Project/preds_bart.json'
with open(preds_file,'r') as f:
    preds = json.load(f)

dataset['test'][0]['dialogue']

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

model = model.to('cuda')
# tokenizer = tokenizer.to('cuda')

all_prompts = []
for i in range(len(dataset['test'])):
    conversation = dataset['test'][i]['dialogue']
    summary = preds[i]
    prompt =f'Given is a conversation {conversation} and its corresponding summary {summary}, please improve upon the summary. Please include the details from the conversation and make the summaries comprehensive'
    all_prompts.append(prompt)


batch_size = 32
total_items = len(dataset['test'])
new_summaries = []
for i in range(0, total_items, batch_size):
    batch_start = i
    batch_end = min(i + batch_size, total_items)
    prompts = all_prompts[batch_start:batch_end]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs)
    generated_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    new_summaries.append(generated_summaries)

import itertools
new_summaries = list(itertools.chain.from_iterable(new_summaries))
new_summaries

with open(f'/content/drive/MyDrive/ECE285-Project/all_summaries_bart_longer_prompt.json','w') as f:
    json.dump(new_summaries,f)

# Metric
rouge = evaluate.load("rouge")
google_bleu = evaluate.load("google_bleu")

def compute_scores(preds, labels):
    metric_dict = {}
    rogue_score = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    gleu_score = google_bleu.compute(predictions=preds, references=labels)
    rogue_score = {k: round(v * 100, 4) for k, v in rogue_score.items()}
    gleu_score = {k: round(v * 100, 4) for k, v in gleu_score.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    metric_dict["gen_len"] = np.mean(prediction_lens)
    for k,v in rogue_score.items():
        metric_dict[k] = v
    for k,v in gleu_score.items():
        metric_dict[k] = v
    return metric_dict

compute_scores(preds, dataset['test']['summary'])

compute_scores(new_summaries, dataset['test']['summary'])





warnings.filterwarnings("ignore")
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

for summary in preds[:10]:
  para_phrases = parrot.augment(input_phrase=summary, max_return_phrases = 1, adequacy_threshold = 0.70)
  print(para_phrases)

preds[:10]

parrot.augment(input_phrase='Can you recommed some upscale restaurants in Newyork?')

