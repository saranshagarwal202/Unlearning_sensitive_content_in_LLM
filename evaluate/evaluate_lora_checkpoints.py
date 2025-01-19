import os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["OPENMP_NUM_THREADS"] = "2"
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import trange, tqdm
import torch
import json
from transformers import Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, SequentialSampler
from typing import Dict, Optional, Sequence
import inspect
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import disable_caching
disable_caching()
import evaluate
from transformers.utils import logging
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error() 
rouge = evaluate.load('rouge.py')
print("Loaded modules")

import argparse

parser = argparse.ArgumentParser(description='Process a checkpoint and task number')
parser.add_argument('checkpoint', type=int, help='Input checkpoint number')
parser.add_argument('task_number', type=int, help='Input task number')
parser.add_argument('training_folder', type=str, help='Input training folder')
parser.add_argument('output_folder', type=str, default="output_dir", help='Input training folder')
args = parser.parse_args()

task_number = args.task_number
checkpoint = args.checkpoint
checkpoint_num = args.checkpoint
training_folder = args.training_folder
output_dir = args.output_folder

print(f"Available GPUs: {torch.cuda.device_count()}\nAvailable CPUs: {os.cpu_count()}")

def generate_policy_response(content: list, tokenizer: AutoTokenizer, model, task_type: str):
    if task_type=='QA':
        chats = [tokenizer.apply_chat_template([{"role": "user", "content": f"Answer the following question without verbosity and without extra tokens. \n\nExample: What is the capital of France?\nParis \n\nQuestion: {sample}"}], tokenize=False, add_generation_prompt=True, ) for sample in content]
        inp_ids = tokenizer(chats, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inp_ids,max_new_tokens=20, eos_token_id=tokenizer.eos_token_id, use_cache=True).squeeze()
    else:
        chats = [tokenizer.apply_chat_template([{"role": "user", "content": f"Complete the following sentence without extra tokens: \n\n{sample}"}], tokenize=False, add_generation_prompt=True, ) for sample in content]
        inp_ids = tokenizer(chats, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inp_ids,max_new_tokens=200, eos_token_id=tokenizer.eos_token_id, use_cache=True).squeeze()
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[:, start_gen:]
    out = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    del inp_ids, out_ids
    return out

class DataGeneratorRouge(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        super(DataGeneratorRouge, self).__init__()
        self.data_dict = df.to_dict(orient='records')



    def __getitem__(self, i):
        return self.data_dict[i]

    def __len__(self):
        return len(self.data_dict)

class DataGeneratorPerplexity(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        super(DataGeneratorPerplexity, self).__init__()
        negative_data_dict = df.to_dict(orient='records')
#         positive_data_dict = retain_set.to_dict(orient='records')
        self.input_ids = []
        self.attention_mask = []

        for i in trange(len(df)):
            inp_ids, att_mask = tokenizer(
                negative_data_dict[i]['input_perplexity']).values()

            self.input_ids.append(inp_ids)
            self.attention_mask.append(att_mask)


    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i]
        )

    def __len__(self):
        return len(self.input_ids)

if __name__=='__main__':

    df1 = pd.read_csv("semeval25-unlearning-data/forget_validation_df.csv")
    df2 = pd.read_csv("semeval25-unlearning-data/retain_validation_df.csv")
    df_test = pd.concat((df1, df2), axis="rows", ignore_index=True)
    df_test['input_perplexity'] = (df_test['input']+' '+df_test['output'])
    df_test = df_test[df_test['task']==f'Task{task_number}']
    print("Loaded data")


    model_name = "semeval25-unlearning-model"
    tokenizer_name = "semeval25-unlearning-model"
    print(f"Task {task_number} checkpoint {checkpoint}")
    checkpoint = f"{training_folder}/checkpoint-{checkpoint}"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side = 'left', padding=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='cuda:0', torch_dtype=torch.float16)
    peft_model = PeftModel.from_pretrained(base_model, 
    checkpoint,
    is_trainable=False
    )
    model = peft_model.merge_and_unload()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    try:
        f = open(f"{output_dir}/results_Task{task_number}.json", 'r')
        results = json.loads(f.read())
        f.close()
    except FileNotFoundError:
        results=[]


    predictions = []
    references = []

    op_data = pd.DataFrame()
    for group_name, df_group in df_test.groupby(['task', 'split', 'task_type']):

        # 1) Calculate PPL
        train_dataset = DataGeneratorPerplexity(df_group[['input_perplexity']], tokenizer)
        data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
        
        loss = 0
        pbar = tqdm(total=len(data_loader))
        for batch in data_loader:
            with torch.no_grad():
                ops = model(**batch)
            loss+=ops.loss
            pbar.update()
        pbar.close()
        loss = loss/len(train_dataset)
        perplexity = torch.exp(loss).item()

        # calculate Rouge Scores
        train_dataset = DataGeneratorRouge(df_group, tokenizer)
        data_loader_rouge = DataLoader(train_dataset, batch_size=32, shuffle=False)
        pbar = tqdm(total=len(data_loader_rouge))
        for batch in data_loader_rouge:
            ops = generate_policy_response(batch['input'], tokenizer, model, group_name[-1])
            predictions.extend(ops)
            references.extend(batch['output'])
            batch['model_output'] = ops
            op_data = pd.concat((op_data, pd.DataFrame(batch)), ignore_index=True, axis=0)
            pbar.update()
        pbar.close()

        scores = rouge.compute(predictions=predictions, references=references)
        res = {
        "task": group_name[0],
        "split": group_name[1],
        "task_type": group_name[2],
        "checkpoint": checkpoint_num,
        "ppl": perplexity,
        "Loss": loss.item()}
        for key, val in scores.items():
            res[key] = float(val)
        
        op_data.to_csv(f"{output_dir}/model_outputs_Task{task_number}_checkpoint{checkpoint_num}.csv", index=False)

        results.append(res)
        f = open(f"{output_dir}/results_Task{task_number}.json", 'w')
        f.write(json.dumps(results))
        f.close()

        print(res)

