from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
import torch
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model loading
# model_path = '/home/jspark/projects/FUse/LLM/Llama-3_2-1B-Instruct'
model_path = '/home/jspark/projects/FUse/LLM/Llama-3_2-3B-Instruct'
# model_path = '/home/jspark/projects/FUse/LLM/Llama-3_1-8B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype= torch.bfloat16,
    device_map = 'auto'
)

# tokenizer loading & setting
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
model.config.pad_token_id = model.config.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type = 'CAUSAL_LM',
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.05,
    # target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
)

model = get_peft_model(model, lora_config)


# 데이터셋 loading
dataset = load_from_disk('/home/jspark/projects/FUse/data/dataset_for_LLM_iupac')


# # training arguments
# training_args = TrainingArguments(
#     output_dir = '/home/jspark/projects/FUse/LLM/script/output',
#     per_device_train_batch_size=8,
#     gradient_accumulation_steps=4,
#     warmup_steps = 5,
#     learning_rate = 2e-4,
#     logging_steps = 10,
#     optim = "adamw_torch",
#     weight_decay = 0.01,
#     lr_scheduler_type = "linear",
#     report_to="none",
#     num_train_epochs= 10,
# )

# # trl library의 SFTTrainer 사용
# trainer = SFTTrainer(
#     peft_config = lora_config,
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset['train'],
#     max_seq_length=512,
#     dataset_text_field = 'text',
#     dataset_num_proc = 2,
#     packing = False,
#     args=training_args,
# )

# training arguments
training_args = SFTConfig(
    dataset_text_field='text',
    max_seq_length=512,
    output_dir = '/home/jspark/projects/FUse/LLM/script/output',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    logging_steps=5,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='linear',
    num_train_epochs= 20,

)

# trl library의 SFTTrainer 사용
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    peft_config=lora_config,
    args=training_args
)


# 학습 시작
stats = trainer.train()

# 모델 저장 (lora)
save_path = '/home/jspark/projects/FUse/LLM/script/output/llama-3B-model_1122'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 모델 로딩 (lora)
model = PeftModel.from_pretrained(model, save_path, torch_dtype= torch.bfloat16, device_map = 'auto')
model = model.merge_and_unload()


TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
eval_dataset = []
eval_top_3_dataset = []
for data in dataset['validation']:
    temp_text = data['text']
    temp_text = temp_text.split('<|begin_of_text|><|start_header_id|>system<|end_header_id|>')[1]
    system_context = temp_text.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[0].strip()
    user_context = temp_text.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[1].strip().split('<|eot_id|>')[0].strip()
    top3_user_context = temp_text.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[1].strip().split('<|eot_id|>')[0].split('Which')[0].strip() + '\nWhich is the top-3 relevant functions and uses of the molecule?'
    assistant_context = temp_text.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip().split('<|eot_id|>')[0]
    
    temp_template = TEMPLATE.format(context = system_context, question = user_context)
    temp_template_top3 = TEMPLATE.format(context = system_context, question = top3_user_context)
    
    eval_dataset.append(temp_template)
    eval_top_3_dataset.append(temp_template_top3)
    

# preds = []
# from tqdm import tqdm
# for text in tqdm(eval_dataset):
#     tokenized_chat = tokenizer.encode(text, return_tensors='pt')
#     tokenized_chat = tokenized_chat.to(device)
#     generate_ids = model.generate(tokenized_chat, max_new_tokens=512)
#     pred = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
#     preds.append(pred)
    

i=141
prompt = eval_top_3_dataset[i]
prompt = eval_dataset[i]
print(prompt)
# structure_description = "This molecule is an ether compound with a central carbon chain that includes ethyl groups and a tertiary alcohol. The CCOC parts at both ends represent ethoxy groups, connected symmetrically around a carbon backbone. The CC(C)CCCC(C)(C)O section in the center denotes a long hydrocarbon chain with a tertiary alcohol (C-OH) at the middle carbon, flanked by methyl and ethyl groups. This structure, featuring ether linkages and a tertiary alcohol, is typical in compounds used for their solubility and branching properties."
# prompt = prompt.split('<|eot_id|><|start_header_id|>assistant')
# prompt = prompt[0] + structure_description + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
model.eval()
with torch.no_grad():
    tokenized_chat = tokenizer.encode(prompt, return_tensors="pt")
    tokenized_chat = tokenized_chat.to(device)
    generate_ids = model.generate(tokenized_chat, max_new_tokens=256, temperature=0.0, do_sample=False)#no_repeat_ngram_size = 2,
    print(tokenizer.decode(generate_ids[0], skip_special_tokens = True))
print(dataset['validation'][i]['text'])

