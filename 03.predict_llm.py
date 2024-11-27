from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import pandas as pd
import re
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model loading
model_path = '/home/jspark/projects/FUse/LLM/Llama-3_2-1B-Instruct'
model_path = '/home/jspark/projects/FUse/LLM/Llama-3_2-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype= torch.bfloat16,
    device_map = 'auto'
)

# tokenizer loading & setting
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model) # Quantization이 수행된 모델로 준비

# 모델 로딩 (lora)
model = PeftModel.from_pretrained(model, '/home/jspark/projects/FUse/LLM/script/output/llama-3B-model_1122/', torch_dtype= torch.bfloat16, device_map = 'auto')
model = model.merge_and_unload()

# dataset = load_from_disk('/home/jspark/projects/FUse/data/dataset_for_LLM3')
dataset = load_from_disk('/home/jspark/projects/FUse/data/dataset_for_LLM_iupac')

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

def make_prediction(prompt):
    tokenized_chat = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generate_ids=  model.generate(tokenized_chat, max_new_tokens=256, temperature=0.0, do_sample=False)
    output = tokenizer.decode(generate_ids[0])
    return output

# 샘플 예측
if True:
    i = 0
    prompt = eval_dataset[i]
    t3_prompt = eval_top_3_dataset[i]
    print('Pred : ', make_prediction(prompt).split('\n\n')[-1])
    print('Top 3 Pred : ', make_prediction(t3_prompt).split('\n\n')[-1])
    print('True : ', dataset['validation']['text'][i].split('\n\n')[-1])

# 전체 예측
predictions = []
top_3_predictions = []
model.eval()
for prompt, top_3_prompt in tqdm(zip(eval_dataset, eval_top_3_dataset)):
    with torch.no_grad():
        output = make_prediction(prompt)
        top_3_output = make_prediction(top_3_prompt)
        predictions.append(output)
        top_3_predictions.append(top_3_output)



# 라벨 리스트 정의
labels = [
    "Adhesive", "Antistatic", "Antioxidant", "Binder", "Biocide", "Catalyst", "Chelating agent",
    "Chemical reaction regulator", "Cleaning agent", "Degradant", "Deodorizer", "Dispersing agent",
    "Dye", "Emulsifier", "film former", "Flame retardant", "Flavouring", "Foamant", "Fragrance",
    "Hardener", "Humectant", "Pharmaceutical", "Pigment", "Plasticizer", "Preservative",
    "Processing aids", "Softener conditioner", "Solubility enhancer", "Solvent", "Stabilizing agent",
    "Surfactant", "Thickening agent", "UV stabilizer", "Viscosity modifier", "Wetting agent",
    "pH regulating agent"
]

# 텍스트 데이터에서 SMILES 및 예측된 라벨 추출 함수
def parse_data(text):
    # SMILES 추출
    smiles_match = re.search(r'SMILES: (\S+)', text)
    smiles = smiles_match.group(1) if smiles_match else None
    
    # 예측 라벨 추출: <|eot_id|> 유무와 관계없이 처리
    predicted_labels_text = re.search(r'assistant<\|end_header_id\|>\n\n(.+)', text)
    if predicted_labels_text:
        # ', '로 구분하여 라벨 추출
        predicted_labels = [label.strip() for label in predicted_labels_text.group(1).split("<|eot_id|>")[0].split(", ")]
    else:
        predicted_labels = []
    
    # 라벨 벡터 생성 (예측된 라벨에만 1 할당)
    label_vector = {label: 1 if label in predicted_labels else 0 for label in labels}
    label_vector["SMILES"] = smiles  # SMILES 추가
    return label_vector

# 모든 데이터 처리
predicted_data = [parse_data(text) for text in predictions]
top_3_predicted_data = [parse_data(text) for text in top_3_predictions]

predicted_data = pd.DataFrame(predicted_data)
top_3_predicted_data = pd.DataFrame(top_3_predicted_data)

predicted_data.to_excel('/home/jspark/projects/FUse/LLM/script/predictions/1122-3B/prediction.xlsx', index=False)
top_3_predicted_data.to_excel('/home/jspark/projects/FUse/LLM/script/predictions/1122-3B/top_3_prediction.xlsx', index=False)