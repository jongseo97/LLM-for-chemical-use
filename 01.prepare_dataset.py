from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel('/home/jspark/projects/FUse/data/20240613_100_processed_data.xlsx')
data.columns = ['SMILES', 'Adhesive', 'Antistatic', 'Antioxidant', 'Binder', 'Biocide', 'Catalyst', 'Chelating agent', 'Chemical reaction regulator', 'Cleaning agent', 'Degradant', 'Deodorizer', 'Dispersing agent', 'Dye',
                'Emulsifier', 'film former', 'Flame retardant', 'Flavouring', 'Foamant', 'Fragrance', 'Hardener', 'Humectant', 'Pharmaceutical', 'Pigment', 'Plasticizer', 'Preservative', 'Processing aids', 'Softener conditioner',
                'Solubility enhancer', 'Solvent', 'Stabilizing agent', 'Surfactant', 'Thickening agent', 'UV stabilizer', 'Viscosity modifier', 'Wetting agent', 'pH regulating agent']
label_list = list(data.columns)[1:]

label_list_str = ', '.join(label_list)

# 프롬프트 템플릿 설정
TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"


def convert_to_template_format(row):
    smiles = row['SMILES']
    labels = [label for label in label_list if row[label] == 1]
    labels_str = ", ".join(labels) if labels else 'None'
    
    context = f"""You are an expert chemist with extensive knowledge of chemical properties and functions. Based on the given SMILES, respond with only the names of the relevant functions and uses from the predefined list below. Do not include any additional explanations or descriptions. Only select from this list, and do not invent new functions.
Available functions: {label_list_str}."""
    question = f"Identify the relevant functions and uses of the following molecule.\nSMILES: {smiles}\nWhich is the relevant functions and uses of the molecule?"
    answer = labels_str
    
    return {"text" : TEMPLATE.format(context = context, question = question, answer = answer)}


# 데이터 프롬프트 템플릿 변환 적용
data = data.apply(convert_to_template_format, axis=1).tolist()

# 데이터 분할
train_data, val_data = train_test_split(data, test_size = 0.2, random_state = 42)

# 데이터셋 변환
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

dataset = DatasetDict({
    'train' : train_dataset,
    'validation' : val_dataset
})

dataset.save_to_disk('/home/jspark/projects/FUse/data/dataset_for_LLM3')
print(dataset['train'][0]['text'])