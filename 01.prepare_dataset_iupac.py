from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_excel('/home/jspark/projects/FUse/data/LLM_training_names.xlsx')
test_data = pd.read_excel('/home/jspark/projects/FUse/data/LLM_test_names.xlsx')

train_data = train_data[['iupac', 'Adhesive', 'Antistatic', 'Antioxidant', 'Binder', 'Biocide', 'Catalyst', 'Chelating agent', 'Chemical reaction regulator', 'Cleaning agent', 'Degradant', 'Deodorizer', 'Dispersing agent', 'Dye',
                'Emulsifier', 'film former', 'Flame retardant', 'Flavouring', 'Foamant', 'Fragrance', 'Hardener', 'Humectant', 'Pharmaceutical', 'Pigment', 'Plasticizer', 'Preservative', 'Processing aids', 'Softener conditioner',
                'Solubility enhancer', 'Solvent', 'Stabilizing agent', 'Surfactant', 'Thickening agent', 'UV stabilizer', 'Viscosity modifier', 'Wetting agent', 'pH regulating agent']]
test_data = test_data[['iupac', 'Adhesive', 'Antistatic', 'Antioxidant', 'Binder', 'Biocide', 'Catalyst', 'Chelating agent', 'Chemical reaction regulator', 'Cleaning agent', 'Degradant', 'Deodorizer', 'Dispersing agent', 'Dye',
                'Emulsifier', 'film former', 'Flame retardant', 'Flavouring', 'Foamant', 'Fragrance', 'Hardener', 'Humectant', 'Pharmaceutical', 'Pigment', 'Plasticizer', 'Preservative', 'Processing aids', 'Softener conditioner',
                'Solubility enhancer', 'Solvent', 'Stabilizing agent', 'Surfactant', 'Thickening agent', 'UV stabilizer', 'Viscosity modifier', 'Wetting agent', 'pH regulating agent']]

label_list = list(train_data.columns[1:])
label_list_str = ', '.join(label_list)

# 프롬프트 템플릿 설정
TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"


def convert_to_template_format(row):
    iupac = row['iupac']
    labels = [label for label in label_list if row[label] == 1]
    labels_str = ", ".join(labels) if labels else 'None'
    
    context = f"""You are an expert chemist with extensive knowledge of chemical functions. Based on the given chemical IUPAC name, list only the functions from the predefined list below.
Available functions: {label_list_str}."""
    question = f"Identify the functions of the following molecule with confidence.\nChemical: {iupac}\n"
    answer = labels_str
    
    return {"text" : TEMPLATE.format(context = context, question = question, answer = answer)}


# 데이터 프롬프트 템플릿 변환 적용
train_data = train_data.apply(convert_to_template_format, axis=1).tolist()
test_data = test_data.apply(convert_to_template_format, axis=1).tolist()

# 데이터셋 변환
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(test_data)

dataset = DatasetDict({
    'train' : train_dataset,
    'validation' : val_dataset
})

dataset.save_to_disk('/home/jspark/projects/FUse/data/dataset_for_LLM_iupac')
print(dataset['train'][0]['text'])