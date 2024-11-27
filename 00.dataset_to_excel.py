from datasets import load_from_disk
import pandas as pd
import re

dataset = load_from_disk('/home/jspark/projects/FUse/data/dataset_for_LLM3')
train_dataset = dataset['train']['text']
test_dataset = dataset['validation']['text']
print(len(train_dataset), len(test_dataset))

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
    
    # assistant의 예측 라벨만 추출
    predicted_labels_text = re.search(r'assistant<\|end_header_id\|>\n\n(.+?)<\|eot_id\|>', text)
    predicted_labels = predicted_labels_text.group(1).split(", ") if predicted_labels_text else []
    
    # 라벨 벡터 생성 (예측된 라벨에만 1 할당)
    label_vector = {label: 1 if label in predicted_labels else 0 for label in labels}
    label_vector["SMILES"] = smiles  # SMILES 추가
    return label_vector

# 모든 데이터 처리
train_parsed_data = [parse_data(text) for text in train_dataset]
test_parsed_data = [parse_data(text) for text in test_dataset]


# DataFrame 생성
train_df = pd.DataFrame(train_parsed_data)
test_df = pd.DataFrame(test_parsed_data)

# 결과 확인
train_df.shape
test_df.shape

train_df.to_excel('/home/jspark/projects/FUse/data/LLM_training.xlsx', index=False)
test_df.to_excel('/home/jspark/projects/FUse/data/LLM_test.xlsx', index=False)
