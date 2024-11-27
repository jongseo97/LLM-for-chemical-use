import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, hamming_loss

targets = pd.read_excel('/home/jspark/projects/FUse/data/LLM_test.xlsx')
outputs = pd.read_excel('/home/jspark/projects/FUse/LLM/script/predictions/1116-3B/prediction.xlsx')
top_3_outputs = pd.read_excel('/home/jspark/projects/FUse/LLM/script/predictions/1116-3B/top_3_prediction.xlsx')
# outputs = pd.read_excel('/home/jspark/projects/FUse/LLM/script/predictions/1115-3B/best_prediction.xlsx')
# top_3_outputs = pd.read_excel('/home/jspark/projects/FUse/LLM/script/predictions/1115-3B/top_2_prediction.xlsx')

targets = targets.iloc[:,:-1]
outputs = outputs.iloc[:,:-1]
top_3_outputs = top_3_outputs.iloc[:,:-1]
targets = np.array(targets)

# outputs = top_3_outputs

print(f'macro f1 score is {f1_score(targets, outputs, average="macro")}')
print(f'micro f1 score is {f1_score(targets, outputs, average="micro")}')
print(f'weighted f1 score is {f1_score(targets, outputs, average="weighted")}')
print(f'accuracy score is {accuracy_score(targets, outputs)}')
print(f'hamming loss is {hamming_loss(targets, outputs)}')

summary = classification_report(targets, outputs, output_dict=True)
summary = pd.DataFrame.from_dict(summary).transpose()
print(summary)