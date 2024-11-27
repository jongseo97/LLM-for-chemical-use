import pubchempy
import pandas as pd
from tqdm import tqdm

df_train = pd.read_excel('/home/jspark/projects/FUse/data/LLM_training_names.xlsx')
df_test = pd.read_excel('/home/jspark/projects/FUse/data/LLM_test.xlsx')

train_names = list(df_train['iupac'][:3700])
test_names = []

if True:
    for smiles in tqdm(df_train['SMILES'][3700:]):
        compounds = pubchempy.get_compounds(smiles, namespace = 'smiles')
        match = compounds[0]
        train_names.append(match.iupac_name)
        if len(train_names) % 100 == 0:
            now_names = train_names.copy()
            while len(now_names) != df_train.shape[0]:
                now_names.append('')
            df_train['iupac'] = now_names
            df_train.to_excel('/home/jspark/projects/FUse/data/LLM_training_names.xlsx')

    for smiles in tqdm(df_test['SMILES']):
        compounds = pubchempy.get_compounds(smiles, namespace = 'smiles')
        match = compounds[0]
        test_names.append(match.iupac_name)
        if len(test_names) % 100 == 0:
            now_names = test_names.copy()
            while len(now_names) != df_test.shape[0]:
                now_names.append('')
            df_test['iupac'] = now_names
            df_test.to_excel('/home/jspark/projects/FUse/data/LLM_test_names.xlsx')

    df_train['iupac'] = train_names
    df_test['iupac'] = test_names

    df_train.to_excel('/home/jspark/projects/FUse/data/LLM_training_names.xlsx')
    df_test.to_excel('/home/jspark/projects/FUse/data/LLM_test_names.xlsx')

