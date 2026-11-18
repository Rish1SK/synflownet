import pandas as pd
from tdc.single_pred import ADME

names = ['CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C19_Veith',
         'CYP2D6_Veith', 'CYP3A4_Veith']

frames = []
for n in names:
    data = ADME(name=n)               
    df = data.get_data()                
    df['isoform'] = n.split('_')[0]
    frames.append(df)

all_cyp = pd.concat(frames, ignore_index=True)