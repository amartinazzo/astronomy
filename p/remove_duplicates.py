import pandas as pd

preds_file = 'predictions/predictions-1129-0.csv'
preds_file_agg = 'predictions/predictions-1129-0-agg.csv'

preds_df = pd.read_csv(preds_file)
grouped = preds_df.groupby('object_id', as_index=False).size()
print('duplicated objects: {}'.format(grouped[grouped>1].count()))

preds_df = preds_df.groupby('object_id', as_index=False).mean()
grouped = preds_df.groupby('object_id', as_index=False).size()
print('duplicated objects after agg: {}'.format(grouped[grouped>1].count()))

preds_df.to_csv(preds_file_agg, index=False)