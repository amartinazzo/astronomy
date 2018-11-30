from train import *

# TODO files must be input arguments

checkpoint_file = 'models/model-1129-0.h5'
preds_file = 'predictions/predictions-1129-0.csv'

classes = ['class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65',
           'class_67','class_88','class_90','class_92','class_95']

# n_features = 79
# n_classes = 14
chunksize = 1e8
# 1e7 -> 45 chunks, 44 timeseries split
# 1e8 -> 5 chunks, 4 timeseries split


# MAIN
# make predictions
# takes around 25 min on deep02


model = build_model(n_features=n_features, n_classes=n_classes)
model.load_weights(checkpoint_file)
print('model loaded')

df_test_metadata = pd.read_csv(test_meta_file)
cols = ['object_id']+classes+['class_99']

print('\n\nbeginning predictions chunk by chunk')
start = time.perf_counter()
for i, df_chunk in enumerate(pd.read_csv(test_file, chunksize=chunksize, iterator=True)):
    df_pred, object_ids = pre_process_df(df_chunk, df_test_metadata, mode='test')
    
    X_pred = get_input(df_pred)
    
    preds = model.predict_proba(X_pred)
    preds_99 = np.ones(preds.shape[0])

    for j in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, j])

    preds_df = pd.DataFrame(preds, columns=classes)
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99)
    preds_df['object_id'] = object_ids
    
    preds_df = preds_df[cols]
    #print(preds_df.loc[:3,:])
    
    header = i<1
    preds_df.to_csv(preds_file, header=header, mode='a', index=False)
    
    print('chunk {} finished after {} min'.format(i, np.round((time.perf_counter()-start)/60)))