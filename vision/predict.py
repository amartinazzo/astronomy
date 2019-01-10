from train import *
from keras_retinanet.models import convert_model
from keras_retinanet.utils.eval import evaluate


model_weights = 'model-190107-1539.h5'
val_file = 'catalog_val_small.csv'

print('loading model...')
model = load_model(model_weights, n_classes=2, anchor_params=anchor_params)
print('converting model...')
model = convert_model(model)

val_gen = CSVGenerator(val_file, classes_file)

print('evaluating...')

ap = evaluate(
	val_gen,
	model,
	iou_threshold=0.3,
	score_threshold=0.3,
	max_detections=100,
	save_path=model_weights.split('.')[0]+'eval'
	)

print(ap)