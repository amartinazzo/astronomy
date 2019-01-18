from train import *
from keras_retinanet.models import convert_model
from keras_retinanet.utils.eval import evaluate


model_weights = 'models/model-190114-1104.h5'
# model_weights = 'models/resnet50_coco_best_v2.1.0.h5'
val_file = 'catalog_val_small.csv'

print('loading model...')
model = load_model(model_weights, n_classes=2)

# for i, layer in enumerate(model.layers):
# 	print('{} {} {}'.format(str(i), str(layer.trainable), layer.name))

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