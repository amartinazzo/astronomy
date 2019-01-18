import pandas as pd
import sys

# get catalog and img from args

if len(sys.argv)!=3:
	print('usage: python gen-debug-file.py catalog_file.csv image_file.png')
	exit()

catalog = sys.argv[1]
img = sys.argv[2]

df = pd.read_csv(catalog, header=None)
df = df[df[0]==img]
df.to_csv('debug.csv', header=None, index=False)