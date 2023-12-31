import os
import csv
import random
from PIL import Image

import os
import glob
import pyrootutils
import pandas as pd

ROOT = pyrootutils.setup_root(
    search_from=os.path.dirname(os.path.realpath('__file__')),
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

def episode_sampling(data_dir, class_list, class_img_dict, episode_num, way_num=5, shot_num=5, query_num=15):
	'''
		Random constructing episodes from the dataset
		episode_num: the total number of episodes 
	'''
	data_list = []
	for e in range(episode_num):

		# construct each episode
		episode = []
		temp_list = random.sample(class_list, way_num)
		label_num = -1 

		for item in temp_list:
			label_num += 1
			imgs_set = class_img_dict[item]
			support_imgs = random.sample(imgs_set, shot_num)
			query_imgs = [val for val in imgs_set if val not in support_imgs]

			if query_num < len(query_imgs):
				query_imgs = random.sample(query_imgs, query_num)

			# the dir of support set
			query_dir = [i for i in query_imgs]
			support_dir = [i for i in support_imgs]

			data_files = {
				"query_img": query_dir,
				"support_set": support_dir,
				"target": label_num
			}
			episode.append(data_files)
		data_list.append(episode)

	return data_list

def RGB_loader(path):
	return Image.open(path).convert('RGB')

def find_classes(dir):
	classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}

	return classes, class_to_idx


def load_csv2dict(csv_path):
	class_img_dict = {}
	with open(csv_path) as csv_file:
		csv_context = csv.reader(csv_file, delimiter=',')
		for line in csv_context:
			if csv_context.line_num == 1:
				continue
			img_name, img_class = line

			if img_class in class_img_dict:
				class_img_dict[img_class].append(img_name)
			else:
				class_img_dict[img_class] = []
				class_img_dict[img_class].append(img_name)

	csv_file.close()
	class_list = list(class_img_dict.keys())
	class_list.sort()
	class_to_idx = {class_list[i]: i for i in range(len(class_list))}

	return class_img_dict, class_list, class_to_idx

def generate_train_and_val_dataframe(data_path, label_mode):
	def generate_dataframe(data_path):
		data = {
			"filename": [],
			"label": []
		}
		for path in glob.glob(data_path):
			splitted = path.split("/")
			splitted = splitted[-1].split("_")
			data["filename"].append(path)
			if label_mode == "make":
				label = splitted[2]
			elif label_mode == "model":
				label = splitted[3]
			elif label_mode == "color":
				label = splitted[4]
			elif label_mode == "make_model":
				label = "_".join(splitted[2:4])
			elif label_mode == "make_model_color":
				label = "_".join(splitted[2:5])
			data['label'].append(label)
		return pd.DataFrame(data)
	
	data_path = f"{ROOT}/{data_path}"
	train_path = f"{data_path}/train/*/*.jpg"
	val_path = f"{data_path}/val/*/*.jpg"
	df_train = generate_dataframe(train_path)
	df_val = generate_dataframe(val_path)
	df_train.to_csv(f"{ROOT}/dataset/train.csv", index=False)
	df_val.to_csv(f"{ROOT}/dataset/val.csv", index=False)