import os
import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=os.path.dirname(os.path.realpath('__file__')),
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import torch
import hydra
import numpy as np
from omegaconf import DictConfig
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Internal package
from src.dataset.utils import *

class FewShotDataset(Dataset):
	'''
		Prepare the datasets of episodes for training, validation and test.
	'''
	def __init__(self, opt: DictConfig, transform=None, support_transform=None, mode='train', loader=RGB_loader):
		super().__init__()

		self.mode = mode
		self.transform = transform
		self.support_transform = support_transform
		self.loader = loader
		self.label_mode = opt.label_mode
		self.way_num = opt.way_num
		self.shot_num = opt.shot_num
		self.query_num = opt.query_num
		self.data_dir = opt.dataset_dir
		self.test_aug = opt.test_aug
		self.augmented_shot_num = opt.aug_shot_num


		assert (mode in ['train', 'val'])

		generate_train_and_val_dataframe(self.data_dir, self.label_mode)
		
		if mode == 'train':
			csv_path    = os.path.join( self.data_dir, 'train.csv')
			self.episode_num = opt.episode_train_num
		
		elif mode == 'val':
			csv_path    = os.path.join( self.data_dir, 'val.csv')
			self.episode_num = opt.episode_val_num

		else:
			raise ValueError('mode ought to be in [train/val]')


		# Construct the few-shot tasks (episodes)
		class_img_dict, class_list, class_to_idx = load_csv2dict(csv_path)
		self.data_list = episode_sampling(self.data_dir, class_list, class_img_dict, 
			self.episode_num, self.way_num, self.shot_num, self.query_num)

		print('Loading dataset -- mode {0}: {1} (Few-shot)'.format(mode, len(self.data_list)))


	def __len__(self):
		return len(self.data_list)


	def __getitem__(self, index):
		'''
			Load an episode for training and validation.          
		'''
		episode_files = self.data_list[index]

		query_images = []
		query_targets = []
		support_images = []
		support_targets = []
		augmented_support_images = []
		augmented_support_targets = []

		if self.test_aug and self.mode == 'test':
		
			for i in range(len(episode_files)):
				data_files = episode_files[i]

				# load query images
				query_dir = data_files['query_img']

				for j in range(len(query_dir)):
					temp_img = self.loader(query_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(temp_img)
					query_images.append(temp_img)


				# load support images
				temp_support = []
				temp_augmented_support = []
				support_dir = data_files['support_set']
				for j in range(len(support_dir)): 
					PIL_img = self.loader(support_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(PIL_img)
						temp_support.append(temp_img.unsqueeze(0))

					if self.support_transform is not None:
						for _ in range(self.augmented_shot_num):
							temp_img = self.support_transform(PIL_img)
							temp_augmented_support.append(temp_img.unsqueeze(0))

				temp_support = torch.cat(temp_support, 0)
				support_images.append(temp_support)
				temp_augmented_support = torch.cat(temp_augmented_support, 0)
				augmented_support_images.append(temp_augmented_support)


				# read the label
				target = data_files['target']
				query_targets.extend(np.tile(target, len(query_dir)))
				support_targets.extend(np.tile(target, len(support_dir)))
				augmented_support_targets.extend(np.tile(target, len(support_dir) * self.augmented_shot_num))

			 
			return (query_images, query_targets, support_images, support_targets, augmented_support_images, augmented_support_targets)

		else:

			for i in range(len(episode_files)):
				data_files = episode_files[i]

				# load query images
				query_dir = data_files['query_img']

				for j in range(len(query_dir)):
					temp_img = self.loader(query_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(temp_img)
					query_images.append(temp_img)


				# load support images
				temp_support = []
				support_dir = data_files['support_set']
				for j in range(len(support_dir)): 
					PIL_img = self.loader(support_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(PIL_img)
						temp_support.append(temp_img.unsqueeze(0))

				temp_support = torch.cat(temp_support, 0)
				support_images.append(temp_support)

				# read the label
				target = data_files['target']
				query_targets.extend(np.tile(target, len(query_dir)))
				support_targets.extend(np.tile(target, len(support_dir)))
			 
			return (query_images, query_targets, support_images, support_targets)

def get_dataloader(opt: DictConfig, modes):
	'''
		Obtain the data loader for training/val/test.
	'''
	loaders = []
	for mode in modes:

		transform = transforms.Compose([
				transforms.Resize((opt.imageSize, opt.imageSize)),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				# transforms.Normalize(mean=mean, std=std)
		])


		""" Use to generate additional support set"""
		support_transform = transforms.Compose([
				transforms.RandomResizedCrop(opt.imageSize),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				# transforms.Normalize(mean=mean, std=std)
			])
	
		dataset = FewShotDataset(opt, transform, support_transform, mode)


		if mode == 'train':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.episodeSize, shuffle=True,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		elif mode == 'val':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.episodeSize, shuffle=True,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		elif mode == 'test':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.testepisodeSize, shuffle=False,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		else:
			raise ValueError('Mode ought to be in [train, val, test]')

		loaders.append(loader)

	return loaders

if __name__=="__main__":
	@hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
	def main(cfg: DictConfig):
		train_loader, val_loader = get_dataloader(opt=cfg.train, modes=["train", "val"])
	main()