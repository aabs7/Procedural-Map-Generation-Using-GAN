.PHONY: help
help::
	@echo ''
	@echo 'DCGAN Targets:'
	@echo 'train-mnist			trains DCGAN on MNIST dataset'
	@echo 'train-celebrity		trains DCGAN on celebrity dataset'
	@echo 'train-floor			trains DCGAN on floor dataset'
	@echo 'train-all			trains DCGAN on all datasets'

.PHONY: train-mnist
train-mnist:
	@$(VENV_PYTHON) GAN/DCGAN/train.py --use_data mnist


unzip-celebrity-dataset = resources/dataset/celeb_dataset
$(unzip-celebrity-dataset):
	@mkdir -p resources/dataset/celeb_dataset
	unzip freeze_files/zip_dataset/celeb_dataset.zip -d resources/dataset/celeb_dataset
.PHONY: train-celebrity
train-celebrity: $(unzip-celebrity-dataset)
train-celebrity:
	@$(VENV_PYTHON) GAN/DCGAN/train.py --use_data celebrity


unzip-floor-dataset = resources/dataset/floor_dataset
$(unzip-floor-dataset):
	@mkdir -p resources/dataset/
	unzip freeze_files/zip_dataset/floor_dataset.zip -d resources/dataset/
.PHONY: train-floor
train-floor: $(unzip-floor-dataset)
train-floor:
	@$(VENV_PYTHON) GAN/DCGAN/train.py --use_data floor


.PHONY: train-all
train-all: train-mnist train-celebrity

.PHONY: visualize-data
visualize-data:
	@tensorboard --logdir resources/logs/