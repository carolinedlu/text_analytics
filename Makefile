#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER=python3


ifeq (,$(shell which conda))
  HAS_CONDA=False
else
  HAS_CONDA=True
endif


#################################################################################
# COMMANDS             	                                                        #
#################################################################################


## Setup the projet
project-up:
	docker-compose -f docker-compose.yaml up

## Teardown the project
project-down:
	docker-compose -f docker-compose.yaml down --rmi all

## Push update to docker container
docker-push:
	docker build -t csanry/text:latest .
	docker login
	docker push csanry/text:latest

## Clean environment
clean:
	pre-commit run --all-files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete


gen_files: 
	python3 text_analytics/sentiment_analysis/bow_preprocessing.py
	python3 text_analytics/sentiment_analysis/split_dataset.py