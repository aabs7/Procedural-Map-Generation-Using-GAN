.PHONY: help
help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Core Targets:'
	@echo 'help 				displays help message'
	@echo 'run 					plots new map'


VENV := venv
VENV_PYTHON = $(VENV)/bin/python3
VENV_PIP = $(VENV)/bin/pip

CORE_ARGS = --grid_size 200 150 \
	--num_hallways 6 \
	--min_spacing_hallways 25 \
	--hallway_width 5 \
	--room_width 10 \
	--room_length 10 20 \
	--seed 2004 \

.PHONY: build
build: requirements.txt
	@python3 -m venv $(VENV)
	@$(VENV_PIP) install -r requirements.txt

.PHONY: test
test: build 
	@. venv/bin/activate && pytest -svk tests

.PHONY: run
run: build 
	@$(VENV_PYTHON) main.py

.PHONY: clean
clean: 
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf $(VENV)

# .PHONY : generate-map
# generate-map:
# 	. /home/venv/ml/bin/activate && @python3 core/hallway.py $(CORE_ARGS)


.PHONY: test
	pytest -v
