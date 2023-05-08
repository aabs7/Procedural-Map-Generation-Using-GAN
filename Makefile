.PHONY: help
help::
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Core Targets:'
	@echo 'help 				displays help message'
	@echo 'build 				builds the virtual environment'
	@echo 'test 				runs all the tests'
	@echo 'save-map-plots 		saves the floor plan plots for specific seeds'

VENV := venv
VENV_PYTHON = $(VENV)/bin/python3
VENV_PIP = $(VENV)/bin/pip

CORE_MAP_ARGS = --grid_size 500 750 \
	--min_spacing_hallways 100 \
	--hallway_width 5 \
	--room_width 15 \
	--boundary_threshold 100 \
	--room_length 15 25 \
	--seed 2008 \

.PHONY: build
build: requirements.txt
	@python3 -m venv $(VENV)
	@$(VENV_PIP) install -r requirements.txt

.PHONY: test
test: build 
	@. venv/bin/activate && pytest -svk tests

.PHONY: run
run: build
	@mkdir -p resources/maps_image
	@mkdir -p resources/maps_txt
	@$(VENV_PYTHON) main.py $(CORE_MAP_ARGS)

.PHONY: clean
clean: 
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf $(VENV)

.PHONY: test
	pytest -v


# Save 10 floor plan each by running main.py for specific seeds
floor-plan-seeds = \
	$(shell for ii in $$(seq 10 20); do echo "resources/maps_image/floor_plan_h4_$${ii}.png"; done) \
	$(shell for ii in $$(seq 30 40); do echo "resources/maps_image/floor_plan_h5_$${ii}.png"; done) \
	$(shell for ii in $$(seq 50 60); do echo "resources/maps_image/floor_plan_h6_$${ii}.png"; done) 
$(floor-plan-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(floor-plan-seeds): hallways = $(shell echo $@ | grep -Eo '[0-9]+' | head -1)
$(floor-plan-seeds):
	@echo "Saving maps = $(seed)"
	@mkdir -p resources/maps_image
	@mkdir -p resources/maps_txt
	@$(VENV_PYTHON) main.py $(CORE_MAP_ARGS) \
			--seed $(seed) \
			--num_hallways $(hallways) \

.PHONY: save-map-plots
save-map-plots: $(floor-plan-seeds)

# import all the makefile from inside the modules
include GAN/DCGAN/Makefile