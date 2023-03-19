CORE_ARGS = --grid_size 200 150 \
	--num_hallways 6 \
	--min_spacing_hallways 25 \
	--hallway_width 5 \
	--room_width 10 \
	--room_length 10 20 \
	--seed 2004 \

.PHONY: help
help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Core Targets:'
	@echo 'help 				displays help message'
	@echo 'generate-map			plots new map'

.PHONY : generate-map
generate-map:
	@python3 hallway.py $(CORE_ARGS)
