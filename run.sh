#!/bin/bash

source config
create_dir "in"
clear_dir "out"

# # # # # # # # # # # # # # # # # #

uv sync
clear

# # # # # # # # # # # # # # # # # #

uv run main.py









# # # # # # # # # # # # # # # # # #
echo -en "\n\n"