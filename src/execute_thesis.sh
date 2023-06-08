#!/bin/bash

export PDF_PATH=/Users/christianbromley/Documents/Personal/PhD/Christian_Bromley_Final_Thesis_20211221_cover.pdf
export OUTPUT_PATH=/Users/christianbromley/Documents/Personal/PhD/Christian_Bromley_Final_Thesis_20211221.csv
export TEXT_PARSE_DIR=/Users/christianbromley/Documents/Personal/PhD/thesis_parsed
export MODELS_DIR=/Users/christianbromley/Documents/Projects/machine-learning-course/bert-transformer/models

python3 /Users/christianbromley/Documents/Projects/machine-learning-course/bert-transformer/src/pdf_to_txt.py ${PDF_PATH} ${OUTPUT_PATH} --start_page 13 --end_page 139
python3 /Users/christianbromley/Documents/Projects/machine-learning-course/bert-transformer/src/prepare_text_samples.py ${OUTPUT_PATH} ${TEXT_PARSE_DIR}
python3 /Users/christianbromley/Documents/Projects/machine-learning-course/bert-transformer/src/tokenize.py ${TEXT_PARSE_DIR} ${MODELS_DIR}
