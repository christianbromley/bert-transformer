#!/bin/bash

export PDF_PATH=/Users/christianbromley/Documents/Personal/PhD/Christian_Bromley_Final_Thesis_20211221_cover.pdf
export OUTPUT_PATH=/Users/christianbromley/Documents/Personal/PhD/Christian_Bromley_Final_Thesis_20211221.txt

python3 /Users/christianbromley/Documents/Projects/machine-learning-course/bert-transformer/src/pdf_to_txt.py ${PDF_PATH} ${OUTPUT_PATH} --start_page 13 --end_page 139
