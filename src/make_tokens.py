import argparse
from pathlib import Path
#import sys
#sys.path.insert(0,'./.env/lib/python3.8/site-packages')
#from tokenizers import ByteLevelBPETokenizer
#from transformers import RobertaTokenizer
import os


# Create the parser
parser = argparse.ArgumentParser(description='Tokenize text')

# Add arguments
parser.add_argument('input_dir', type=str, help='Directory with your input text files')
parser.add_argument('output_dir', type=str, help='Directory to save model')


def parse_arguments():
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_dir = args.input_dir
    output_dir = args.output_dir

    return input_dir, output_dir

def extract_file_paths(input_dir):
    paths = [str(x) for x in Path(input_dir).glob('**/*.txt')]
    return paths

def run(paths, output_dir):
    # intialise tokeniser
    tokenizer = ByteLevelBPETokenizer()
    # train
    tokenizer.train(files=paths,
                    vocab_size=30_522,
                    min_frequency=2,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

    tokenizer.save_model(output_dir)

def main():

    input_dir, output_dir = parse_arguments()

    # create directory for the text files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get file paths
    paths = extract_file_paths(input_dir)

    # now execute
    run(paths, output_dir)

if __name__ == "__main__":
     main()