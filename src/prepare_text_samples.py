import argparse
import pandas as pd
import os
import spacy

# Create the parser
parser = argparse.ArgumentParser(description='Prepare text for tokenizer.')

# Add arguments
parser.add_argument('text_path', type=str, help='File path to your text')
parser.add_argument('output_path', type=str, help='File path to your text outputs.')

def parse_arguments():
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_file = args.text_path
    output_dir = args.output_path

    return input_file, output_dir

def read_text_process_write(input_file, output_dir):
    # read input data
    full_text = pd.read_csv(input_file, sep=',')

    print(full_text.shape)

    # loop through the data and chunk it up
    text_data = []
    file_count = 0

    for i, row in full_text.iterrows():
        sample = str(row['sentence']).replace('\n', '').replace('\t', '')
        #print(len(sample))
        text_data.append(sample)

        if len(text_data) == 10_000:
            # once we get to the 10K mark, save to file
            with open(f'{output_dir}/text_{file_count}.txt', 'w',
                      encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            print(i)
            text_data = []
            file_count += 1

    with open(f'{output_dir}/text_{file_count}.txt', 'w',
              encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))

def main():
    # parse arguments
    input_file, output_dir = parse_arguments()

    # create directory for the text files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    read_text_process_write(input_file, output_dir)


if __name__ == "__main__":
    main()
