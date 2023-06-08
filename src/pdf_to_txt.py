import argparse
import PyPDF2


# Create the parser
parser = argparse.ArgumentParser(description='Convert PDF to .txt.')

# Add arguments
parser.add_argument('pdf', type=str, help='File path to your PDF')
parser.add_argument('out', type=str, help='File path to your text output.')
parser.add_argument('--start_page', type=int, default=0, help='Start page number')
parser.add_argument('--end_page', type=int, default=None, help='End page number')


def parse_arguments():
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    input_file = args.pdf
    output_file = args.out
    start_page = args.start_page
    end_page = args.end_page

    return input_file, output_file, start_page, end_page


def extract_text_from_pdf(pdf_path, txt_path, start_page: int = None, end_page: int = None):
    with open(pdf_path, 'rb') as pdf_file, open(txt_path, 'w', encoding='utf-8') as txt_file:
        reader = PyPDF2.PdfReader(pdf_file)
        if end_page is None:
            end_page = len(reader.pages)
        for page_num in range(start_page,end_page):
            page = reader.pages[page_num]
            text = page.extract_text()
            txt_file.write(text)
            txt_file.write('\n\n')


def main():
    input, output, start_page, end_page = parse_arguments()

    # extract text
    extract_text_from_pdf(input, output, start_page, end_page)

if __name__ == "__main__":
     main()