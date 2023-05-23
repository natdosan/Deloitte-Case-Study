import PyPDF2
import re

def extract_annotations(page):
    annotations = page.get('/Annots')
    if annotations:
        for annotation in annotations:
            subtype = annotation.get('/Subtype')
            if subtype and subtype.lower() == '/underline':
                yield annotation

def column_names(begin_page, end_page, filepath):
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        all_matches = []
        for page in reader.pages[begin_page - 1:end_page]:
            text = page.extract_text()
            matches = re.findall(r'([A-Za-z0-9]+)[ ]+Len:', text, re.IGNORECASE)
            print(matches)
            all_matches += matches
        return all_matches
    

def get_bad_columns(filepath):

    bad_column_names = []
    # This gets all the column names for Stimulants
    bad_column_names += column_names(128, 139, '../data/Data-Codebook.pdf')
    # This gets all the drugs from cocaine, crack, heroin, hallucinogens
    bad_column_names += column_names(67, 92, '../data/Data-Codebook.pdf')
    # This gets all the drugs from pain relievers and tranquilizers
    bad_column_names += column_names(102, 127, '../data/Data-Codebook.pdf')
    # This gets all the sedative columns
    bad_column_names += column_names(140, 148, '../data/Data-Codebook.pdf')
    return bad_column_names
    
if __name__ == '__main__':
    get_bad_columns('../data/Data-Codebook.pdf')
    