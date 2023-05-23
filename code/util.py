import PyPDF2
import re
import os
import contextlib

def extract_annotations(page):
    """
    Extracts and yields underlined annotations from a given PDF page

    Parameters
    ----------
    page : PageObject
        A single page from a PDF as a PyPDF2 PageObject

    Yields
    ------
    annotation : DictionaryObject
        Underlined annotations from the given PDF page, if any exist

    Returns
    -------
    None
    """
    annotations = page.get('/Annots')
    if annotations:
        for annotation in annotations:
            subtype = annotation.get('/Subtype')
            if subtype and subtype.lower() == '/underline':
                yield annotation

def column_names(begin_page, end_page, filepath):
    """
    Gets the column names that match the specific regex from a PDF file

    Parameters
    ----------
    begin_page : int
        begin page in pdf to scrape from
    end_page : int
        begin page in pdf to scrape from
    filepath : str
        local filepath to the pdf

    Returns
    -------
    all_matches : list
        list of matching column names
    """
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        all_matches = []
        for page in reader.pages[begin_page - 1:end_page]:
            text = page.extract_text()
            matches = re.findall(r'([A-Za-z0-9]+)[ ]+Len:', text, re.IGNORECASE)
            all_matches += matches
    return all_matches
    

def get_bad_columns(filepath):
    """
    Gets the column names that match the specific regex from a PDF file

    Parameters
    ----------
    filepath
        local filepath to the pdf

    Returns
    -------
    bad_column_names : list
        list of all column names concatonated by calling column_names
    """

    bad_column_names = []
    # This gets all the column names for Stimulants
    bad_column_names += column_names(128, 139, filepath)
    # This gets all the drugs from cocaine, crack, heroin, hallucinogens
    bad_column_names += column_names(67, 92, filepath)
    # This gets all the drugs from pain relievers and tranquilizers
    bad_column_names += column_names(102, 127, filepath)
    # This gets all the sedative columns
    bad_column_names += column_names(140, 148, filepath)
    return bad_column_names
    
if __name__ == '__main__':
    get_bad_columns('../data/Data-Codebook.pdf')
    