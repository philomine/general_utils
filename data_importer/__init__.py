import os

__all__ = ['input', 'output', 'raw']

def get_files():
    files = {}
    for subfolder in __all__:
        path = './data/' + subfolder + '/'
        subfolder_files = os.listdir(path)
        subfolder_files = [subfolder_file for subfolder_file in subfolder_files if subfolder_file[0] != '_']

        files[subfolder] = subfolder_files
    
    return files

def get_files_string():
    files = get_files()
    
    files_string = ''
    for subfolder in files.keys():
        files_string += subfolder + '\n'
        for subfolder_file in files[subfolder]:
            files_string += '\t' + subfolder_file + '\n'
    
    return files_string