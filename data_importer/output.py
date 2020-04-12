from ._file_extractor import extractor as _extractor

_subfolder = 'output'
_imported = {}

def __getattr__(name):
    if name == '__path__':
        res = None 
    else:
        if name not in _imported.keys():
            _imported[name] = _extractor(_subfolder, name)
        res = _imported[name]
    return res