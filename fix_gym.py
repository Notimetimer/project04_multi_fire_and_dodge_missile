import glob
import os

for f in glob.glob('**/*.py', recursive=True):
    try:
        with open(f, 'rb') as fp:
            raw = fp.read()
        
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue
            
            continue
            
        with open(f, 'r', encoding='utf-8', newline='') as fp:
            lines = fp.readlines()
            
        
        with open(f, 'w', encoding='utf-8', newline='') as fp:
            fp.writelines(newlines)
            
        print(f"Fixed {f}")
    except Exception as e:
        print(f"Error on {f}: {e}")
