import sys
import csv
from pathlib import Path
import argparse

def get_headers1(line):
    d = {}
    for h in ['Checked', 'Name', 'Calc. MW','RT [min]', 'Area (Max.)', 'MS2'] :
        d[h] = line.index(h)
    for h in line:
        if 'Area: ' in h:
            d[h] = line.index(h)
    return d

def get_headers2(line):
    d = {}
    for h in ['Checked', 'Calc. MW', 'RT [min]', 'FWHM [min]', 'Max. # MI', '# Adducts', 
              'Area (All Ions)', 'Study File ID']:
        d[h] = line.index(h)
    return d

def get_headers3(line):
    d = {}
    for h in ['Checked', 'Ion', 'Charge', 'Molecular Weight', 'm/z', 'RT [min]', 'FWHM [min]', 
              '# MI', 'Area', 'Parent Area [%]', 'Study File ID']:
        d[h] = line.index(h)
    return d

def get_unaligned_headers(line):
    d = {}
    for h in ['Checked', 'Ion', 'Charge', 'Molecular Weight', 'm/z', 'RT [min]',
              'FWHM [min]', '# MI', 'Area', 'Parent Area [%]', 'Study File ID'] :
        d[h] = line.index(h)
    return d  

def convert_cd33(filepath, is_aligned: bool, outfilename: str=None):
    filepath = Path(filepath)
    if outfilename is None:
        outfilepath = filepath.parent / (filepath.stem + '_CD33converted' + filepath.suffix)
    else:
        if '.csv' not in outfilename:
            outfilename += '.csv'
        outfilepath = filepath.parent / outfilename
    if is_aligned: 
        with open(str(filepath), encoding='utf8') as inf:
            with open(str(outfilepath), 'w', encoding='utf8', newline='') as out:
                r = csv.reader(inf)
                w = csv.writer(out)
                # Get indexes of headers
                first_line = next(r)
                h1 = get_headers1(first_line) 
                # Get headers2 and headers3
                for line in r:
                    if line[0] == '' and line[1] == 'Tags':
                        h2 = get_headers2(line)
                    if line[1] == '' and line[2] == 'Tags':
                        h3 = get_headers3(line)
                        break
                mod_h2 = list(h2.keys())
                for i, h in enumerate(mod_h2):
                    if h == 'Calc. MW':
                        mod_h2[i] = 'Molecular Weight'
                    if h == 'Area (All Ions)':
                        mod_h2[i] = 'Area'
                inf.seek(0)  # Reset reader position to top of file
                r = csv.reader(inf)  # Reset the csv reader
                for i, line in enumerate(r):
                    # Compound header
                    if 'Tags' in line[0]:
                        w.writerow([('Molecular Weight' if 'Calc. MW' in key else key) for key in h1.keys()] + [''])
                    # Compound data
                    elif line[0] == '' and line[1] == 'FALSE':
                        row = [line[i] for i in h1.values()] + ['']
                        row[list(h1.keys()).index('Name')] = ''  # Name field = Empty String because LipiDex parser doesn't like it
                        w.writerow(row)
                    # Compound per File header:
                    elif line[0] == '' and line[1] == 'Tags':
                        w.writerow([''] + mod_h2 + [''])
                    # Compound per File data:
                    elif line[1] == '' and line[2] == 'FALSE':
                        w.writerow([''] + [line[i] for i in h2.values()] + [''])
                    # Feature header:
                    elif line[1] == '' and line[2] == 'Tags':
                        w.writerow(['', ''] + list(h3.keys()) + [''])
                    # Feature data:
                    elif line[2] == '':
                        w.writerow(['', ''] + [line[i] for i in h3.values()] + [''])
                    else:
                        raise ValueError('line ' + str(i) + ' was not written\n' + str(line))
    else: 
        with open(str(filepath), encoding='utf8') as inf:
            with open(str(outfilepath), 'w', encoding='utf8', newline='') as out:
                r = csv.reader(inf)
                w = csv.writer(out)
                # Get indexes of headers
                first_line = next(r)
                header = get_unaligned_headers(first_line) 
                w.writerow(list(header.keys()) + [''])
                for line in r:
                    w.writerow([line[i] for i in header.values()] + [''])
        

if __name__ == "__main__":
	if sys.version_info[0] < 3 or sys.version_info[1] < 7:
		raise Exception('Must use Python version 3.7 or higher')
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', dest='filepath', type=str, 
						help='CD 3.3 .csv file to convert')
	aligned_flag = parser.add_mutually_exclusive_group()
	aligned_flag.add_argument('-a', '--aligned', dest='is_aligned', action='store_true')
	aligned_flag.add_argument('-u', '--unaligned', dest='is_aligned', action='store_false')
	parser.add_argument('-o', '--outfilename', dest='outfilename',
						help='(Optional) Filename for converted file')
	args = parser.parse_args(sys.argv[1:])
	
	convert_cd33(
		filepath=args.filepath, 
		is_aligned=args.is_aligned, 
		outfilename=args.outfilename)
	
	
	
	
	
	