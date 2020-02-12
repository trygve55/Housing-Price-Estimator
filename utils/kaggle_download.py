''' Kaggle Download
This script is used to set up the initial directory structure and
download and extract the necessary resource files from Kaggle.

If you're unable to download the Kaggle files, please see
https://github.com/Kaggle/kaggle-api for install guides.
'''

import shutil
import zipfile
from os import mkdir, chdir, listdir
from os.path import exists, isfile, join
from subprocess import Popen, PIPE, STDOUT


def preliminary():
    # Create input directory
    if (exists('input') == False):
        mkdir('input')
    
    # Set directory root for download of data sources
    chdir('input')

    # Download all data source files from Kaggle
    input_cmd = "kaggle competitions download -c house-prices-advanced-regression-techniques"
    proc = Popen(input_cmd, stdin = PIPE, stdout = PIPE, shell=True)
    stdout, stderr = proc.communicate(bytes(input_cmd, 'utf-8'))
    print(stdout)

    # Create models directory
    if (exists('../models') == False):
        mkdir('../models')
    
    if (exists('../submissions') == False):
        mkdir('../submissions')
    

if __name__ == '__main__':
    preliminary()
