import os
import shutil
import subprocess
import sys

DATA_SETS_FOLDER = 'MNIST_data'
DATA_SETS_URL_PATTERN = 'http://yann.lecun.com/exdb/mnist/%s'
DATA_SETS_FILE_NAMES = ['train-images-idx3-ubyte.gz',
                        'train-labels-idx1-ubyte.gz',
                        't10k-images-idx3-ubyte.gz',
                        't10k-labels-idx1-ubyte.gz']


def main():
  if os.path.exists(DATA_SETS_FOLDER):
    print >> sys.stderr, 'WARNING: folder %s already exists.' % DATA_SETS_FOLDER
    while True:
      response = raw_input(
          'Do you want to delete this folder and redownload the data sets? (y/n) ')
      if response == 'n':
        sys.exit('Aborted by the user.')
      if response == 'y':
        break
    shutil.rmtree(DATA_SETS_FOLDER)
  os.makedirs(DATA_SETS_FOLDER)

  for file_name in DATA_SETS_FILE_NAMES:
    print >> sys.stderr, 'Downloading %s...' % file_name
    rc = subprocess.call(['wget', DATA_SETS_URL_PATTERN % file_name, '-o',
                          '%s/%s' % (DATA_SETS_FOLDER, file_name)])
    if rc != 0:
      sys.exit('Failed to download %s' % file_name)


if __name__ == '__main__':
  main()
