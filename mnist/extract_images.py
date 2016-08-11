import argparse
import gzip
import inspect
import os
import shutil
import struct
import sys
from PIL import Image


def main():
  parser = argparse.ArgumentParser(
      description='Extract images from the compressed file.')
  parser.add_argument('input_path',
                      type=str,
                      help='the path of the compressed file')
  parser.add_argument('output_folder',
                      type=str,
                      help='the path of the output folder')
  args = parser.parse_args()

  if os.path.exists(args.output_folder):
    shutil.rmtree(args.output_folder)
  os.makedirs(args.output_folder)

  IMAGES_PER_ROW = 50
  IMAGES_PER_COL = 50
  with gzip.open(args.input_path, 'r') as input_file:
    input_file.read(4)  # skip the magic number
    (image_count, row_count, col_count) = struct.unpack('>lll',
                                                        input_file.read(12))
    print >> sys.stderr, 'Image count =', image_count
    print >> sys.stderr, 'Image size = %d x %d' % (row_count, col_count)
    images_per_file = IMAGES_PER_ROW * IMAGES_PER_COL
    for image_no_start in range(0, image_count, images_per_file):
      image_no_end = min(image_no_start + images_per_file, image_count)
      print >> sys.stderr, 'Extracting images (%d to %d)...' % (
          image_no_start, image_no_end - 1)
      image = Image.new('L', (row_count * IMAGES_PER_ROW,
                              col_count * IMAGES_PER_COL))
      output_pixels = image.load()
      for image_no in range(image_no_start, image_no_end):
        x_start = (image_no - image_no_start) % IMAGES_PER_COL * col_count
        y_start = (image_no - image_no_start) / IMAGES_PER_COL * row_count
        input_pixels = input_file.read(row_count * col_count)
        for y in range(row_count):
          for x in range(col_count):
            output_pixels[(x_start + x,
                           y_start + y)] = ord(input_pixels[y * col_count + x])
      output_file_name = 'image-%d-to-%d.jpg' % (image_no_start,
                                                 image_no_end - 1)
      output_path = os.path.join(args.output_folder, output_file_name)
      with open(output_path, 'w') as output_file:
        image.save(output_file)


if __name__ == '__main__':
  main()
