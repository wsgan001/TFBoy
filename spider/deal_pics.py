# coding: utf-8
import os
import sys
import time
from config import *
from PIL import Image
from multiprocessing import Pool


def run(input):
    pics = os.listdir(input)
    for pic in pics:
        try:
            in_pic = os.path.join(sys.path[0], input, pic)
            im = Image.open(in_pic)

            new_img = im.resize(format_size(im.size), Image.ANTIALIAS)
            new_img.save(in_pic, "jpeg")
        except:
            print("Error When Deal: ", in_pic)
            os.remove(in_pic)


def format_size(ori_size):
    if max(ori_size) > 500:
        ratio = 500.0 / max(ori_size)
        dst_size = (int(ori_size[0] * ratio), int(ori_size[1] * ratio))
        return dst_size
    else:
        return ori_size


if __name__ == '__main__':
    p = Pool(processes=POOLNUM)
    for name, dir in PIC_TYPES.items():
        print(name, dir)
        p.apply_async(run, (dir,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
