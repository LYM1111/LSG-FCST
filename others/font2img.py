# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json
import collections

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "../charset/cjk.json"

def load_global_charset():
    global CN_CHARSET, JP_CHARSET, CN_T_CHARSET, KR_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk['gbk']
    JP_CHARSET = cjk['jp']
    KR_CHARSET = cjk['kr']
    CN_T_CHARSET = cjk['gb2312_t']
def get_character():
    with open ('char_english.txt','r') as file_to_read:
        lines = file_to_read.readline()
        arr = []
        for l in lines:
            arr.append(l)
        print(arr)
        return arr
def font2img( dst, charset, char_size, canvas_size, x_offset, y_offset,
            sample_count, sample_dir, label=0, filter_by_hash=True):
    # src_font = ImageFont.truetype(src, size=char_size)
    for root, dirs, files in os.walk(dst):
        for file in files:

            dst_font = ImageFont.truetype(os.path.join(dst,file), size=char_size)
            str=get_character()
            count = 0
            if not os.path.exists(os.path.join(sample_dir, file.split('.')[0])):
                os.makedirs(os.path.join(sample_dir, file.split('.')[0]))
            for i in range(len(str)):
                for c in charset:
                    if count == sample_count:
                        break
                    if c == str[i]:
                        e = draw_example(c,  dst_font, canvas_size, x_offset, y_offset, filter_by_hash)
                    else:continue
                    if e:
                        e.save(os.path.join(sample_dir, file.split('.')[0],"{}.png".format(c)))
                        count += 1
                        if count % 100 == 0:
                            print("processed %d chars" % count)
                if i==499:
                    break


def is_monochromatic_image(img):
    extr = img.getextrema()
    a = 0
    for i in extr:
        if isinstance(i, tuple):
            a += abs(i[0] - i[1])
        else:
            a = abs(extr[0] - extr[1])
            break
    return a == 0

def draw_example(ch, dst_font, canvas_size, x_offset, y_offset, filter_by_img=True):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    if filter_by_img and is_monochromatic_image(dst_img): 
        return None
    # src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    # if filter_by_img and is_monochromatic_image(src_img):
    #     return None

    example_img = Image.new("L", (canvas_size , canvas_size), 255)
    example_img.paste(dst_img, (0, 0))
    # example_img.paste(src_img, (canvas_size, 0))
    return example_img

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
# parser.add_argument('--src_font', required=True, help='path of the source font')
parser.add_argument('--dst_font', default='/data1/LYM/font_translator_gan-master/ttf',#required=True,
                    help='path of the target font')
parser.add_argument('--charset', type=str, default='CN',
                    help='charset, can be either: CN, CN_T, JP, KR or a one line file')
parser.add_argument('--filter', action='store_false', help='filter recurring characters')
parser.add_argument('--shuffle', action='store_false', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=62, help='character size')
parser.add_argument('--canvas_size', type=int, default=64, help='canvas size')
parser.add_argument('--x_offset', type=int, default=1, help='x offset')
parser.add_argument('--y_offset', type=int, default=1, help='y offset')
parser.add_argument('--sample_count', type=int, default=1000, help='number of characters to draw')
parser.add_argument('--sample_dir', default='../datasets_fine/font/unseen_font_unseen_character/english',type=str, help='directory to save examples')
parser.add_argument('--label', type=int, default=0, help='label as the prefix of examples')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
        charset = locals().get("%s_CHARSET" % args.charset)
    else:
        charset = [c for c in open(args.charset).readline()[:-1].decode("gbk")]
    if args.shuffle:
        np.random.shuffle(charset)
    
    font2img(args.dst_font, charset, args.char_size,
            args.canvas_size, args.x_offset, args.y_offset,
            args.sample_count, args.sample_dir, args.label, args.filter)

    