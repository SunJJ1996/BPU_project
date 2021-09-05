# -*- coding: UTF-8 -*-
import os

dirpath = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/BPUtrain200809test/image_valid'
savepath = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/BPUtrain200809test'
targetfile = 'BPUvalid200809.txt'


def is_image(fn):
    return os.path.splitext(fn)[-1] in (
        '.jpg', '.JPG', '.png', '.PNG')


def main():
    imagelist = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if not is_image(fn):
                continue
            fname = os.path.join(r, fn)
            print(fname)
            imagelist.append(fname)
    if not imagelist:
        print('image not found')
        return
    # target = os.path.join(dirpath, targetfile)
    target = os.path.join(savepath, targetfile)
    with open(target, 'w') as f:
        f.write('\n'.join(imagelist))
    print('the path of images have been wirte to',
          target)


if __name__ == '__main__':
    main()