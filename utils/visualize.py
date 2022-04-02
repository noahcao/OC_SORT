'''
    The script to visualize 
'''
import cv2 
import torch
import os 
import numpy as np
import colorsys
import seaborn as sns 

platte = sns.color_palette("Spectral", 100, as_cmap=True) # doesn't work

from typing import Iterable, Tuple
import colorsys
import itertools
from fractions import Fraction
from pprint import pprint


######## The code to generate high-contrastive colors for visualization ##########
def zenos_dichotomy() -> Iterable[Fraction]:
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1,2**k)

def fracs() -> Iterable[Fraction]:
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield Fraction(0)
    for k in zenos_dichotomy():
        i = k.denominator # [1,2,4,8,16,...]
        for j in range(1,i,2):
            yield Fraction(j,i)

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
# bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)

HSVTuple = Tuple[Fraction, Fraction, Fraction]
RGBTuple = Tuple[float, float, float]

def hue_to_tones(h: Fraction) -> Iterable[HSVTuple]:
    for s in [Fraction(6,10)]: # optionally use range
        for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
            yield (h, s, v) # use bias for v here if you use range

def hsv_to_rgb(x: HSVTuple) -> RGBTuple:
    return colorsys.hsv_to_rgb(*map(float, x))

flatten = itertools.chain.from_iterable

def hsvs() -> Iterable[HSVTuple]:
    return flatten(map(hue_to_tones, fracs()))

def rgbs() -> Iterable[RGBTuple]:
    return map(hsv_to_rgb, hsvs())

def rgb_to_css(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y*255), x)
    rgb_str =  "{},{},{}".format(*uint8tuple)
    rgb_value = rgb_str.split(",")
    rgb_value = [int(d) for d in rgb_value]
    return (rgb_value[0], rgb_value[1], rgb_value[2])

def css_colors() -> Iterable[str]:
    return map(rgb_to_css, rgbs())

sample_colors = list(itertools.islice(css_colors(), 400))

##########  ########## ########## ########## ########## ########## ##########  


def draw_box(im, box, thickness=1, color=(255,0,0), trackids=[]):
    x1, y1, w, h = box
    x2 = x1+w 
    y2 = y1+h 
    cv2.rectangle(im, (x1, y1), (x2, y2), color=color, thickness=thickness)
    trackids = [str(id) for id in trackids]
    cv2.putText(im, ",".join(trackids), (int(x1+2), int(y1+12)), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
    return im 

def draw_pieces(pieces_annos, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    frames = torch.unique(pieces_annos[:,:,0])
    for frame in frames:
        frame_bboxes = pieces_annos[pieces_annos[:, :, 0]==frame]
        img = os.path.join(img_dir, "%06d.jpg" % frame)
        im = cv2.imread(img)
        for bbox in frame_bboxes:
            trackid = bbox[1]
            if trackid == 0:
                continue 
            else:
                coord = bbox[2:6]
                occupy_indices = (frame_bboxes[:, 2:6] == coord)[:,0]
                bboxes = frame_bboxes[occupy_indices]
                trackids = torch.unique(bboxes[:, 1]).int().tolist()
                im = draw_box(im, coord.numpy(), thickness=2, color=sample_colors[int(trackid)], trackids=trackids)
        save_path = os.path.join(save_dir, "vis_%06d.jpg" % frame)
        cv2.imwrite(save_path, im)

if __name__ == "__main__":
    seq_name = "MOT17-02-DPM"
    img_dir = "data/MOT17/train/{}/img1".format(seq_name)
    anno = "data/MOT17/train_pieces/{}".format(seq_name)
    index = 2895
    piece_anno = torch.load("{}/{}_pieces.pth".format(anno, index))
    # import pdb; pdb.set_trace()
    for i in range(piece_anno.shape[0]):
        piece_anno[i, piece_anno[i, :, 1] !=0, 1] = i+1
    tracklet_anno = torch.load("{}/{}_tracklets.pth".format(anno, index))
    draw_pieces(piece_anno, img_dir, "visualizations/{}/{}".format(seq_name, index))
    cmd = "ffmpeg  -y -r 10 -i visualizations/{}/{}/vis_%06d.jpg vis_{}_{}.mp4".format(seq_name, index, seq_name, index)
    os.system(cmd)
