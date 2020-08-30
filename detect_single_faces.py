from __future__ import print_function

import os
from tqdm import tqdm
import argparse
import pickle

import cv2
import numpy as np
import torch
from torchvision import transforms as T
import torch.nn.functional as F
torch.backends.cudnn.bencmark = True

import net_s3fd
from bbox import decode, nms

tensor2pil = T.ToPILImage()


def detect_single_faces(args):
    print('loading model...')
    model = load_model(args)

    dname = args.dname
    dname_result = args.dname_result
    fnames = os.listdir(dname)
    print('found {} images from {}'.format(len(fnames), dname))
    fnames = [os.path.join(dname, fname) for fname in fnames if 'gif' not in os.path.splitext(fname)[1]]
    print('found {} non-gif images from {}'.format(len(fnames), dname))

    results = detect_faces(fnames, model, args.pre_thresh)

    fname_pkl = os.path.join(dname_result, 'results.pkl')
    with open(fname_pkl, 'wb') as f:
        pickle.dump(results, f)
    print('wrote', fname_pkl)


@torch.no_grad()
def detect(net, img):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = torch.from_numpy(img).float().cuda()
    BB, CC, HH, WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)//2):
        olist[i*2] = F.softmax(olist[i*2])
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist)//2):
        ocls, oreg = olist[i*2], olist[i*2+1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride/2+windex*stride, stride/2+hindex*stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            if score < 0.05:
                continue
            priors = torch.Tensor([[axc/1.0, ayc/1.0, stride*4/1.0, stride*4/1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))
    return bboxlist


def detect_faces(fnames, model, pre_thresh=0.3, resize=512):
    results = []
    for fname in tqdm(fnames, desc='detecting faces'):
        try:
            img = cv2.imread(fname)
            height, width, _ = img.shape
            ratio = resize / height
            if ratio < 1:
                img = cv2.resize(img, (resize, int(ratio * width)))
        except Exception as e:
            print(e + ' @' + fname)

        bboxlist = detect(model, img)

        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        keep = bboxlist[:, 4] > pre_thresh
        bboxlist = bboxlist[keep, :]
        bboxlist[:, :4] = bboxlist[:, :4] / ratio

        results.append({"fname": os.path.basename(fname),
                        "bboxes": bboxlist})

    return results


def load_model(args):
    net = getattr(net_s3fd, args.net)()
    if args.model != '':
        net.load_state_dict(torch.load(args.model))
    else:
        print('Please set --model parameter!')
    net.cuda()
    net.eval()
    return net


def open_image(fname, resize=128):
    img = cv2.imread(fname)
    img = cv2.resize(img, (resize, resize))
    img = torch.FloatTensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255
    # image = Image.open(fname).convert('RGB')
    # totensor = T.Compose([T.Resize((resize, resize)), T.ToTensor()])
    # return totensor(image)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch face detect')
    parser.add_argument('--dname', type=str, default='/home/data/hairstyle/images')
    parser.add_argument('--dname_result', type=str, default='results')
    parser.add_argument('--pre_thresh', type=float, default=0.3)
    parser.add_argument('--thresh', type=float, default=0.85)
    parser.add_argument('--net', '-n', default='s3fd', type=str)
    parser.add_argument('--model', default='s3fd_convert.pth', type=str)
    parser.add_argument('-f', default='s3fd_convert.pth', type=str)

    args = parser.parse_args()
    os.makedirs(args.dname_result, exist_ok=True)

    detect_single_faces(args)
