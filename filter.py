from __future__ import print_function

import os
from collections import namedtuple
from tqdm import tqdm
import argparse
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
import torch.nn.functional as F
torch.backends.cudnn.bencmark = True

import net_s3fd
from bbox import decode, nms

Face = namedtuple('Face', ['fname', 'bbox', 'score'])
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

    results, faces = detect_faces(fnames, model, args.pre_thresh)

    results_noface, results_oneface, results_multiple = split_results_by_num_faces(results)

    scores = [face.bbox[4] for face in faces]
    scores_oneface = [result.bbox[0, 4] for result in results_oneface]
    num_faces = [result.bbox.shape[0] for result in results]

    results_oneface_belowthresh = [result for result in results_oneface if result.bbox[0, 4] < args.thresh]
    results_oneface_valid = [result for result in results_oneface if result.bbox[0, 4] >= args.thresh]

    print('noface: {}'.format(len(results_noface)))
    print('oneface_belowthresh: {}'.format(len(results_oneface_belowthresh)))
    print('oneface_valid: {}'.format(len(results_oneface_valid)))
    print('multiple: {}'.format(len(results_multiple)))

    fname_pkl = os.path.join(dname_result, 'faces_valid.pkl')
    with open(fname_pkl, 'wb') as f:
        pickle.dump(results_oneface_valid, f)
    print('wrote', fname_pkl)

    save_histogram(scores, os.path.join(dname_result, 'scores.png'))
    save_histogram(scores_oneface, os.path.join(dname_result, 'scores_oneface.png'))
    save_histogram_discrete(num_faces, os.path.join(dname_result, 'num_faces.jpg'))

    thumbnail_results(results_noface, os.path.join(dname_result, 'results_noface.jpg'))
    thumbnail_results(results_oneface_belowthresh, os.path.join(dname_result, 'results_oneface_belowthresh.jpg'))
    thumbnail_results(results_oneface_valid, os.path.join(dname_result, 'results_oneface_valid.jpg'))
    thumbnail_results(results_multiple, os.path.join(dname_result, 'results_multiple.jpg'))


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


def detect_faces(fnames, model, pre_thresh, resize=512):
    results, faces = [], []
    for fname in tqdm(fnames, desc='detecting faces'):
        try:
            img = cv2.imread(fname)
            height, width, _ = img.shape
            ratio = resize / height
            img = cv2.resize(img, (resize, int(ratio * width)))
        except Exception as e:
            print(e + ' @' + fname)

        bboxlist = detect(model, img)

        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        keep = bboxlist[:, 4] > pre_thresh
        bboxlist = bboxlist[keep, :]
        bboxlist[:, :4] = bboxlist[:, :4] / ratio

        results.append(Face(fname, bboxlist, None))
        for b in bboxlist:
            # x1, y1, x2, y2, s = b
            face = Face(fname, b, b[4])
            faces.append(face)
    return results, faces


def save_histogram(data, fname, num_bins=10):
    fig = plt.figure()
    n, bins, patches = plt.hist(data, num_bins, facecolor='gray', alpha=0.5)
    fig.savefig(fname)


def save_histogram_discrete(data, fname):
    fig = plt.figure()
    data = np.array(data)
    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d)/2
    right_of_last_bin = data.max() + float(d)/2
    plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d))
    fig.savefig(fname)


def split_results_by_num_faces(results):
    results_noface, results_oneface, results_multiple = [], [], []
    for result in results:
        num_face = result.bbox.shape[0]
        if num_face == 0:
            results_noface.append(result)
        elif num_face == 1:
            results_oneface.append(result)
        else:
            results_multiple.append(result)
    return results_noface, results_oneface, results_multiple


def thumbnail_results(results, fname):
    if len(results) == 0:
        print('There is nothing to be saved at ' + fname)
        return
    images = []
    i = 0
    tokens = os.path.splitext(fname)
    for result in tqdm(results, 'saving to ' + fname):
        image = open_image(result.fname)
        images.append(image)
        if len(images) == 64:
            images = torch.stack(images)
            save_image(images, '_{:04d}'.format(i).join(tokens))
            images = []
            i += 1
    if len(images) != 0:
        images = torch.stack(images)
        save_image(images, '_{:04d}'.format(i).join(tokens))


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
    parser.add_argument('--pre_thresh', type=float, default=0.5)
    parser.add_argument('--thresh', type=float, default=0.85)
    parser.add_argument('--net', '-n', default='s3fd', type=str)
    parser.add_argument('--model', default='s3fd_convert.pth', type=str)
    parser.add_argument('-f', default='s3fd_convert.pth', type=str)

    args = parser.parse_args()
    os.makedirs(args.dname_result, exist_ok=True)

    detect_single_faces(args)
