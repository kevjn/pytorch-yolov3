import argparse

import torch
import torch.nn as nn

import numpy as np
import cv2

class OutputLayer(nn.Module):
    def forward(self, x, output, **kwargs):
        output.append(x)
        return x

class ForkLayer(nn.Module):
    def forward(self, x, route_layers, **kwargs):
        route_layers.append(x)
        return x

class ResetLayer(nn.Module):
    def forward(self, x, route_layers, **kwargs):        
        return route_layers.pop()

class ConcatLayer(nn.Module):
    def forward(self, x, route_layers, **kwargs):        
        return torch.cat((x, route_layers.pop()), 1)

class Convolutional(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        super().__init__()
        # batch Normalization 2d
        self.bn_bias = nn.Parameter(torch.empty(out_ch))
        self.bn_weight = nn.Parameter(torch.empty(out_ch))
        self.register_buffer('running_mean', torch.zeros(out_ch))
        self.register_buffer('running_var', torch.ones(out_ch))

        # convolution 2d
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride, pad, bias=False)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x, **kwargs):
        x = self.conv(x)
        x = torch.functional.F.batch_norm(x, self.running_mean, self.running_var, self.bn_weight, self.bn_bias, False, 0.1, 1e-5)
        return self.leaky(x)

class Resblock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Convolutional(ch, ch//2, 1, 1, pad=0))
            resblock_one.append(Convolutional(ch//2, ch, 3, 1, pad=1))
            self.module_list.append(resblock_one)

    def forward(self, x, **kwargs):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, **kwargs):
        return self.upsample(x)

class YOLOLayer(nn.Module):
    def __init__(self, anchors, anch_mask, n_classes, layer_no, in_ch, ignore_thre=0.7):
        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8] # fixed (send this as param)
        IMG_SIZE = 416
        self.n_classes = n_classes

        anch_mask = anch_mask[layer_no]
        n_anchors = len(anch_mask)
        self.bias = nn.Parameter(torch.empty(1, n_anchors, 1, 1, self.n_classes+5))
        self.weight = nn.Parameter(torch.empty(n_anchors, self.n_classes+5, in_ch, 1, 1))
        
        stride = strides[layer_no] # (send as param)
        fsize = int(IMG_SIZE / stride)

        x_shift = torch.FloatTensor(np.tile(np.arange(fsize),fsize*3))
        y_shift = torch.FloatTensor(np.tile(np.repeat(np.arange(fsize),fsize),3))
        self.xy_shift = torch.cat([torch.stack([x_shift, y_shift], 1), torch.zeros(fsize*fsize*3,83)], 1)

        masked_anchors = (np.array(anchors) / stride)[anch_mask]
        w_anchors = torch.FloatTensor(np.repeat(masked_anchors[:,0], fsize*fsize))
        h_anchors = torch.FloatTensor(np.repeat(masked_anchors[:,1], fsize*fsize))
        self.wh_anchors = torch.cat([torch.ones(fsize**2*3,2), torch.stack([w_anchors, h_anchors], 1), torch.ones(fsize**2*3,81)], 1)

        self.strides = torch.ones(85)
        self.strides[:4] = stride

    def forward(self, xin, **kwargs):
        batchsize = xin.shape[0]

        # sliding window
        xin = xin.unfold(2, 1, 1).unfold(3, 1, 1)
        # convolution
        output = torch.einsum("abcdef,ghbef->agcdh", xin, self.weight) + self.bias

        output = output.reshape(batchsize, -1, self.n_classes + 5)

        m = torch.cat([torch.ones(2), torch.zeros(2), torch.ones(81)]).bool()
        p = torch.exp(output)
        p = p * (1 + p)**-1 * m + ~m * p * self.wh_anchors
        return (p + self.xy_shift) * self.strides


class YOLOv3(nn.Module):
    def __init__(self, ignore_thre=0.7):
        super(YOLOv3, self).__init__()

        anchors = [[10, 13], [16, 30], [33, 23],
                   [30, 61], [62, 45], [59, 119],
                   [116, 90], [156, 198], [373, 326]]

        anch_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        n_classes = 80

        self.module_list = nn.ModuleList([
            # DarkNet53
            Convolutional(in_ch=3,   out_ch=32,   ksize=3, stride=1, pad=1),
            Convolutional(in_ch=32,  out_ch=64,   ksize=3, stride=2, pad=1),
            Resblock(ch=64),
            Convolutional(in_ch=64,  out_ch=128,  ksize=3, stride=2, pad=1),
            Resblock(ch=128, nblocks=2),
            Convolutional(in_ch=128, out_ch=256,  ksize=3, stride=2, pad=1),
            Resblock(ch=256, nblocks=8),
            ForkLayer(),
            Convolutional(in_ch=256, out_ch=512,  ksize=3, stride=2, pad=1),
            Resblock(ch=512, nblocks=8),
            ForkLayer(),
            Convolutional(in_ch=512, out_ch=1024, ksize=3, stride=2, pad=1),
            Resblock(ch=1024, nblocks=4),

            # YOLOv3
            Resblock(ch=1024, nblocks=2, shortcut=False),
            Convolutional(in_ch=1024, out_ch=512, ksize=1, stride=1, pad=0),
            ForkLayer(),

            # 1st yolo branch
            Convolutional(in_ch=512, out_ch=1024, ksize=3, stride=1, pad=1),
            YOLOLayer(anchors, anch_mask, n_classes, layer_no=0, in_ch=1024, ignore_thre=ignore_thre),
            OutputLayer(),
            ResetLayer(),
            Convolutional(in_ch=512, out_ch=256, ksize=1, stride=1, pad=0),
            Upsample(scale_factor=2, mode='nearest'),
            ConcatLayer(),
            Convolutional(in_ch=768, out_ch=256, ksize=1, stride=1, pad=0),
            Convolutional(in_ch=256, out_ch=512, ksize=3, stride=1, pad=1),
            Resblock(ch=512, nblocks=1, shortcut=False),
            Convolutional(in_ch=512, out_ch=256, ksize=1, stride=1, pad=0),
            ForkLayer(),

            # 2nd yolo branch
            Convolutional(in_ch=256, out_ch=512, ksize=3, stride=1, pad=1),
            YOLOLayer(anchors, anch_mask, n_classes, layer_no=1, in_ch=512, ignore_thre=ignore_thre),
            OutputLayer(),
            ResetLayer(),
            Convolutional(in_ch=256, out_ch=128, ksize=1, stride=1, pad=0),
            Upsample(scale_factor=2, mode='nearest'),
            ConcatLayer(),
            Convolutional(in_ch=384, out_ch=128, ksize=1, stride=1, pad=0),
            Convolutional(in_ch=128, out_ch=256, ksize=3, stride=1, pad=1),

            # 3rd yolo branch
            Resblock(ch=256, nblocks=2, shortcut=False),
            YOLOLayer(anchors, anch_mask, n_classes, layer_no=2, in_ch=256, ignore_thre=ignore_thre),
            OutputLayer()
        ])

    def forward(self, x):
        output, route_layers = [], []
        for module in self.module_list:
            x = module(x, output=output, route_layers=route_layers)

        return torch.cat(output, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default=None, help='path to weights file')
    parser.add_argument('--labels_path', type=str, default=None, help='path to labels file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--detect_thresh', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.45, help='non-maximum suppression threshold')
    args = parser.parse_args()

    imgsize = 416
    model = YOLOv3()

    img = cv2.imread(args.image)[..., ::-1] # read and convert BGR to RGB
    img_raw = img.copy()

    # preprocess image for 416x416 input
    h, w, _ = img.shape
    ratio = (max(h,w) / imgsize)**-1
    resized_img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    nh, nw, _ = resized_img.shape
    dx, dy = (imgsize - np.array([nw, nh])) // 2
    canvas = np.full((imgsize, imgsize, 3), 127)
    canvas[dy:dy+nh, dx:dx+nw, :] = resized_img
    img = canvas
    # normalize and convert to pytorch tensor
    img = torch.from_numpy(img.transpose(2,0,1) / 255.).float().unsqueeze(0)

    # load and set pretrained weights
    with open(args.weights_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    offset = 0
    for param in model.state_dict().values():
        w = torch.from_numpy(weights[offset:offset + param.numel()]).view_as(param)
        param.data.copy_(w)
        offset += param.numel()

    # load coco_class_names
    with open(args.labels_path, 'rb') as f:
        labels = [line.strip().decode() for line in f]

    # forward pass for single image 
    model.eval()
    output = model(img)
    assert output.size(0) == 1, "batch size > 1 not supported"
    output = output[0]

    # filter out confidence scores below threshold
    img_pred = output[output[:, 4] * output[:, 5:].max(1).values >= args.detect_thresh]

    if not img_pred.size(0):
        exit("No objects detected")

    # convert (cx, cy, h, w) attribute to box corner coordinates
    box_corner = torch.empty(img_pred .size(0), 4)
    box_corner[:, :2]  = img_pred[:, :2] - img_pred[:, 2:4] / 2
    box_corner[:, 2:4] = img_pred[:, :2] + img_pred[:, 2:4] / 2

    # setup plotting
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.imshow(img_raw)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    import matplotlib.colors as mcolors
    colors = 'yrgbcmk'

    # suppress non-maximal bounding boxes for each unique class
    _max = img_pred[:,5:].max(1)
    for c in _max.indices.unique():
        mask = _max.indices == c
        bbox = box_corner[mask]

        # area and scores for all bboxes in class
        bbox_area = torch.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        scores = img_pred[mask, 4] * _max.values[mask]

        order = scores.argsort(descending=True).tolist()
        select = [order[0]]
        for i in order:
            # top left and bottom right corner
            tl = torch.maximum(bbox[i, :2], bbox[select, :2])
            br = torch.minimum(bbox[i, 2:], bbox[select, 2:])

            # area of intersection
            area = torch.prod(br - tl, axis=1).clamp(0)

            iou = area / (bbox_area[i] + bbox_area[select] - area)
            if (iou < args.nms_thresh).all():
                select.append(i)

        # scale bboxes back to original image
        bboxes = box_corner[mask].detach().numpy()
        bboxes[:, (0, 2)] -= dx
        bboxes[:, (1, 3)] -= dy
        bboxes /= ratio

        # for each selected class box, add it to the figure
        for i in select:
            bb = bboxes[i]

            xy = bb[0], bb[1]
            height = bb[3] - bb[1]
            width = bb[2] - bb[0]

            # debug print
            print(f"Label: {labels[c]}, Confidence: {scores[i]:.2f}")

            color = colors[c % len(colors)]
            ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=2)) 
            ax.text(*xy, labels[c], bbox={'facecolor': color, 'alpha': 0.5}, verticalalignment='bottom')

    plt.show() # finally, show plot