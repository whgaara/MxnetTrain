import time
import d2lzh as d2l

from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


# 类别预测层，因为有背景层所以要增加1
def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# 边界框预测层
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)


# 连结多尺度的预测
def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()


def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


# 高宽减半模块，通过一个池化将特征图的长宽减半
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk


# 基础网络块
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk


# 完整模块
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # reshape函数中的0表示保持批量大小不变
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                concat_preds(bbox_preds))


def train():
    batch_size = 32
    train_iter, _ = d2l.load_data_pikachu(batch_size)
    ctx, net = d2l.try_gpu(), TinySSD(num_classes=1)
    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.2, 'wd': 5e-4})
    cls_loss = gloss.SoftmaxCrossEntropyLoss()
    bbox_loss = gloss.L1Loss()

    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        cls = cls_loss(cls_preds, cls_labels)
        bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox

    def cls_eval(cls_preds, cls_labels):
        # 由于类别预测结果放在最后一维，argmax需要指定最后一维
        return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    for epoch in range(20):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()  # 从头读取数据
        start = time.time()
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = net(X)
                # 为每个锚框标注类别和偏移量
                bbox_labels, bbox_masks, cls_labels = \
                    contrib.nd.MultiBoxTarget(anchors, Y, cls_preds.transpose((0, 2, 1)))
                # 根据类别和偏移量的预测和标注值计算损失函数
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            trainer.step(batch_size)
            acc_sum += cls_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size

        if (epoch + 1) % 5 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))


if __name__ == '__main__':
    train()
