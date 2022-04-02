'''
    The implementation is modified from the Pytorch official implementation of TripletLoss:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn._reduction as _Reduction
import warnings
from torch import Tensor
from typing import Callable, Optional
import random

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class TripletAdaptiveMarginLoss(_Loss):
    '''
    The implementation is modified to handle adaptative and unbalanced number of 
    positive and negative samples referring to each anchor. And the batch size is usually just 1
    '''
    r"""
    #NOTE: belows are the comments from original implementation of TripletMarginLoss
    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    See also :class:`~torch.nn.TripletMarginWithDistanceLoss`, which computes the
    triplet margin loss for input tensors using a custom distance function.

    Args:
        margin (float, optional): Default: :math:`1`.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, D)` or :math`(D)` where :math:`D` is the vector dimension.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'`` and
                  input shape is :math`(N, D)`; a scalar otherwise.

    Examples::

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(self, margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False, size_average=None,
                 reduce=None, reduction: str = 'mean'):
        super(TripletAdaptiveMarginLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, features: Tensor, piece_ids: Tensor) -> Tensor:
        '''
            input: 
                features [NxD]: the feature maps 
                piece_ids [N]: the group ids for feature maps, positive pairs should have the same id 
        '''
        triplet_loss = 0 
        # ids = torch.unique(piece_ids)
        '''
            anchor order is ordered by index:
            piece_ids[positive[i]] == piece_ids[anchor[i]]
            piece_ids[negative[i]] != piece_ids[anchor[i]]
        '''
        feat_num = piece_ids.shape[0]
        anchor = torch.arange(feat_num)
        positive = [random.sample(set(torch.where(piece_ids == piece_ids[i])[0]), 1)[0] for i in range(feat_num)]
        negative = [random.sample(set(torch.where(piece_ids != piece_ids[i])[0]), 1)[0] for i in range(feat_num)]
        positive = torch.Tensor(positive).long()
        negative = torch.Tensor(negative).long()
        anchor_feats = features[anchor]
        positive_feats = features[positive]
        negative_feats = features[negative]
        # NOTE: in fact, the loss has been reduced by "mean" already, but still very large, so I divide it by feat_num AGAIN
        return F.triplet_margin_loss(anchor_feats, positive_feats, negative_feats,
                margin=self.margin, p=self.p, eps=self.eps, swap=self.swap, reduction=self.reduction) / feat_num
        # return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
        #                              eps=self.eps, swap=self.swap, reduction=self.reduction)