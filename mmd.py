
from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os, sys, math


def MMD(x,y):

	x = x.view(x.size(0), x.size(2) * x.size(3)).permute(1,0)
	y = y.view(y.size(0), y.size(2) * y.size(3)).permute(1,0)

	N = 784

	kernel = [10., 1., 0.1, 0.01, 0.001, 0.00001]

	mmd_ = 0.
	for k in range(len(kernel)):
		K, L, P = pairwise(x, y, kernel[k])
		mmd_ += (1./(N*N)) * torch.sum(K) + (1./(N*N)) * torch.sum(L) - (2./(N*N)) * torch.sum(P)

	return mmd_ / len(kernel)

def pairwise(x, y, k):

	sigma = -k

	xx = torch.mm(x,x.t())
	yy = torch.mm(y,y.t())
	zz = torch.mm(x,y.t())

	rx = (xx.diag().unsqueeze(0).expand_as(xx))
	ry = (yy.diag().unsqueeze(0).expand_as(yy))

	K = torch.exp(sigma * (rx + rx.t() - 2*xx))
	L = torch.exp(sigma * (ry + ry.t() - 2*yy))
	P = torch.exp(sigma * (rx.t() + ry - 2*zz))

	return K, L, P
