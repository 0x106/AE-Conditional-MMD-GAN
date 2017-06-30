import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math, sys
import numpy as np

class AEGenerator(nn.Module):
	def __init__(self, isize, nz, ngf, ngpu, hidden):
		super(AEGenerator, self).__init__()
		self.ngpu = ngpu

		# hidden + noise -> image
		self.main = nn.Sequential(
			nn.Linear(nz, ngf),
			nn.ReLU(True),
			nn.Linear(ngf, ngf),
			nn.ReLU(True),
			nn.Linear(ngf, ngf),
			nn.ReLU(True),
			nn.Linear(ngf, isize*isize),
		)

		# image -> hidden
		self.encoder = nn.Sequential(
			nn.Linear(isize*isize, ngf*2),
			nn.ReLU(True),
			nn.Linear(ngf*2, ngf*4),
			nn.ReLU(True),
			nn.Linear(ngf*4, hidden)
		)

		# hidden -> image
		self.decoder = nn.Sequential(
			nn.Linear(hidden, ngf*4),
			nn.ReLU(True),
			nn.Linear(ngf*4, ngf*2),
			nn.ReLU(True),
			nn.Linear(ngf*2, isize*isize)
		)

		self.size = isize

	def forward(self, noise, image, probs=None, training=True):

		image = image.view(image.size(0), self.size * self.size)
		noise = noise.view(noise.size(0), noise.size(1))

		hidden_state = self.encoder(image)
		x = torch.cat((noise,hidden_state),1)
		decoded = self.decoder(hidden_state)

		x = self.main(x)

		x, decoded = x.view(x.size(0), 1, self.size, self.size), decoded.view(decoded.size(0), 1, self.size, self.size)
		return x, decoded




class Generator(nn.Module):
	def __init__(self, isize, nz, ngf, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu

		main = nn.Sequential(
			nn.Linear(nz, ngf),
			nn.ReLU(True),
			nn.Linear(ngf, ngf),
			nn.ReLU(True),
			nn.Linear(ngf, ngf),
			nn.ReLU(True),
			nn.Linear(ngf, isize*isize),
		)

		# self.last_layer = nn.Linear(n_units, isize)

		self.main = main
		self.size = isize

	def forward(self, noise, image=None, probs=None, training=True):
		noise = noise.view(noise.size(0), noise.size(1))
		# x = torch.cat((noise,image),1)
		x = self.main(noise)
		x = x.view(x.size(0), 1, self.size, self.size)
		return x


class Critic(nn.Module):
	def __init__(self, isize, nz, ndf, ngpu):
		super(Critic, self).__init__()
		self.ngpu = ngpu

		main = nn.Sequential(
			nn.Linear(isize*isize, ndf),
			nn.ReLU(True),
			nn.Linear(ndf, ndf),
			nn.ReLU(True),
			nn.Linear(ndf, ndf),
			nn.ReLU(True),
			nn.Linear(ndf, 1),
		)
		self.main = main
		self.size = isize

	def forward(self, x):
		x = x.view(x.size(0), x.size(2) * x.size(3))
		if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
		else:
			output = self.main(x)
		output = output.mean(0)
		return output.view(1)

class MNISTAutoEncoder(nn.Module):
	def __init__(self, opt):
		super(MNISTAutoEncoder, self).__init__()

		self.dims = opt['dims']

		self.encode = nn.Sequential(
			nn.Linear(opt['dims']*opt['dims'], opt['nh']),
			nn.ReLU(True),
			nn.Linear(opt['nh'], opt['nh']*2),
			nn.BatchNorm1d(opt['nh']*2),
			nn.ReLU(True),
			nn.Linear(opt['nh']*2, opt['nh']*4),
			nn.BatchNorm1d(opt['nh']*4),
			nn.ReLU(True),
			nn.Linear(opt['nh']*4, opt['n_units']),
		)

		self.decode = nn.Sequential(
			nn.BatchNorm1d(opt['n_units']),
			nn.ReLU(True),
			nn.Linear(opt['n_units'], opt['nh']*4),
			nn.BatchNorm1d(opt['nh']*4),
			nn.ReLU(True),
			nn.Linear(opt['nh']*4, opt['nh']*2),
			nn.BatchNorm1d(opt['nh']*2),
			nn.ReLU(True),
			nn.Linear(opt['nh']*2, opt['nh']),
			nn.BatchNorm1d(opt['nh']),
			nn.ReLU(True),
			nn.Linear(opt['nh'], opt['dims']*opt['dims']),
			# nn.Tanh()
		)

		self.num_weights = 0
		for l in self.encode.parameters():
			if l.dim() == 1:
				self.num_weights += l.size(0)
			else:
				self.num_weights += l.size(0) * l.size(1)
		for l in self.decode.parameters():
			if l.dim() == 1:
				self.num_weights += l.size(0)
			else:
				self.num_weights += l.size(0) * l.size(1)

	def forward(self, x):
		x = x.view(x.size(0), self.dims*self.dims)
		encoded = self.encode(x)
		decoded = self.decode(encoded)
		return decoded.view(x.size(0), 1, self.dims, self.dims), nn.Sigmoid()(encoded) # decoded should match input, encoded are the
								#   dropout probabilities.
