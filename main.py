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
import mlp, mmd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--ndf', type=int, default=256)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

opt.dataroot = '/Users/jordancampbell/helix/phd/AE-Conditional-MMD-GAN/data'
opt.experiment = '/Users/jordancampbell/helix/phd/AE-Conditional-MMD-GAN/output/samples'

if opt.cuda:
	opt.dataroot = '/input'
	opt.experiment = '/output/samples'
else:
	import matplotlib.pyplot as plt
	plt.ion()

if opt.experiment is None:
	opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print("Input:", opt.dataroot)
print("Output:", opt.experiment)

class MNISTDataGenerator():

<<<<<<< HEAD
	def __init__(self, opt):

		self.B = opt.batchSize
		self.cuda = opt.cuda
		self.num_local_classes = 10
		self.C = self.B // self.num_local_classes

		data_path = opt.dataroot#'../data/mnist'
		if self.cuda:
			data_path = '/input'

		self.trData = dset.MNIST(data_path, train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.testData = dset.MNIST(data_path, train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.N = len(self.trData)

		self.labels = [ [] for i in range(10) ]
		for i in range(self.N):
			self.labels[self.trData[i][1]].append(i)

		self.sample = torch.FloatTensor(self.B, 1, opt.imageSize, opt.imageSize)
		self.local = torch.FloatTensor(self.B, 1, opt.imageSize, opt.imageSize)
		self.point = torch.FloatTensor(self.B, 1, opt.imageSize, opt.imageSize)

		self.train_loader = torch.utils.data.DataLoader(self.trData, batch_size=self.B, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(self.testData, batch_size=self.B, shuffle=True)

		self.train_iter = iter(self.train_loader)

	def next(self):
		if self.cuda:
			self.sample = self.sample.cpu()
			self.local = self.local.cpu()
			self.point = self.point.cpu()

		# sample comes from the train_loader
		try:
			self.sample, _ = self.train_iter.next()
		except:
			self.train_iter = iter(self.train_loader)
			self.sample, _ = self.train_iter.next()

		# a point is a random image from a random class

		random_labels = np.random.permutation(10)

		for i in range(self.num_local_classes):
			label = random_labels[i]#np.random.randint(10)
			class_size = len(self.labels[label])
			image = np.random.randint(class_size)

			self.point[i*self.C].copy_(self.trData[ self.labels[label][image] ][0])
			self.point[i*self.C:(i+1)*self.C] = self.point[i*self.C].unsqueeze(0).expand_as(self.point[i*self.C:(i+1)*self.C])

			# local is a random selection from the same class as point
			subset = torch.randperm(class_size)[:self.C]
			for k in range(self.C):
				self.local[(i*self.C) + k].copy_( self.trData[ self.labels[label][subset[k]] ][0] )

		if self.cuda:
			self.sample = self.sample.cuda()
			self.local = self.local.cuda()
			self.point = self.point.cuda()

		return self.sample, self.local, self.point
=======
    def __init__(self, opt):

        self.B = opt.batchSize
        self.cuda = opt.cuda
        self.num_local_classes = 10
        self.C = self.B // self.num_local_classes

        data_path = '../data/mnist'
        if self.cuda:
            data_path = '/input'

        self.trData = dset.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        self.testData = dset.MNIST(data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        self.N = len(self.trData)

        self.labels = [ [] for i in range(10) ]
        for i in range(self.N):
            self.labels[self.trData[i][1]].append(i)

        self.sample = torch.FloatTensor(self.B, 1, opt.size, opt.size)
        self.local = torch.FloatTensor(self.B, 1, opt.size, opt.size)
        self.point = torch.FloatTensor(self.B, 1, opt.size, opt.size)

        self.train_loader = torch.utils.data.DataLoader(self.trData, batch_size=self.B, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.testData, batch_size=self.B, shuffle=True)

        self.train_iter = iter(self.train_loader)

    def next(self):
        if self.cuda:
            self.sample = self.sample.cpu()
            self.local = self.local.cpu()
            self.point = self.point.cpu()

        # sample comes from the train_loader
        try:
            self.sample, _ = self.train_iter.next()
        except:
            self.train_iter = iter(self.train_loader)
            self.sample, _ = self.train_iter.next()

        # a point is a random image from a random class

		random_labels = np.random.permutation(10)

        for i in range(self.num_local_classes):
            label = random_labels[i]#np.random.randint(10)
            class_size = len(self.labels[label])
            image = np.random.randint(class_size)

            self.point[i*self.C].copy_(self.trData[ self.labels[label][image] ][0])
            self.point[i*self.C:(i+1)*self.C] = self.point[i*self.C].unsqueeze(0).expand_as(self.point[i*self.C:(i+1)*self.C])

            # local is a random selection from the same class as point
            subset = torch.randperm(class_size)[:self.C]
            for k in range(self.C):
                self.local[(i*self.C) + k].copy_( self.trData[ self.labels[label][subset[k]] ][0] )

        if self.cuda:
            self.sample = self.sample.cuda()
            self.local = self.local.cuda()
            self.point = self.point.cuda()

        return self.sample, self.local, self.point
>>>>>>> 549a3107f0d167f1857829ac978903c80b3c593f

# trData = dset.MNIST(opt.dataroot, train=True, download=True,
# 			   transform=transforms.Compose([
# 				#    transforms.ToPILImage(),
# 				#    transforms.Scale(32),
# 				   transforms.ToTensor(),
# 				   transforms.Normalize((0.1307,), (0.3081,))
# 			   ]))
# testData = dset.MNIST(opt.dataroot, train=False, download=False,
# 			   transform=transforms.Compose([
# 				#    transforms.ToPILImage(),
# 				#    transforms.Scale(32),
# 				   transforms.ToTensor(),
# 				   transforms.Normalize((0.1307,), (0.3081,))
# 			   ]))

# dataloader = torch.utils.data.DataLoader(trData, batch_size=opt.batchSize,
# 										 shuffle=True, num_workers=int(opt.workers))
# gen_dataloader = torch.utils.data.DataLoader(trData, batch_size=opt.batchSize,
# 										 shuffle=True, num_workers=int(opt.workers))

data_loader = MNISTDataGenerator(opt)
sys.exit()

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

netG = mlp.AEGenerator(opt.imageSize, nz, ngf, ngpu, 40)
netG.apply(weights_init)

if opt.netG != '': # load checkpoint if needed
	netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = mlp.Critic(opt.imageSize, nz, ndf, ngpu)
netD.apply(weights_init)

if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
fixed_input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)

data_iter = iter(dataloader)
data = data_iter.next()
real_cpu, _ = data
if opt.cuda:
	real_cpu = real_cpu.cuda()
fixed_input.copy_(real_cpu)

one = torch.FloatTensor([1])
mone = one * -1

AE_criterion = nn.BCELoss()

if opt.cuda:
	netD.cuda()
	netG.cuda()
	input = input.cuda()
	fixed_input = fixed_input.cuda()
	one, mone = one.cuda(), mone.cuda()
	AE_criterion = AE_criterion.cuda()
	noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
	optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
	optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
	optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0
logs = [[], [], [], []]
for epoch in range(opt.niter):
	# data_iter = iter(dataloader)
	# gen_data_iter = iter(gen_dataloader)
	i = 0
	while i < 100:#len(dataloader):
		############################
		# (1) Update D network
		###########################
		for p in netD.parameters(): # reset requires_grad
			p.requires_grad = True # they are set to False below in netG update

		# train the discriminator Diters times
		if gen_iterations < 25 or gen_iterations % 500 == 0:
			Diters = 100
		else:
			Diters = opt.Diters
		j = 0
		while j < Diters and i < 100:#len(dataloader):
			j += 1

			# clamp parameters to a cube
			for p in netD.parameters():
				p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

			# data = data_iter.next()
			data, local, point = data_loader.next()
			i += 1

			# train with real
			real_cpu, _ = data
			netD.zero_grad()
			batch_size = real_cpu.size(0)

			if opt.cuda:
				real_cpu = real_cpu.cuda()
			input.resize_as_(real_cpu).copy_(real_cpu)
			inputv = Variable(input)

			errD_real = netD(inputv)
			errD_real.backward(one)

			# train with fake
			try:
				gen_data = gen_data_iter.next()
			except:
				gen_data_iter = iter(gen_dataloader)
				gen_data = gen_data_iter.next()
			real_cpu, _ = gen_data
			if opt.cuda:
				real_cpu = real_cpu.cuda()
			input.resize_as_(real_cpu).copy_(real_cpu)
			inputv = Variable(input)
			noise.resize_(input.size(0), nz, 1, 1).normal_(0, 1)
			noisev = Variable(noise, volatile = True) # totally freeze netG

			fake, decoded = netG(noisev, inputv)
			fake = Variable(fake.data)

			inputv = fake
			errD_fake = netD(inputv)
			errD_fake.backward(mone)
			errD = errD_real - errD_fake
			optimizerD.step()

		############################
		# (2) Update G network
		###########################
		for p in netD.parameters():
			p.requires_grad = False # to avoid computation

		# j = 0
		# while j < Diters and i < len(gen_dataloader):
		# 	j += 1

		netG.zero_grad()

		try:
			gen_data = gen_data_iter.next()
		except:
			gen_data_iter = iter(gen_dataloader)
			gen_data = gen_data_iter.next()
		real_cpu, _ = gen_data
		if opt.cuda:
			real_cpu = real_cpu.cuda()
		input.resize_as_(real_cpu).copy_(real_cpu)
		inputv = Variable(input)

		noise.resize_(input.size(0), nz, 1, 1).normal_(0, 1)
		noisev = Variable(noise)
		fake, decoded = netG(noisev, inputv)

		mmd_loss = mmd.MMD(fake, decoded)
		mmd_loss.backward(retain_variables=True)

		reconstruction_loss = AE_criterion(nn.Sigmoid()(decoded), nn.Sigmoid()(Variable(inputv.data)))
		reconstruction_loss.backward(retain_variables=True)

		errG = netD(fake)
		errG.backward(one)

		optimizerG.step()
		gen_iterations += 1

		print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f reconstruction: %f mmd: %f'
			% (epoch, opt.niter, i, len(dataloader), gen_iterations,
			errD.data[0], errG.data[0], reconstruction_loss.data[0], mmd_loss.data[0]))
		if gen_iterations % 50 == 0:
			real_cpu = real_cpu.mul(0.5).add(0.5)
			vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
			fake, decoded = netG(Variable(fixed_noise, volatile=True), Variable(fixed_input, volatile=True))
			fake.data = fake.data.mul(0.5).add(0.5)
			decoded.data = decoded.data.mul(0.5).add(0.5)
			vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
			vutils.save_image(decoded.data, '{0}/decoded_samples_{1}.png'.format(opt.experiment, gen_iterations))

		if not opt.cuda:
			logs[0].append(errD.data[0])
			logs[1].append(errG.data[0])
			logs[2].append(reconstruction_loss.data[0])
			logs[3].append(mmd_loss.data[0])

	if not opt.cuda:
		plt.subplot(311)
		plt.plot(logs[0][-500:])
		plt.plot(logs[1][-500:])
		plt.subplot(312)
		plt.plot(logs[2][-500:])
		plt.subplot(313)
		plt.plot(logs[3][-500:])
		plt.pause(0.001)
		plt.clf()

	# do checkpointing
	if epoch % 10 == 0:
		torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
		torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
