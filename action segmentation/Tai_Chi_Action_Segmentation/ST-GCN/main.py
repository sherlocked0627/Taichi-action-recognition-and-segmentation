import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import random

from torch.utils.data import Dataset, DataLoader
from st_gcn import *
from metric import accuracy
from metric import f1
from config import get_args
from dataloder import YangDataset
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

train_data = YangDataset('train')
valid_data = YangDataset('valid')
test_data = YangDataset('test')
train_data.data = train_data.data.to(device)
train_data.label = train_data.label.to(device)
valid_data.data = valid_data.data.to(device)
valid_data.label = valid_data.label.to(device)
test_data.data = test_data.data.to(device)
test_data.label = test_data.label.to(device)


train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

model = Model(in_channels=3, num_class=10, graph_args={'layout':'TaiChi', 'strategy':'spatial'}, edge_importance_weighting=True)
if device.type == 'cuda':
	model = model.cuda()


num_params = 0
for p in model.parameters():
	num_params += p.numel()
print(model)
print('The number of parameters: {}'.format(num_params))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, betas=[args.beta1, args.beta2], weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.1)

best_epoch = 0
best_acc = 0
def train():
	global best_epoch, best_acc

	if args.start_epoch:
		model.load_state_dict(torch.load(os.path.join(args.model_path,
													  'model-%d.pkl'%(args.start_epoch))))

	# Training
	for epoch in range(args.start_epoch, args.num_epochs):
		train_loss = 0
		train_acc  = 0
		scheduler.step()
		model.train()
		for i, x in enumerate(train_loader):
			logit = model(x[0].float())
			target = x[1]
			# target = train_label[i]

			loss = criterion(logit, target)

			model.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss = loss.item()
			train_acc  = accuracy(logit, target)

		print('[epoch',epoch+1,'] Train loss:',train_loss, 'Train Acc:',train_acc)

		if (epoch+1) % args.val_step == 0:
			model.eval()
			val_loss = 0
			val_acc  = 0 
			with torch.no_grad():
				for i, x in enumerate(valid_loader):
					logit = model(x[0].float())
					target = x[1]

					val_loss += criterion(logit, target).item()
					val_acc += accuracy(logit, target)

				if best_acc <= (val_acc/i):
					best_epoch = epoch+1
					best_acc = (val_acc/i)
					torch.save(model.state_dict(), os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))

			print('Val loss:',val_loss/i, 'Val Acc:',val_acc/i)

def test():
	global best_epoch

	model.load_state_dict(torch.load(os.path.join(args.model_path, 
												  'model-%d.pkl'%(best_epoch))))
	print("load model from 'model-%d.pkl'"%(best_epoch))

	model.eval()
	test_loss = 0
	test_acc  = 0
	predit = []
	target_seq = []
	with torch.no_grad():
		for i, x in enumerate(test_loader):
			logit = model(x[0].float())
			#print(F.softmax(logit, 1).cpu().numpy(), torch.max(logit, 1)[1].float().cpu().numpy())
			target = x[1]

			test_loss += criterion(logit, target).item()
			test_acc  += accuracy(logit, target)
			result = torch.max(logit, 1)[1].int()
			predit.append(result.cpu().numpy().tolist()[0])
			target_seq.append(target.cpu().numpy().tolist()[0])
			pass

	print('Test loss:',test_loss/i, 'Test Acc:',test_acc/i)
	f1(target_seq, predit, [0.1,0.25,0.5])

if __name__ == '__main__':
	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		best_epoch = args.test_epoch
	test()

