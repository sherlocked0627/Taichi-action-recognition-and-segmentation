import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='predict')
# parser.add_argument('--dataset', default="fog")
parser.add_argument('--dataset', default="TaiChi")

args = parser.parse_args()

num_stages = 4
num_layers_PG = 10
num_layers_RF = 10
num_f_maps = 64
features_dim = 6
bz = 4
lr = 0.0005
num_epochs = 10

dil = [1,2,4,8,16,32,64,128,256,512]


sample_rate = 2

vid_list_file = "./data/" + args.dataset + "/split1/train" + ".txt"
vid_list_file_tst = "./data/" + args.dataset + "/split1/test" + ".txt"
features_path = "./data/" + args.dataset + "/feature/"
gt_path = "./data/" + args.dataset + "/label/"
mapping_file = "./data/" + args.dataset + "/mapping.txt"
model_dir = "./models/"+args.dataset+"/batch4_s2_ctc_0.0005_epoch100"
results_dir = "./results/"+args.dataset+"/batch4_s2_ctc_0.0005_epoch100"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
num_classes = 10
trainer = Trainer(dil, num_layers_RF, num_stages, num_f_maps, features_dim, num_classes)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    for num_epochs in range(1,5,1):
        print("epoch: %d" % num_epochs)
        trainer.predict(model_dir, gt_path, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
