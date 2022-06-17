import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from batch_gen import get_features
from ms_gcn import MultiStageModel
import label_eval as eval

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, dil, num_layers_R, num_R, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(dil, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        # self.model = nn.DataParallel(self.model)
        self.step = [20, 40]
        self.base_lr = learning_rate

        optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask, weight = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, weight = batch_input.to(device), batch_target.to(device), mask.to(
                    device), weight.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                predictions = torch.transpose(predictions,1,0)
                for p in predictions:
                    loss += self.ce(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :, 1:])
                    ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
                    log_probs = p.transpose(2,0).transpose(1,2).log_softmax(2).requires_grad_()
                    target = []
                    target_length = []
                    # input_length = [batch_target.shape[1]] * batch_size
                    input_length = []
                    for d in range(0,batch_size):
                        target_every = [batch_target[d,0]]
                        for b in range(1,batch_target[d,:].shape[0]):
                            if (batch_target[d,b] != batch_target[d,b-1]) and batch_target[d,b] > 0:
                                target_every.append(batch_target[d,b])
                        target = target + target_every
                        target_length.append(len(target_every))
                        mask_input = mask[d,0,:].bool()
                        input_length.append(torch.masked_select(p[d,0,:],mask_input).shape[0])
                    target = torch.Tensor(target)
                    loss += 0.0005 * ctc_loss(log_probs, target, input_length, target_length)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            # if (epoch + 1) % 5 == 0:
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total))

    def predict(self, model_dir, gt_path, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            self.model.to(device)
            # self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')
            file_ptr.close()
            for vid in list_of_vids:
                string2 = vid[:-10]
                features = np.load(features_path + vid)
                features = get_features(features)
                features = features[:, ::sample_rate, :, :]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                N, C, T, V, M = input_x.size()
                input_x = input_x.to(device)

                file_ptr = np.load(gt_path + vid)
                classes = np.zeros(min(np.shape(features)[1], len(file_ptr[0])), dtype=int)
                for i in range(len(classes)):
                    classes[i] = file_ptr[0][i].astype(int)
                sample_rate = 1
                classes = classes[::sample_rate].copy()
                
                predictions = self.model(input_x, torch.ones(N,2,T).to(device))
                predictions = torch.transpose(predictions,1,0)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze().data.detach().cpu().numpy()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                correct += (predicted == classes).sum()
                total += len(predicted)
                f_name = vid[:-4]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
        # print("acc = %f" % (float(correct) / total))
        acc, f1 = eval.label_eval('batch4_s2_ctc_0.0005_epoch100')
