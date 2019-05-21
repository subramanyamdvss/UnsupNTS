import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
# random.seed(7)
# torch.manual_seed(7)
# torch.cuda.manual_seed_all(7)

class CNN(nn.Module):
    
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        
        # V = args.embed_num
        D = args.embedd_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        # self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.add_control = args.add_control
        self.control_num = args.control_num
        self.dropout = nn.Dropout(args.dropout)
        if not self.add_control:
            self.fc3 = nn.Linear(len(Ks)*Co, C)

            if C ==2:
                self.fc2 = nn.Linear(len(Ks)*Co, C)
                self.fc1 = nn.Linear(len(Ks)*Co, C)
                self.fc4 = nn.Linear(len(Ks)*Co, C)
        else:
            self.classf = nn.Linear(len(Ks)*Co, C*self.control_num)
            for i in range(1,C*self.control_num+1):
                setattr(self,'disc{}'.format(i),nn.Linear(len(Ks)*Co,C))
            # self.discs = [nn.Linear(len(Ks)*Co,C) for i in range(C*control_num)]

    def _train(self, mode):
        self.train(mode)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x,target,type=None, train=True,ncontrol=0):
        # x = self.embed(x)  # (N, W, D)
        self._train(train)        
        

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print('input to conv',x.size())
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        if not self.add_control:
            if type == "discsim":
                logit = self.fc1(x)  # (N, C)
            elif type == "disccom":
                logit = self.fc2(x)
            elif type == "classsim":
                logit = self.fc3(x)
            elif type == "classcom":
                logit = self.fc4(x)
        else:
            if type == "discsim":
                logit = getattr(self,"disc{}".format(ncontrol))(x)  # (N, C)
            elif type == "disccom":
                logit = getattr(self,"disc{}".format(ncontrol+self.control_num))(x)
            elif type == "classsim":
                logit = self.classf(x)

        loss = F.cross_entropy(logit, target,size_average=False)
        return loss


