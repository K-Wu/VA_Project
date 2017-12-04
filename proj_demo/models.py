import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, out_size=128):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, out_size)

    def forward(self, feats):
        h_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        c_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        h_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)
        c_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)

        if feats.is_cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
            h_t2 = h_t2.cuda()
            c_t2 = c_t2.cuda()

        for _, feat_t in enumerate(feats.chunk(feats.size(1), dim=1)):
            h_t, c_t = self.lstm1(feat_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

        # aggregated feature
        feat = h_t2
        return feat

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 512, 128)
        self.AFeatPool = FeatAggregate(128, 128, 128)
        self.fc = nn.Linear(128, 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = F.relu(self.vfc(vfeat))
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


# Visual-audio multimodal metric learning: MaxPool+FC
class LSTMFastForwardVMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(LSTMFastForwardVMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        #self.vfc = nn.Linear(1024, 128)
        #self.fc = nn.Linear(128, 96)

        self.fc1=nn.Linear(1024+128,512)
        #self.lstm1_num_layers=3
        self.lstm1_num_layers=1
        #self.lstm1_hidden_size=512
        self.lstm1_hidden_size=256
        #self.lstm1_hidden=self.init_lstm_hidden()
        self.lstm1=nn.LSTM(input_size=1152,hidden_size = self.lstm1_hidden_size,bias=True,batch_first=True,num_layers=self.lstm1_num_layers,bidirectional=True)
        self.lstm2=nn.LSTM(input_size=self.lstm1_hidden_size*2,hidden_size = 1,bias=True,batch_first=True,num_layers=1,bidirectional=False)

        self.fc2=nn.Linear(self.lstm1_hidden_size*2,2)

        self.sm1=torch.nn.Softmax()
    def init_lstm_hidden(self,hidden_size,batch_size):
        #return (
        #torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)),
        #torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)))

        return torch.autograd.Variable(torch.zeros(hidden_size,batch_size,self.lstm1_hidden_size))#,
                           #torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for weight in m.all_weights:
                    nn.init.xavier_uniform(weight)
    def forward(self, vfeat, afeat):
        # aggregate the visual features

        vafeat = torch.cat([vfeat,afeat],1)#vfeat [128,1024,120] afeat [128,128,120]
        vafeat = torch.transpose(vafeat, 2, 1)#after transpose vafeat [128,120,1152]
        #vafeat = self.fc1(vafeat)# after  [128,120,512]

        vafeat,hidden = self.lstm1(vafeat)#,(self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0)),self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0))))#after [128,120,1024]
        attention,hidden = self.lstm2(vafeat)#,(self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0)),self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0))))#after [128,120,1]
        vafeat = vafeat*attention
        vafeat = torch.transpose(vafeat, 2, 1)  # after [128,1024,120]

        vafeat = self.mp(vafeat)#maxpool after [128,1024,1]
        vafeat=vafeat.view(vafeat.size(0),-1)
        vafeat = self.fc2(vafeat)

        vafeat = self.sm1(vafeat)
        return (vafeat[:,0]).expand(1,vafeat[:,0].size(0))

        # vfeat = self.mp(vfeat)
        # vfeat = vfeat.view(-1, 1024)
        # vfeat = F.relu(self.vfc(vfeat))
        # vfeat = self.fc(vfeat)
        #
        # # aggregate the auditory features
        # afeat = self.mp(afeat)
        # afeat = afeat.view(-1, 128)
        # afeat = self.fc(afeat)
        # return F.pairwise_distance(vfeat, afeat)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        return loss

class MyCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(MyCrossEntropyLoss,self).__init__()

    def forward(self,softmax,label):
        loss = torch.mean(-(1.0-label)*torch.log(softmax))
        return loss