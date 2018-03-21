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


class Delta(nn.Module):
    def __init__(self, framenum=120):
        super(Delta, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vlstm = nn.LSTM(input_size=1024, hidden_size=512, bias=True, batch_first=True,
                             num_layers=1, bidirectional=True)

        self.alstm = nn.LSTM(input_size=128, hidden_size=512, bias=True, batch_first=True,
                             num_layers=1, bidirectional=True)

        self.relative=torch.nn.Parameter(torch.eye(1024,1024))
        self.matmp=nn.MaxPool1d(framenum)
        self.sm1 = torch.nn.Softmax()

    def init_lstm_hidden(self, hidden_size, batch_size):
        # return (
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)),
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)))

        return torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))  # ,
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))

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
        vfeat= torch.transpose(vfeat,1,2)
        afeat= torch.transpose(afeat,1,2)
        vfeat,_ = self.vlstm(vfeat)
        afeat,_ = self.alstm(afeat)
        # vfeat = torch.transpose(vfeat,2,1)
        # afeat = torch.transpose(afeat,2,1)
        # vfeat = self.mp(vfeat)
        # afeat = self.mp(afeat)
        # vfeat = vfeat.view(vfeat.size(0),-1)
        # afeat = afeat.view(afeat.size(0),-1)
        #mat =torch.autograd.Variable(torch.zeros(vfeat.size(0),vfeat.size(1),afeat.size(1)))
        # for i in range(vfeat.size(0)):
        #    mat[i] = vfeat[i].view(1,-1).mm(self.relative).mm(afeat[i].view(-1,1))
        # mat = torch.nn.functioanl.tanh(mat)

        relative = self.relative.view(1,self.relative.size(0),self.relative.size(1)).expand(vfeat.size(0),self.relative.size(0),self.relative.size(1))
        afeat=torch.transpose(afeat,2,1)
        mat=torch.nn.functional.tanh(torch.bmm(torch.bmm(vfeat,relative),afeat))
        vec1 = torch.squeeze(self.matmp(mat))
        vec2 = torch.squeeze(self.matmp(torch.transpose(mat,1,2)))

        return F.pairwise_distance(vec1, vec2)


class CANE(nn.Module):
    def __init__(self, framenum=120):
        super(CANE, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vlstm = nn.LSTM(input_size=1024, hidden_size=512, bias=True, batch_first=True,
                             num_layers=1, bidirectional=True)

        self.alstm = nn.LSTM(input_size=128, hidden_size=512, bias=True, batch_first=True,
                             num_layers=1, bidirectional=True)

        self.relative=torch.nn.Parameter(torch.eye(1024,1024))
        self.matmp=nn.MaxPool1d(framenum)
        self.sm1 = torch.nn.Softmax()

    def init_lstm_hidden(self, hidden_size, batch_size):
        # return (
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)),
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)))

        return torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))  # ,
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))

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
        vfeat= torch.transpose(vfeat,1,2)
        afeat= torch.transpose(afeat,1,2)
        vfeat,_ = self.vlstm(vfeat)
        afeat,_ = self.alstm(afeat)
        # vfeat = torch.transpose(vfeat,2,1)
        # afeat = torch.transpose(afeat,2,1)
        # vfeat = self.mp(vfeat)
        # afeat = self.mp(afeat)
        # vfeat = vfeat.view(vfeat.size(0),-1)
        # afeat = afeat.view(afeat.size(0),-1)
        #mat =torch.autograd.Variable(torch.zeros(vfeat.size(0),vfeat.size(1),afeat.size(1)))
        # for i in range(vfeat.size(0)):
        #    mat[i] = vfeat[i].view(1,-1).mm(self.relative).mm(afeat[i].view(-1,1))
        # mat = torch.nn.functioanl.tanh(mat)

        relative = self.relative.view(1,self.relative.size(0),self.relative.size(1)).expand(vfeat.size(0),self.relative.size(0),self.relative.size(1))
        afeat=torch.transpose(afeat,2,1)
        mat=torch.nn.functional.tanh(torch.bmm(torch.bmm(vfeat,relative),afeat))
        afeat = torch.transpose(afeat, 2, 1)
        vec1_weight = torch.squeeze(self.matmp(mat))
        vec2_weight = torch.squeeze(self.matmp(torch.transpose(mat,1,2)))
        vec1=torch.sum(vfeat*torch.unsqueeze(vec1_weight,-1),1) #sum up along frame
        vec2=torch.sum(afeat*torch.unsqueeze(vec2_weight,-1),1)
        return F.pairwise_distance(vec1, vec2)

class InterestingCNN(nn.Module):
    def __init__(self, framenum=120):
        #vfeat 128x1024x120
        #afeat 128x128x120
        super(InterestingCNN, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.vconv1a = nn.Conv2d(1,32,(1024,5),padding=(0,2))
        self.vconv1b = nn.Conv2d(1,32,(1024,7),padding=(0,3))
        self.vconv1c = nn.Conv2d(1,32,(1024,9),padding=(0,4))
        self.vconv1d = nn.Conv2d(1,32,(1024,11),padding=(0,5))
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                nn.init.constant(m.bias,0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat_1a=self.vconv1a(vfeat.unsqueeze(1))#128x32x1x120
        vfeat_1a=vfeat_1a.squeeze()
        vfeat_1b=self.vconv1b(vfeat.unsqueeze(1))#128x32x1x120
        vfeat_1b = vfeat_1b.squeeze()
        vfeat_1c=self.vconv1c(vfeat.unsqueeze(1))#128x32x120
        vfeat_1c = vfeat_1c.squeeze()
        vfeat_1d=self.vconv1d(vfeat.unsqueeze(1))#128x32x120
        vfeat_1d = vfeat_1d.squeeze()
        vfeat=torch.cat((vfeat_1a,vfeat_1b,vfeat_1c,vfeat_1d),1)
        vfeat=torch.nn.functional.relu(vfeat)
        vfeat=vfeat.view(vfeat.size(0),-1)
        #vfeat = self.mp(vfeat)
        #vfeat = vfeat.view(-1, 1024)
        #vfeat = F.relu(self.vfc(vfeat))
        #vfeat = self.fc(vfeat)

        # aggregate the auditory features
        #print("afeat")
        #print(afeat)
        print("vfeat")
        print(vfeat)
        #afeat = self.mp(afeat)
        print("afeat_size")
        print(afeat)
        afeat = afeat.contiguous().view(afeat.size(0),-1)
        #afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


# def regularization_loss(model):
#     l1_crit = nn.L1Loss(size_average=False)
#     reg_loss = 0
#     for param in model.parameters():
#         reg_loss += l1_crit(param)
#
#     factor = 0.0005
#     loss = factor * reg_loss
#     return loss

class DeepCNN(nn.Module):
    def __init__(self, framenum=120):
        #vfeat 128x1024x120
        #afeat 128x128x120
        super(DeepCNN, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.vconv1a = nn.Conv2d(1,64,(1152,13),padding=(0,2))
        self.vconv1b = nn.Conv2d(1,64,(1152,15),padding=(0,3))
        self.vconv1c = nn.Conv2d(1,64,(1152,17),padding=(0,4))
        self.vconv1d = nn.Conv2d(1,64,(1152,19),padding=(0,5))

        self.vconv2 = nn.Conv2d(1,256,(256,3),padding=(0,1))
        self.vconv3 = nn.Conv2d(1,128,(256,3),padding=(0,1))
        self.vconv4 = nn.Conv2d(1,64,(128,3),padding=(0,1))
        
        self.vconv5 = nn.Conv2d(1,32,(64,3),padding=(0,1))
        #self.vconv6 = nn.Conv2d(1,16,(32,3),padding=(0,1))
        self.convMP = nn.MaxPool1d(2)
        self.fc = nn.Linear(14*64, 1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                nn.init.constant(m.bias,0)



    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vafeat=torch.cat((vfeat,afeat),1)
        vfeat_1a=self.vconv1a(vafeat.unsqueeze(1))#128x64x1x120
        vfeat_1a=vfeat_1a.squeeze()
        vfeat_1b=self.vconv1b(vafeat.unsqueeze(1))#128x64x1x120
        vfeat_1b = vfeat_1b.squeeze()
        vfeat_1c=self.vconv1c(vafeat.unsqueeze(1))
        vfeat_1c = vfeat_1c.squeeze()#128x64x120
        vfeat_1d=self.vconv1d(vafeat.unsqueeze(1))
        vfeat_1d = vfeat_1d.squeeze()#128x64x120
        #print("0000000000")
        #print(vfeat_1a.size())
        #print(vfeat_1b.size())
        #print(vfeat_1c.size())
        #print(vfeat_1d.size())
        vfeat=torch.cat((vfeat_1a,vfeat_1b,vfeat_1c,vfeat_1d),1)#128x256x120
        vfeat=torch.nn.functional.relu(vfeat)
        #print("=-=-=-=-=-=--=-=-")
        #print(vfeat.size())
        vfeat=self.vconv2(vfeat.unsqueeze(1)).squeeze()
        #print("-------")
        #print(vfeat.size())
        vfeat=self.convMP(vfeat)
        #print("##########")
        #print(vfeat.size())
        
        vfeat=self.vconv3(vfeat.unsqueeze(1)).squeeze()
        #print("********")
        #print(vfeat.size())
        vfeat=self.convMP(vfeat)
        #print("========")
        #print(vfeat.size())
        
        
        #vfeat=self.vconv5(vfeat.unsqueeze(1)).squeeze()
        
        
        vfeat = self.vconv4(vfeat.unsqueeze(1)).squeeze()

        vfeat = self.convMP(vfeat)

        #vfeat = self.vconv5(vfeat.unsqueeze(1)).squeeze()

        #vfeat = self.convMP(vfeat)
        #vfeat = self.vconv6(vfeat.unsqueeze(1)).squeeze()
        #print("&*&*&*&*&*&*&")
        #print(vfeat.size())
        #vfeat = self.convMP(vfeat)
        
        vfeat = vfeat.view(vfeat.size(0),-1)
        vfeat = self.fc(vfeat)
        #print(vfeat)
        #vfeat = torch.nn.functional.sigmoid(vfeat)

        return vfeat

class MutualAttention(nn.Module):
    def __init__(self, framenum=120):
        super(MutualAttention, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        # self.vfc = nn.Linear(1024, 128)
        # self.fc = nn.Linear(128, 96)

        self.fc1 = nn.Linear(1024 + 128, 512)
        # self.lstm1_num_layers=3
        self.lstm1_num_layers = 1
        # self.lstm1_hidden_size=512
        self.lstm1_hidden_size = 256
        # self.lstm1_hidden=self.init_lstm_hidden()
        self.lstm1 = nn.LSTM(input_size=1152, hidden_size=self.lstm1_hidden_size, bias=True, batch_first=True,
                             num_layers=self.lstm1_num_layers, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=self.lstm1_hidden_size * 2, hidden_size=1, bias=True, batch_first=True,
                             num_layers=1, bidirectional=False)

        self.fc2 = nn.Linear(self.lstm1_hidden_size * 2, 128)

        self.fc3 = nn.Linear(128, 2)
        self.sm1 = torch.nn.Softmax()

    def init_lstm_hidden(self, hidden_size, batch_size):
        # return (
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)),
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size)))

        return torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))  # ,
        # torch.autograd.Variable(torch.zeros(hidden_size, batch_size, self.lstm1_hidden_size))

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

        vafeat = torch.cat([vfeat, afeat], 1)  # vfeat [128,1024,120] afeat [128,128,120]
        vafeat = torch.transpose(vafeat, 2, 1)  # after transpose vafeat [128,120,1152]
        # vafeat = self.fc1(vafeat)# after  [128,120,512]

        vafeat, hidden = self.lstm1(
            vafeat)  # ,(self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0)),self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0))))#after [128,120,1024]
        attention, hidden = self.lstm2(
            vafeat)  # ,(self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0)),self.init_lstm_hidden(self.lstm1_hidden_size*2,vafeat.size(0))))#after [128,120,1]
        vafeat = vafeat * attention

        vafeat = torch.transpose(vafeat, 2, 1)  # after [128,1024,120]

        vafeat = self.mp(vafeat)  # maxpool after [128,1024,1]
        vafeat = vafeat.view(vafeat.size(0), -1)
        vafeat = self.fc2(vafeat)
        vafeat = torch.nn.functional.tanh(vafeat)
        vafeat = self.fc3(vafeat)
        vafeat = self.sm1(vafeat)
        return (vafeat[:, 0]).expand(1, vafeat[:, 0].size(0))

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

        self.fc2=nn.Linear(self.lstm1_hidden_size*2,128)

        self.fc3=nn.Linear(128,2)
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
        vafeat = torch.nn.functional.tanh(vafeat)
        vafeat = self.fc3(vafeat)
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
        loss = torch.mean((1-label) * torch.pow(dist, 2).squeeze() +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2).squeeze())

        return loss
class MyCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(MyCrossEntropyLoss,self).__init__()

    def forward(self,softmax,label):
        input=softmax.squeeze()
        print(input.size())
        target=label
        max_val= (-input).clamp(min=0)
        loss = torch.mean(input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        return loss