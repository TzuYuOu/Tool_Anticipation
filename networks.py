import torch
from torch import nn
import torchvision.models as models

class FeatureAlexNet(nn.Module):
  def __init__(self, num_features = 4096):
    super().__init__()
    self.alexnet = models.alexnet(pretrained=True)
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.fc = nn.Sequential(nn.Linear(256 * 6 * 6, num_features),nn.ReLU(inplace=True))
  
  def forward(self, x):
    x = self.alexnet.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
  
    return x

  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

class FeatureVGGNet(nn.Module):
  def __init__(self, num_features = 4096):
    super().__init__()
    self.vggnet = models.vgg16(pretrained=True)
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.fc = nn.Sequential(nn.Linear(512*7*7, num_features),nn.ReLU(inplace=True))
  
  def forward(self, x):
    x = self.vggnet.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
  
    return x

  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

class FeatureResNet(nn.Module):
  def __init__(self, num_features=1024):
    super().__init__()
    self.resnet = models.resnet18(pretrained=True)
    # self.fc = nn.Sequential(nn.Linear(2048, num_features),nn.ReLU(inplace=True))
  
  def forward(self, x):
    x = self.resnet.conv1(x)
    x = self.resnet.bn1(x)
    x = self.resnet.relu(x)
    x = self.resnet.maxpool(x)

    x = self.resnet.layer1(x)
    x = self.resnet.layer2(x)
    x = self.resnet.layer3(x)
    x = self.resnet.layer4(x)

    x = self.resnet.avgpool(x)
    x = x.view(x.size(0), -1)
    # x = self.fc(x)
    
    return x

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size):
      super().__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.cell = nn.LSTMCell(input_size,hidden_size)

  def forward(self, sequence, hidden_state):
      h,c = hidden_state
      output = []
      for x in sequence:
        h,c = self.cell(x,(h,c))
        output.append(h.view(1,h.size(0),-1))
      output = torch.cat(output,0)
      return output, (h,c)
      
  def init_hidden(self):
    return (torch.zeros(1,self.hidden_size).cuda(),
            torch.zeros(1,self.hidden_size).cuda())
  
  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

class LSTMAlexNet(nn.Module):
  def __init__(self, num_classes, lstm_size=512, lstm_input_size=4096, pretrain=None):
    super().__init__()
    self.featureNet = FeatureAlexNet(num_features=lstm_input_size)
    self.lstm = LSTM(lstm_input_size, lstm_size)
    self.classifier = nn.Linear(lstm_size, num_classes)

    if pretrain is not None:
      self.load(pretrain)

  def forward(self, x, hidden_state):
    x = self.featureNet.forward(x)
    x = x.view(x.size(0), 1, -1)
    x, hidden_state = self.lstm(x, hidden_state)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x, hidden_state

  def init_hidden(self):
    return self.lstm.init_hidden()

  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

class LSTMResNet(nn.Module):
  def __init__(self, num_classes, lstm_size=256, lstm_input_size=512, pretrain=None):
    super().__init__()
    self.featureNet = FeatureResNet(num_features=lstm_input_size)
    self.lstm = LSTM(lstm_input_size, lstm_size)
    self.classifier = nn.Linear(lstm_size, num_classes)

    if pretrain is not None:
      self.load(pretrain)

  def forward(self, x, hidden_state):
    x = self.featureNet.forward(x)
    x = x.view(x.size(0), 1, -1)
    x, hidden_state = self.lstm(x, hidden_state)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x, hidden_state

  def init_hidden(self):
    return self.lstm.init_hidden()

  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

class TimeLSTMAlexNet(nn.Module):
  def __init__(self, num_classes, lstm_size=512, lstm_input_size=4096, pretrain=None):
    super().__init__()
    self.featureNet = FeatureAlexNet(num_features=lstm_input_size)
    self.lstm = nn.LSTM(lstm_input_size+1, lstm_size, batch_first=True)
    self.classifier = nn.Linear(lstm_size, num_classes)
    nn.init.xavier_normal_(self.lstm.all_weights[0][0])
    nn.init.xavier_normal_(self.lstm.all_weights[0][1])
    nn.init.xavier_uniform_(self.classifier.weight)

    if pretrain is not None:
      self.load(pretrain)

  def forward(self, x, t_elapsed):
    x = self.featureNet.forward(x)
    t_elapsed = t_elapsed.view(t_elapsed.size(0), -1)
    x = torch.cat((x, t_elapsed), 1)
    x = x.view(1, x.size(0), -1)
    y, _ = self.lstm(x)
    y = y.view(y.size(0), -1)
    y = self.classifier(y)
    return y

  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

