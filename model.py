import torch
from torch import nn
from torch.nn import functional as F
 


class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.name, " : ", x.shape)
        return x
  
class KeyGenerator(nn.Module):

    def __init__(self,
                 mul=5,
                 time_step=80,
                 n_mels=80,  
                 frame = 5,
                 keypoints = 37, 
                 multi_landmark = 1
                 ):
        super(KeyGenerator,self).__init__() 

        #[batch, 80, 80] => [batch, 1, 68, 2]
        '''
        self.conv1 = self.conv_block(1, channel,kernel_size=(3,32),stride=(1,1))   
        self.dec1 = self.conv_t_block(channel,channel,kernel_size=(3,3),stride=(1,2)) 
        self.final_pool = nn.AdaptiveAvgPool2d((width, input_size))
        '''  
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=time_step, nhead=8, dropout=0.3, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=time_step//4, nhead=2, dropout=0.3, batch_first=True) 
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=time_step//8, nhead=2, dropout=0.3, batch_first=True) 
        # self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=time_step//8, nhead=4, dropout=0.2, batch_first=True) 
        '''
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=time_step, nhead=4, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
            '''
        
        self.linear1 = self.linear(time_step, time_step//4)
        self.linear2 = self.linear(time_step//4, time_step//8)
        self.linear3 = self.linear(time_step//8  * time_step, time_step//32 * time_step)
        self.final = nn.Linear((time_step//32) * time_step, multi_landmark * keypoints * 2)
        # self.norm = nn.BatchNorm2d()
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.drops = nn.Dropout(0.2)
        self.init_weights() 
        self.keypoints = keypoints 
        self.multi_landmark = multi_landmark  

    def conv_block(self, input_size, output_size, kernel_size=11, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(),
        )
        
    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
 
    def linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size), 
            nn.Hardswish(),
            nn.Dropout(0.3)
        )

    def forward(self, input):  
        # x = self.transformer(input)  
        # x = x.permute(1,0,2)
        # input = input.cuda() 
        x = input + self.encoder_layer1(input)
        x = self.linear1(x)
        x = x + self.encoder_layer2(x) 
        x = self.linear2(x)
        x = x + self.encoder_layer3(x) 
        x = self.flatten(x)
        x = self.linear3(x)  
        x = self.relu(self.final(x))

        return x.view(-1, self.multi_landmark, self.keypoints, 2)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out) 

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)), # torch.Size([16, 16, 128, 128])
            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)), # torch.Size([16, 32, 64, 64])

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)), # torch.Size([16, 64, 32, 32])

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)), # torch.Size([16, 128, 16, 16])

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),      
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),  # torch.Size([16, 256, 8, 8])

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),    
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 512, 4, 4])
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),]) # torch.Size([16, 512, 2, 2])
 

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),), # torch.Size([16, 512, 2, 2])

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 512, 4, 4])

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 512, 8, 8])

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 384, 16, 16])

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 256, 32, 32])

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),),  # torch.Size([16, 128, 64, 64])

            nn.Sequential( # Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            Conv2d(160, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), 
            ),])  # torch.Size([16, 64, 128, 128])
            #Conv2d(40, 64, kernel_size=3, stride=1, padding=1),
        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardsigmoid()) 
        self.pixel = nn.PixelShuffle(2)
    def forward(self, x):
         
        feats = [] 
         
        for f in self.face_encoder_blocks:
            x = f(x)   
            feats.append(x)  

        for f in self.face_decoder_blocks:
            x = f(x) 
            try: 
                x = torch.cat((x, feats[-1]), dim=1)  
            except Exception as e: 
                # print(feats[-1].size())
                raise e
            
            feats.pop()    
        outputs = self.output_block(x)   
        return outputs



if __name__ == '__main__':
    from torchsummary import summary
    # print(torch.__version__)
    m = KeyGenerator().cuda()
    rand = torch.rand((80, 80))#.cuda()
    y = m(rand)
    # print(summary(m,(80,80)))
    # print(y.shape) 
