
import torch
import torch.nn as nn


from torch.optim import SGD
import torch.nn.functional as F


from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


from torchinfo import summary

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()

        # Depthwise convolution - The number of groups is equivalent to the number of channels which makes the convolution be performed to each channel independently.
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)

        # Pointwise convolution = i used a 1x1 kernel to combine  information accross channels and project the features to a new space.It transforms teh number of channesl from in_channel to out_channels 
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        #Input Channels: The number of input channels (in_channels) corresponds to the depth or the number of features at each spatial location.
        #Output Channels: The number of output channels (out_channels) corresponds to the number of filters or features that the convolutional layer is going to produce.
        # pointwise convolution (1x1) performs a linear combination of input channels at each spatial location, resulting in an output with a new set of channels. The weights for this linear combination are learned during the training process, providing the model with the flexibility to capture different relationships and patterns across channels


        #Normalizes the output of the pointwise convolution. Batch normalization helps stabilize and accelerate the training process by normalizing the activations
        self.bn = nn.BatchNorm2d(out_channels) #self.bn = nn.BatchNorm2d(out_channels)
        #choice of applying the BatchNorm2d fter the pointwise conv and before the PReLu to stabilize te inputs of the activation function

        #Applies the PReLU activation function to the batch-normalized output. PReLU introduces learnable parameters to the standard ReLU activation.
        self.relu = nn.PReLU()
        #PReLU introduces a learnable parameter, allowing the slope of the negative part of the activation to be adjusted during training.
        #mathematically is it equivalento to:
        #PReLU(x) -> x;x>=0
        #PReLU(x) -> alpha . x;x<0

    #defines the forward pass of the network, specifying how the input data is transformed through the layers of the network to produce the final output
    def forward(self, x):
        #Applies depthwise separable convolution operation, which consists of depthwise convolution, pointwise convolution, batch normalization, and activation. 
        out = self.relu(self.bn(self.pointwise_conv(self.depthwise_conv(x))))
        return out

#see InvertedResidualBlock , essentially the same explanations but without the connection to output
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super(BottleneckBlock, self).__init__()

        expanded_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, in_channels //2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels //2)
        self.relu = nn.PReLU()

        #depthwise conv
        self.depthwise_conv = DepthwiseSeparableConv2d(in_channels //2, in_channels //2, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels //2)


        self.conv3 = nn.Conv2d(in_channels //2, expanded_channels , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded_channels )
        

        # Remove the shortcut connection
        self.shortcut = nn.Sequential()


    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.depthwise_conv(out)))
        out = self.bn3(self.conv3(out))
 
        return out

 

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion,stride=1):
        #is calling the constructor of the parent class nn.Module 
        super(InvertedResidualBlock, self).__init__()

        
        ###### Bloque de Expansion: narrow to wide
        # 1x1 convolution is applied to the input tensor, changing the number of channels.The key role of the 1x1 convolution with expansion is to change the number of channels. 
        # this convolution is used to expand the low-dimensional input feature map to a higher-dimensional space suited to non-linear activations
        expanded_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)


        self.bn1 = nn.BatchNorm2d(expanded_channels)
        #MobiFace uses PReLU for non linearity
        #PReLU introduces a learnable parameter, allowing the slope of the negative part of the activation to be adjusted during training.
        #mathematically is it equivalent to:
        #PReLU(x) -> x;x>=0
        #PReLU(x) -> alpha . x;x<0
        self.relu = nn.PReLU()

        ####### Wide to wide
        #A depthwise separable convolution is applied to the result of the previous step to achieve spatial filtering of hight dimensional tensor
        self.depthwise_conv = DepthwiseSeparableConv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(expanded_channels)


        ###### wide to narrow
        #pointwise convolution linear convolution
        #spatially-filtered feature map is projected back to a low-dimensional subspace
        self.conv3 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)


        #he shortcut connection is a form of a residual connection/skip connection
        #Its purpose is to enable the smooth flow of gradients during backpropagation, aiding in the training of deep networks.
        #The shortcut helps mitigate potential vanishing or exploding gradient problems by providing a direct path for information flow.
        #shortcut is a sequential module that represents a shortcut connection.It is designed to connect the input directly to the output of the block, bypassing the internal transformations, if certain conditions are met.
        self.shortcut = nn.Sequential()
        #checks whether the number of input channels is not equal to the number of output channels after expansion. If this condition is true, it implies that there is a change in the number of channels, and a shortcut connection is needed to match dimensions.
        if stride != 1 or in_channels != out_channels:
            print('shortcut')
            self.shortcut = nn.Sequential(
                #If the condition is met, a shortcut connection is created using a 1x1 convolution followed by batch normalization.The 1x1 convolution adjusts the number of channels, ensuring compatibility for element-wise addition with the output of the block.
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels ),
                
            )

    #defines the forward pass of the network, specifying how the input data is transformed through the layers of the network to produce the final outpu
    def forward(self, x):
        #expansion
        out = self.relu(self.bn1(self.conv1(x)))

        #depthwise conv
        out = self.relu(self.bn2(self.depthwise_conv(out)))

        #linear activation
        out = self.bn3(self.conv3(out))
        

        # Shortcut Connection -  the shortcut is applied during the forward pass - it is adding the original input tensor x to output tensor
        # effectively acting as a residual connection, helping to create a shortcut path for information flow and facilitating gradient propagation during backpropagation.
        out += self.shortcut(x)


        return out

class MobiFace(nn.Module):
    def __init__(self):
        super(MobiFace, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.depthwise_conv = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bottleneck blocks followed by nverted Residual bottleneck blocks
        self.bottleneck_block1 = BottleneckBlock(64, 64, expansion=1, stride=2)
        self.residual_block1 = InvertedResidualBlock(64, 64, expansion=2)
        self.residual_block1_2 = InvertedResidualBlock(64, 64, expansion=2)
        
        self.bottleneck_block2 = BottleneckBlock(64, 128, expansion=2, stride=2)

        self.residual_block2 = InvertedResidualBlock(128, 128, expansion=2)
        self.residual_block2_2 = InvertedResidualBlock(128, 128, expansion=2)
        self.residual_block2_3 = InvertedResidualBlock(128, 128, expansion=2)
 
        
        self.bottleneck_block3 = BottleneckBlock(128, 256, expansion=2, stride=2)
        self.residual_block3 = InvertedResidualBlock(256, 256, expansion=2)
        self.residual_block3_2 = InvertedResidualBlock(256, 256, expansion=2)
        self.residual_block3_3 = InvertedResidualBlock(256, 256, expansion=2)
        self.residual_block3_4 = InvertedResidualBlock(256, 256, expansion=2)
        self.residual_block3_5 = InvertedResidualBlock(256, 256, expansion=2)
        self.residual_block3_6 = InvertedResidualBlock(256, 256, expansion=2)

        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512,512)
        

    def forward(self, x):

        #some print statements were added in case of checking the input shape transformation accross the network 
        

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.depthwise_conv(out)))
        
        
        # First bottleneck block followed by residual block
        out = self.bottleneck_block1(out)
        
        out = self.residual_block1(out)
        out = self.residual_block1_2(out)
        
        
        # Second bottleneck block followed by residual block
        out = self.bottleneck_block2(out)
        out = self.residual_block2(out)
        out = self.residual_block2_2(out)
        out = self.residual_block2_3(out)
        

        # Third bottleneck block followed by residual block
        out = self.bottleneck_block3(out)
        out = self.residual_block3(out)
        out = self.residual_block3_2(out)
        out = self.residual_block3_3(out)
        out = self.residual_block3_4(out)
        out = self.residual_block3_5(out)
        out = self.residual_block3_6(out)

        
        
        out = self.bn3(self.conv3(out))
        out = torch.mean(out, dim=[2, 3])  # Global Average Pooling
        out = self.fc(out)
        return out

# Create an instance of the MobiFace model
model = MobiFace()

# Print the model architecture
print(model)