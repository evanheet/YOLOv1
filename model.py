import torch
import torch.nn as nn


class YOLOv1(nn.Sequential):
    def __init__(self):
        super().__init__(

            # Input dimensions: 448x448x3
            
            # Step 1: Convolution with 64 filters of kernel size 7x7x3, stride 2, padding 3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 224x224x64 - new height/width = floor((448-7+2(3))/2)+1 = 223+1 = 224, new depth = 64

            ### First block: Shows the action of one of the 64 filters of kernel size 7x7x3 sweeping over the 448x448x3 input ###

            # Step 2: Max pooling with size 2x2, stride 2 on input 224x224x64
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output dimensions: 112x112x64 - new height/width = floor(((224-2)/(2))+1) = floor(112) = 112, depth stays the same

            # Step 3: Convolution with 192 filters of kernel size 3x3x64, stride 1, padding 3 on input 112x112x64
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 112x112x192 - new height/width = floor((112-3+2(1))/1)+1 = 111+1 = 112, new depth = 192

            ### Second block: Result of the convolution in step 3 in which the results of each of the 192 filters of size 3x3x64 are stacked ###
            ### Does not represent the action of the convolutions as in the first block ###

            # Step 4: Max pooling with size 2x2, stride 2 on input 112x112x192
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output dimensions: 56x56x192 - new height/width = floor(((112-2)/(2))+1) = floor(56) = 56, depth stays the same

            # Step 5: Convolution with 128 filters of kernel size size 1x1x192, stride 1, padding 0 on input 56*56*192
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 56x56x128 - new height/width = floor((56-1+2(0))/1)+1 = 55+1 = 56, new depth = 128

            # Step 6: Convolution with 256 filters of kernel size 3x3x128, stride 1, padding 1 on input 56x56x128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 56x56x256 - new height/width = floor((56-3+2(1))/1)+1 = 55+1 = 56, new depth = 256

            ### Third block: Result of the convolution in step 6 in which the results of each of the 256 filters of size 3x3x128 are stacked ###

            # Step 7: Convolution with 256 filters of kernel size 1x1x256, stride 1, padding 0 on input 56x56x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 56x56x256 - new height/width = floor((56-1+2(0))/1)+1 = 55+1 = 56, new depth = 256

            # Step 8: Convolution with 512 filters of kernel size 3x3x256, stride 1, padding 1 on input 56x56x256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 56x56x512 - new height/width = floor((56-3+2(1))/1)+1 = 55+1 = 56, new depth = 512

            # Step 9: Max pooling with size 2x2, stride 2 on input 56x56x512
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output dimensions: 28x28x512 - new height/width = floor(((56-2)/(2))+1) = floor(28) = 28, depth stays the same

            # Note: Steps 10 and 11 are done 4 times. I have implemented them once here. The 3 other times are repeated as steps 12-17
            # without explanations as the dimensions from the start of step 10 to the end of step 11 do not change

            # Step 10: Convolution with 256 filters of kernel size 1x1x512, stride 1, padding 0 on input 28x28x512
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 28x28x256 - new height/width = floor((28-1+2(0))/1)+1 = 27+1 = 28, new depth = 256

            # Step 11: Convolution with 512 filters of kernel size 3x3x256, stride 1, padding 1 on input 28x28x256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 28x28x512 - new height/width = floor((28-3+2(1))/1)+1 = 27+1 = 28, new depth = 512

            ### Fourth block: Result of the convolution in step 11 (and 13, 15, and 17) in which the results of each of the 512 filters ###
            ### of size 3x3x256 are stacked ###

            # Steps 12-17:
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 28x28x512

            # Step 18: Convolution with 512 filters of kernel size 1x1x512, stride 1, padding 0 on input 28x28x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 28x28x512 - new height/width = floor((28-1+2(0))/1)+1 = 27+1 = 28, depth stays the same

            # Step 19: Convolution with 1024 filters of kernel size 3x3x512, stride 1, padding 1 on input 28x28x512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 28x28x1024 - new height/width = floor((28-3+2(1))/1)+1 = 27+1 = 28, new depth = 1024

            # Step 20: Max pooling with size 2x2, stride 2 on input 28x28x1024
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output dimensions: 14x14x1024 - new height/width = floor(((28-2)/(2))+1) = floor(14) = 14, depth stays the same

            # Note: Steps 21 and 22 are done 2 times. I have implemented them once here. The other time is repeated as steps 23-24
            # without explanations as the dimensions from the start of step 21 to the end of step 22 do not change

            # Step 21: Convolution with 512 filters of kernel size 1x1x1024, stride 1, padding 0 on input 14x14x1024
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 14x14x512 - new height/width = floor((14-1+2(0))/1)+1 = 13+1 = 14, new depth = 512

            # Step 22: Convolution with 1024 filters of kernel size 3x3x512, stride 1, padding 1 on input 14x14x512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 14x14x1024 - new height/width = floor((14-3+2(1))/1)+1 = 13+1 = 14, new depth = 1024

            ### Fifth block: Result of the convolution in step 22 (and 24 and 25) in which the results of each of the 1024 filters ###
            ### of size 3x3x1024 are stacked ###

            # Steps 23-24:
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 14x14x1024

            # Step 25: Convolution with 1024 filters of kernel size 3x3x1024, stride 1, padding 1 on input 14x14x1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 14x14x1024 - new height/width = floor((14-3+2(1))/1)+1 = 13+1 = 14, depth stays the same

            # Step 26: Convolution with 1024 filters of kernel size 3x3x1024, stride 2, padding 1 on input 14x14x1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 7x7x1024 - new height/width = floor((14-3+2(1))/2)+1 = 6+1 = 7, depth stays the same

            # Step 27: Convolution with 1024 filters of kernel size 3x3x1024, stride 1, padding 1 on input 7x7x1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 7x7x1024 - new height/width = floor((7-3+2(1))/1)+1 = 6+1 = 7, depth stays the same

            ### Sixth block: Result of the convolution in step 27 (and 28) in which the results of each of the 1024 filters ###
            ### of size 3x3x1024 are stacked ###

            # Step 28: Convolution with 1024 filters of kernel size 3x3x1024, stride 1, padding 1 on input 7x7x1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Output dimensions: 7x7x1024 - new height/width = floor((7-3+2(1))/1)+1 = 6+1 = 7, depth stays the same

            # Step 29: Flatten 7x7x1024 input into a 1x50176 tensor and pass through linear layer to obain 1x4096 feature vector
            nn.Flatten(),
            nn.Linear(in_features=50176, out_features=4096),
            nn.Dropout(0.5),

            # Step 30: Pass the 1x4096 feature vector though another linear layer to obtain an 1x1470 feature vector. Unflatten
            # this 1x1470 feature vector into the 30x7x7 detection head of the model
            nn.Linear(in_features=4096, out_features=1470),
            nn.Unflatten(1, (30, 7, 7))
        )
    
model = YOLOv1()
#print(model)
x = torch.rand(1, 3, 448, 448)
result = model(x)
print(result)
