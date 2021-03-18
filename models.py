import torch
import torch.nn as nn
import SiLU
import common


class EACD(nn.Module): # Edge Module, ASPP Channel Attention Block and Dual Network
    def __init__(self, channels):
        super(EACD, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        self.aspp = common.ASPP(features)
        self.kernelbox = common.kernelblock(channels)
        self.RCAB = common.RCA_Block(features)
        self.RDB = common.RDB(features, 3)
        self.RG = common.RG(features)

        edgeblock = []
        edgeblock.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        edgeblock.append(SiLU.SiLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        edgeblock.append(SiLU.SiLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        edgeblock.append(SiLU.SiLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        edgeblock.append(SiLU.SiLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        edgeblock.append(SiLU.SiLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,bias=False))
        self.edgeblock = nn.Sequential(*edgeblock)

        layers1 = []
        layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers1.append(SiLU.SiLU())
        for _ in range(15):
            layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers1.append(nn.BatchNorm2d(features))
            layers1.append(SiLU.SiLU())
        layers1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))

        self.one = nn.Sequential(*layers1)

        layers2 = []
        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers2.append(SiLU.SiLU())

        for _ in range(6):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, dilation=2, bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(SiLU.SiLU())

        for _ in range(2):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(SiLU.SiLU())

        for _ in range(6):
            layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, dilation=2, bias=False))
            layers2.append(nn.BatchNorm2d(features))
            layers2.append(SiLU.SiLU())

        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers2.append(SiLU.SiLU())

        layers2.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        self.two = nn.Sequential(*layers2)

        block1 = []
        block1.append(nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=kernel_size, padding=padding))
        # block1.append(nn.BatchNorm2d(features*2))
        block1.append(SiLU.SiLU())
        block1.append(nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=kernel_size,padding=padding))
        self.block1 = nn.Sequential(*block1)

        block2 = []
        block2.append(nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=kernel_size, padding=padding))
        block2.append(nn.BatchNorm2d(features*2))
        block2.append(SiLU.SiLU())
        self.block2 = nn.Sequential(*block2)

        lastblock = []
        lastblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.lastblock = nn.Sequential(*lastblock)

        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding)
        self.conv32 = nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=kernel_size, padding=padding)
        self.featconv = nn.Conv2d(in_channels=features*3, out_channels=features, kernel_size=kernel_size, padding=padding)
        self.feat2conv = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=channels*4, out_channels=features, kernel_size=kernel_size, padding=padding)
        self.feat4conv = nn.Conv2d(in_channels=features*6, out_channels=features*2, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        edge = x - self.edgeblock(x)
        featout = self.kernelbox(x)
        catdata = self.conv2(torch.cat((x, featout, x+edge, edge), 1))

        rdbdata = self.RDB(self.RDB(catdata))
        asppdata = self.aspp(rdbdata) + rdbdata
        rcab1 = self.RCAB(asppdata)
        rcab2 = self.RCAB(rcab1)
        inputdata = self.conv3(self.RCAB(rcab2)) + asppdata

        # Layer1
        out1 = self.one(inputdata)
        out1cat = self.feat2conv(torch.cat((out1,inputdata), 1))
        # Layer2
        out2 = self.two(inputdata)
        out2cat = self.feat2conv(torch.cat((out2, inputdata), 1))

        dual_cat = self.feat2conv(torch.cat((out1cat + inputdata, out2cat + inputdata), 1))
        rdbdata2 = self.RDB(self.RDB(dual_cat))
        asppdata2 = self.aspp(rdbdata2) + rdbdata2
        aspp = self.aspp(asppdata2) + asppdata2
        rcab21 = self.RCAB(aspp)
        rcab22 = self.RCAB(rcab21)
        lastdata = self.conv3(self.RCAB(rcab22)) + aspp

        rg1 = self.RG(lastdata)
        rg2 = self.RG(rg1)
        rg3 = self.RG(rg2)
        lastrg = self.conv3(rg3) + lastdata

        last = self.lastblock(lastrg)

        out = last + residual

        return out, edge