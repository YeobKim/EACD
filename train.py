import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
# from dataset import prepare_data, Dataset
from utils import *
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from imageio import imwrite
import PIL
from PIL import Image
from PIL import ImageFilter
from datetime import datetime
from dataset import *
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import torchsummary.summary as summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="NTIRE_DeBlur_Challenge")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=100, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--patchsize", type=int, default=128, help='patch size of image')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, batchSize=opt.batchSize).data
    dataset_val = Dataset(train=False).data
    # loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True) # if debug, num_workers=0 not 4
    # loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # toPILimg = transforms.Compose([transforms.ToPILImage()])
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Build model
    net = EACD(channels=3)
    # net.apply(weights_init_kaiming)
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()
    # Move to GPU
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = net.cuda()
    # summary(net, (3, 128, 128))
    criterion.cuda()
    criterion2.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    start_time = datetime.now()
    print('Training Start!!')
    print(start_time)

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr # 1e-4
        elif (epoch >= opt.milestone) and (epoch < 200):
            current_lr = opt.lr / 10. # 1e-5
        else:
            current_lr = opt.lr / 100. # 1e-6
        # current_lr = opt.lr * (0.5 ** ((epoch+ 1) // opt.step))

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        loss_val = 0

        # train
        for i, (blur_train, gt_train) in enumerate(dataset_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # Make Edge Image.
            original_edge = torch.FloatTensor(blur_train.shape[0], 3, opt.patchsize, opt.patchsize)

            for j in range(blur_train.shape[0]):
                original_edge[j] = toTensor((toPILImg(gt_train[j]).filter(ImageFilter.FIND_EDGES)))

            # gt = utils.make_grid(gt_train.data, nrow=8, normalize=True, scale_each=True)
            # blur = utils.make_grid(blur_train.data, nrow=8, normalize=True, scale_each=True)
            # edge = utils.make_grid(original_edge.data, nrow=8, normalize=True, scale_each=True)
            # fig = plt.figure()
            # rows = 1
            # cols = 3
            #
            # ax1 = fig.add_subplot(rows, cols, 1)
            # # tensor는 cuda에서 처리하지 못하기 때문에 .cpu()로 보내줌.
            # ax1.imshow(np.transpose(gt.cpu(), (1, 2, 0)))
            # ax1.set_title('clean image')
            #
            # ax2 = fig.add_subplot(rows, cols, 2)
            # ax2.imshow(np.transpose(blur.cpu(), (1, 2, 0)))
            # ax2.set_title('noisy image')
            #
            # ax3 = fig.add_subplot(rows, cols, 3)
            # ax3.imshow(np.transpose(edge.cpu(), (1, 2, 0)))
            # ax3.set_title('edge image')
            #
            # plt.show()
            # activation = {}
            # def get_activation(name):
            #     def hook(model, input, feat):
            #         activation[name] = feat.detach()
            #     return hook
            # model.Edge_Net.register_forward_hook(get_activation('Edge_Net'))

            blur_train, gt_train = Variable(blur_train.cuda()), Variable(gt_train.cuda())
            original_edge = Variable(original_edge.cuda())

            out_train, edge = model(blur_train)

            # featmap = activation['Edge_Net']
            # featmapplot = utils.make_grid(featmap.data, nrow=8, normalize=True, scale_each=True)
            #
            # plt.imshow(np.transpose(featmapplot.cpu(), (1, 2, 0)), cmap="gray")
            # plt.title('Edge_Net')
            # plt.show()
            if epoch < 200:
                blur_loss = criterion(out_train, gt_train)
                edge_loss = criterion(edge, original_edge)
            else :
                blur_loss = criterion2(out_train, gt_train)
                edge_loss = criterion(edge, original_edge)
            # edge_loss = 1 - ms_ssim(edge, original_edge, data_range=1, win_size=5)

            loss = blur_loss + 0.5*edge_loss
            # loss = 0.5 * noise_loss + 0.5 * edge_loss
            loss_val += loss.item()

            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train, edge = model(blur_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, gt_train, 1.)
            # i%100 == 0 -> each 100 epochs, print loss and psnr.
            if i % 1000 == 0 :
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch+1, i+1, len(dataset_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if i % 2000 == 0:
                Img = utils.make_grid(gt_train[0].data, nrow=8, normalize=True, scale_each=True)
                blurImg = utils.make_grid(blur_train[0].data, nrow=8, normalize=True, scale_each=True)
                edgeImg = utils.make_grid(edge[0].data, nrow=8, normalize=True, scale_each=True)
                Irecon = utils.make_grid(out_train[0].data, nrow=8, normalize=True, scale_each=True)
                # writer.add_image('clean image', Img, epoch)
                # writer.add_image('noisy image', Imgn, epoch)
                # writer.add_image('reconstructed image', Irecon, epoch)

                # Compare clean, noisy, denoising image
                fig = plt.figure()
                fig.suptitle('Unet_Test %d' % (epoch + 1))
                rows = 2
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                # tensor는 cuda에서 처리하지 못하기 때문에 .cpu()로 보내줌.
                ax1.imshow(np.transpose(Img.cpu(), (1,2,0)), cmap="gray")
                ax1.set_title('gt image')

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(np.transpose(blurImg.cpu(), (1,2,0)), cmap="gray")
                ax2.set_title('blur image')

                ax3 = fig.add_subplot(rows, cols, 3)
                ax3.imshow(np.transpose(edgeImg.cpu(), (1,2,0)), cmap="gray")
                ax3.set_title('edge image')

                ax4 = fig.add_subplot(rows, cols, 4)
                ax4.imshow(np.transpose(Irecon.cpu(), (1, 2, 0)), cmap="gray")
                ax4.set_title('deblur image [%.4f]' % loss.item())

                # plt.savefig('./fig_result/epoch_{:d}.png'.format(epoch + 1))
                plt.show()

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # the end of each epoch
        # model.eval()
        loss_val /= len(dataset_train)
        print("Average Loss : %.4f" % (loss_val))

        midtime = datetime.now() - start_time
        print(midtime)

        if ((epoch + 1) % 5 == 0) or (loss_val <= 0.058) or (epoch == 0):
            torch.save(model.state_dict(), os.path.join(opt.outf, 'EACD_' + str(epoch + 1) + "_" + str(round(loss_val, 4)) + '.pth'))

    end_time = datetime.now()
    print('Training Finished!!')
    print(end_time)

if __name__ == "__main__":
    main()
