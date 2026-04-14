import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from network import mapping_dataset
from MVCUnet_model import MVCUnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import torch.optim as optim
from funct import calculate_psnr_or_ssim, figure_save

import matplotlib
matplotlib.use('TKAgg')

def main():

    pha = 256
    fre = 256
    train_root_dir = 'your_traindata'
    test_root_dir = 'your_valid_data'

    pth_save_file = 'weight_save'

    test_figure_save_file = 'figure_save'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_rate = 5e-4
    epoch_num = 200

    transform1 = transforms.ToTensor()
    batch_size = 1
    map_mission = 6

    input_ch = [0,1,2,3,4,5]
    input_chnum = 6

    label_scale = 10
    label_incre_degree = 20


    train_dataset = mapping_dataset(train_root_dir, transform1, pha, fre, map_mission, input_ch, is_aug=False)
    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=batch_size, shuffle=True,
                              drop_last=True)

    test_dataset = mapping_dataset(test_root_dir, transform1, 256, 256, map_mission, input_ch, is_aug=False)



    net = MVCUnet(in_cn=input_chnum,out_cn=1,hidden_cn=32)
    net = net.to(device)



    criterion = nn.L1Loss()
    criterion = criterion.to(device)


    optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=0)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, min_lr=5e-8,
                                                    verbose=True)


    writer = SummaryWriter("log")
    step = 0

    #  ——————————————————————————————————train————————————————————————————————————
    for epoch in range(epoch_num):

        print(f'current lr: {optimizer.param_groups[0]["lr"]}')

        net.train()
        for i, data_tem in enumerate(train_loader):
            net.zero_grad()
            optimizer.zero_grad()  
            in_data, label_data = data_tem
            in_data = in_data.to(device)
            label_data = label_data.to(device)
            net_out = net(in_data)
            train_loss = criterion(net_out, label_data)
            train_loss.backward()
            optimizer.step()
            print(f'Training ^_^----'
                f'epoch:{epoch + 1}, number:{i + 1}/minibatch_num:{len(train_loader)}, train_loss:{train_loss.item()}')

        if epoch % 1 == 0:
            writer.add_scalar('train_loss', train_loss, global_step=epoch+1)
        # ——————————————————————————————————valid————————————————————————————————————
        net.eval()
        total_test_loss = 0
        total_test_psnr = 0
        total_test_ssim = 0

        with torch.no_grad():
            for i1, (test_data, test_label) in enumerate(test_dataset):
                test_data = test_data.to(device)
                test_label = test_label.to(device)
                test_data = torch.unsqueeze(test_data, dim=0)
                test_label = torch.unsqueeze(test_label, dim=0)
                testdata_out = net(test_data)

                test_loss_temp = criterion(testdata_out, test_label)
                test_psnr_temp = calculate_psnr_or_ssim(test_label, testdata_out, data_range=label_scale, goal=1, incre_degree=label_incre_degree)
                test_ssim_temp = calculate_psnr_or_ssim(test_label, testdata_out, data_range=label_scale, goal=2, incre_degree=label_incre_degree)

                total_test_loss += test_loss_temp
                total_test_psnr += test_psnr_temp
                total_test_ssim += test_ssim_temp

                print(f'validDataSet--'
                      f'epoch:{epoch + 1}, number:{i1 + 1}/{len(test_dataset)}, test_loss:{test_loss_temp.item()}, '
                      f'psnr:{test_psnr_temp}, ssim:{test_ssim_temp}')

                if (epoch % 25) == 0 and (i1 % 1) == 0:
                    label_img_file = f'{test_figure_save_file}/label_epoch{epoch+1}_{i1}.png'
                    testout_img_file = f'{test_figure_save_file}/testout_epoch{epoch+1}_{i1}.png'
                    figure_save(label_img_file, testout_img_file, test_label, testdata_out, scale_degree=label_incre_degree)

            aver_test_loss = total_test_loss / len(test_dataset)
            aver_test_psnr = total_test_psnr / len(test_dataset)
            aver_test_ssim = total_test_ssim / len(test_dataset)

            print(f'TestDataSet--'
                  f'epoch:{epoch}  aver_loss:{aver_test_loss.item()}, aver_psnr:{aver_test_psnr}, aver_ssim:{aver_test_ssim}')

            if epoch % 1 == 0:
                writer.add_scalar('test_loss', aver_test_loss, global_step=epoch+1)
                writer.add_scalar('test_psnr', aver_test_psnr, global_step=epoch+1)
                writer.add_scalar('test_ssim', aver_test_ssim, global_step=epoch+1)

        torch.save(net.state_dict(), pth_save_file)
        print(f'save weight after epoch:{epoch + 1}')
        schedule.step(aver_test_loss)

    writer.close()


if __name__ == "__main__":
    main()
