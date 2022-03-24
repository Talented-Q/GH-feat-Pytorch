import torch
from Encoder import Encoder,Bottleneck
from stylegan import GMapping,GSynthesis,Discriminator
import numpy as np
import random
from torch.optim import lr_scheduler
from loss import vector_loss,construct_loss,dis_loss,gen_loss
from utils import show_train, show_test

class GH_FEAT:

    def __init__(self,resolution, train_data, test_data, epochs, device, latent_size, img_size, structure):
        self.data = train_data
        self.test = test_data
        self.epochs = epochs
        self.device = device
        self.latent_size = latent_size
        self.style_mixing_prob = 0.9
        self.alpha = 1.0
        self.num_layers = (int(np.log2(resolution)) - 1) * 2  # 14
        self.depth = int(np.log2(resolution)) - 2  # 6
        self.G = GSynthesis(resolution=img_size,structure=structure).to(device)
        self.G.load_state_dict(torch.load(r'C:\Users\Tom.riddle\Desktop\code\ghfeat-pytorch\model\G.pth'))
        self.S = GMapping(dlatent_broadcast=self.num_layers).to(device)
        self.S.load_state_dict(torch.load(r'C:\Users\Tom.riddle\Desktop\code\ghfeat-pytorch\model\S.pth'))
        self.D = Discriminator(resolution=img_size,structure=structure).to(self.device)
        self.E = Encoder(block=Bottleneck,layers=[3,4,6,3]).to(self.device)

        self.E_opt = torch.optim.Adam(self.E.parameters(), lr=0.0001, betas=(0.9,0.99))
        self.E_scheduler = lr_scheduler.StepLR(self.E_opt, step_size=30000, gamma=0.8)
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.9,0.99))
        self.D_scheduler = lr_scheduler.StepLR(self.D_opt, step_size=30000, gamma=0.8)

    def train(self):
        self.G.eval()
        self.S.eval()

        for epoch in range(self.epochs):
            e_loss = 0
            d_loss = 0
            train_img,_ = next(iter(self.data))
            train_img = train_img.to(self.device)
            train_img = train_img.requires_grad_()

            con_vec = self.E(train_img)
            con_images = self.G(con_vec, self.depth, self.alpha)
            fake_out = self.D(con_images, self.depth, self.alpha)

            # v_loss = vector_loss(dlatents_in,con_vec)
            con_loss = construct_loss(train_img,con_images)
            g_loss = gen_loss(fake_out)

            E_loss = con_loss + g_loss

            self.E_opt.zero_grad()
            E_loss.backward()
            self.E_opt.step()

            real_out = self.D(train_img, self.depth, self.alpha)
            wrong_out = self.D(con_images.detach(), self.depth, self.alpha)
            D_loss = dis_loss(real_out,wrong_out,train_img)

            self.D_opt.zero_grad()
            D_loss.backward()
            self.D_opt.step()

            self.E_scheduler.step()
            self.D_scheduler.step()

            with torch.no_grad():
                e_loss += E_loss.item()
                d_loss += D_loss.item()
                e_loss = e_loss/train_img.shape[0]
                d_loss = d_loss/train_img.shape[0]
                print("Epoch: {}, Discrimiantor Loss: {:.3f}, Encoder Loss: {:.3f}, E lr:{}, D lr:{}".format(
                        epoch, d_loss, e_loss, self.E_opt.state_dict()['param_groups'][0]['lr'], self.D_opt.state_dict()['param_groups'][0]['lr']
                    ))

            show_train(train_img,con_images)
            torch.save(self.E.state_dict(), r'C:\Users\Tom.riddle\Desktop\code\ghfeat-pytorch\model/encoder.pth')
            torch.save(self.D.state_dict(), r'C:\Users\Tom.riddle\Desktop\code\ghfeat-pytorch\model/discriminator.pth')
            # if epoch % 4000 == 0:
            #     test_img = next(iter(self.test))
            #     test_vec = self.E(test_img)
            #     con_test = self.G(test_vec, self.depth, self.alpha)
            #     show_test(test_img,con_test)





