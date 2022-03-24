import torch
import torch.nn as nn
import torchvision

l2_loss = nn.MSELoss()
# Encoder loss function on W space.

def vector_loss(input_vec,encode_vec):
    return l2_loss(encode_vec,input_vec)

# Encoder loss function on image space.

VGG16=torchvision.models.vgg16(pretrained=True)
VGG16=VGG16.to('cuda')
for parm in VGG16.parameters():
    parm.requires_grad=False

def get_feature(model,x):
    out=[]
    for i,hidden in enumerate(list(model.features)):
        x=hidden(x)
        if i==2 or i==5 or i==9 or i==13 or i==17:
             out.append(torch.flatten(x,start_dim=1))
    out = torch.cat([feature for feature in out], dim=1)
    return out

def construct_loss(input_img,con_img,feature_scale=0.00005):
    vgg_real_input = ((input_img + 1) / 2) * 255
    vgg_fake_input = ((con_img + 1) / 2) * 255
    vgg_feature_real = get_feature(VGG16,vgg_real_input)
    vgg_feature_fake = get_feature(VGG16,vgg_fake_input)
    recon_loss_feats = feature_scale * l2_loss(vgg_feature_fake,vgg_feature_real)
    recon_loss_pixel = l2_loss(con_img,input_img)
    recon_loss = recon_loss_pixel + recon_loss_feats
    return recon_loss

# adv loss function on discriminator and generator
def R1Penalty(real_img, real_logit):

    # real_img = torch.autograd.Variable(real_img, requires_grad=True)
    # real_logit = self.dis(real_img, height, alpha)
    # real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                     grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                     create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
    # real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty


def dis_loss(r_preds, f_preds, real_img, r1_gamma=10.0):
    # Obtain predictions
    # r_preds = self.dis(real_samps, height, alpha)
    # f_preds = self.dis(fake_samps, height, alpha)

    loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

    if r1_gamma != 0.0:
        r1_penalty = R1Penalty(real_img, r_preds) * (r1_gamma * 0.5)
        loss += r1_penalty

    return loss


def gen_loss(f_preds, D_scale=0.08):
    # f_preds = self.dis(fake_samps, height, alpha)
    return D_scale * torch.mean(nn.Softplus()(-f_preds))

