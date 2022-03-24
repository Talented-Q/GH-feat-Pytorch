from train_ghfeat import GH_FEAT
from dataset import get_dataset

train_dl, test_dl, length_train, length_test = get_dataset(image_size=256, name='bird')

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    ghfeat = GH_FEAT(resolution=256, train_data=train_dl, test_data=test_dl, epochs=3500000, device='cuda',
                     latent_size=512, img_size=256, structure='fixed')
    ghfeat.train()