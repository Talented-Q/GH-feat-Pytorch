from visdom import Visdom

vis = Visdom(env='show')

def show_train(fake_img,con_img):
    vis.image(win='fake_img', img=fake_img[0, :], opts=dict(title='fake_img'))
    vis.image(win='con_img', img=con_img[0, :], opts=dict(title='con_img'))

def show_test(real_img,new_img):
    vis.image(win='real_img', img=real_img[0, :], opts=dict(title='real_img'))
    vis.image(win='new_img', img=new_img[0, :], opts=dict(title='new_img'))