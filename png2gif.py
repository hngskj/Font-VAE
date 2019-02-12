from PIL import Image, ImageFont, ImageDraw
import glob

dir = "font_vae_cnn\plot_e200c10/"
epoch = 80

font=ImageFont.truetype("font_data/NotoSans-Regular.ttf", 30)

name_list = [filename for filename in glob.glob(dir+'font_vae_mean*.png')]
name_list = name_list[:epoch]
img0_name = name_list[0]

image = Image.open(img0_name)
epoch_draw0 = Image.new('RGBA', (200,50), "white")
text_draw0 = ImageDraw.Draw(epoch_draw0)
text_draw0.text((0,0), "epoch = 1/"+str(epoch), font=font, fill="black")
image.paste(epoch_draw0, (100, 50))

images_list = []

for idx, img in enumerate(name_list):
    _img = Image.open(img)

    epoch_draw = Image.new('RGBA', (200,50), "white")
    text_draw = ImageDraw.Draw(epoch_draw)
    text = "epoch = "+str(idx+1)+"/"+str(epoch)
    text_draw.text((0,0), text, font=font, fill="black")
    _img.paste(epoch_draw, (100, 50))
    images_list.append(_img)

# print(images_list)
image.save(dir+"_plot.gif",
           save_all=True,
           append_images=images_list[1:],
           duration=100,
           loop=0)
