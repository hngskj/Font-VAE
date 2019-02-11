import glob
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
from random import randrange, randint


canvas = 112

english = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"
hiragana = "あかさたなはまやらわいきしちにひみりうくすつぬふむゆるえけせてねへめれおこそとのほもよろをん"
katakana = "アカサタナハマヤラワイキシチニヒミリウクスツヌフムユルエケセテネヘメレオコソトノホモヨロヲン"
japanese = hiragana + katakana
cyrillic = "АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя"
greek = "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"

case = list(english)
font_list = [filename for filename in glob.glob('font_data/*')]

print(font_list)
print(len(font_list))

# for font_dir in font_list:
#     font_dir = font_dir[10:-4]
#     if not os.path.exists('data/dataset/'+ font_dir):
#        os.makedirs('data/dataset/'+ font_dir)


def gen_letter(font_name, i):
    image = Image.new('L', (canvas, canvas), (0))
    drawer = ImageDraw.Draw(image)

    letter = ""
    for n in range(100):
        letter += case[randrange(len(case))]

    lines = textwrap.fill(letter, width=15)
    margin = randint(-20, 0)
    drawer.text((margin, margin), lines, fill='white', font=font)
    font_name = font_name[9:-4]
    image.save('data/dataset/' + font_name +'/' + str(i) +'.png', 'PNG')

# train 5000 / validation 1000 samples per class
sample_n = 1000
for j in range(len(font_list)):
    print(font_list[j])
    font = ImageFont.truetype(font_list[j], 25)

    for i in range(sample_n):
        # gen_letter(font_list[j], i)
        pass
