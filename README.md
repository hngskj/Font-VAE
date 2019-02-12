# Analysis of font shape <br/> using Variational Autoencoder with Convnets



---


### VAE architecture

The VAE has a modular design. The encoder, decoder and VAE are 3 models that share weights. After training the VAE model, the encoder can be used to generate latent vectors. The decoder can be used to generate font images by sampling the latent vector from a Gaussian distribution with mean=0 and std=1.

encoder | decoder
------------ | -------------
![](summary/font_vae_cnn_encoder.png) | ![](summary/font_vae_cnn_decoder.png)

---

### Used dataset
dataset size = train 5000 & validation 1000 per each class<br/>
width, height = 112, 112<br/>
font size = 25<br/>
used characters = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"<br/>
images are generated with [font2png.py](font2png.py)

idx | font name | sample images
------------ | ------------ | -------------
0 | EBGaramond | ![](example_dataset/0_EBGaramond-Regular1.png) ![](example_dataset/0_EBGaramond-Regular2.png) ![](example_dataset/0_EBGaramond-Regular3.png) ![](example_dataset/0_EBGaramond-Regular4.png) ![](example_dataset/0_EBGaramond-Regular5.png)
1 | PT_Serif | ![](example_dataset/1_PT_Serif-Web-Regular1.png) ![](example_dataset/1_PT_Serif-Web-Regular2.png) ![](example_dataset/1_PT_Serif-Web-Regular3.png) ![](example_dataset/1_PT_Serif-Web-Regular4.png) ![](example_dataset/1_PT_Serif-Web-Regular5.png)
2 | NotoSans | ![](example_dataset/2_NotoSans-Regular1.png) ![](example_dataset/2_NotoSans-Regular2.png) ![](example_dataset/2_NotoSans-Regular3.png) ![](example_dataset/2_NotoSans-Regular4.png) ![](example_dataset/2_NotoSans-Regular5.png)
3 | Roboto | ![](example_dataset/3_Roboto-Regular1.png) ![](example_dataset/3_Roboto-Regular2.png) ![](example_dataset/3_Roboto-Regular3.png) ![](example_dataset/3_Roboto-Regular4.png) ![](example_dataset/3_Roboto-Regular5.png)
4 | Righteous | ![](example_dataset/4_Righteous-Regular1.png) ![](example_dataset/4_Righteous-Regular2.png) ![](example_dataset/4_Righteous-Regular3.png) ![](example_dataset/4_Righteous-Regular4.png) ![](example_dataset/4_Righteous-Regular5.png)
5 | Bangers | ![](example_dataset/5_Bangers-Regular1.png) ![](example_dataset/5_Bangers-Regular2.png) ![](example_dataset/5_Bangers-Regular3.png) ![](example_dataset/5_Bangers-Regular4.png) ![](example_dataset/5_Bangers-Regular5.png)
6 | Pacifico | ![](example_dataset/6_Pacifico-Regular1.png) ![](example_dataset/6_Pacifico-Regular2.png) ![](example_dataset/6_Pacifico-Regular3.png) ![](example_dataset/6_Pacifico-Regular4.png) ![](example_dataset/6_Pacifico-Regular5.png)
7 | DancingScript | ![](example_dataset/7_DancingScript-Regular1.png) ![](example_dataset/7_DancingScript-Regular2.png) ![](example_dataset/7_DancingScript-Regular3.png) ![](example_dataset/7_DancingScript-Regular4.png) ![](example_dataset/7_DancingScript-Regular5.png)
8 | Inconsolata | ![](example_dataset/8_Inconsolata-Regular1.png) ![](example_dataset/8_Inconsolata-Regular2.png) ![](example_dataset/8_Inconsolata-Regular3.png) ![](example_dataset/8_Inconsolata-Regular4.png) ![](example_dataset/8_Inconsolata-Regular5.png)
9 | VT323 | ![](example_dataset/9_VT323-Regular1.png) ![](example_dataset/9_VT323-Regular2.png) ![](example_dataset/9_VT323-Regular3.png) ![](example_dataset/9_VT323-Regular4.png) ![](example_dataset/9_VT323-Regular5.png)

---

### Trainging history 

epoch | 20 | 50 | 200
------------ | ------------ | ------------- | -------------
log | ![](plot/epoch20/history.png) | ![](plot/epoch50/history.png) | ![](plot/epoch200/history.png)


---

### 2 dimensional latent space 

epoch=20 | epoch=50 | epoch=200
:------------: | :------------: | :------------:
![](plot/epoch20/_plot.gif) | ![](plot/epoch50/_plot.gif) | ![](plot/epoch200/_plot.gif)

### Training progress in epochs
1 frame = 1 batch <br/>
1 epoch = 196 batchs

epoch=1 | epoch=2 | epoch=3
:------------: | :------------: | :------------:
![](plot/epoch10/_plot_e1.gif) | ![](plot/epoch10/_plot_e2.gif) | ![](plot/epoch10/_plot_e3.gif)
**epoch=4** | **epoch=5** | **epoch=6**
![](plot/epoch10/_plot_e4.gif) | ![](plot/epoch10/_plot_e5.gif) | ![](plot/epoch10/_plot_e6.gif)

### Experiment with 3 dimentional latent space 

![](plot/epoch10/plot_3d.gif)

---

#### References:

[1] [Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."](https://arxiv.org/abs/1312.6114) <br/>
[2] [Keras Blog: Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html) <br/>
[3] [Keras example: VAE](https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py)


#### Dataset:

- https://fonts.google.com/



