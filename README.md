# Variational Autoencoder with Convnets for Font analysis

The VAE has a modular design. The encoder, decoder and VAE are 3 models that share weights. After training the VAE model, the encoder can be used to generate latent vectors. The decoder can be used to generate font images by sampling the latent vector from a Gaussian distribution with mean=0 and std=1.

---


### VAE architecture<br/>

encoder | decoder
------------ | -------------
![](summary/font_vae_cnn_encoder.png) | ![](summary/font_vae_cnn_decoder.png)

---

### Used dataset
idx | Font | sample images
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

### Trainging history w/ 20 epochs
![](font_vae_cnn/plot_e20c10/history.png)

---

### 2 Dimensional latent space
![](font_vae_cnn/plot_e20c10/_plot.gif)

---

#### References:<br/>
[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes. "https://arxiv.org/abs/1312.6114 <br/>
[2] https://blog.keras.io/building-autoencoders-in-keras.html <br/>
[3] https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py


#### Dataset:<br/>
- https://fonts.google.com/



