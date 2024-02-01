#!/usr/bin/env python
# coding: utf-8

import cv2 as Vision

#from PIL import Image
import PIL.Image as Image

import numpy as Number

#from utils.webui_screentone_text2img import colorize_sketch

import utils.webui_screentone_text2img as ConvertTextToImage

#from utils.blend_screentone import blend_screentone, fgbg_hist_matching
import utils.blend_screentone as Blend_Screentone

import Common.Constant as Constant
import Common.Utility as Utility


# 1. 线稿上色
# 注意: 执行这一步时要确保webui->Settings->VAE->SD VAE是Automatic

Positive_prompt = 'masterpiece, best quality, illustration, beautiful and aesthetic, (best illumination, an extremely delicate and beautiful),simple background'
Negative_prompt = 'nsfw, nude, lowres, ((bad anatomy)), ((bad hands)), worst quality, low quality, normal quality, jpeg, ((jpeg artifacts))'


Directory_Output_image = Utility.Generate_Directory_Automatically(Constant.File_Output)


Utility.Update_Configuration\
(
    Name_Parameter_In = F"sd_model_checkpoint",
    # Value_Parameter_In = F"anything-v4.5-vae-swapped.ckpt [b0147c33be]",
    Value_Parameter_In = F"anything-v4.5-vae-swapped.safetensors [a504b5b137]",
)

# load image from url
# Image_Input = Image.open(F"{Constant.File_Workplace}{Constant.Image_Name}{Constant.Img_JPG}")
Image_Input = Image.open(F"{Constant.File_Workplace}{Constant.Image_Name}{Constant.Img_PNG}")

Image_Input.save(F"{Constant.File_Output}Image_Input{Constant.Img_PNG}")

#sketch.show()
#sketch.save("C:/Users/schoy/Downloads/New folder/sketch.PNG")


Utility.Update_Configuration\
(
    Name_Parameter_In = F"sd_vae",
    Value_Parameter_In = F"Automatic",
)


List_Colored_Img = ConvertTextToImage.colorize_sketch\
(
        Image_Input,
        prompt=Positive_prompt,
        negative_prompt=Negative_prompt,
        long_side=Constant.long_side,
        seed=Constant.seed,
        url= Constant.Address_SDWUI_TextToImage
)
Colored_Img = List_Colored_Img[0]

Colored_Img.save(F"{Constant.File_Output}Colored_Img{Constant.Img_PNG}")

# colored.save("C:/Users/schoy/Downloads/New folder/colored.PNG")
# colored.show()

# Colored image to screentone
Utility.Update_Configuration\
(
    Name_Parameter_In = F"sd_model_checkpoint",
    Value_Parameter_In = F"manga_34000.ckpt [29c34037ac]",
)

# make vae setting to "mangatone_default"
Utility.Update_Configuration\
(
    Name_Parameter_In = F"sd_vae",
    Value_Parameter_In = F"mangatone_default.ckpt",
)


List_screentone_rough = ConvertTextToImage.colorize_sketch\
(
    Colored_Img,
    url=Constant.Address_SDWUI_TextToImage,
    prompt='greyscale, monochrome, screentone',
    negative_prompt='',
    long_side=Constant.long_side,
    seed=Constant.seed
)

Screentone_Rough = List_screentone_rough[0]

Screentone_Rough.save(F"{Constant.File_Output}Screen_Rough{Constant.Img_PNG}")

#screentone_rough.save("C:/Users/schoy/Downloads/New folder/Rough.PNG")
#screentone_rough.show()



# 3. 后处理
# 下面两张图是最后需要得到的结果


Colored_Array = Number.array(Colored_Img)
Screentone_Rough_Array = Number.array(Screentone_Rough)

Screentone_Colorized_Array, \
Layers, \
Layers_Visible = Blend_Screentone.blend_screentone\
(
    Colored_Array,
    Screentone_Rough_Array,
    seed=Constant.seed,
    cluster_n = Constant.Cluster_Number,
)

Colored_Screentone = Image.fromarray(Screentone_Colorized_Array)

Colored_Screentone.save(F"{Constant.File_Output}Colored_Screentone{Constant.Img_PNG}")

#colored_tone.save("C:/Users/schoy/Downloads/New folder/colored_tone.PNG")
#colored_tone.show()


# screentone_final = Constant.screentone_scale * Screentone_Rough_Array + Constant.color_scale * Vision.cvtColor(Colored_Array, Vision.COLOR_RGB2GRAY)[..., None]
# screentone_final = Number.clip(screentone_final, 0, 255).astype(Number.uint8)

Colored_To_GrayScale = Vision.cvtColor\
(
    Colored_Array,
    Vision.COLOR_RGB2GRAY
)

Colored_To_GrayScale_Array = Colored_To_GrayScale[..., None]

Screentone_Final_Array = Constant.screentone_scale * Screentone_Rough_Array + Constant.color_scale * Colored_To_GrayScale_Array

Screentone_Final_Clipped_Array = Number.clip\
(
        Screentone_Final_Array,
        0,
        255,
).astype(Number.uint8)

# sc = Vision.cvtColor(Screentone_Colorized_Array, Vision.COLOR_RGB2GRAY)
# sc_list = [sc[..., None]]
# Blend_Screentone.fgbg_hist_matching(sc_list, Screentone_Final_Clipped_Array)
# final = Image.fromarray(sc_list[0][..., 0])

SColored_To_Gray = Vision.cvtColor\
(
    Screentone_Colorized_Array,
    Vision.COLOR_RGB2GRAY
)

SColored_To_Gray_List = [SColored_To_Gray[..., None]]

Blend_Screentone.fgbg_hist_matching\
(
    SColored_To_Gray_List,
    Screentone_Final_Clipped_Array,
)

Final_Image = Image.fromarray(SColored_To_Gray_List[0][..., 0])

Final_Image.save(F"{Constant.File_Output}Final_Image{Constant.Img_PNG}")

# final.save('workspace/teaser/final.png')
#final.save("C:/Users/schoy/Downloads/New folder/final.PNG")
#final.show()

