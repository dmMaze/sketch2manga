import inspect

import gradio as Framework

import os

import cv2 as Vision

import PIL.Image as Image

import numpy as Number

import utils.webui_screentone_text2img as ConvertTextToImage

import utils.blend_screentone as Blend_Screentone

from Common import Constant, Utility

#region Pre-processing

# Debug
# import os as OS
# OS.chdir(F"..")
Directory_Current=os.getcwd()
#
print(F"{Directory_Current=}")

#region Pre-processing

Positive_prompt = 'masterpiece, best quality, illustration, beautiful and aesthetic, (best illumination, an extremely delicate and beautiful),simple background'
Negative_prompt = 'nsfw, nude, lowres, ((bad anatomy)), ((bad hands)), worst quality, low quality, normal quality, jpeg, ((jpeg artifacts))'

def Tab():
    Title_In = F"Line to Screentone"

    with Framework.Blocks(title = Title_In):

        Framework.Markdown(F"---")

        with Framework.Row():
            with Framework.Column():
                Title_Image_Input = F"Input Sketch Image"
                Framework.Markdown(F"## {Title_Image_Input}")
                ImageBox_Input = Framework.Image\
                (
                    show_label = False,
                    type = F"filepath",
                )

                Framework.Examples\
                (
                    examples = \
                    [
                        os.path.join(Constant.File_Workplace, img_name)\
                            for img_name\
                            in os.listdir(Constant.File_Workplace)
                    ],
                    inputs = \
                    [
                        ImageBox_Input,
                    ],
                )

            with Framework.Column():
                Title_Image_Output = F"Result"
                Framework.Markdown(F"## {Title_Image_Output}")
                ImageBox_Output_Color_Screentone1 = Framework.Image\
                (
                    show_label = False,
                )

        Framework.Markdown(F"---")

        Title_Image_Output = F"Process"
        Framework.Markdown(F"## {Title_Image_Output}")

        with Framework.Row():
            with Framework.Column():
                Title_Image_Output = F"Colorized Input"
                Framework.Markdown(F"### {Title_Image_Output}")
                ImageBox_Output_Color = Framework.Image\
                (
                    show_label = False,
                )

            with Framework.Column():
                Title_Image_Output = F"Rough Screentone"
                Framework.Markdown(F"### {Title_Image_Output}")
                ImageBox_Output_Rough = Framework.Image\
                (
                    show_label = False,
                )

        with Framework.Row():
            with Framework.Column():
                Title_Image_Output = F"Colored Screentone"
                Framework.Markdown(F"### {Title_Image_Output}")
                ImageBox_Output_Color_Screentone = Framework.Image\
                (
                    show_label = False,
                )

            with Framework.Column():
                Title_Image_Output = F"Gray Scale"
                Framework.Markdown(F"### {Title_Image_Output}")
                ImageBox_Output_GrayScale = Framework.Image \
                (
                    show_label = False,
                )

        ImageBox_Input.change\
        (
            fn = Change_Input_Sketch,
            inputs = [ImageBox_Input],
            outputs =\
            [
                ImageBox_Output_Color_Screentone1,
                ImageBox_Output_Color,
                ImageBox_Output_Rough,
                ImageBox_Output_Color_Screentone,
                ImageBox_Output_GrayScale,
            ],
        )

def Change_Input_Sketch(Path_img:str,):


    #Debug
    print(F"{inspect.stack()[0][3]}")

    List_Image_Return = list()

    if(Path_img is None):

        #Debug 
        print(F"{Path_img=}")

        pass

    elif(Path_img == ""):
    
        #Debug
        print(F"{Path_img=}")

        pass
    
    else:
        #
        #Debug 
        print(F"{Path_img=}")
        #

        Utility.Update_Configuration \
            (
                Name_Parameter_In=F"sd_model_checkpoint",
                # Value_Parameter_In = F"anything-v4.5-vae-swapped.ckpt [b0147c33be]",
                Value_Parameter_In=F"anything-v4.5-vae-swapped.safetensors [a504b5b137]",
            )

        # load image from url
        # Image_Input = Image.open(F"{Constant.File_Workplace}{Constant.Image_Name}{Constant.Img_JPG}")
        Image_Input = Image.open(Path_img)
        
        #Debug
        print(F"{Image_Input=}" )
        #

        Utility.Update_Configuration \
            (
                Name_Parameter_In=F"sd_vae",
                Value_Parameter_In=F"Automatic",
            )

        List_Colored_Img = ConvertTextToImage.colorize_sketch \
            (
                Image_Input,
                prompt=Positive_prompt,
                negative_prompt=Negative_prompt,
                long_side=Constant.long_side,
                seed=Constant.seed,
                url=Constant.Address_SDWUI_TextToImage
            )
        Colored_Img = List_Colored_Img[0]

        #Debug
        print(F"{Colored_Img=}")
        #

        # Colored_Img.save(F"{Constant.File_Output}Colored_Img{Constant.Img_PNG}")
        List_Image_Return.append(Colored_Img)


        # Colored image to screentone
        Utility.Update_Configuration \
            (
                Name_Parameter_In=F"sd_model_checkpoint",
                Value_Parameter_In=F"mangatone.ckpt [9381c1a502]",
            )

        # make vae setting to "mangatone_default"
        Utility.Update_Configuration \
            (
                Name_Parameter_In=F"sd_vae",
                Value_Parameter_In=F"mangatone_default.ckpt",
            )

        List_screentone_rough = ConvertTextToImage.colorize_sketch \
            (
                Colored_Img,
                url=Constant.Address_SDWUI_TextToImage,
                prompt='greyscale, monochrome, screentone',
                negative_prompt='',
                long_side=Constant.long_side,
                seed=Constant.seed
            )

        Screentone_Rough = List_screentone_rough[0]

        #Debug
        print(F"{Screentone_Rough=}")
        #

        # Screentone_Rough.save(F"{Constant.File_Output}Screen_Rough{Constant.Img_PNG}")
        List_Image_Return.append(Screentone_Rough)


        # 3. 后处理
        # 下面两张图是最后需要得到的结果

        Colored_Array = Number.array(Colored_Img)
        Screentone_Rough_Array = Number.array(Screentone_Rough)

        Screentone_Colorized_Array, \
            Layers, \
            Layers_Visible = Blend_Screentone.blend_screentone \
            (
                Colored_Array,
                Screentone_Rough_Array,
                seed=Constant.seed,
                cluster_n=Constant.Cluster_Number,
            )

        Colored_Screentone = Image.fromarray(Screentone_Colorized_Array)


        #Debug
        print(F"{Colored_Screentone=}")

        print(F"{len(Layers)=}")

        print(F"{len(Layers_Visible)=}")
        #


        #Colored_Screentone.save(F"{Constant.File_Output}Colored_Screentone{Constant.Img_PNG}")
        List_Image_Return.append(Colored_Screentone)

        Colored_To_GrayScale = Vision.cvtColor \
            (
                Colored_Array,
                Vision.COLOR_RGB2GRAY
            )

        Colored_To_GrayScale_Array = Colored_To_GrayScale[..., None]

        Screentone_Final_Array = Constant.screentone_scale * Screentone_Rough_Array + Constant.color_scale * Colored_To_GrayScale_Array

        Screentone_Final_Clipped_Array = Number.clip \
            (
                Screentone_Final_Array,
                0,
                255,
            ).astype(Number.uint8)


        SColored_To_Gray = Vision.cvtColor \
            (
                Screentone_Colorized_Array,
                Vision.COLOR_RGB2GRAY
            )

        SColored_To_Gray_List = [SColored_To_Gray[..., None]]

        Blend_Screentone.fgbg_hist_matching \
                (
                SColored_To_Gray_List,
                Screentone_Final_Clipped_Array,
            )

        Final_Image = Image.fromarray(SColored_To_Gray_List[0][..., 0])

        #Debug
        print(F"{Final_Image=}")
        #

        # Final_Image.save(F"{Constant.File_Output}Final_Image{Constant.Img_PNG}")
        List_Image_Return.append(Final_Image)
        List_Image_Return.insert(0,Colored_Screentone)

    return List_Image_Return
