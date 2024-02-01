

import pathlib as Pather


import cv2 as Vision
import requests as Requesting


from . import Constant


def Save_Image ( Path_Image_Output_In , Image_In ) :

	IsSuccessful=Vision.imwrite\
	(
		# ！ 需要重新确认语法
		Path_Image_Output_In ,
		Image_In\
		[
			0: Image_In.shape [0],
			0: Image_In.shape [1],
		],
	)

	if(IsSuccessful == False):
		print(F"Image saving failed: {Path_Image_Output_In}")
	else:
		pass


def Generate_Directory_Automatically (Directory_Output_In) :
	"""Initialise base directory"""
	# !Need to be extracted to separate utility method and constant globally
	# Check Directory existence before saving output image to avoid exception occurred
	Directory_Base_Relative_Return=Pather.Path(Directory_Output_In)
	#
	Directory_Base_Relative_Return.mkdir\
	(
		parents = True,
		exist_ok = True,
	)

	return Directory_Base_Relative_Return


def Update_Configuration(Name_Parameter_In, Value_Parameter_In):

	List_Option_SDWUI = Requesting.get \
	(
		url = Constant.Address_SDWUI_Option ,
		# json=payload,
	)

	#
	#Debug
	print(F"{List_Option_SDWUI=}")
	#

	List_Option_SDWUI_JSON = List_Option_SDWUI.json ()
	#
	# Debug
	print(F"{List_Option_SDWUI_JSON=}")
	print ( F"{List_Option_SDWUI_JSON[Name_Parameter_In]=}" )
	#
	List_Option_SDWUI_JSON [ Name_Parameter_In ] = Value_Parameter_In
	#
	# Debug
	# print(F"{List_Option_SDWUI_JSON=}")
	print ( F"{List_Option_SDWUI_JSON[Name_Parameter_In]=}" )
	#
	Requesting.post \
	(
		url = Constant.Address_SDWUI_Option ,
		json = List_Option_SDWUI_JSON ,
	)
	#
	# Debug
	List_Option_SDWUI = Requesting.get \
	(
		url = Constant.Address_SDWUI_Option ,
		# json=payload,
	)
	List_Option_SDWUI_JSON = List_Option_SDWUI.json ()
	print ( F"{List_Option_SDWUI_JSON[Name_Parameter_In]=}" )
