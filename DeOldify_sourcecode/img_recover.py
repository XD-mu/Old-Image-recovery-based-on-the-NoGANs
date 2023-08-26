#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
import fastai
from deoldify.visualize import *
import warnings
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU1)

if torch.cuda.is_available():
    print('GPU available.')

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

source_url = 'abc' #@param {type:"string"}
render_factor = 21 #@param {type: "slider", min: 7, max: 40}
watermarked = False #@param {type:"boolean"}
#artistic=False使用老模型、artistic=True使用训练出的新模型
colorizer = get_image_colorizer(artistic=True)
if source_url is not None and source_url !='':
    image_path = colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=True, watermarked=watermarked)
    show_image_in_notebook(image_path)
else:
    print('Provide an image url and try again.')