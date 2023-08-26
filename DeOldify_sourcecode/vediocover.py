#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from deoldify.visualize import *
import warnings
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
if torch.cuda.is_available():
    print('GPU is available.')

torch.backends.cudnn.benchmark=True
# jupyter-lab
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

colorizer = get_video_colorizer()

source_url = 'https://twitter.com/i/status/1654737303204671494' #@param {type:"string"}
render_factor = 25  #@param {type: "slider", min: 5, max: 40}
watermarked = False #@param {type:"boolean"}

if source_url is not None and source_url !='':
    video_path = colorizer.colorize_from_url(source_url, 'video.mp4', render_factor, watermarked=watermarked)
    show_video_in_notebook(video_path)
else:
    print('Provide a video url and try again.')

# for i in range(10,40,2):
#     colorizer.vis.plot_transformed_image('video/bwframes/video/00001.jpg', render_factor=i, display_render_factor=True, figsize=(8,8))