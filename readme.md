# Sketch2Manga

[![arXiv](https://img.shields.io/badge/arXiv-2403.08266-<COLOR>)](https://arxiv.org/abs/2403.08266)

Apply screentone to line drawings or colored illustrations with diffusion models.

<p float="center">
  <img src="https://github.com/dmMaze/sketch2manga/assets/51270320/85098012-68d8-471f-b8ed-0476d856cce5" />
  <br>
    <em>Sketch2Manga - Drag and drop into ComfyUI to load the workflow </em>
  <a href="https://twitter.com/ini_pmh/status/715578786830417921/photo/1">(Source @ini_pmh)</a>
</p>

<p float="center">
  <img src="https://github.com/dmMaze/sketch2manga/assets/51270320/ecaf6632-e108-4a8d-9e7e-8882bdb2e620" />
  <br>
    <em>Illustration2Manga - Drag and drop into ComfyUI to load the workflow </em>
  <a href="https://danbooru.donmai.us/posts/5493050">(Source @curecu8)</a>
</p>




## Usage

### Preperation

Install [pytorch](https://pytorch.org/get-started/locally/)  

Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), download a diffusion model for colorization (the demo used [meinapastel](https://civitai.com/models/11866/meinapastel) for ComfyUI, [anything-v4.5](https://huggingface.co/ckpt/anything-v4.5-vae-swapped/tree/main) for sd-webui) and [control_v11p_sd15_lineart](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_lineart.pth).  
Download the finetuned [vae](https://huggingface.co/dreMaz/sketch2manga/blob/main/vae/mangatone_default.ckpt) and [diffusion model](https://huggingface.co/dreMaz/sketch2manga/blob/main/mangatone.ckpt) for screening.

### ComfyUI
Clone this repo to the ComfyUI directory and install dependencies:
``` bash
git clone https://github.com/dmMaze/sketch2manga [ComfyUI Directory]/custom_nodes/sketch2manga
cd [ComfyUI Directory]/custom_nodes/sketch2manga 
pip install -r requirements.txt
```
Launch ComfyUI, drag and drop the figure above to load the workflow.


### Gradio Demo
Clone this repo and install dependencies, launch sd-webui with argument ```--api``` (We are using stable-diffusion-webui @ bef51aed and sd-webui-controlnet @ aa2aa81), and run
```
python gradio_demo/launch.py
```

#### SD-WebUI API
There is an example ```webuiapi_demo.ipynb``` showcasing inference using SD-WebUI API, it is a bit outdated though, but the logic applied is the same.

## Comparsion
Our Illustration to Manga method compared with [Mimic Manga](https://lllyasviel.github.io/MangaFilter/) (considered as SOTA)
<table>
  <thead>
    <tr>
      <th align="center" width="33%">Illustration (Input)</th>
      <th align="center" width="33%">Mimic Manga</th>
      <th align="center" width="33%">Ours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" >
        <a href="https://github.com/dmMaze/sketch2manga/assets/51270320/50977ee7-d1a6-4fa0-a0b3-7cbb22b4b317">
          <img alt="Input" src="https://github.com/dmMaze/sketch2manga/assets/51270320/50977ee7-d1a6-4fa0-a0b3-7cbb22b4b317" />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/dmMaze/sketch2manga/assets/51270320/75704188-3d2e-4358-8142-f17ecdf06c84">
          <img alt="MimicManga" src="https://github.com/dmMaze/sketch2manga/assets/51270320/75704188-3d2e-4358-8142-f17ecdf06c84" />
        </a>
      </td>
      <td align="center" >
        <a href="https://github.com/dmMaze/sketch2manga/assets/51270320/2ad2947d-aaf9-428e-bace-d33a3b9679e3">
          <img alt="Ours" src="https://github.com/dmMaze/sketch2manga/assets/51270320/2ad2947d-aaf9-428e-bace-d33a3b9679e3" />
        </a>
      </td>
    </tr>
  </tbody>
</table>



