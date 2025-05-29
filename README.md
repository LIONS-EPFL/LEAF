# Robustness in Both Domains: CLIP Needs a Robust Text Encoder

<span style="color:rgb(255, 0, 0);">Elias Abad Rocamora</span>,  <span style="color:rgb(133 203 210);">Christian Schlarmann</span>, <span style="color:rgb(133 203 210);">Naman Deep Singh</span>, <span style="color:rgb(255, 0, 0);">Yongtao Wu</span>, <span style="color:rgb(133 203 210);">Matthias Hein</span> and <span style="color:rgb(255, 0, 0);">Volkan Cevher</span>

<span style="color:rgb(255, 0, 0);">LIONS @ EPFL</span> and <span style="color:rgb(133 203 210);">Tübingen AI Center</span>


In this repo, you will find all the models trained for our paper. 

### Loading CLIPModels

You can load our models as any other CLIP model, for example, loading `LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2` can be done by following the "openai/clip-vit-large-patch14" example snippet:

```python

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model_name = "LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2"
processor_name = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(processor_name)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities


```

When loading other model sizes, the `processor_name` needs to be changed accordingly as:
| Model Size | Processor Name |
|     -      |       -        |
| ViT-L-14   |`"openai/clip-vit-large-patch14"`|
| ViT-H-14   |`"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"`|
| ViT-g-14   |`"laion/CLIP-ViT-g-14-laion2B-s12B-b42K"`|
| ViT-bigG-14   |`"laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"`|

### Loading CLIPTextModels

If just need the text encoder, you can load it with the following snippet:

```python
from transformers import CLIPTokenizer, CLIPTextModel

model_name = "LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2"
processor_name = "openai/clip-vit-large-patch14"

model = CLIPTextModel.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(processor_name)

inputs = tokenizer(["a photo of a cat", "a photo of a dog"],  padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooled_output # pooled (EOS token) states
```

### Acknowledgements

Our codebase is based in the [OpenCLIP codebase](https://github.com/mlfoundations/open_clip), we appreciate the effort of the OpenCLIP team and the release of their code and model weights.


