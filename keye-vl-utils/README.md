# keye-vl-utils

Keye-vl-utils contains a set of helper functions for processing and integrating visual language information with Keye Series Model.

## Install

```bash
pip install keye-vl-utils
```

## Usage


### KeyeVL

```python
from transformers import AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model_path = "Keye/Keye-8B-Instruct"

model = AutoModel.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2", trust_remote_code=True,
).to('cuda')

# You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
# based on the maximum tokens that the model can accept. 
# export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9


# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Image URL
    [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Base64 encoded image
    [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}],
    ## PIL.Image.Image
    [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "Describe this image."}]}],
    ## Model dynamically adjusts image size, specify dimensions if required.
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}],
    # Video
    ## Local video path
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4"}, {"type": "text", "text": "Describe this video."}]}],
    ## Local video frames
    [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/extracted_frame1.jpg", "file:///path/to/extracted_frame2.jpg", "file:///path/to/extracted_frame3.jpg"],}, {"type": "text", "text": "Describe this video."},],}],
    ## Model dynamically adjusts video nframes, video height and width. specify args if required.
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", "fps": 2.0, "resized_height": 280, "resized_width": 280}, {"type": "text", "text": "Describe this video."}]}],
]

processor = AutoProcessor.from_pretrained(model_path)
model = KeyeForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs).to("cuda")
print(inputs)
generated_ids = model.generate(**inputs)
print(generated_ids)
```

## Deployment
The Keye-8B-Instruct series maintain full compatibility with the `Qwen2_5_VLForConditionalGeneration` architecture for deployment and inference.
