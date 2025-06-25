# Evaluation Datasets 
Based on the in-house short video data, we constructed 6 datasets for **Keye** and other Vision-Language Models (VLMs) like **Qwen2.5-VL** and **InternVL** to evaluate performance.

## Tasks
| Task           | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| CPV            | The task of predicting product attributes in e-commerce.                    |
| Video_Topic    | The task of determining whether multiple videos belong to the same topic.   |
| Video_Order    | The task of determining the logical order between multiple videos with the same topic. |
| PornComment    | The task of whether short video comments contain pornographic content.      |
| High_Like      | A binary classification task to determine the rate of likes of a short video. |
| SPU            | The task of determining whether two items are the same product in e-commerce. |

These datasets can be downloaded from [Hugging Face (HF)](https://huggingface.co/). 

## Performance 
| Task           | Qwen2.5-VL-3B | Qwen2.5-VL-7B | InternVL-3-8B | MiMo-VL | Keye |
| -------------- | ------------- | ------------- | ------------- | ------- | ---- |
| CPV            | 12.80         | 20.10         | 15.00         | 17.10   | 55.13 |
| Video_Topic    | 43.90         | 46.95         | 51.21         | 49.39   | 54.30 |
| Video_Order    | 36.80         | 58.40         | 64.80         | 78.40   | 84.43 |
| PornComment    | 57.10         | 56.50         | 57.60         | 68.60   | 71.96 |
| High_Like      | 48.20         | 48.70         | 47.80         | 50.40   | 55.25 |
| SPU            | 74.10         | 81.30         | 75.60         | 81.90   | 87.05 |

## Example of Evaluation

Here is an example of an evaluation using VLMs on our datasets. The following configuration needs to be added to the config file.
```python
{

    "model":'...'
    "data": {
        "CPV": {
            "class": "KwaiVQADataset",
            "dataset": "CPV"
        },
        "Video_Topic": {
            "class": "KwaiVQADataset",
            "dataset": "Video_Topic"
        },
        "Video_Order": {
            "class": "KwaiVQADataset",
            "dataset": "Video_Order"
        },
        "PornComment": {
            "class": "KwaiYORNDataset",
            "dataset": "PornComment"
        },
        "High_like":{
            "class":"KwaiYORNDataset",
            "dataset":"High_like"
        },
        "SPU": {
            "class": "KwaiYORNDataset",
            "dataset": "SPU"
        }
    }
}
