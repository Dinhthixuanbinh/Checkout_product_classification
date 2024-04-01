## Introduction
This is classification module for this dataset [data](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
## Performance
Evaluations on val data
| Method              | Input Size        | Easy  | Medium | Hard  | Average  | 
| ------------------- | --------------- | ----- | ------ | ----- | ----- |
| EfficientNet-B0     | 128              | 73.26%| 70.34% | 69.94%| 71.18%   |
| EfficientNet-B0     | 256              | 83.58%| 80.93% | 79.75%| 81.42%   |
| EfficientNet-B0     | 512              | 85.71%| 83.10% | 82.63%| 83.82%   |
| EfficientNet-B2     | 256              | 82.62%| 81.09% | 78.5% | 80.73%   |
| **EfficientNet-B2**    | **512**              | 86.88%| 85.05% | 83.51%| 85.14%   |

Evaluation on test data 
| Clutter mode | Methods           | Input size | Accuracy |
|--------------|-------------------|------------|----------|
| Easy         | EfficientNet-B2  | 512        | 86.21%   |
| Medium       | EfficientNet-B2  | 512        | 84.74%   |
| Hard         | EfficientNet-B2  | 512        | 83.50%   |
| Average      | EfficientNet-B2  | 512        | 84.81%   |

## Installation
Install build requirements.

 ```
 pip install -r requirements
 ```
## Data preparation
1. Download datasets - anotations from [kaggle](https://www.kaggle.com/datasets/phamvoquoclong/automated-retail-checkout-classification) and set it according to the name of the folder.
2. The structure of file images should look like.
   ```
       data_folder
           easy/
               images/
           medium/
               images/
           hard/
               images/      
   ```
3. The structure of file label should look like.
   ```
         label_folder/
             hinh1.txt/    
   ```
#### Annotation Format 

*Please refer to hinh1.txt for detail*

For each image:
  ```
  # <object_id> x_center y_center height_bbox weight_bbox
  ...
  ...
  # <object_id> x_center y_center height_bbox weight_bbox
  ...
  ...
  ```
## Model Evaluation
Using this code to get accuracy of the model
```
python val.py --image_folder val_level_img --label_folder labels_val
```
## Model Inference
Please refer to `predict.py` to do inference. The output image may look like.
![image](https://github.com/Dinhthixuanbinh/Automated_Retail_Checkout/assets/136946649/da87a02c-d47b-41dd-8a22-f1b5d1f6155b)


