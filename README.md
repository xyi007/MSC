# MSC:Image-text Retrieval with Main Semantics Consistency
## Requirements and Installation
---------------
We recommended the following dependencies.
* Python 3
* PyTorch (>1.0)
* torchvision
* TensorBoard
* matplotlib
* NumPy (>1.12.1)
* Punkt Sentence Tokenizer:
  ```python
  import nltk
  nltk.download()
  ```

## Download data and vocab
---------------------------
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, which can be downloaded by using:
```python
wget https://iudata.blob.core.windows.net/scan/data.zip
wget https://iudata.blob.core.windows.net/scan/vocab.zip
```

Another download link is available below：
```python
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```

## Introduction
---------------
Image-text retrieval (ITR) has been one of the primary tasks in cross-modal
retrieval, serving as a crucial bridge between computer vision and natural
language processing. Significant progress has been made to achieve global
alignment and local alignment between images and texts by mapping images
and texts into a common space to establish correspondences between these
two modalities. 

![image](https://github.com/xyi007/MSC/blob/main/Framework_revision.pdf)

## Training new models
------------------------
Modify the data_path, vocab_path, model_name, logger_name in the opts.py file. Then run train.py:

For MSCOCO:
```Python
(For SGR) python train.py --data_name coco_precomp --num_epochs 25 --lr_update 10 --module_name SGR
(For SAF) python train.py --data_name coco_precomp --num_epochs 25 --lr_update 10 --module_name SAF
```
For Flickr30K:
```Python
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --lr_update 25 --module_name SGR
(For SAF) python train.py --data_name f30k_precomp --num_epochs 30 --lr_update 15 --module_name SAF
```

## Evaluate trained models
--------------------------
For SGR and SAF: Modify the model_path, data_path, vocab_path in the evaluation.py file. Then run evaluation.py:
```Python
python evaluation.py
```

For SGRAF: Modify the paths in the eval_ensemble.py file. Then run eval_ensemble.py:
```Python
python eval_ensemble.py
```

Note that ```fold5=True``` is only for evaluation on mscoco1K (5 folders average) while fold5=False for mscoco5K and flickr30K. 

## Visualization
-----------------
![i2t](https://github.com/xyi007/MSC/blob/main/Image_text_retrieval.pdf)
![t2i](https://github.com/xyi007/MSC/blob/main/Text_image_retrieval.pdf)
