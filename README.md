# Real-time Scene Text Detection with Differentiable Binarization

## Requirements
* pytorch 1.4+
* torchvision 0.5+
* gcc 4.9+

## Data Preparation

Training data: prepare a text `train.json` in the following format, use '\t' as a separator
```json
{
    "data_root": "E:/dataset/text/icdar2015/dual_detection/train/imgs",
    "data_list": [
        {
            "img_name": "img_1.jpg",
            "fore_annotations": [
                {
                    "polygon": [
                        [
                            377.0,
                            117.0
                        ],
                        [
                            463.0,
                            117.0
                        ],
                        [
                            465.0,
                            130.0
                        ],
                        [
                            378.0,
                            130.0
                        ]
                    ],
                    "text": "Genaxis Theatre",
                    "illegibility": false,
                    "language": "Latin",
                    "chars": [
                        {
                            "polygon": [],
                            "char": "",
                            "illegibility": false,
                            "language": "Latin"
                        }
                    ]
				}
            ],
            "back_annotations": [
                {
                    "polygon": [
                        [
                            377.0,
                            117.0
                        ],
                        [
                            463.0,
                            117.0
                        ],
                        [
                            465.0,
                            130.0
                        ],
                        [
                            378.0,
                            130.0
                        ]
                    ],
                    "text": "Genaxis Theatre",
                    "illegibility": false,
                    "language": "Latin",
                    "chars": [
                        {
                            "polygon": [],
                            "char": "",
                            "illegibility": false,
                            "language": "Latin"
                        }
                    ]
                }
			]
        }
	]
}
```


## Train
1. config the `dataset['train']['dataset'['data_path']'`,`dataset['validate']['dataset'['data_path']`in [config/dual_resnet18_FPN_DBhead_polyLR.yaml](cconfig/icdar2015_resnet18_fpn_DBhead_polyLR.yaml)
* . single gpu train
```bash
bash singlel_gpu_train.sh
```
* . Multi-gpu training
```bash
bash multi_gpu_train.sh
```

### todo
- [x] predict, eval

### reference
1. https://arxiv.org/pdf/1911.08947.pdf
2. https://github.com/WenmuZhou/DBNet.pytorch
3. https://github.com/MhLiao/DB

**If this repository helps you，please star it. Thanks.**