# Capstone Design 2021-2

## Monitor kickboard driver

### Overview

"공유 킥보드" 서비스를 이용하는 사용자들이 도로교통법을 준수하는지 감시하는 딥러닝 모델입니다

- 킥보드를 이용 중 헬멧의 착용 여부를 감시합니다
- 동승자가 탑승했는지 여부를 감시합니다

[YOLOv5](https://github.com/ultralytics/yolov5) 를 베이스로 제작하였으며, [open-images-dataset-v6](https://opensource.google/projects/open-images-dataset) 와 [cooc-2017](https://cocodataset.org/#home) 을 데이터셋으로 활용하였습니다.

### Contributers

- 김영교
- 원지운

### How to start

1. Create virtual environment with anaconda and activate

```bash
$ conda env create -f environment.yaml
$ conda activate kickboard
```

2. Install required packages via `pip`

```bash
$ pip install -r 
```

3. Execute python files via [YOLOv5](https://github.com/ultralytics/yolov5) gives you a guide 

### Results

- Model: [Latest train result](https://wandb.ai/skywrace/YOLOv5/runs/2zjutty7?workspace=user-skywrace)


- Dataset

> Dataset should be placed on outside of this repository

```yaml
# datasets/custom/dataset.yaml

train: ../datasets/custom/train/images 
val: ../datasets/custom/valid/images
nc: 2
names: ['Helmet', 'Person']
```

## Conclusion

### References

- https://github.com/ultralytics/yolov5
- https://opensource.google/projects/open-images-dataset
- https://cocodataset.org/#home