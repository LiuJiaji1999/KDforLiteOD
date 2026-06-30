import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics/cfg/models/11/yolo11m.yaml', # stu
        'data':'./dataset/powerdata.yaml',
        'imgsz': 640,
        'epochs': 100,
        'batch': 8,
        'workers': 4,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'amp': False, # 蒸馏损失为nan，把amp设置为False
        'project':'runs/distill',
        # 'project':'runs/debug',
        'name':'yolo11m-mimic-review-exp',
        
        # distill
        'prune_model': False,
        'teacher_weights': 'runs/train/11exp2/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/11/yolo11m.yaml',
        'kd_loss_type': 'feature', # 'logical', 'feature', 'all'
        'kd_loss_decay': 'constant', # 'cosine', 'linear', 'cosine_epoch', 'linear_epoch', 'constant'

        'logical_loss_type': 'BCKD', # 'l1', 'l2', 'BCKD'
        'logical_loss_ratio': 0.5,
        
        'teacher_kd_layers': '16,19,22',
        'student_kd_layers': '16,19,22',
        'feature_loss_type': 'mimic', # 'mimic', 'mgd'-loss=nan, 'cwd', 'chsim', 'sp'
        'feature_loss_ratio': 0.5
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()