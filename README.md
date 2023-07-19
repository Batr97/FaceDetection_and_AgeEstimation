# Face detection and Age estimation from videos and images

Files: 
- `scripts/prepare_new_dataset_RF.py` - Script for creation new dataset from APPA dataset by cropping images via RetinaFace and generation of new augmented images (x2)
- `scripts/UTK_generation.py` - Script for generation of new augmented images (x2) from provided UTK dataset
- `notebooks/age_estimator.ipynb` - Notebook containing:
    - visualization of RetinaFace model outputs
    - dataset preprocessing
    - training of Vision Transformer
    - pushing weights to model hub
- `app/models/ViT_age_estimation.py` - Class of the ViT model for age prediction
- `app/utils/image_utils.py` - Script with functions to detect faces in video/image via RetinaFace
- `app/main.py` - FastAPI app

FastAPI service is runned inside `/app` directory with command: 
```
python -m uvicorn main:app --reload
```

It contains two endpoints for uploading image and video and further age estimation: 
- *predict/image*
- *predict/video*

Read detailed report in `report/report.md`.

Model weights are pushed to hub: https://huggingface.co/Batr97/ViT_ordinary/tree/refs%2Fpr%2F1/
