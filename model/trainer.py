from wandb.api_key import WANDB_API_KEY
from wandb.config import PROJECT_NAME
import wandb
import yaml
from ultralytics import YOLO

def login_and_init_wandb(WANDB_API_KEY, PROJECT_NAME):
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME)

def train_model(model_path, data_path, epochs, img_size, device_ids, batch_size):
    model = YOLO(model_path)  # load a pretrained model (recommended for training)
    results = model.train(data=data_path, epochs=epochs, imgsz=img_size, device=device_ids, batch=batch_size)
    return results

def main(model_path='yolov8s.pt', data_path='detect.yaml', epochs=100, img_size=640, device_ids=[0, 1], batch_size=8):
    # Execute the functions
    login_and_init_wandb(WANDB_API_KEY, PROJECT_NAME)
    # create_yaml(detect_yaml_path)
    results = train_model(model_path, data_path, epochs, img_size, device_ids, batch_size)

    # Log the results to wandb
    wandb.log(results)

if __name__ == "__main__":
    main()
