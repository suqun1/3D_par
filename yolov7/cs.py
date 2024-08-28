import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized

def simple_detect(weights='path/to/your/weights', source='path/to/your/images', img_size=640, conf_thres=0.4, iou_thres=0.4):
    # Initialize
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = img_size  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # scale (0 - 255) to (0.0 - 1.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        # Apply NMS
        det = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)[0]

        # Rescale boxes from imgsz to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

        # Rescale predictions from imgsz to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()

        # Flatten the predictions
        flattened_preds = [single_pred.unsqueeze(0) for single_pred in pred.view(-1, pred.size(-1))]

        # Display some flattened predictions
        num_elements_to_display = 5
        selected_elements = flattened_preds[:num_elements_to_display]
        for element in selected_elements:
            print(element)

        # Further processing can be done here as required

# Replace 'path/to/your/weights' and 'path/to/your/images' with your actual file paths
simple_detect(weights='runs/train/exp127/weights/best.pt', source='inference/images')
