from ultralytics import YOLO
import cv2
import PIL.Image

class YOLORouter:
    def __init__(self, model_path='models/yolo/best.pt'):
        """
        Initializes the YOLOv11 router.
        If the model file doesn't exist yet, it will use a pretrained fallback.
        """
        try:
            self.model = YOLO(model_path)
        except:
            print(f"Warning: model at {model_path} not found. Using pretrained yolo11n.pt.")
            self.model = YOLO('yolo11n.pt')

    def detect_geometry(self, image_source):
        """
        Detects the medicine packaging and classifies its geometry.
        
        Returns:
            - packaging_type: "flat" or "cylindrical"
            - bbox: [x1, y1, x2, y2]
            - confidence: float
        """
        # Week 2 Task: Implement YOLO detection logic here
        # For now, let's return a dummy result
        results = self.model(image_source)
        
        # Placeholder logic:
        # 0: flat_label, 1: cylindrical_label
        packaging_type = "flat" 
        bbox = [0, 0, 100, 100]
        confidence = 0.95
        
        return packaging_type, bbox, confidence

    def crop_label(self, image, bbox):
        """
        Crops the image to the detected label bounding box.
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Week 2 Task: Implement cropping logic
        return image # return original for now

if __name__ == "__main__":
    # Test stub
    router = YOLORouter()
    print("YOLO Router initialized.")
