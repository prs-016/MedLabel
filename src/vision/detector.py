"""
src/vision/detector.py
======================
YOLOv11 router — detects medicine label region and classifies geometry.

Class map (matches Roboflow training annotation):
    0  flat_label      → Path A (PaddleOCR)
    1  cylindrical     → Path B (Gemini Vision)

Falls back to "flat" when:
  - trained model weights are not present
  - no detections are returned
"""

from __future__ import annotations

from pathlib import Path


CLASS_NAMES = {0: "flat", 1: "cylindrical"}
_DEFAULT_MODEL = "models/yolo/best.pt"
_FALLBACK_MODEL = "yolo11n.pt"


class YOLORouter:

    def __init__(self, model_path: str = _DEFAULT_MODEL):
        self._trained = False
        self.model    = None
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        from ultralytics import YOLO

        if Path(model_path).is_file():
            try:
                self.model    = YOLO(model_path)
                self._trained = True
                return
            except Exception as exc:
                print(f"Warning: could not load {model_path}: {exc}")

        print(
            f"Trained model not found at '{model_path}'. "
            "Defaulting to 'flat' until the YOLO model is trained."
        )
        try:
            self.model = YOLO(_FALLBACK_MODEL)
        except Exception:
            self.model = None

    def detect_geometry(
        self,
        image_source,  # path str | np.ndarray | PIL.Image
    ) -> tuple[str, list[float], float]:
        """
        Detect label region and classify packaging geometry.

        Returns
        -------
        (packaging_type, bbox, confidence)
            packaging_type : "flat" | "cylindrical"
            bbox           : [x1, y1, x2, y2]  (pixels)
            confidence     : 0-1
        """
        if self.model is None or not self._trained:
            return "flat", [0.0, 0.0, 100.0, 100.0], 0.5

        results = self.model(image_source, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return "flat", [0.0, 0.0, 100.0, 100.0], 0.5

        boxes    = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        cls_id   = int(boxes.cls[best_idx].item())
        conf     = float(boxes.conf[best_idx].item())
        bbox     = boxes.xyxy[best_idx].tolist()

        packaging_type = CLASS_NAMES.get(cls_id, "flat")
        return packaging_type, bbox, conf

    def crop_label(self, image, bbox: list[float]):
        """
        Crop a cv2 BGR image to the detected bounding box.
        Returns the original image unchanged if the crop is degenerate.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]


if __name__ == "__main__":
    router = YOLORouter()
    print(f"YOLORouter initialised — trained model loaded: {router._trained}")
