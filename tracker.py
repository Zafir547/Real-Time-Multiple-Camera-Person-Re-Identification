import numpy as np
import torch
import torchvision.transforms as T
import torchreid
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

class Tracker:
    def __init__(self, use_half=True):
        self.use_half = use_half
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
        self.tracker = DeepSortTracker(metric)

        # Load OSNet model
        self.reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1,
            loss='softmax',
            pretrained=True
        ).to(self.device).eval()

        # Enable half precision if supported
        if self.use_half and self.device.type == 'cuda':
            self.reid_model = self.reid_model.half()

        # Image transformation
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.tracks = []

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        crops = []
        valid_indices = []
        for i, bbox in enumerate(bboxes):
            x, y, w, h = [int(v) for v in bbox]
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            tensor_crop = self.transform(crop)
            crops.append(tensor_crop)
            valid_indices.append(i)

        # Fallback if no valid crops
        if not crops:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        batch = torch.stack(crops).to(self.device)
        if self.use_half:
            batch = batch.half()

        with torch.no_grad():
            features = self.reid_model(batch).cpu().numpy()

        # DeepSORT format detections
        detections_ds = []
        for i, idx in enumerate(valid_indices):
            det = Detection(bboxes[idx], scores[idx], features[i])
            detections_ds.append(det)

        self.tracker.predict()
        self.tracker.update(detections_ds)
        self.update_tracks()

    def update_tracks(self):
        self.tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            self.tracks.append(Track(track.track_id, bbox))

class Track:
    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox