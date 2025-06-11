import colorsys
import hashlib
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction


class BaseTracker(ABC):
    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        nr_classes: int = 80,
        per_class: bool = False,
        asso_func: str = "iou",
        is_obb: bool = False,
    ):
        """
        Initialize the BaseTracker object with detection threshold, maximum age, minimum hits,
        and Intersection Over Union (IOU) threshold for tracking objects in video frames.

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age of a track before it is considered lost.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func + "_obb" if is_obb else asso_func
        self.is_obb = is_obb

        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        self._first_dets_processed = False

        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []

        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)

    @abstractmethod
    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
    
    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size == 0:
            return class_dets, class_embs

        class_indices = np.where(dets[:, 5] == cls_id)[0]
        class_dets = dets[class_indices]

        if embs is None:
            return class_dets, class_embs

        # Assert that if embeddings are provided, they have the same number of elements as detections
        assert dets.shape[0] == embs.shape[0], (
            "Detections and embeddings must have the same number of elements when both are provided"
        )
        class_embs = None
        if embs.size > 0:
            class_embs = embs[class_indices]
            self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
        return class_dets, class_embs

    @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """

        def wrapper(self, *args, **kwargs):
            # Extract detections and image from args
            dets = args[0]

            # Unwrap `data` attribute if present
            if hasattr(dets, 'data'):
                dets = dets.data

            # Convert memoryview to numpy array if needed
            if isinstance(dets, memoryview):
                dets = np.array(dets, dtype=np.float32)  # Adjust dtype if needed

            img = None

            # First-time detection setup
            if not self._first_dets_processed and dets is not None:
                if dets.ndim == 2 and dets.shape[1] == 6:
                    self.is_obb = False
                    self._first_dets_processed = True
                elif dets.ndim == 2 and dets.shape[1] == 7:
                    self.is_obb = True
                    self._first_dets_processed = True

            # First frame image-based setup
            if not self._first_frame_processed and img is not None:
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(
                    w=self.w,
                    h=self.h,
                    asso_mode=self.asso_func_name
                ).asso_func
                self._first_frame_processed = True

            # Call the original method with the unwrapped `dets`
            return method(self, dets, **kwargs)

        return wrapper
    
    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """

        def wrapper(self, dets: np.ndarray):
            # handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))

            if not self.per_class:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets)
            # else:
            # Initialize an array to store the tracks for each class
            per_class_tracks = []

            # same frame count for all classes
            frame_count = self.frame_count

            for cls_id in range(self.nr_classes):
                # Get detections and embeddings for the current class
                class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)

                LOGGER.debug(f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings"
                      f" {class_embs.shape if class_embs is not None else None}")

                # Activate the specific active tracks for this class id
                self.active_tracks = self.per_class_active_tracks[cls_id]

                # Reset frame count for every class
                self.frame_count = frame_count

                # Update detections using the decorated method
                tracks = update_method(self, dets=class_dets)

                # Save the updated active tracks
                self.per_class_active_tracks[cls_id] = self.active_tracks

                if tracks.size > 0:
                    per_class_tracks.append(tracks)

            # Increase frame count by 1
            self.frame_count = frame_count + 1
            return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))

        return wrapper

    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()

        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xFFFFFFFF

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = "#%02x%02x%02x" % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip("#")[i : i + 2], 16) for i in (0, 2, 4))

        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]

        return bgr
