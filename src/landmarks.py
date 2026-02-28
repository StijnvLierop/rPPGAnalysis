import itertools
import tempfile
from typing import Tuple, List, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils


# SELECTED_LANDMARKS = [10, 34, 35, 36, 47, 50, 53, 67, 69, 70, 100, 101, 104, 108, 109, 111, 116,
#                       117, 118, 119, 121, 123, 124, 126, 127, 139, 143, 147, 151, 187, 189, 203,
#                       205, 206, 207, 216, 222, 228, 230, 234, 244, 264, 266, 276, 280, 282, 283,
#                       299, 300, 329, 330, 337, 338, 340, 346, 347, 348, 353, 355, 368, 371, 372,
#                       411, 417, 423, 425, 426, 427, 436, 441, 444, 446, 448, 450, 452, 454, 464]

# MediaPipe Face Mesh Landmark Regions
# These indices are based on the standard 468-point face mesh.
LANDMARK_REGIONS = {
    # High Priority Zones (Best for rPPG)
    "high_prio_forehead": [10, 67, 69, 104, 108, 109, 151, 299, 337, 338],
    "high_prio_nose": [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198, 209,
                       217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440,
                       456],
    "high_prio_left_cheek": [36, 47, 50, 100, 101, 116, 117, 118, 119, 123, 126, 147, 187, 203, 205,
                             206, 207, 216],
    "high_prio_right_cheek": [266, 280, 329, 330, 346, 347, 348, 355, 371, 411, 423, 425, 426, 427,
                              436],

    # Mid Priority Zones
    "mid_prio_forehead": [8, 9, 21, 68, 103, 251, 284, 297, 298, 301, 332, 333, 372, 383],
    "mid_prio_nose": [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412],
    "mid_prio_left_cheek": [34, 111, 137, 156, 177, 192, 213, 227, 234],
    "mid_prio_right_cheek": [340, 345, 352, 361, 454],
    "mid_prio_chin": [135, 138, 169, 170, 199, 208, 210, 211, 214, 262, 288, 416, 428, 430, 431,
                      432, 433, 434],
    "mid_prio_mouth": [92, 164, 165, 167, 186, 212, 322, 391, 393, 410],

    # Specific Anatomical Segments
    "forehead_left": [21, 71, 68, 54, 103, 104, 63, 70, 53, 52, 65, 107, 66, 108, 69, 67, 109, 105],
    "forehead_center": [10, 151, 9, 8, 107, 336, 285, 55],
    "forehead_right": [338, 337, 336, 296, 285, 295, 282, 334, 293, 301, 251, 298, 333, 299, 297,
                       332, 284],
    "cheek_left_top": [116, 111, 117, 118, 119, 100, 47, 126, 101, 123, 137, 177, 50, 36, 209, 129,
                       205, 147, 187, 215, 206, 203],
    "cheek_right_top": [349, 348, 347, 346, 345, 447, 323, 280, 352, 330, 371, 358, 423, 426, 425,
                        427, 411, 376],
    "cheek_left_bottom": [215, 138, 135, 210, 212, 57, 216, 207, 192],
    "cheek_right_bottom": [435, 427, 416, 364, 394, 422, 287, 410, 434, 436],
    "nose_full": [193, 417, 168, 188, 6, 412, 197, 174, 399, 456, 195, 236, 131, 51, 281, 360, 440,
                  4, 220, 219, 305],
    "chin_full": [204, 170, 140, 194, 201, 171, 175, 200, 418, 396, 369, 421, 431, 379, 424],

    # Exclusion Zones (Dense masks)
    "left_eye": [157, 144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65,
                 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124],
    "right_eye": [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300,
                  441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381,
                  373, 249, 253, 255],
    "mouth_full": [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186,
                   57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106],

    # Global Coverage
    "equispaced_facial_points": [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50,
                                 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117,
                                 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152,
                                 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216,
                                 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297,
                                 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364,
                                 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430,
                                 432, 436]
}

def extract_landmarks(frames: List[np.ndarray],
                      fps: float,
                      selected_landmark_regions: List[str]) -> Tuple[np.ndarray, List]:
    """
    Extracts facial landmarks from a list of frames and returns them as a numpy array.

    :param frames: Frames to extract landmarks from.
    :param fps: The frame rate of the video.
    :param selected_landmark_regions: List of landmark regions to look at.
    :return: - A numpy array containing the mean RGB value
               of the selected landmarks with shape (num_frames, 3).
             - A list of FaceLandmarker detection results with length num_frames.
    """
    # Initialize MediaPipe Face Landmarker
    base_options = mp.tasks.BaseOptions(model_asset_path='src/face_landmarker.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    landmark_detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # Get selected landmarks
    selected_landmarks = [v for l, v in LANDMARK_REGIONS.items() if l in selected_landmark_regions]
    selected_landmarks = list(itertools.chain.from_iterable(selected_landmarks))

    # Get face landmarks for all frames
    landmarks_list = []
    landmark_detector_results = []
    for frame_idx, frame in enumerate(list(frames)):

        # Get timestamp
        timestamp_ms = int((frame_idx / fps) * 1000)

        # Detect landmarks for frame
        landmark_detector_result = landmark_detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
            timestamp_ms
        )

        # If face landmarks were found
        if landmark_detector_result.face_landmarks:

            image_height, image_width, _ = frame.shape

            # Get landmarks
            landmarks = landmark_detector_result.face_landmarks[0]

            # Extract pixel values at the specific locations
            pixel_colors = []
            roi_size = 5  # We will take a 5x5 square around each landmark (25 pixels total)
            offset = roi_size // 2
            for index in selected_landmarks:
                landmark = landmarks[index]

                # Get landmark center coordinates
                x_px = int(landmark.x * image_width)
                y_px = int(landmark.y * image_height)

                # Define ROI boundaries and clip to image dimensions
                y1 = max(0, y_px - offset)
                y2 = min(image_height, y_px + offset + 1)
                x1 = max(0, x_px - offset)
                x2 = min(image_width, x_px + offset + 1)

                # Extract the square patch and average its pixels
                roi_patch = frame[y1:y2, x1:x2]
                if roi_patch.size > 0:
                    # Average over the patch (spatial filtering)
                    mean_color = np.mean(roi_patch, axis=(0, 1))
                    pixel_colors.append(mean_color)

            # Add mean RGB value of landmark locations to landmarks list
            landmarks_list.append(np.mean(pixel_colors, axis=0))

        # Add predicted landmark locations result
        landmark_detector_results.append(landmark_detector_result)

    # Close landmark detector
    landmark_detector.close()

    return np.array(landmarks_list), landmark_detector_results


def draw_landmark_video(frames: List[np.ndarray],
                        detection_results: List,
                        selected_landmark_regions: List,
                        fps: float = 25) -> Optional[str]:
    """
    Render facial landmarks onto given frames and return an MP4 video.

    :param frames: A list of frames to render landmarks on.
    :param detection_results: A list of FaceLandmarker detection results.
    :param selected_landmark_regions: List of landmark regions to look at.
    :param fps: The frame rate of the video.
    :return: The path to the rendered video.
    """
    # Ensure frames and detection results have the same length
    if len(frames) != len(detection_results):
        raise ValueError("frames and detection_results must have the same length")

    # If no frames, return None
    if not frames:
        return None

    # Create a secure temporary file (auto-deleted after reading)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        temp_path = tmp.name

    # Get selected landmarks
    selected_landmarks = [v for l, v in LANDMARK_REGIONS.items() if
                          l in selected_landmark_regions]
    selected_landmarks = list(itertools.chain.from_iterable(selected_landmarks))

    # Setup VideoWriter once
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    # Loop over frames
    for frame, detection_result in zip(frames, detection_results):
        # Work on a copy to avoid modifying the original frames list
        annotated_image = np.copy(frame)

        # Some detection results may have empty detections; then continue
        if not getattr(detection_result, 'face_landmarks', None):
            bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            continue

        # Get face landmarks for the frame (only look at the first face)
        face_landmarks = detection_result.face_landmarks[0]

        # Only visualize selected landmarks
        subset_landmarks = [face_landmarks[i] for i in selected_landmarks]

        # Draw landmarks on the frame
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=subset_landmarks,
            connections=None,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0),
                                                            thickness=2,
                                                            circle_radius=2)
        )

        # Convert RGB to BGR for OpenCV Writing
        # MediaPipe processing is usually done in RGB, but VideoWriter expects BGR
        bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Write frame to file
        out.write(bgr_frame)

    # Clean up writer before reading the file
    out.release()

    return temp_path


def extract_face_crops(frames: List, target_size=(128, 128)) -> np.ndarray:
    """
        Detects faces using MediaPipe GPU and returns a stacked array of crops.
        """
    # Initialize MediaPipe Task
    base_options = mp.tasks.BaseOptions(
        model_asset_path='src/face_landmarker.task'
    )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    face_crops = []

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            h, w, _ = frame.shape
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Detect landmarks (using frame index as timestamp in ms)
            result = landmarker.detect_for_video(mp_image, i * 33)  # 33ms ~ 30fps

            if result.face_landmarks:
                # Get landmarks for the first face detected
                landmarks = result.face_landmarks[0]

                # Calculate Bounding Box based on all landmarks
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                # Add a 20% margin to include the forehead and cheeks better
                margin_w = (x_max - x_min) * 0.2
                margin_h = (y_max - y_min) * 0.2

                x1 = max(0, int(x_min - margin_w))
                y1 = max(0, int(y_min - margin_h))
                x2 = min(w, int(x_max + margin_w))
                y2 = min(h, int(y_max + margin_h))

                # Crop and Resize
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    resized_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                    face_crops.append(resized_crop)
            else:
                # If a face is missing in one frame, duplicate the last successful crop
                # to keep the temporal sequence consistent for Mamba
                if face_crops:
                    face_crops.append(face_crops[-1])
                else:
                    # If the very first frame fails, use a black placeholder (or skip)
                    face_crops.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))

    # Return as [T, H, W, C] array
    return np.array(face_crops)
