import os
import time
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepfake_detection.data.split_dataset import split_dataset

from load_dataset import load_dataset

# -------------------------- CONFIGURATION --------------------------
logging.basicConfig(level=logging.INFO)

WINDOW_SIZE = 6
STRIDE = 1

def set_landmarks():
    return [10, 34, 35, 36, 47, 50, 53, 67, 69, 70, 100, 101, 104, 108, 109, 111, 116,
            117, 118, 119, 121, 123, 124, 126, 127, 139, 143, 147, 151, 187, 189, 203,
            205, 206, 207, 216, 222, 228, 230, 234, 244, 264, 266, 276, 280, 282, 283,
            299, 300, 329, 330, 337, 338, 340, 346, 347, 348, 353, 355, 368, 371, 372,
            411, 417, 423, 425, 426, 427, 436, 441, 444, 446, 448, 450, 452, 454, 464]

# ------------------------ VIDEO PROCESSING -------------------------

def write_video(output_path, images, fps):
    # Setup video writer
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

    # Write video
    for image in images:
        video.write(image[...,::-1].copy())

    # Release video
    video.release()


def process_video(instance, method, calc_bpm=False, landmarks_video_dir: str = None):
    videofilepath = instance.path
    video_type = instance.label
    try:
        from pyVHR.extraction import SkinExtractionConvexHull
        from pyVHR.extraction.utils import get_fps, sig_windowing
        from pyVHR.extraction.sig_processing import SignalProcessing
        from pyVHR.BPM.BPM import BVP_to_BPM, BPM_median
        from pyVHR.analysis.pipeline import RGB_sig_to_BVP
        from pyVHR.BVP.methods import cpu_GREEN, cpu_POS, cpu_OMIT, cpu_CHROM, cpu_PBV, cpu_PCA, cpu_LGI, cpu_ICA, cpu_SSR
        from pyVHR.deepRPPG.mtts_can import MTTS_CAN_deep

        sig_processing = SignalProcessing()
        sig_processing.set_landmarks(set_landmarks())
        sig_processing.set_square_patches_side(28.0)
        sig_processing.choose_cuda_device(0)
        sig_processing.skin_extractor = SkinExtractionConvexHull('GPU')

        if landmarks_video_dir:
            sig_processing.set_visualize_skin_and_landmarks(visualize_skin=True,
                                                            visualize_landmarks=True,
                                                            visualize_landmarks_number=True,
                                                            visualize_patch=True)

        logging.info(f"Processing video: {videofilepath}")

        if method == 'green':
            m = cpu_GREEN
        elif method == 'pos':
            m = cpu_POS
        elif method == 'omit':
            m = cpu_OMIT
        elif method == 'chrom':
            m = cpu_CHROM
        elif method == 'pbv':
            m = cpu_PBV
        elif method == 'pca':
            m = cpu_PCA
        elif method == 'lgi':
            m = cpu_LGI
        elif method == 'ica':
            m = cpu_ICA
        elif method == 'ssr':
            m = cpu_SSR

        sig = sig_processing.extract_patches(str(videofilepath), "squares", "mean")
        fps = get_fps(str(videofilepath))
        windowed_sig, timesES = sig_windowing(sig, WINDOW_SIZE, STRIDE, fps)

        if landmarks_video_dir:
            landmarks_video_path = os.path.join(landmarks_video_dir, str(videofilepath.name).split('.')[0] + '_landmarks.mp4')
            write_video(landmarks_video_path, sig_processing.get_visualize_patches(), fps)

        if method == 'MTTS_CAN':
            bvp = MTTS_CAN_deep(windowed_sig, fps)
        elif method == 'pos':
            bvp = RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu', method=m, params={'fps': fps})
        elif method == 'green_chrom':
            bvps = [RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu', method=i) for i in [cpu_CHROM, cpu_GREEN]]
            bvp = np.mean(bvps, axis=0)
        elif method == 'green_chrom_pos':
            bvps = [RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu', method=i) for i in [cpu_CHROM, cpu_GREEN]]
            bvps.append(RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu', method=cpu_POS, params={'fps': fps}))
            bvp = np.mean(bvps, axis=0)
        else:
            bvp = RGB_sig_to_BVP(windowed_sig, fps=fps, device_type='cpu', method=m)

        if bvp is None:
            logging.warning(f"No BVP extracted for {videofilepath}. Skipping.")
            return None

        if calc_bpm:
            bpmES = BVP_to_BPM(bvp, fps, minHz=0.65, maxHz=4.0)
            bpm, uncertainty = BPM_median(bpmES)
        else:
            bpmES, bpm, uncertainty = None, None, None

        return {
            "Filename": str(videofilepath),
            "BVPS": bvp,
            "timesES": timesES,
            "Type": video_type,
            "BPMES": bpmES,
            "BPM": bpm,
            "Uncertainty": uncertainty,
            "FPS": fps
        }
    except:
        print("Something went wrong with ", videofilepath)

# ----------------------------- MAIN -------------------------------- #
def main():

    # Load dataset
    # dataset = load_dataset('ubfc2')
    # dataset = load_dataset('ground-truth')
    # dataset = load_dataset('ff_raw_test')
    dataset = load_dataset('Sanne')
    # dataset = load_dataset('df-1.0')

    # Set method
    method = 'green'

    # Set video output path (if applicable)
    landmarks_video_dir = None
    # landmarks_video_dir = '/mnt/extern/FBS/Beeld/Sanne de Wit/Dataset/Ground Truth/landmark_videos'

    # If dataset is larger than 1000 instances, split into parts
    if len(dataset) > 1000:
        set1, set2 = split_dataset(dataset, test_size=0.5, random_state=42)
        datasets = [set1, set2]
    else:
        datasets = [dataset]

    # Loop over datasets
    start_time = time.perf_counter()
    for dataset in datasets:

        # Do not run in parallel when exporting video files
        if landmarks_video_dir:
            results = [process_video(i, method, True, landmarks_video_dir)
                       for i in tqdm(dataset, total=len(dataset))]
        else:
            # Extract signals (parallelize over CPU cores)
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(executor.map(process_video,
                                                 dataset,
                                                 repeat(method),
                                                 repeat(True)),
                                    total=len(dataset)))

        # Filter out Nones
        results = [r for r in results if r is not None]

        # Write to file
        outfile = f"/home/stijn/repositories/rPPG_paper/extracted_signals/extracted_signals_{dataset.name}_{method}.json"
        df = pd.DataFrame(results)
        df.to_json(outfile,
                   orient='records',
                   lines=True
                   )
        logging.info(f"Saved extracted data to {outfile}!")
        logging.info(f"Total time: {time.perf_counter() - start_time:.2f}s")

if __name__ == "__main__":
    main()