from collections import defaultdict
from tap import Tap
from pathlib import Path
import tempfile
import subprocess
import hashlib
import cv2
import math
import numpy as np
import copyreg

def patch_Keypoint_pickling():
    # Create the bundling between class and arguments to save for Keypoint class
    # See : https://stackoverflow.com/questions/50337569/pickle-exception-for-cv2-boost-when-using-multiprocessing/50394788#50394788
    def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )
    # C++ Constructor, notice order of arguments : 
    # KeyPoint (float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)

    # Apply the bundling to pickle
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)



class Args(Tap):
    video: str
    slides: str


def resizeimg(arr: np.ndarray):
    return arr
    width = 500
    return cv2.resize(
        arr,
        (width, int(arr.shape[0] / arr.shape[1] * width)),
        interpolation=cv2.INTER_LANCZOS4,
    )

def unique_temp_path_for_str(s: str):
    tmp = Path(tempfile.gettempdir())
    hash = hashlib.sha1(s.encode('utf8')).hexdigest()
    out_dir = tmp / "match-slides-to-recording-2" / hash
    return out_dir

def unique_temp_path_for_file(file: Path):
    tmp = Path(tempfile.gettempdir())
    hash = hashlib.sha1(file.read_bytes()).hexdigest()
    out_dir = tmp / "match-slides-to-recording" / hash
    return out_dir


def pdf_to_images(pdf: Path) -> list[Path]:
    # todo: cleanup
    out_dir = unique_temp_path_for_file(pdf)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        print(f"writing pdf pages to {out_dir}")
        try:
            subprocess.run(["pdftocairo", pdf, "-png", out_dir / "page"])
        except Exception as e:
            out_dir.rmdir()
            raise e
    else:
        print(f"loading existing pdf extracted pages from {out_dir}")
    return list(out_dir.glob("page*.png"))


def video_to_images_cv(video: Path) -> list[Path]:
    out_dir = unique_temp_path_for_file(video)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        print(f"writing video frames to {out_dir}")
        cap = cv2.VideoCapture(str(video))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        print(f"{fps=}")
        if fps == 0:
            raise Exception("fps unknown")
        while cap.isOpened():
            frame_number = cap.get(1)
            ret, frame = cap.read()
            if ret != True:
                break
            frame_number = int(round(frame_number))
            if frame_number % fps == 0:
                print(f"{frame_number=}")
                out_file = out_dir / f"second-{frame_number//fps:05d}.png"
                cv2.imwrite(str(out_file), frame)
    else:
        print(f"loading existing extracted frames from {out_dir}")
    return list(out_dir.glob("frame*.png"))


def video_to_images_ffmpeg(video: Path) -> list[Path]:
    out_dir = unique_temp_path_for_file(video)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        print(f"writing video frames to {out_dir}")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video,
                "-vf",
                "fps=1/5",
                out_dir / "second-%05d.png",
            ]
        )
    else:
        print(f"loading existing extracted frames from {out_dir}")
    return list(out_dir.glob("*.png"))


_orb = None
_mser = None

def get_mser_singleton():
    global _mser 
    if  _mser is None:
     _mser  = cv2.MSER_create(nfeatures=1000)
    return _mser 

# def get_orb_singleton():
#     global _orb
#     if _orb is None:
#         _orb = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     return _orb

def get_orb_singleton():
    global _orb
    if _orb is None:
        FAST_SCORE=1
        _orb = cv2.ORB_create(nfeatures=2000, patchSize=62, edgeThreshold=62, scoreType=FAST_SCORE)
    return _orb

def detect_features_in_frame_2(frame: Path):
    import pickle
    out_file = frame.with_suffix(".orb.pickle")
    if out_file.exists():
        with out_file.open("rb") as f:
            return pickle.load(f)
    else:
        mser = get_mser_singleton()
        orb = get_orb_singleton()
        image = cv2.imread(str(frame))
        image = resizeimg(image)
        regions = mser.detectRegions(image)
        kp, des = orb.cCompute(image, regions) # todo: points and bounding boxes to keypoints
        with out_file.open("wb") as f:
            pickle.dump((kp,des), f, protocol=pickle.HIGHEST_PROTOCOL)
        return kp, des

def cache_decorator(fn):
    from functools import wraps
    import pickle
    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = str((fn.__name__, args, kwargs))
        print(f"cache key {key}")
        out_file = unique_temp_path_for_str(key).with_suffix(".pickle")
        out_file.parent.mkdir(exist_ok=True)
        if out_file.exists():
            print("cache hit")
            with out_file.open("rb") as f:
                return pickle.load(f)
        else:
            print("cache miss")
            ret = fn(*args, **kwargs)
            with out_file.open("wb") as f:
                pickle.dump(ret, f)
            return ret
    return wrapper


def detect_features_in_frame(frame: Path):
    orb = get_orb_singleton()
    image = cv2.imread(str(frame))
    image = resizeimg(image)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


@cache_decorator
def detect_features_in_frames_parallel(frames: list[Path]):
    import multiprocessing
    from tqdm import tqdm

    with multiprocessing.Pool() as p:
        features = tqdm(
            p.imap(detect_features_in_frame, frames),
            desc="detecting features",
            total=len(frames),
        )

        return dict(zip(frames, features))


display = False
def draw_to_file(frame: Path, slide: Path, fnamesuffix: str, matches):
    orb = get_orb_singleton()
    frame_img = cv2.imread(str(frame))
    slide_img = cv2.imread(str(slide))

    frame_img = resizeimg(frame_img)
    slide_img = resizeimg(slide_img)

    frame_m = orb.detectAndCompute(frame_img, None)
    slide_m = orb.detectAndCompute(slide_img, None)

    ii = cv2.drawMatchesKnn(
        frame_img,
        frame_m[0],
        slide_img,
        slide_m[0],
        [[m] for m in matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    out_file = frame.parent.parent / "out2" / (frame.stem + "-" + fnamesuffix + "-" + slide.stem + ".jpg")
    out_file.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_file), ii)
    print(f"written to file {out_file}")

    if display:
        import matplotlib.pyplot as plt

        plt.imshow(ii)
        plt.show()
    pass

def get_rating_of_affine_transform(kp1, kp2):
    a = np.asarray([p.pt for p in kp1])
    b = np.asarray([p.pt for p in kp2])
    retval, inliers = cv2.estimateAffine2D(a, b, None)
    assert len(inliers) == len(kp1)
    inlier_ratio = np.count_nonzero(inliers) / len(inliers)
    return inlier_ratio
def get_best_match_for(
    frames: list[Path], slides: list[Path], features: dict[Path, np.ndarray]
):
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    flann = cv2.FlannBasedMatcher(index_params, {})
    flann.add([features[c][1] for c in slides])
    flann.train()
    for frame in frames:
        frame_keypoints, frame_feat_descriptors = features[frame]
        matches = flann.knnMatch(frame_feat_descriptors, k=30)  # max 30 x same slides??
        mo = defaultdict(lambda: [])
        for matchiii in matches:
            best_dist = matchiii[0].distance
            for match in matchiii:
                if match.distance < best_dist * 1.05:
                    mo[match.imgIdx].append(match)
        molist = sorted(mo.items(), key=lambda e: -len(e[1]))
        print(f"best matches for frame {frame.name}")
        slides_with_ratings = []
        for slide_idx, matches in molist[0:40]:
            slide = slides[slide_idx]
            slide_keypoints, slide_feat_descriptors = features[slide]
            matching_keypoint_pairs = [(slide_keypoints[match.trainIdx], frame_keypoints[match.queryIdx]) for match in matches]
            left, right = zip(*matching_keypoint_pairs)
            affine_rating = get_rating_of_affine_transform(list(left), list(right))
            print(f"affine transform inlier ratio: {affine_rating:.2f}")
            rating = affine_rating * len(matches)
            slides_with_ratings.append((slide, rating, matches))
            print(f"slide {slide.name}: {len(matches)} matches, ratio={affine_rating:.2f}, score: {rating:.0f}")
        slides_with_ratings.sort(key=lambda e: -e[1])
        # if frame.name == "second-00034.png":
        for slide, rating, matches in slides_with_ratings[0:1]:
            draw_to_file(frame, slide, f"rating{rating:03.0f}", matches)


if __name__ == "__main__":
    patch_Keypoint_pickling()
    args = Args().parse_args()
    pngs = sorted(pdf_to_images(Path(args.slides)))
    frames = sorted(video_to_images_ffmpeg(Path(args.video)))

    features = detect_features_in_frames_parallel([*pngs, *frames])

    get_best_match_for(frames, pngs, features)
