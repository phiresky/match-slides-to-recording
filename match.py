from collections import defaultdict
from tap import Tap
from pathlib import Path
import tempfile
import subprocess
import hashlib
import cv2
import math
import numpy as np

img_size = (400, (400 * 9) // 16)

class Args(Tap):
    video: str
    slides: str


def resizeimg(arr: np.ndarray):
    return cv2.resize(arr, (400, int(arr.shape[0] / arr.shape[1] * 400)), interpolation=cv2.INTER_LANCZOS4)

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


def get_orb_singleton():
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=2000)
    return _orb


def detect_features_in_frame(frame: Path):
    out_file = frame.with_suffix(".orbmax.npy")
    if False and out_file.exists():
        return np.load(out_file)
    else:
        orb = get_orb_singleton()
        image = cv2.imread(str(frame))
        image = resizeimg(image)
        kp1, des1 = orb.detectAndCompute(image, None)
        np.save(out_file, des1)
        return des1


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

def draw_to_file(frame: Path, slide: Path, matches):
    orb = get_orb_singleton()
    frame_img = cv2.imread(str(frame))
    slide_img = cv2.imread(str(slide))

    frame_img = resizeimg(frame_img)
    slide_img = resizeimg(slide_img)


    frame_m = orb.detectAndCompute(frame_img, None)
    slide_m = orb.detectAndCompute(slide_img, None)

    ii = cv2.drawMatchesKnn(frame_img, frame_m[0], slide_img, slide_m[0], [[m] for m in matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    import matplotlib.pyplot as plt
    plt.imshow(ii)
    plt.show()
    pass
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
    flann.add([features[c] for c in slides])
    flann.train()
    for frame in frames:
        matches = flann.knnMatch(features[frame], k=30) # max 30 x same slides??
        mo = defaultdict(lambda: [])
        for matchiii in matches:
            best_dist = matchiii[0].distance
            for match in matchiii:
                if match.distance < best_dist * 1.05:
                    mo[match.imgIdx].append(match)
        molist = sorted(mo.items(), key=lambda e: -len(e[1]))
        print(f"best matches for frame {frame.name}")
        for slide_idx, matches in molist[0:10]:
            print(f"slide {slides[slide_idx].name}: {len(matches)} matches")
        if frame.name == "second-00035.png":
            for slide_idx, matches in molist[0:10]:
                slide = slides[slide_idx]
                draw_to_file(frame, slide, matches)
            pass
        


if __name__ == "__main__":
    args = Args().parse_args()
    pngs = sorted(pdf_to_images(Path(args.slides)))
    frames = sorted(video_to_images_ffmpeg(Path(args.video)))

    features = detect_features_in_frames_parallel([*pngs, *frames])

    get_best_match_for(frames, pngs, features)
