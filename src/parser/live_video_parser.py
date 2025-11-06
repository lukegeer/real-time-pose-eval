import cv2, time

class VideoParser:
    def __init__(self, source, resize=None, show_fps=False, live=False):
        """
        Opens a live video file and returns the real time frames for annotation

        Args:
            source (str): Path to .mp4 file or webcam index
            resize (tuple[int,int], optional): resize frames to a certain (width, height)
            show_fps (bool): prints fps every few frames if True
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        self.resize = resize
        self.show_fps = show_fps
        self.live = live

    def __iter__(self):
        frame_idx = 0
        last = time.time()
        fps = 0.0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam")
                break
            
            if self.live:
                frame = cv2.flip(frame, 1)
            
            if self.resize:
                frame = cv2.resize(frame, self.resize)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            if self.show_fps and frame_idx % 30 == 0:
                now = time.time()
                fps = 30 / (now - last)
                print(f"FPS: {fps}")

            yield {"frame": frame, "timestamp": timestamp, "frame_idx": frame_idx, "fps": fps}

            frame_idx += 1

        self.cap.release()