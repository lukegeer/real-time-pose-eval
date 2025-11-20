import cv2
import numpy as np
import time

def dense_optical_flow(method, video_path, params=[], to_gray=None, overlay=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)

    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    # Decide if grayscale is needed
    if to_gray is None:
        if method == cv2.calcOpticalFlowFarneback:
            to_gray = True
        else:
            to_gray = False

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) if to_gray else old_frame.copy()

    # HSV image for visualization
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    while True:
        ret, new_frame = cap.read()
        if not ret:
            break

        start = time.time()

        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY) if to_gray else new_frame.copy()

        # Compute optical flow
        flow = method(old_gray, new_gray, None, *params) if callable(method) else method.calc(old_gray, new_gray, None)

        # Convert flow to HSV
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Overlay optical flow
        if overlay:
            display_frame = cv2.addWeighted(new_frame, 0.4, flow_bgr, 0.6, 0)
        else:
            display_frame = flow_bgr

        # Compute and display FPS
        end = time.time()
        fps_real = 1 / (end - start + 1e-6)
        cv2.putText(display_frame, f"FPS: {fps_real:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
        cv2.imshow("Optical Flow", display_frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == 27:  # ESC to quit
            break

        old_gray = new_gray.copy()

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = "data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"

    # Farneback method
    method = cv2.calcOpticalFlowFarneback
    params = [0.5, 3, 15, 3, 5, 1.2, 0]

    # # TV-L1 alternative
    # method = cv2.optflow.DualTVL1OpticalFlow_create()
    # params = []

    dense_optical_flow(method, video_path, params=params, overlay=True)


if __name__ == "__main__":
    main()
