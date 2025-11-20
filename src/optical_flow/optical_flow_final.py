import cv2
import numpy as np
import time

def dense_optical_flow(method, video_path, params=[], to_gray=None, overlay=True, 
                       scale=0.25, skip_frames=0, profile=False):
    """
    Optimized optical flow with aggressive speed improvements.
    
    Args:
        scale: Resolution scale (0.25 = 4x faster, 0.5 = 2x faster)
        skip_frames: Process every Nth frame (0=all, 1=every other, 2=every third)
        profile: Print timing breakdown
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return
    
    print(f"Original frame size: {old_frame.shape[:2]}")
    
    # Resize frame 
    old_frame = cv2.resize(old_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    print(f"Resized frame size: {old_frame.shape[:2]} (scale={scale})")
    print(f"Pixel reduction: {(1-scale**2)*100:.1f}%")
    print()

    # Decide if grayscale is needed
    if to_gray is None:
        to_gray = (method == cv2.calcOpticalFlowFarneback)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) if to_gray else old_frame.copy()

    # Pre-allocate arrays (avoid reallocations)
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255
    flow_bgr = np.zeros_like(old_frame)
    
    # For FPS calculation
    fps_values = []
    frame_count = 0
    
    # For profiling
    if profile:
        time_resize = []
        time_flow = []
        time_viz = []
    
    # Create window once
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)

    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue

        start = time.time()

        # Resize frame - resolution/speed trade-off
        t0 = time.time()
        new_frame = cv2.resize(new_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        t1 = time.time()
        
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY) if to_gray else new_frame.copy()

        # Compute optical flow
        if callable(method):
            flow = method(old_gray, new_gray, None, *params)
        else:
            flow = method.calc(old_gray, new_gray, None)
        
        t2 = time.time()

        # Convert flow to HSV (optimized)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 90 / np.pi  # Simplified: 180/π/2 = 90/π
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, flow_bgr)

        # Overlay optical flow
        if overlay:
            cv2.addWeighted(new_frame, 0.3, flow_bgr, 0.7, 0, new_frame)
            display_frame = new_frame
        else:
            display_frame = flow_bgr
        
        t3 = time.time()
        
        if profile:
            time_resize.append((t1-t0)*1000)
            time_flow.append((t2-t1)*1000)
            time_viz.append((t3-t2)*1000)

        # Compute FPS (smoothed)
        end = time.time()
        fps_real = 1 / (end - start + 1e-6)
        fps_values.append(fps_real)
        if len(fps_values) > 30:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        
        # cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Optical Flow", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

        old_gray = new_gray

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAverage FPS: {avg_fps:.2f}")
    
    """
    if profile and len(time_resize) > 0:
        print(f"\nTiming breakdown (ms per frame):")
        print(f"  Resize:       {np.mean(time_resize):.2f}")
        print(f"  Optical Flow: {np.mean(time_flow):.2f}")
        print(f"  Visualization:{np.mean(time_viz):.2f}")
        print(f"  Total:        {np.mean(time_resize)+np.mean(time_flow)+np.mean(time_viz):.2f}")
    """

def main():
    video_path = "../../data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"

    method = cv2.calcOpticalFlowFarneback
    
    params = [
        0.5,   # pyr_scale 
        1,     # levels (single level for speed)
        8,     # winsize (small window)
        1,     # iterations (single iteration)
        5,     # poly_n 
        1.1,   # poly_sigma
        0      # flags
    ]
    
    print("Starting optical flow processing...")
    print("Press ESC to quit\n")
    

    dense_optical_flow(method, video_path, params=params, 
                      overlay=True, scale=0.25, skip_frames=0, profile=True)

if __name__ == "__main__":
    main()