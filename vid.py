from IPython.display import display, clear_output
import time

def play_video_in_notebook(video_path, fps=30, max_frames=500):
    """Display a saved video frame-by-frame using matplotlib.
       Works even when the system ffmpeg/codec is broken."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    try:
        fig, ax = plt.subplots()
        ax.axis('off')
        fig.show()
        fig.canvas.draw()

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV uses BGR, matplotlib needs RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.cla()
            ax.imshow(frame_rgb)
            ax.axis('off')
            fig.canvas.draw()
            clear_output(wait=True)
            display(fig)
            time.sleep(1 / fps)
    finally:
        cap.release()
        plt.close(fig)