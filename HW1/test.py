import os
import numpy as np

def save_agent_video(env, agent, filename="agent_demo.mp4", max_steps=1000):
    """
    Record one episode and save it as a video file.
    Tries H.264 (avc1), then mp4v, then XVID/MJPG in an AVI container.
    """
    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state
        if len(frames) >= max_steps:
            break

    height, width, layers = frames[0].shape

    # Codec / container combinations to attempt
    codec_tries = [
        (cv2.VideoWriter_fourcc(*"avc1"), ".mp4"),
        (cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"),
        (cv2.VideoWriter_fourcc(*"XVID"), ".avi"),
        (cv2.VideoWriter_fourcc(*"MJPG"), ".avi"),
    ]

    # Ensure filename has the right extension for the final attempt
    base, _ = os.path.splitext(filename)

    for fourcc, ext in codec_tries:
        trial_filename = f"{base}{ext}"
        out = cv2.VideoWriter(trial_filename, fourcc, 30.0, (width, height))
        if not out.isOpened():
            continue
        try:
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        finally:
            out.release()

        # Check if the file is non-trivial (at least ~1 KB per frame)
        if os.path.exists(trial_filename) and os.path.getsize(trial_filename) > len(frames) * 500:
            print(f"Video saved successfully as {trial_filename} (codec: {fourcc})")
            return trial_filename
        else:
            os.remove(trial_filename)   # discard unusable file

    # Last resort: save raw frames as numpy file
    np_filename = f"{base}_frames.npy"
    np.save(np_filename, np.array(frames))
    print(f"All codecs failed. Raw frames saved as {np_filename}")
    return None