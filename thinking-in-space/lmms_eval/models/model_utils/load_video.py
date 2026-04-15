import av
import numpy as np
from av.codec.context import CodecContext


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def sample_frame_indices(total_frames, num_frm=8):
    if total_frames <= 0:
        raise ValueError(f"Video contains no decodable frames: total_frames={total_frames}")

    sampled_frm = min(total_frames, num_frm)
    return np.linspace(0, total_frames - 1, sampled_frm, dtype=int).astype(int)


def read_video_pyav_with_indices(video_path, num_frm=8):
    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            indices = sample_frame_indices(total_frames, num_frm)
            frames = record_video_length_stream(container, indices)
        except Exception:
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            indices = sample_frame_indices(total_frames, num_frm)
            frames = [frames[i] for i in indices]
    else:
        container = av.open(video_path)
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        indices = sample_frame_indices(total_frames, num_frm)
        frames = [frames[i] for i in indices]

    video = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    return video, indices.tolist()


def read_video_pyav(video_path, num_frm=8):
    video, _ = read_video_pyav_with_indices(video_path, num_frm=num_frm)
    return video
