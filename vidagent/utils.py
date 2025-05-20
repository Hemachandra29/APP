import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from pathlib import Path

def extract_frames(video_path: str, fps: int = 1) -> List[np.ndarray]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
        
    cap.release()
    return frames

def detect_scene_changes(frames: List[np.ndarray], threshold: float = 30.0) -> List[int]:
    """
    Detect scene changes in video frames.
    
    Args:
        frames: List of video frames
        threshold: Threshold for scene change detection
        
    Returns:
        List of frame indices where scene changes occur
    """
    scene_changes = []
    prev_frame = None
    
    for i, frame in enumerate(frames):
        if prev_frame is None:
            prev_frame = frame
            continue
            
        # Calculate frame difference
        diff = cv2.absdiff(frame, prev_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > threshold:
            scene_changes.append(i)
            
        prev_frame = frame
        
    return scene_changes

def create_transition(clip1: VideoFileClip, 
                     clip2: VideoFileClip, 
                     transition_type: str = "crossfade",
                     duration: float = 1.0) -> VideoFileClip:
    """
    Create a transition between two video clips.
    
    Args:
        clip1: First video clip
        clip2: Second video clip
        transition_type: Type of transition (crossfade, fade, slide)
        duration: Duration of transition in seconds
        
    Returns:
        Combined video clip with transition
    """
    if transition_type == "crossfade":
        return CompositeVideoClip([
            clip1.crossfadein(duration),
            clip2.crossfadeout(duration)
        ])
    elif transition_type == "fade":
        return CompositeVideoClip([
            clip1.fadeout(duration),
            clip2.fadein(duration)
        ])
    else:
        raise ValueError(f"Unsupported transition type: {transition_type}")

def create_subtitle_clip(text: str, 
                        duration: float,
                        position: str = "bottom",
                        font: str = "Arial",
                        font_size: int = 24,
                        color: str = "white",
                        stroke_color: str = "black",
                        stroke_width: int = 2) -> TextClip:
    """
    Create a subtitle clip with specified styling.
    
    Args:
        text: Subtitle text
        duration: Duration of subtitle
        position: Position of subtitle (top, bottom)
        font: Font name
        font_size: Font size
        color: Text color
        stroke_color: Stroke color
        stroke_width: Stroke width
        
    Returns:
        TextClip with subtitle
    """
    clip = TextClip(
        text,
        font=font,
        fontsize=font_size,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width
    )
    
    # Position the subtitle
    if position == "bottom":
        clip = clip.set_position(("center", "bottom"))
    else:
        clip = clip.set_position(("center", "top"))
        
    return clip.set_duration(duration)

def save_video(clip: VideoFileClip, 
              output_path: str,
              fps: int = 30,
              codec: str = "libx264",
              audio_codec: str = "aac") -> None:
    """
    Save video clip with specified settings.
    
    Args:
        clip: Video clip to save
        output_path: Output file path
        fps: Frames per second
        codec: Video codec
        audio_codec: Audio codec
    """
    clip.write_videofile(
        output_path,
        fps=fps,
        codec=codec,
        audio_codec=audio_codec
    ) 