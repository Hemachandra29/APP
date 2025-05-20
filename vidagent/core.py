import os
import torch # type: ignore
import ffmpeg # type: ignore
import whisper # type: ignore
import openai # type: ignore
from pathlib import Path
from typing import Optional, Dict, List, Union
from dotenv import load_dotenv # type: ignore
from transformers import ViTFeatureExtractor, ViTForImageClassification # type: ignore
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip # type: ignore
from .utils import extract_frames, detect_scene_changes, create_transition, create_subtitle_clip, save_video
import cv2 # type: ignore

class VidAgent:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the VidAgent with configuration settings."""
        load_dotenv()
        self.config = config or {}
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize AI models for video processing."""
        # Initialize Whisper for audio transcription
        self.whisper_model = whisper.load_model("base")
        
        # Initialize ViT for scene detection
        self.vit_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit_model.to(self.device)
        
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     add_subtitles: bool = True,
                     add_transitions: bool = True,
                     style: str = "documentary") -> str:
        """
        Process a video file with AI-powered editing.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save processed video
            add_subtitles: Whether to add AI-generated subtitles
            add_transitions: Whether to add smart transitions
            style: Video style (documentary, cinematic, etc.)
            
        Returns:
            Path to the processed video
        """
        # Load video
        video = VideoFileClip(input_path)
        
        # Generate script and subtitles
        if add_subtitles:
            script = self.generate_script(input_path, style)
            subtitles = self._generate_subtitles(video, script)
            video = self._add_subtitles(video, subtitles)
        
        # Add transitions if requested
        if add_transitions:
            video = self._add_transitions(video)
        
        # Save processed video
        save_video(video, output_path)
        return output_path
    
    def generate_script(self, video_path: str, style: str = "documentary") -> str:
        """
        Generate a script based on video content.
        
        Args:
            video_path: Path to video file
            style: Desired script style
            
        Returns:
            Generated script text
        """
        # Extract key frames and transcribe audio
        frames = self._extract_key_frames(video_path)
        transcription = self._transcribe_audio(video_path)
        
        # Generate script using OpenAI
        prompt = f"""Generate a {style} script based on the following video content:
        Transcription: {transcription}
        Key scenes: {frames}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional video script writer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def _extract_key_frames(self, video_path: str) -> List[Dict]:
        """Extract and analyze key frames from video."""
        frames = extract_frames(video_path)
        scene_changes = detect_scene_changes(frames)
        
        # Process frames with ViT
        processed_frames = []
        for i, frame in enumerate(frames):
            if i in scene_changes:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with ViT
                inputs = self.vit_processor(images=frame_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.vit_model(**inputs)
                    predictions = outputs.logits.softmax(dim=1)
                
                processed_frames.append({
                    "frame_index": i,
                    "scene_change": True,
                    "predictions": predictions.cpu().numpy()
                })
        
        return processed_frames
    
    def _transcribe_audio(self, video_path: str) -> str:
        """Transcribe audio from video using Whisper."""
        result = self.whisper_model.transcribe(video_path)
        return result["text"]
    
    def _generate_subtitles(self, video: VideoFileClip, script: str) -> List[Dict]:
        """Generate subtitle timings and text."""
        # Split script into sentences
        sentences = [s.strip() for s in script.split('.') if s.strip()]
        
        # Calculate duration per sentence
        duration_per_sentence = video.duration / len(sentences)
        
        subtitles = []
        for i, sentence in enumerate(sentences):
            start_time = i * duration_per_sentence
            end_time = (i + 1) * duration_per_sentence
            
            subtitles.append({
                "text": sentence,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            })
        
        return subtitles
    
    def _add_subtitles(self, video: VideoFileClip, subtitles: List[Dict]) -> VideoFileClip:
        """Add subtitles to video."""
        subtitle_clips = []
        
        for subtitle in subtitles:
            clip = create_subtitle_clip(
                text=subtitle["text"],
                duration=subtitle["duration"],
                position=self.config.get("subtitles", {}).get("position", "bottom"),
                font=self.config.get("subtitles", {}).get("font", "Arial"),
                font_size=self.config.get("subtitles", {}).get("font_size", 24),
                color=self.config.get("subtitles", {}).get("color", "white"),
                stroke_color=self.config.get("subtitles", {}).get("stroke_color", "black"),
                stroke_width=self.config.get("subtitles", {}).get("stroke_width", 2)
            )
            clip = clip.set_start(subtitle["start_time"])
            subtitle_clips.append(clip)
        
        return CompositeVideoClip([video] + subtitle_clips)
    
    def _add_transitions(self, video: VideoFileClip) -> VideoFileClip:
        """Add smart transitions between scenes."""
        # Extract frames and detect scene changes
        frames = extract_frames(video.filename)
        scene_changes = detect_scene_changes(frames)
        
        # Split video into clips at scene changes
        clips = []
        start_time = 0
        
        for change in scene_changes:
            end_time = change / video.fps
            clip = video.subclip(start_time, end_time)
            clips.append(clip)
            start_time = end_time
        
        # Add final clip
        clips.append(video.subclip(start_time))
        
        # Apply transitions between clips
        final_clips = []
        for i in range(len(clips) - 1):
            transition = create_transition(
                clips[i],
                clips[i + 1],
                transition_type=self.config.get("transitions", {}).get("types", ["crossfade"])[0],
                duration=self.config.get("transitions", {}).get("duration", 1.0)
            )
            final_clips.append(transition)
        
        return CompositeVideoClip(final_clips) 