from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    # Video processing settings
    "video": {
        "fps": 30,
        "resolution": (1920, 1080),
        "codec": "libx264",
        "audio_codec": "aac",
    },
    
    # AI model settings
    "models": {
        "whisper": {
            "model_size": "base",
            "language": "en",
            "task": "transcribe"
        },
        "vit": {
            "model_name": "google/vit-base-patch16-224",
            "batch_size": 8,
            "confidence_threshold": 0.7
        }
    },
    
    # Script generation settings
    "script": {
        "max_length": 1000,
        "temperature": 0.7,
        "style_prompts": {
            "documentary": "Write in a professional, informative documentary style",
            "cinematic": "Write in a dramatic, cinematic style with emphasis on visual storytelling",
            "educational": "Write in a clear, educational style suitable for tutorials"
        }
    },
    
    # Subtitle settings
    "subtitles": {
        "font": "Arial",
        "font_size": 24,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 2,
        "position": "bottom"
    },
    
    # Transition settings
    "transitions": {
        "duration": 1.0,
        "types": ["fade", "crossfade", "slide"],
        "min_scene_duration": 3.0
    }
}

def get_config() -> Dict[str, Any]:
    """Get the default configuration."""
    return DEFAULT_CONFIG.copy() 