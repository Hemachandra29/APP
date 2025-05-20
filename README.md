# VidAgent: Autonomous Video Editing and Script Writing System

VidAgent is an AI-powered system that automates video editing, script generation, and content enhancement using state-of-the-art language and vision models.

## Features

- Automated script generation based on video content
- Scene detection and segmentation using Vision Transformers
- Audio transcription and subtitle generation
- Smart video editing with transitions and effects
- FFmpeg-based video rendering pipeline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vidagent.git
cd vidagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Basic video processing:
```python
from vidagent import VidAgent

agent = VidAgent()
agent.process_video("input.mp4", "output.mp4")
```

2. Custom script generation:
```python
agent.generate_script("input.mp4", style="documentary")
```

3. Advanced editing with effects:
```python
agent.process_video("input.mp4", "output.mp4", 
                   add_subtitles=True,
                   add_transitions=True,
                   style="cinematic")
```

## Architecture

- `vidagent/`: Main package directory
  - `core.py`: Core video processing functionality
  - `models/`: AI model implementations
  - `utils/`: Utility functions
  - `config.py`: Configuration settings

## Requirements

- Python 3.8+
- FFmpeg
- CUDA-capable GPU (recommended)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 