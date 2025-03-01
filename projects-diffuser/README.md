# Stable Diffusion Web UI

A Gradio-based web interface for Stable Diffusion image generation.

## Features

- Support for multiple Stable Diffusion models
- Real-time image generation
- Customizable generation parameters
- Memory-efficient processing
- Support for CUDA, MPS, and CPU devices

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python 01-gradio.py
```

The web interface will be available at `http://localhost:7860`

## Parameters

- **Prompt**: Text description of the image you want to generate
- **Negative Prompt**: Text description of what you don't want in the image
- **Model**: Choose from various Stable Diffusion models
- **Width/Height**: Image dimensions (128-1024 pixels)
- **Seed**: Random seed for reproducible results
- **Steps**: Number of denoising steps (higher = better quality but slower)
- **Guidance Scale**: How closely to follow the prompt (higher = more faithful but less creative)

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM (16GB+ recommended)

## Performance Optimization

The code includes several optimizations:
- Automatic device selection (CUDA > MPS > CPU)
- Memory-efficient processing with attention and VAE slicing
- Optimized model loading
- Type hints for better code maintainability

## Error Handling

The application includes comprehensive error handling and logging:
- Detailed error messages
- Proper exception handling
- Logging system for debugging

## Contributing

Feel free to submit issues and enhancement requests! 