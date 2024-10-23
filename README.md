# GeekyRemB: Advanced Background Removal and Image/Video Manipulation Extension for Automatic1111 Web UI

## Overview

**GeekyRemB** is a comprehensive extension for the **Automatic1111 Web UI**, built to bring advanced background removal, image/video manipulation, and blending capabilities into your projects. It offers precise background removal with support for multiple models, chroma keying, foreground adjustments, and advanced effects. Whether working with images or videos, this extension provides everything you need to manipulate visual content efficiently within the **Automatic1111** environment.

---

## Key Features

- **Multi-model Background Removal**: Supports `u2net`, `isnet-general-use`, and other models.
- **Chroma Key Support**: Remove specific colors (green, blue, or red) from backgrounds.
- **Blending Modes**: 10 powerful blend modes for image compositing.
- **Foreground Adjustments**: Scale, rotate, flip, and position elements precisely.
- **Video and Image Support**: Process images and videos seamlessly.
- **Batch and Multi-threaded Processing**: Handle large files efficiently using threading and GPU support.
- **Customizable Output Formats**: Export in PNG, JPEG, MP4, AVI, and more.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GeekyGhost/Automatic1111-Geeky-Remb.git
   ```

2. **Move to the Extensions Folder:**
   Navigate to your **Automatic1111 Web UI** installation directory and move the repository into the `extensions` folder:
   ```bash
   mv Automatic1111-Geeky-Remb ./extensions/
   ```

3. **Restart the Web UI** or use the **Reload** button within the Web UI to register the extension.

4. **Access the Extension**: Open the **GeekyRemB** tab within the Web UI to begin.

---

## Usage Instructions

### Basic Workflow

1. **Select Input Type**: Choose between **Image** or **Video** as your input.
2. **Upload Foreground Content**: Provide the image or video you want to manipulate.
3. **Adjust Foreground Settings**: Modify scaling, aspect ratio, position, rotation, and blending modes.
4. **Apply Background Removal**: Use AI models or chroma key to remove backgrounds.
5. **Choose Output Format**: Select PNG, JPEG, MP4, or other formats.
6. **Click ‘Run GeekyRemB’**: Process the input to generate your final output.

---

## Detailed Settings

### Input/Output Configuration

- **Input Type**:  
  Choose between **Image** or **Video**.

- **Foreground Upload**:  
  Upload your image or video content.

- **Output Type**:  
  Define whether the result will be an **Image** or **Video**.

---

### Foreground Adjustments

- **Scale**:  
  Adjust the size of the foreground between 0.1 to 5.0.

- **Aspect Ratio**:  
  Use ratios like `16:9` or terms like `portrait` or `square`.

- **Rotation & Position**:  
  Rotate the element from -360° to 360° and adjust **X/Y positions** within a -1000 to 1000 range.

- **Flip Options**:  
  Flip the foreground horizontally or vertically.

---

### Background Options

- **Remove Background**:  
  Use AI models (e.g., `u2net`) for automatic background removal.

- **Chroma Key**:  
  Select a chroma key color (green, blue, red) and set tolerance levels.

- **Background Mode**:  
  Options include **transparent**, **solid color**, **image**, or **video** backgrounds.

---

### Advanced Effects

- **Blending Modes**:  
  Choose from 10 blend modes:
  - **Normal**, **Multiply**, **Screen**, **Overlay**, **Soft Light**, **Hard Light**, **Difference**, **Exclusion**, **Color Dodge**, **Color Burn**.

- **Shadow and Edge Detection**:  
  Add shadows and edges with adjustable thickness and blur.

- **Alpha Matting**:  
  Fine-tune mask edges using alpha matting thresholds.

---

### Output Settings

- **Custom Dimensions**:  
  Enable to specify width and height manually.

- **Output Formats**:  
  Export images as PNG, JPEG, WEBP, and videos as MP4, AVI, or MOV.

- **Video Quality**:  
  Set video quality from 0-100 for optimized exports.

---

## Developer Guide

This extension is built with modular, extensible code. Below is an in-depth look at the core classes and methods.

---

### Core Classes and Functions

#### 1. **`GeekyRemB` Class**  
Manages sessions, background removal, threading, and GPU support.

- **`__init__()`**:  
  Initializes the session, checks for CUDA availability, and sets threading parameters.

#### 2. **`remove_background()` Method**  
Handles background removal, blending, chroma keying, and effect applications.  
**Parameters**:
- `model`: AI model to use for removal.
- `alpha_matting`: Enables edge refinement.
- `chroma_key`: Applies chroma key color.
- `blend_mode`: Blend mode for the result.
- `foreground_scale`: Controls the scale of the foreground.

This function processes both image and video inputs, performing transformations and applying custom background settings.

---

#### 3. **`apply_blend_mode()` Method**  
Applies one of 10 blending modes for compositing.

```python
def apply_blend_mode(target, blend, mode="normal", opacity=1.0):
    # Ensures both images have the same dimensions and channels
    result = self.blend_modes[mode](target, blend, opacity)
    return np.clip(result * 255, 0, 255).astype(np.uint8)
```

Supported blend modes:
- **Normal, Multiply, Screen, Overlay, Soft Light, Hard Light, Difference, Exclusion, Color Dodge, Color Burn**

---

#### 4. **`process_video()` Method**  
Processes video frame-by-frame using threading for efficient execution.

```python
def process_video(input_path, output_path, background_video_path, *args):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Process frames and write output
    ...
```

This function supports **CUDA-accelerated encoding** and uses **ffmpeg** for post-processing.

---

#### 5. **`parse_aspect_ratio()` Method**  
Interprets aspect ratios provided by the user. Accepts terms like `portrait`, `landscape`, or numerical ratios.

---

#### 6. **`calculate_new_dimensions()` Method**  
Calculates new dimensions based on scaling and aspect ratio.

```python
def calculate_new_dimensions(self, orig_width, orig_height, scale, aspect_ratio):
    new_width = int(orig_width * scale)
    new_height = int(new_width / aspect_ratio) if aspect_ratio else int(orig_height * scale)
    return new_width, new_height
```

---

## Performance Optimizations

- **GPU Support**: Automatically detects and leverages CUDA for faster video processing.
- **Thread Pooling**: Uses `ThreadPoolExecutor` to batch-process multiple frames.
- **Memory Management**: Dynamically controls batch size to optimize performance.

---

## Troubleshooting Tips

- **Edges are not smooth?**  
  Enable **alpha matting** and adjust thresholds for better results.

- **Chroma key not working correctly?**  
  Increase the **color tolerance** to capture more shades of the key color.

- **Output size not as expected?**  
  Use the **custom dimensions** feature to manually set the desired size.

---

## Future Enhancements

- **Real-time Preview**: Provide live feedback for adjustments.
- **Animation Support**: Add keyframe-based animation for dynamic videos.
- **New Models**: Incorporate additional models for niche use cases.

---

## Acknowledgments

- **rembg Library**: This extension is built on top of the [rembg](https://github.com/danielgatis/rembg) library.
- **Automatic1111 Community**: Thanks to the community for continuous inspiration and support.

---

## Contributing

We welcome contributions!  
Feel free to submit pull requests or open issues with ideas, improvements, or bug reports.

---

With **GeekyRemB**, unlock new possibilities in creative projects. Whether you need to fine-tune images or apply advanced effects to videos, this extension empowers your workflow with precision and control.

---

Happy creating! 🎨
