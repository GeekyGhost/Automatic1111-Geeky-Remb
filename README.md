# GeekyRemB: Advanced Background Removal for Automatic1111 Web UI

update 8/18/2024 

Key Updates to the Code:

New Methods in GeekyRemB Class:
a) parse_aspect_ratio(self, aspect_ratio_input):
This method interprets the user's aspect ratio input. It can handle:

Ratio strings (e.g., "16:9")
Decimal values (e.g., "1.77")
Standard terms (e.g., "portrait", "landscape")

b) calculate_new_dimensions(self, orig_width, orig_height, scale, aspect_ratio):
This method calculates the new dimensions based on the original size, scale, and aspect ratio.
Updated remove_background Method:
The method now uses parse_aspect_ratio and calculate_new_dimensions to determine the new image dimensions.
UI Changes:
The foreground_aspect_ratio input is now a gr.Textbox with a more descriptive placeholder, allowing for more flexible input.

How to Use the New Aspect Ratio Feature:

Basic Usage:
In the "Foreground Adjustments" section, you'll find an input field labeled "Aspect Ratio". Here are the ways you can use it:
a) Leave it blank: This will maintain the original aspect ratio of your image.
b) Enter a ratio: You can input ratios like "16:9", "4:3", "1:1", etc. The format is "width:height".
c) Use decimal values: You can enter a single number representing width divided by height, e.g., "1.77" for 16:9.
d) Use standard terms: You can enter "portrait", "landscape", or "square".
Examples:

For a widescreen format, enter "16:9"
For a traditional TV format, enter "4:3"
For a square image, enter "1:1" or "square"
For a vertical video format, you might enter "9:16"
For a slight landscape format, you could enter "1.2" (which is equivalent to 6:5)


Combining with Scale:
The aspect ratio works in conjunction with the "Scale" slider. Here's how they interact:

Scale determines the overall size of the image relative to its original size.
Aspect ratio determines the shape of the image.

For example:

If you set scale to 1.5 and aspect ratio to "16:9", the image will be enlarged by 50% and shaped to a 16:9 ratio.
If you set scale to 0.5 and aspect ratio to "1:1", the image will be shrunk by 50% and made square.


Special Considerations:

If you enter an invalid aspect ratio, the script will default to maintaining the original aspect ratio.
The aspect ratio is applied before other transformations like rotation or flipping.
When using custom terms like "portrait" or "landscape", the script uses preset ratios (3:4 for portrait, 4:3 for landscape).


Tips for Best Results:

For precise control, use numeric ratios (like "16:9") rather than terms like "landscape".
If you're targeting a specific output size, you may need to experiment with both the scale and aspect ratio settings.
Remember that extreme changes in aspect ratio might distort your image, especially if it contains people or recognizable objects.


Interaction with Other Features:

The aspect ratio is applied before positioning (X and Y position) and rotation.
If you're using a background image, you might want to consider its aspect ratio when setting the foreground aspect ratio for a cohesive look.



![download](https://github.com/user-attachments/assets/9a23a8aa-9ab8-4c6a-ae1a-44879b4a696d)

## Overview

GeekyRemB is a powerful extension for Automatic1111 that provides advanced background removal and image/video manipulation capabilities. It is a port of the ComfyUI node, bringing its functionality to the Automatic1111 environment. The extension allows users to remove backgrounds from images and videos, apply various effects, and manipulate foreground elements with precision.

## Key Features

- Background removal for images and videos
- Support for multiple background removal models
- Chroma key functionality
- Foreground manipulation (scaling, rotation, positioning)
- Various image effects (edge detection, shadow, color adjustments)
- Mask processing options
- Custom output dimensions
- Video background support

## How to Use

1. Install the extension by placing the `geeky_remb.py` file in the `scripts` folder of your Automatic1111 installation.
2. Restart Automatic1111 or reload the UI.
3. Navigate to the "GeekyRemB" tab in the Automatic1111 interface.
4. Choose your input type (Image or Video) and upload your content.
5. Adjust the settings as needed (described in detail below).
6. Click the "Run GeekyRemB" button to process your input.

## UI Settings and Their Functions

### Input/Output Settings

- **Input Type**: Choose between "Image" or "Video" as your input source.
- **Foreground Image/Video**: Upload your image or video file to be processed.
- **Output Type**: Select whether you want the output to be an "Image" or "Video".

### Foreground Adjustments

- **Scale**: Adjust the size of the foreground element (0.1 to 5.0).
- **Aspect Ratio**: Modify the aspect ratio of the foreground (0.1 to 10.0).
- **X Position**: Move the foreground horizontally (-1000 to 1000 pixels).
- **Y Position**: Move the foreground vertically (-1000 to 1000 pixels).
- **Rotation**: Rotate the foreground (-360 to 360 degrees).
- **Opacity**: Adjust the transparency of the foreground (0.0 to 1.0).
- **Flip Horizontal**: Mirror the foreground horizontally.
- **Flip Vertical**: Mirror the foreground vertically.

### Background Options

- **Remove Background**: Toggle background removal on/off.
- **Background Mode**: Choose between "transparent", "color", "image", or "video" backgrounds.
- **Background Color**: Select a color when "color" mode is chosen.
- **Background Image**: Upload an image to use as the background.
- **Background Video**: Upload a video to use as the background.

### Advanced Settings

#### Removal Settings

- **Model**: Select the background removal model (e.g., "u2net", "isnet-general-use").
- **Output Format**: Choose between "RGBA" (with transparency) or "RGB".
- **Alpha Matting**: Enable for improved edge detection in complex images.
- **Alpha Matting Foreground Threshold**: Adjust sensitivity for foreground detection (0-255).
- **Alpha Matting Background Threshold**: Adjust sensitivity for background detection (0-255).
- **Post Process Mask**: Apply additional processing to the generated mask.

#### Chroma Key Settings

- **Chroma Key**: Choose a color ("none", "green", "blue", "red") for chroma keying.
- **Chroma Threshold**: Adjust the sensitivity of the chroma key effect (0-255).
- **Color Tolerance**: Set the range of colors to be considered part of the chroma key (0-255).

#### Effects

- **Invert Mask**: Invert the generated mask.
- **Feather Amount**: Soften the edges of the mask (0-100).
- **Edge Detection**: Apply an edge detection effect.
- **Edge Thickness**: Adjust the thickness of detected edges (1-10).
- **Edge Color**: Choose the color for detected edges.
- **Shadow**: Add a shadow effect to the foreground.
- **Shadow Blur**: Adjust the blurriness of the shadow (0-20).
- **Shadow Opacity**: Set the transparency of the shadow (0.0-1.0).
- **Color Adjustment**: Enable color adjustments for the result.
- **Brightness**: Adjust the brightness of the result (0.0-2.0).
- **Contrast**: Adjust the contrast of the result (0.0-2.0).
- **Saturation**: Adjust the color saturation of the result (0.0-2.0).
- **Mask Blur**: Apply blur to the mask (0-100).
- **Mask Expansion**: Expand or contract the mask (-100 to 100).

### Output Settings

- **Image Format**: Choose the output format for images (PNG, JPEG, WEBP).
- **Video Format**: Select the output format for videos (MP4, AVI, MOV).
- **Video Quality**: Adjust the quality of the output video (0-100).
- **Use Custom Dimensions**: Enable to specify custom output dimensions.
- **Custom Width**: Set a custom width for the output.
- **Custom Height**: Set a custom height for the output.

## Technical Implementation for Developers

The GeekyRemB extension is implemented as a Python class (`GeekyRemB`) with several key methods:

1. `__init__`: Initializes the class and prepares for background removal sessions.

2. `apply_chroma_key`: Implements chroma key functionality using OpenCV.

3. `process_mask`: Handles mask processing operations like inversion, feathering, and expansion.

4. `remove_background`: The core method that processes images, removing backgrounds and applying effects.

5. `process_frame`: Processes individual video frames.

6. `process_video`: Handles video processing, including background video integration.

The UI is built using Gradio components, with the `on_ui_tabs` function setting up the interface. Key functions include:

- `update_input_type`, `update_output_type`, `update_background_mode`, `update_custom_dimensions`: Dynamic UI updates based on user selections.
- `process_image` and `process_video`: Wrapper functions for image and video processing.
- `run_geeky_remb`: The main function that orchestrates the entire process based on user inputs.

The extension uses libraries like `rembg` for background removal, `PIL` for image processing, `cv2` for video handling, and `numpy` for array operations.

Developers can extend the functionality by adding new background removal models, implementing additional effects, or enhancing video processing capabilities. The modular structure allows for easy integration of new features.

## Performance Considerations

- Background removal and video processing can be computationally intensive. Consider implementing progress bars or asynchronous processing for better user experience with large files.
- The extension currently processes videos frame-by-frame. For longer videos, consider implementing batch processing or multi-threading for improved performance.
- Memory usage can be high when processing large images or videos. Implement memory management techniques for handling large files.

## Future Enhancements

Potential areas for improvement include:
- Support for more background removal models
- Advanced video editing features (e.g., keyframe animation for foreground properties)
- Integration with other Automatic1111 extensions or workflows
- GPU acceleration for video processing
- Real-time preview for adjustments

This extension provides a powerful set of tools for background removal and image/video manipulation, bringing the capabilities of the ComfyUI node to the Automatic1111 environment.

## Troubleshooting

- If the background removal is imperfect, try adjusting the alpha matting thresholds or using a different model.
- For subjects with similar colors to the background, experiment with the chroma key feature in combination with AI removal.
- If the resulting image looks unnatural, play with the shadow and color adjustment settings to better integrate the subject with the new background.

## Contributing

We welcome contributions to GeekyRemB! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## License

GeekyRemB is released under the MIT License. Feel free to use, modify, and distribute it as you see fit.

## Acknowledgments

GeekyRemB is built upon the excellent [rembg](https://github.com/danielgatis/rembg) library and integrates seamlessly with the Automatic1111 Stable Diffusion Web UI. We're grateful to the developers of these projects for their fantastic work.

---

<img width="1247" alt="Screenshot 2024-08-08 123752" src="https://github.com/user-attachments/assets/2491ce81-09a7-4d8a-9bdc-58d48f82dfaf">


<img width="1235" alt="Screenshot 2024-08-08 124700" src="https://github.com/user-attachments/assets/cd672db7-97fe-4c1b-b8a3-2b50fb152d04">


<img width="1238" alt="Screenshot 2024-08-08 123732" src="https://github.com/user-attachments/assets/f7b91764-041e-4ae6-a46a-a81eaa8692c9">




We hope you enjoy using GeekyRemB to create stunning images with ease! If you find it useful, consider starring the repository and sharing it with your friends and colleagues.
