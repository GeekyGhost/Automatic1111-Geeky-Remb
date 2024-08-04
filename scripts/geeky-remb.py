import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import cv2
from tqdm import tqdm

import gradio as gr
from modules import script_callbacks, shared, images as image_utils
import torch

class GeekyRemB:
    def __init__(self):
        self.session = None

    def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if color == "green":
            lower = np.array([40 - color_tolerance, 40, 40])
            upper = np.array([80 + color_tolerance, 255, 255])
        elif color == "blue":
            lower = np.array([90 - color_tolerance, 40, 40])
            upper = np.array([130 + color_tolerance, 255, 255])
        elif color == "red":
            lower = np.array([0, 40, 40])
            upper = np.array([20 + color_tolerance, 255, 255])
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
        return mask

    def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
        if invert_mask:
            mask = 255 - mask

        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)

        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)

        return mask

    def remove_background(self, image, model, alpha_matting, alpha_matting_foreground_threshold, 
                          alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                          color_tolerance, background_mode, background_color, background_image, output_format="RGBA", 
                          invert_mask=False, feather_amount=0, edge_detection=False, 
                          edge_thickness=1, edge_color="#FFFFFF", shadow=False, shadow_blur=5, 
                          shadow_opacity=0.5, color_adjustment=False, brightness=1.0, contrast=1.0, 
                          saturation=1.0, x_position=0, y_position=0, rotation=0, opacity=1.0, 
                          flip_horizontal=False, flip_vertical=False, mask_blur=0, mask_expansion=0,
                          foreground_scale=1.0, foreground_aspect_ratio=None):
        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        edge_color = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        pil_image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        original_image = np.array(pil_image)

        if chroma_key != "none":
            chroma_mask = self.apply_chroma_key(original_image, chroma_key, chroma_threshold, color_tolerance)
            input_mask = chroma_mask
        else:
            input_mask = None

        removed_bg = remove(
            pil_image,
            session=self.session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            post_process_mask=post_process_mask,
        )
        
        rembg_mask = np.array(removed_bg)[:,:,3]

        if input_mask is not None:
            final_mask = cv2.bitwise_and(rembg_mask, input_mask)
        else:
            final_mask = rembg_mask

        final_mask = self.process_mask(final_mask, invert_mask, feather_amount, mask_blur, mask_expansion)

        orig_width, orig_height = pil_image.size
        new_width = int(orig_width * foreground_scale)
        if foreground_aspect_ratio is not None:
            new_height = int(new_width / foreground_aspect_ratio)
        else:
            new_height = int(orig_height * foreground_scale)

        fg_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        fg_mask = Image.fromarray(final_mask).resize((new_width, new_height), Image.LANCZOS)

        if background_mode == "transparent":
            result = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))
        elif background_mode == "color":
            result = Image.new("RGBA", (orig_width, orig_height), bg_color)
        else:  # background_mode == "image"
            if background_image is not None:
                result = background_image.resize((orig_width, orig_height), Image.LANCZOS).convert("RGBA")
            else:
                result = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))

        if flip_horizontal:
            fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
            fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            fg_image = fg_image.transpose(Image.FLIP_TOP_BOTTOM)
            fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

        fg_image = fg_image.rotate(rotation, resample=Image.BICUBIC, expand=True)
        fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

        paste_x = x_position + (orig_width - fg_image.width) // 2
        paste_y = y_position + (orig_height - fg_image.height) // 2

        fg_rgba = fg_image.convert("RGBA")
        fg_with_opacity = Image.new("RGBA", fg_rgba.size, (0, 0, 0, 0))
        for x in range(fg_rgba.width):
            for y in range(fg_rgba.height):
                r, g, b, a = fg_rgba.getpixel((x, y))
                fg_with_opacity.putpixel((x, y), (r, g, b, int(a * opacity)))

        fg_mask_with_opacity = fg_mask.point(lambda p: int(p * opacity))

        result.paste(fg_with_opacity, (paste_x, paste_y), fg_mask_with_opacity)

        if edge_detection:
            edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
            edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
            edge_overlay = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))
            edge_overlay.paste(Image.new("RGB", fg_image.size, edge_color), (paste_x, paste_y), Image.fromarray(edge_mask))
            result = Image.alpha_composite(result, edge_overlay)

        if shadow:
            shadow_mask = fg_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
            shadow_image = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))
            shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (paste_x, paste_y), shadow_mask)
            result = Image.alpha_composite(result, shadow_image.filter(ImageFilter.GaussianBlur(shadow_blur)))

        if color_adjustment:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation)

        if output_format == "RGB":
            result = result.convert("RGB")

        return torch.from_numpy(np.array(result).astype(np.float32) / 255.0).unsqueeze(0), torch.from_numpy(np.array(fg_mask).astype(np.float32) / 255.0).unsqueeze(0)

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as geeky_remb_tab:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input image", type="pil")
                run_button = gr.Button(label="Run GeekyRemB")
                
            with gr.Column():
                result_image = gr.Image(label="Result image", type="pil")

        with gr.Accordion("GeekyRemB Settings", open=False):
            model = gr.Dropdown(label="Model", choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime"], value="u2net")
            output_format = gr.Radio(label="Output Format", choices=["RGBA", "RGB"], value="RGBA")
            alpha_matting = gr.Checkbox(label="Alpha Matting", value=False)
            alpha_matting_foreground_threshold = gr.Slider(label="Alpha Matting Foreground Threshold", minimum=0, maximum=255, value=240, step=1)
            alpha_matting_background_threshold = gr.Slider(label="Alpha Matting Background Threshold", minimum=0, maximum=255, value=10, step=1)
            post_process_mask = gr.Checkbox(label="Post Process Mask", value=False)
            chroma_key = gr.Dropdown(label="Chroma Key", choices=["none", "green", "blue", "red"], value="none")
            chroma_threshold = gr.Slider(label="Chroma Threshold", minimum=0, maximum=255, value=30, step=1)
            color_tolerance = gr.Slider(label="Color Tolerance", minimum=0, maximum=255, value=20, step=1)
            background_mode = gr.Radio(label="Background Mode", choices=["transparent", "color", "image"], value="transparent")
            background_color = gr.ColorPicker(label="Background Color", value="#000000")
            background_image = gr.Image(label="Background Image", type="pil", visible=False)
            invert_mask = gr.Checkbox(label="Invert Mask", value=False)
            feather_amount = gr.Slider(label="Feather Amount", minimum=0, maximum=100, value=0, step=1)
            edge_detection = gr.Checkbox(label="Edge Detection", value=False)
            edge_thickness = gr.Slider(label="Edge Thickness", minimum=1, maximum=10, value=1, step=1)
            edge_color = gr.ColorPicker(label="Edge Color", value="#FFFFFF")
            shadow = gr.Checkbox(label="Shadow", value=False)
            shadow_blur = gr.Slider(label="Shadow Blur", minimum=0, maximum=20, value=5, step=1)
            shadow_opacity = gr.Slider(label="Shadow Opacity", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
            color_adjustment = gr.Checkbox(label="Color Adjustment", value=False)
            brightness = gr.Slider(label="Brightness", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
            contrast = gr.Slider(label="Contrast", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
            saturation = gr.Slider(label="Saturation", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
            x_position = gr.Slider(label="X Position", minimum=-1000, maximum=1000, value=0, step=1)
            y_position = gr.Slider(label="Y Position", minimum=-1000, maximum=1000, value=0, step=1)
            rotation = gr.Slider(label="Rotation", minimum=-360, maximum=360, value=0, step=0.1)
            opacity = gr.Slider(label="Opacity", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
            flip_horizontal = gr.Checkbox(label="Flip Horizontal", value=False)
            flip_vertical = gr.Checkbox(label="Flip Vertical", value=False)
            mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=100, value=0, step=1)
            mask_expansion = gr.Slider(label="Mask Expansion", minimum=-100, maximum=100, value=0, step=1)
            foreground_scale = gr.Slider(label="Foreground Scale", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
            foreground_aspect_ratio = gr.Slider(label="Foreground Aspect Ratio", minimum=0.1, maximum=10.0, value=1.0, step=0.1)

        def update_background_options(mode):
            return {
                background_color: gr.update(visible=mode == "color"),
                background_image: gr.update(visible=mode == "image")
            }

        background_mode.change(
            fn=update_background_options,
            inputs=[background_mode],
            outputs=[background_color, background_image]
        )

        geeky_remb = GeekyRemB()

        def run_geeky_remb(image, model, output_format, alpha_matting, alpha_matting_foreground_threshold, 
                           alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                           color_tolerance, background_mode, background_color, background_image, invert_mask, feather_amount, 
                           edge_detection, edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity, 
                           color_adjustment, brightness, contrast, saturation, x_position, y_position, rotation,
                           opacity, flip_horizontal, flip_vertical, mask_blur, mask_expansion, foreground_scale, 
                           foreground_aspect_ratio):
            if image is None:
                return None
            
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            
            # Ensure background_color is a valid hex color string
            if not background_color.startswith('#'):
                background_color = '#000000'  # Default to black if invalid
            
            result, _ = geeky_remb.remove_background(
                image_tensor, model, alpha_matting, alpha_matting_foreground_threshold, 
                alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                color_tolerance, background_mode, background_color, background_image, output_format, invert_mask, 
                feather_amount, edge_detection, edge_thickness, edge_color, shadow, shadow_blur,
                shadow_opacity, color_adjustment, brightness, contrast, saturation, x_position, 
                y_position, rotation, opacity, flip_horizontal, flip_vertical, mask_blur, 
                mask_expansion, foreground_scale, foreground_aspect_ratio
            )
            return Image.fromarray(np.clip(255. * result[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        run_button.click(
            fn=run_geeky_remb,
            inputs=[input_image, model, output_format, alpha_matting, alpha_matting_foreground_threshold, 
                    alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                    color_tolerance, background_mode, background_color, background_image, invert_mask, feather_amount, 
                    edge_detection, edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity, 
                    color_adjustment, brightness, contrast, saturation, x_position, y_position, rotation,
                    opacity, flip_horizontal, flip_vertical, mask_blur, mask_expansion, foreground_scale, 
                    foreground_aspect_ratio],
            outputs=[result_image]
        )

    return [(geeky_remb_tab, "GeekyRemB", "geeky_remb_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
