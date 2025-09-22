#!/usr/bin/env python3
"""
ArchRend.py - Professional Architectural Rendering Generator
Creates watercolor-style architectural illustrations like hand-drawn architectural renderings

Features:
- Watercolor-style color washes and soft edges
- Precise architectural detail preservation
- Natural color palette enhancement
- Professional architectural illustration quality
- Soft shadows and depth rendering
- Landscape and vegetation enhancement
- Compatible with PhotStration dynamic image loading system

Author: PhotStration Project  
Date: September 2025
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
import sys
from typing import Optional

# Import our dynamic image utility
try:
    from image_utils import get_render_image_path
except ImportError:
    print("Error: image_utils.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

class ArchitecturalRenderer:
    """
    Professional architectural rendering generator for watercolor-style illustrations
    """
    
    def __init__(self, image_path):
        """Initialize with image path"""
        self.image_path = image_path
        self.original_image: Optional[Image.Image] = None
        self.cv_image: Optional[np.ndarray] = None
        self.layers = {}
        
    def load_image(self):
        """Load and prepare the input image"""
        try:
            # Load with PIL for better color handling
            self.original_image = Image.open(self.image_path)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            
            # Also load with OpenCV for advanced processing
            self.cv_image = cv2.imread(self.image_path)  # type: ignore
            if self.cv_image is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            
            print(f"Loaded image: {self.image_path}")
            print(f"Size: {self.original_image.size}")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def enhance_architectural_colors(self):
        """Enhance colors for architectural realism - warm bricks, natural tones"""
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Convert to working format
        working_image = self.original_image.copy()
        
        # Enhance saturation for rich architectural colors
        enhancer = ImageEnhance.Color(working_image)
        working_image = enhancer.enhance(1.3)  # Boost color saturation
        
        # Enhance contrast for architectural definition
        enhancer = ImageEnhance.Contrast(working_image)
        working_image = enhancer.enhance(1.2)  # Slightly more contrast
        
        # Warm up the colors (architectural renderings often have warm tones)
        cv_image = cv2.cvtColor(np.array(working_image), cv2.COLOR_RGB2BGR)
        
        # Apply color temperature adjustment (warmer)
        warming_filter = np.array([[[1.0, 1.05, 1.15]]], dtype=np.float32)
        cv_image = cv_image.astype(np.float32) * warming_filter
        cv_image = np.clip(cv_image, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        enhanced_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        enhanced_colors = Image.fromarray(enhanced_rgb)
        
        self.layers['enhanced_colors'] = enhanced_colors
        return enhanced_colors
    
    def create_watercolor_base(self, smoothing_strength=5):
        """Create watercolor-like base with flat color regions like hand-painted illustrations"""
        if self.cv_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Start with the image
        working_image = self.cv_image.copy()
        
        # Apply initial smoothing to reduce noise
        working_image = cv2.bilateralFilter(working_image, 15, 120, 120)
        
        # Color quantization using a simpler approach for flat watercolor washes
        # Reduce bit depth to create flat color regions
        div = 64  # Higher value = fewer colors, more stylized
        quantized_image = working_image // div * div + div // 2
        quantized_image = np.clip(quantized_image, 0, 255).astype(np.uint8)
        
        # Apply additional smoothing to create soft watercolor regions
        for i in range(2):  # Less aggressive than before
            quantized_image = cv2.bilateralFilter(quantized_image, 9, 50, 50)
        
        # Convert to PIL
        watercolor_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)
        watercolor_base = Image.fromarray(watercolor_rgb)
        
        # Very slight blur for watercolor softness
        watercolor_base = watercolor_base.filter(ImageFilter.GaussianBlur(0.8))
        
        self.layers['watercolor_base'] = watercolor_base
        return watercolor_base
    
    def create_architectural_details(self, detail_strength=0.8):
        """Create strong architectural outlines like hand-drawn watercolor illustrations"""
        if self.cv_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use Canny edge detection for clean architectural lines
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Dilate edges to make them more prominent (like pen strokes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create clean line overlay
        lines = np.zeros_like(self.cv_image)
        lines[edges == 255] = [20, 20, 20]  # Dark gray lines, not pure black
        
        # Convert to PIL
        lines_rgb = cv2.cvtColor(lines, cv2.COLOR_BGR2RGB)
        architectural_details = Image.fromarray(lines_rgb)
        
        self.layers['architectural_details'] = architectural_details
        return architectural_details
    
    def create_soft_shadows(self):
        """Create soft, realistic shadows for architectural depth"""
        if self.cv_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Convert to grayscale for shadow analysis
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        
        # Create shadow mask from darker regions
        _, shadow_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Apply heavy blur for soft shadows
        shadow_mask = cv2.GaussianBlur(shadow_mask, (25, 25), 10)
        
        # Create soft shadow overlay
        shadow_overlay = np.zeros_like(self.cv_image)
        shadow_overlay[:, :, :] = [50, 50, 80]  # Soft blue-gray shadow
        
        # Apply shadow mask
        shadow_mask_3d = np.stack([shadow_mask, shadow_mask, shadow_mask], axis=2)
        shadow_overlay = (shadow_overlay * (shadow_mask_3d / 255.0)).astype(np.uint8)
        
        # Convert to PIL
        shadow_rgb = cv2.cvtColor(shadow_overlay, cv2.COLOR_BGR2RGB)
        soft_shadows = Image.fromarray(shadow_rgb)
        
        self.layers['soft_shadows'] = soft_shadows
        return soft_shadows
    
    def enhance_vegetation(self):
        """Enhance green vegetation areas for natural architectural setting"""
        if self.cv_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Convert to HSV for better green detection
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        
        # Define range for green colors (vegetation)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Create enhanced vegetation
        vegetation_enhanced = self.cv_image.copy()
        
        # Enhance green saturation in vegetation areas
        vegetation_enhanced[green_mask > 0] = cv2.addWeighted(
            vegetation_enhanced[green_mask > 0], 0.7,
            np.full_like(vegetation_enhanced[green_mask > 0], [30, 180, 30]), 0.3, 0
        )
        
        # Convert to PIL
        vegetation_rgb = cv2.cvtColor(vegetation_enhanced, cv2.COLOR_BGR2RGB)
        enhanced_vegetation = Image.fromarray(vegetation_rgb)
        
        self.layers['enhanced_vegetation'] = enhanced_vegetation
        return enhanced_vegetation
    
    def blend_architectural_layers(self):
        """Blend all layers to create stylized watercolor architectural illustration"""
        if not self.layers:
            print("No layers created yet!")
            return None
        
        if self.original_image is None:
            print("No original image loaded!")
            return None
        
        # Start with the quantized watercolor base (this is key for stylization)
        if 'watercolor_base' in self.layers:
            result = self.layers['watercolor_base'].copy()
        elif 'enhanced_colors' in self.layers:
            result = self.layers['enhanced_colors'].copy()
        else:
            result = self.original_image.copy()
        
        # Add strong architectural lines on top (like ink over watercolor)
        if 'architectural_details' in self.layers:
            result = self._multiply_blend(result, self.layers['architectural_details'], 0.7)
        
        # Enhance vegetation areas lightly
        if 'enhanced_vegetation' in self.layers:
            result = Image.blend(result, self.layers['enhanced_vegetation'], 0.2)
        
        # Apply very subtle shadows (watercolors have soft shadows)
        if 'soft_shadows' in self.layers:
            result = self._overlay_blend(result, self.layers['soft_shadows'], 0.15)
        
        # Final artistic enhancement - increase saturation for watercolor vibrancy
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1.2)
        
        # Slight contrast boost for definition
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.1)
        
        self.layers['final_result'] = result
        return result
    
    def _multiply_blend(self, base, overlay, opacity=1.0):
        """Multiply blend mode with opacity control"""
        base_array = np.array(base, dtype=np.float32) / 255.0
        overlay_array = np.array(overlay, dtype=np.float32) / 255.0
        
        result_array = base_array * overlay_array
        result_array = base_array * (1 - opacity) + result_array * opacity
        result_array = (result_array * 255).astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    def _overlay_blend(self, base, overlay, opacity=1.0):
        """Overlay blend mode with opacity control"""
        base_array = np.array(base, dtype=np.float32) / 255.0
        overlay_array = np.array(overlay, dtype=np.float32) / 255.0
        
        # Overlay formula
        mask = base_array < 0.5
        result_array = np.where(mask, 
                               2 * base_array * overlay_array,
                               1 - 2 * (1 - base_array) * (1 - overlay_array))
        
        # Apply opacity
        result_array = base_array * (1 - opacity) + result_array * opacity
        result_array = np.clip(result_array, 0, 1)
        result_array = (result_array * 255).astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    def save_as_web_jpeg(self, output_path, max_size=(1920, 1080), quality=85):
        """Save architectural rendering as web-optimized JPEG"""
        try:
            if 'final_result' not in self.layers:
                print("No final result to save!")
                return False
            
            final_image = self.layers['final_result']
            
            # Resize to web-friendly dimensions while maintaining aspect ratio
            final_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (JPEG doesn't support transparency)
            if final_image.mode != 'RGB':
                final_image = final_image.convert('RGB')
            
            # Save as optimized JPEG
            final_image.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            print(f"Saved web-optimized JPEG to: {output_path}")
            print(f"Final size: {final_image.size}")
            return True
            
        except Exception as e:
            print(f"Error saving JPEG: {e}")
            return False
    
    def save_as_layered_tiff(self, output_path):
        """Save as layered TIFF with individual layers"""
        try:
            if 'final_result' not in self.layers:
                print("No final result to save!")
                return False
            
            # Save the final result
            final_image = self.layers['final_result']
            final_image.save(output_path, 'TIFF', save_all=True, compression='tiff_lzw')
            
            # Also save individual layers
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            layer_dir = os.path.join(base_dir, f"{base_name}_layers")
            os.makedirs(layer_dir, exist_ok=True)
            
            for layer_name, layer_image in self.layers.items():
                layer_path = os.path.join(layer_dir, f"{layer_name}.png")
                layer_image.save(layer_path, 'PNG')
            
            print(f"Saved architectural rendering to: {output_path}")
            print(f"Individual layers saved to: {layer_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving layered TIFF: {e}")
            return False
    
    def create_architectural_rendering(self, 
                                     smoothing_strength=3,
                                     detail_strength=0.8,
                                     enhance_colors=True):
        """
        Complete architectural rendering creation pipeline for hand-drawn watercolor style
        
        Args:
            smoothing_strength (int): Watercolor base processing intensity (2-5, lower = more stylized)
            detail_strength (float): Architectural line strength (0.5-1.0, higher = stronger lines)
            enhance_colors (bool): Enable color enhancement for artistic vibrancy
        """
        print("Starting stylized architectural watercolor rendering...")
        
        # Load the image
        if not self.load_image():
            return False
        
        # Create architectural rendering layers
        if enhance_colors:
            print("Enhancing colors for watercolor vibrancy...")
            self.enhance_architectural_colors()
        
        print("Creating flat watercolor color regions...")
        self.create_watercolor_base(smoothing_strength=smoothing_strength)
        
        print("Adding strong architectural line work...")
        self.create_architectural_details(detail_strength=detail_strength)
        
        print("Creating subtle watercolor shadows...")
        self.create_soft_shadows()
        
        print("Enhancing vegetation areas...")
        self.enhance_vegetation()
        
        # Blend all layers for stylized result
        print("Blending layers for watercolor illustration style...")
        final_result = self.blend_architectural_layers()
        
        if final_result:
            print("Stylized architectural watercolor rendering completed!")
            return True
        else:
            print("Error creating architectural rendering.")
            return False

def main():
    """Main execution function"""
    print("=" * 70)
    print("PhotStration ArchRend - Professional Architectural Rendering Generator")
    print("=" * 70)
    
    try:
        # Get the image path using our dynamic loading system
        image_path = get_render_image_path()
        print(f"Processing image: {os.path.basename(image_path)}")
        
        # Create the architectural renderer
        renderer = ArchitecturalRenderer(image_path)
        
        # Create the architectural rendering with stylized parameters
        success = renderer.create_architectural_rendering(
            smoothing_strength=3,    # Lower value for more stylized effect
            detail_strength=0.8,     # Higher value for stronger architectural lines
            enhance_colors=True      # Enhanced colors for watercolor vibrancy
        )
        
        if success:
            # Generate output filename - save to OneDrive Pictures folder
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = r"C:\Users\jaa15\OneDrive\Pictures\APS images"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as single web-optimized JPEG
            jpg_path = os.path.join(output_dir, f"{base_name}_architectural_rendering.jpg")
            
            if renderer.save_as_web_jpeg(jpg_path):
                print(f"\n[SUCCESS] Architectural watercolor rendering created successfully!")
                print(f"Web-optimized JPEG saved to: {jpg_path}")
            
            print(f"\n[COMPLETE] Architectural rendering processing complete!")
            
        else:
            print("[ERROR] Failed to create architectural rendering.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()