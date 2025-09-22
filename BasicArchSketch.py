#!/usr/bin/env python3
"""
BasicArchSketch.py - Minimalistic Architectural Line Drawing Generator
Creates simple, clean line drawings with strong structural lines and minimal detail

Features:
- Strong, bold architectural lines
- Minimal detail for clean aesthetic
- Simplified geometry and forms
- Clean white background
- Focus on essential structural elements
- Compatible with PhotStration dynamic image loading system

Author: PhotStration Project  
Date: September 2025
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import os
import sys
from typing import Optional

# Import our dynamic image utility
try:
    from image_utils import get_render_image_path
except ImportError:
    print("Error: image_utils.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

class BasicArchitecturalSketch:
    """
    Minimalistic architectural sketch generator for clean line drawings
    """
    
    def __init__(self, image_path):
        """Initialize with image path"""
        self.image_path = image_path
        self.original_image: Optional[Image.Image] = None
        self.cv_image: Optional[np.ndarray] = None
        
    def load_image(self):
        """Load and prepare the input image"""
        try:
            # Load with PIL for better color handling
            self.original_image = Image.open(self.image_path)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            
            # Convert to OpenCV format for processing
            self.cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            
            print(f"Loaded image: {self.cv_image.shape[1]}x{self.cv_image.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def create_clean_structural_sketch(self):
        """Create a clean sketch focusing only on major building structure"""
        if self.cv_image is None:
            raise ValueError("No image loaded")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        
        # Light smoothing to reduce texture but preserve structure
        smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Detect edges with moderate settings
        edges = cv2.Canny(smooth, 50, 100, apertureSize=3)
        
        # Remove small noise components aggressively
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        # Create clean image keeping only significant structural elements
        clean_edges = np.zeros_like(edges)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Keep only substantial structural elements (remove small ornate details)
            if area > 100:  # Minimum size for structural elements
                clean_edges[labels == i] = 255
        
        # Strengthen major lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_edges = cv2.dilate(clean_edges, kernel, iterations=2)  # Thicker dilation
        
        # Close small gaps in structural lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # Larger kernel
        clean_edges = cv2.morphologyEx(clean_edges, cv2.MORPH_CLOSE, kernel)
        
        # Final cleanup - remove very small artifacts again
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_edges, connectivity=8)
        
        final_clean = np.zeros_like(clean_edges)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Be even more aggressive about removing small details
            if area > 150:  # Only keep significant structural lines
                final_clean[labels == i] = 255
        
        return final_clean
    
    def enhance_structural_lines(self, structural_lines, image_shape):
        """Create enhanced structural line image"""
        enhanced = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Draw all structural lines
        for x1, y1, x2, y2, length in structural_lines:
            # Thicker lines for more prominent structures
            thickness = 2 if length > 80 else 1
            cv2.line(enhanced, (x1, y1), (x2, y2), 255, thickness)
        
        return enhanced
    
    def create_clean_sketch(self):
        """Create the final clean architectural sketch"""
        # Get clean structural sketch (removes ornate details)
        clean_structural = self.create_clean_structural_sketch()
        
        # Convert to PIL Image for final processing
        sketch_pil = Image.fromarray(255 - clean_structural)  # Invert so lines are black on white
        sketch_pil = sketch_pil.convert('RGB')
        
        # Enhance contrast for bold lines
        enhancer = ImageEnhance.Contrast(sketch_pil)
        sketch_pil = enhancer.enhance(1.5)
        
        return sketch_pil
    
    def save_as_web_jpeg(self, output_image, base_name="basic_sketch"):
        """Save as web-optimized JPEG"""
        output_dir = os.path.expanduser("~/OneDrive/Pictures/APS images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Resize for web optimization (max 1920x1080)
        max_size = (1920, 1080)
        output_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save as JPEG with optimization
        output_path = os.path.join(output_dir, f"{base_name}.jpg")
        output_image.save(output_path, 'JPEG', quality=85, optimize=True)
        
        print(f"Saved web-optimized JPEG to: {output_path}")
        print(f"Final size: {output_image.size}")
        return output_path

def main():
    """Main execution function"""
    print("BasicArchSketch - Minimalistic Architectural Line Drawing Generator")
    print("=" * 60)
    
    try:
        # Get the input image path dynamically
        image_path = get_render_image_path()
        
        # Create sketch generator
        sketch_gen = BasicArchitecturalSketch(image_path)
        
        # Load the image
        if not sketch_gen.load_image():
            print("Failed to load image. Exiting.")
            return
        
        print("Creating minimalistic architectural sketch...")
        
        # Generate the basic sketch
        final_sketch = sketch_gen.create_clean_sketch()
        
        # Save as web-optimized JPEG
        sketch_gen.save_as_web_jpeg(final_sketch, "basic_architectural_sketch")
        
        print("\n✅ Basic architectural sketch generation complete!")
        print("Check the APS images folder for your minimalistic line drawing.")
        
    except Exception as e:
        print(f"\n❌ Error during sketch generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()