<<<<<<< HEAD
# PhotStration - Professional Photo to Artistic Illustration Renderer

🎨 **Transform photographs into stunning artistic illustrations** with three distinct rendering styles using advanced computer vision and image processing techniques.

## 🚀 Features

### Three Rendering Engines:

#### 🏠 **ArchRend.py** - Architectural Watercolor Renderings
- **Professional watercolor-style illustrations** similar to hand-drawn architectural renderings
- Enhanced architectural colors with warm, realistic building tones
- Soft watercolor base with flowing color regions
- Preserved architectural details (bricks, windows, trim)
- Realistic soft shadows for depth and dimension
- Enhanced vegetation and landscaping elements

#### 🏗️ **PhotStration.py** - Technical Architectural Drawings
- **Fine line technical drawings** in the style of architectural maps
- Precise edge detection for building features
- Clean technical drawing aesthetic
- Map-like illustration rendering
- Structural detail preservation
- Crosshatching and depth contours

#### 🎭 **PosterRend.py** - Cartoon/Comic Style Illustrations
- **Bold cartoon and comic book style** illustrations
- Posterization with flat color regions
- Bold black outlines for comic book aesthetic  
- Color quantization for simplified palettes
- Perfect for comic strips and animation-style graphics

## 📁 Project Structure

```
PhotStration/
├── ArchRend.py           # Architectural watercolor renderer
├── PhotStration.py       # Technical drawing renderer  
├── PosterRend.py         # Cartoon/comic renderer
├── image_utils.py        # Dynamic image loading utility
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── PhotRend/           # Place input images here
└── .vscode/            # VS Code configuration
    ├── settings.json   # Python development settings
    ├── tasks.json      # Automated tasks
    └── PhotStration.code-workspace  # Workspace file
```

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.13+** (configured: `C:/Users/jaa15/AppData/Local/Programs/Python/Python313/python.exe`)
- **VS Code** with Python extension

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Your Images
- Copy any image (JPEG, PNG, TIFF) to the `PhotRend/` folder
- The system automatically detects and uses the first image found

### 3. Run Rendering Scripts

#### Option A: Command Line
```bash
python ArchRend.py      # Watercolor architectural rendering
python PhotStration.py  # Technical drawing style
python PosterRend.py    # Cartoon/comic style
```

#### Option B: VS Code Tasks (Recommended)
- Press `Ctrl+Shift+P` → "Tasks: Run Task"
- Choose from:
  - 🎨 **Run ArchRend** (Architectural Rendering)
  - 🏗️ **Run PhotStration** (Technical Drawing)  
  - 🎭 **Run PosterRend** (Cartoon Style)
  - 🚀 **Run All Rendering Scripts** (Sequential)

## 📤 Output

All rendered images are saved to:
**`C:\Users\jaa15\OneDrive\Pictures\APS images`**

### Output Formats:
- **TIFF** - Layered file with individual processing layers
- **PNG** - Final rendered image for easy viewing/sharing
- **Individual Layers** - Separate PNG files for each processing step

### File Naming:
- `[originalname]_architectural_rendering.tiff/png` (ArchRend)
- `[originalname]_architectural_map.tiff/png` (PhotStration)  
- `[originalname]_cartoon_comic.tiff/png` (PosterRend)

## 🎛️ Customization

Each renderer supports parameter adjustment:

### ArchRend.py Parameters:
```python
renderer.create_architectural_rendering(
    smoothing_strength=4,    # Watercolor effect intensity (3-7)
    detail_strength=0.5,     # Architectural detail preservation (0.3-0.8)
    enhance_colors=True      # Color enhancement toggle
)
```

### PhotStration.py Parameters:
```python
renderer.create_architectural_illustration(
    line_detail=1.2,         # Fine line detail level (0.5-2.0)
    edge_threshold=40,       # Edge detection sensitivity (20-100)
    detail_threshold=80,     # Structural detail threshold (50-150)
    contour_levels=4,        # Depth contour levels (3-8)
    hatch_density=0.2        # Hatching texture density (0.1-0.5)
)
```

### PosterRend.py Parameters:
```python
renderer.create_cartoon_illustration(
    color_levels=6,          # Posterization levels (4-12)
    line_thickness=7,        # Outline thickness (3-9)
    blur_strength=7          # Smoothing strength (5-15)
)
```

## 🔧 Technical Details

### Core Technologies:
- **OpenCV** - Advanced computer vision and image processing
- **PIL/Pillow** - Image manipulation and enhancement
- **NumPy** - Numerical operations and array processing
- **scikit-image** - Additional image processing algorithms

### Key Algorithms:
- **Bilateral Filtering** - Edge-preserving smoothing
- **K-means Clustering** - Color quantization and posterization  
- **Adaptive Thresholding** - Detail preservation
- **Canny Edge Detection** - Precise edge detection
- **Hough Line Transform** - Structural line detection
- **Morphological Operations** - Image structure analysis

### Processing Pipeline:
1. **Dynamic Image Loading** - Automatic image detection
2. **Color Enhancement** - Saturation, contrast, temperature adjustment
3. **Layer Generation** - Multiple processing layers (base, details, shadows, etc.)
4. **Advanced Blending** - Multiply, overlay, and alpha blending techniques
5. **Output Generation** - Layered TIFF and PNG export

## 🎯 Use Cases

### Architectural Presentations
- Professional project visualizations
- Client presentation materials
- Portfolio illustrations

### Creative Projects  
- Artistic photo transformations
- Comic book/graphic novel illustrations
- Technical documentation graphics
- Educational materials

### Commercial Applications
- Real estate marketing materials
- Architectural visualization services
- Graphic design projects
- Social media content

## 🤝 Development

### VS Code Integration
- **Auto-formatting** with Black
- **Type checking** with Pylance
- **Integrated debugging** with breakpoint support
- **Task automation** for common workflows

### Code Quality
- Type hints for better code maintainability
- Comprehensive error handling
- Modular architecture with reusable components
- Clean separation of concerns

## 📝 License

This project is part of the PhotStration suite developed for professional artistic rendering and image processing.

## 🔄 Version History

- **v1.0** - Initial release with three rendering engines
- **v1.1** - Enhanced VS Code integration and documentation
- **v1.2** - Improved layer blending and output quality

---

**Created by:** PhotStration Project Team  
**Date:** September 2025  
**Python Version:** 3.13+
=======
# Photostration
Create web ready simplified images of photographs
>>>>>>> 28bcbe930907f09446be5c3ac4c797cbf8266de9
