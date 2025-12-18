# GEAhancer: Evolutionary Generation of Creative Images Using a Polygon-Based Genetic Algorithm

This study presents a novel genetic algorithm that generates creative and visually stunning 512×512 artistic enhancements of input images by evolving populations of colored semi-transparent polygons. Heavily inspired by the rigorous, medically-proven evolutionary framework of MedGA [Rundo et al., 2019] (see [repo MedGA](https://github.com/Medga-eth)), this work significantly extends its capabilities by transposing its power from pixel-level optimization to a higher-order domain of geometric primitives while maintaining support for both traditional bimodal medical imaging and modern multimodal color image processing.

**Key Innovations in GEAhancer:**
- 🎨 **Polygon-Based Evolution**: Replaces pixel-level optimization with geometric primitive evolution
- 🖼️ **Creative Image Generation**: Produces artistic 512×512 enhancements
- 🔄 **Dual Processing Modes**: Supports both medical (bimodal) and artistic (general) image processing
- 🏗️ **Modular Architecture**: Inherits MedGA's robust GA framework while extending its capabilities
- 📊 **Advanced Analytics**: Comprehensive fitness tracking and visualization tools

## 🚀 Quick Start

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/geahancer.git
cd geahancer

# Install dependencies
pip install -r requirements.txt

# Run a simple enhancement
python geahancer.py -i input_image.jpg -g 500 --both

# Enhanced Examples

# Artistic enhancement with high population
python geahancer.py -i artwork.png -g 1000 -p 200 --color --format png

# Batch process a folder of medical images
python geahancer.py -f medical_images/ -g 300 -s ranking -e 2

# Generate comparison frames (grayscale + color)
python geahancer.py -i creative_input.jpg -g 800 --both -v

# Launch guided configuration
python3 geahancer.py --interactive
```
<a name="references"></a>References

## Core Methodology
A detailed description of the original MedGA framework, which serves as the foundation for GEAhancer:

- Rundo L., Tangherloni A., Nobile M.S., Militello C., Besozzi D., Mauri G., and Cazzaniga P.: "MedGA: a novel evolutionary method for image enhancement in medical imaging systems", Expert Systems with Applications, 119, 387-399, 2019. doi: 10.1016/j.eswa.2018.11.013

Medical Applications
The original MedGA has been applied as a preprocessing step in various clinical scenarios:

- Rundo L., Tangherloni A., Cazzaniga P., Nobile M.S., Russo G., Gilardi M.C., Vitabile S., Mauri G., Besozzi D., and Militello, C.: "A novel framework for MR image segmentation and quantification by using MedGA", Computer Methods and Programs in Biomedicine, 2019. doi: 10.1016/j.cmpb.2019.04.016

## GEAhancer Extensions
This work extends the above methodologies by introducing polygon-based evolution for creative image generation while maintaining backward compatibility with medical imaging applications.

<a name="required-libraries"></a>Required Libraries
GEAhancer has been developed in Python 3.7+ and tested on Ubuntu Linux, macOS, and Windows systems.

## Core Dependencies
```bash
# Essential libraries
numpy>=1.19.0
matplotlib>=3.3.0
Pillow>=8.0.0
opencv-python>=4.5.0
scipy>=1.5.0

# Enhanced CLI (optional but recommended)
rich>=10.0.0

# Parallel processing (optional)
mpi4py>=3.0.0
```
## Installation Methods

```bash
pip3 install -r requirements.txt
```

## Version Notes
- The sequential version can process single images or folders without MPI
- The parallel version uses Master-Slave paradigm with mpi4py for HPC resources
- `rich` library provides enhanced terminal interface but is optional

<a name="enhanced-input-parameters"></a>Enhanced Input Parameters
GEAhancer extends the original MedGA parameter set with new features for creative image generation:

## Basic Parameters (Required)
```bash
Parameter	Short	Description	Default
--image	-i	Input image file path	Required
--folder	-f	Input folder containing images	Required
--output	-o	Output directory	output
Note: -i and -f are mutually exclusive
```
## Genetic Algorithm Parameters
```bash
Parameter	Short	Description	Default	Range
--population	-p	Population size (chromosomes)	100	10-1000
--generations	-g	Number of evolution generations	100	10-5000
--selection	-s	Selection method: tournament, wheel, ranking	tournament	-
--cross-rate	-c	Crossover rate	0.9	0.0-1.0
--mut-rate	-m	Mutation rate	0.01	0.0-1.0
--pressure	-k	Tournament selection pressure	20	2-100
--elitism	-e	Number of elite chromosomes preserved	1	0-10
Enhanced Processing Modes (New in GEAhancer)
Parameter	Description	Use Case
--color	Process color images in color mode	Artistic images, photographs
--both	Process both grayscale AND color with comparison frame	Medical+Artistic comparison
--format	Output format: jpg, png, tiff, bmp	Format selection
Advanced Features
Parameter	Short	Description	Default
--distributed	-d	Enable MPI parallel processing	False
--cores	-t	Number of MPI cores	4
--verbose	-v	Enable detailed output	False
--interactive		Launch interactive configuration	False
```

<a name="architecture-overview"></a>Architecture Overview

## Core Components
```bash
GEAhancer Architecture
├── 🎨 Image Processing Layer
│   ├── processing class (enhanced)
│   ├── Automatic histogram modality detection
│   ├── Polygon-based representation
│   └── Multi-format support (TIFF, PNG, JPEG, BMP)
├── 🧬 Genetic Algorithm Engine
│   ├── chromosome class (extended)
│   ├── gene class with polygon attributes
│   ├── Dual fitness functions (bimodal/general)
│   └── Enhanced genetic operations
├── ⚡ Processing Modes
│   ├── Sequential processing
│   ├── MPI Parallel processing
│   └── Interactive CLI
├── 📊 Analytics & Visualization
│   ├── Fitness progression tracking
│   ├── Comparison frame generation
│   └── Performance metrics
└── 🖥️ User Interface
    ├── Rich CLI interface
    ├── Parameter configuration
    └── Progress visualization
```

## Key Algorithmic Extensions

1. Polygon Representation: Genes now represent geometric primitives rather than pixel intensities

2. Dual Fitness Evaluation:

3. Bimodal fitness: Preserves medical image characteristics

4. General fitness: Optimizes for artistic quality in color images

5. Adaptive Mutation: Mutation strategy adjusts based on histogram modality detection

6. Comparison Framework: Side-by-side evaluation of different enhancement strategies

<a name="output-structure"></a>Output Structure

## Directory Organization

```bash
output/
├── 📁 image_basename/                    # Per-image results
│   ├── 📄 fitness                        # Fitness values per generation
│   ├── 📄 information                    # GA settings and parameters
│   ├── 📄 matrixBest                     # Best image as TSV (optional)
│   ├── 📄 terms                          # Fitness component values
│   ├── 📄 threshold                      # Optimal thresholds per generation
│   ├── 📊 fitness_analysis.png           # Fitness progression plot
│   └── 📁 images/                        # Enhanced images
│       ├── 🖼️ imageOriginal.png          # Original image
│       ├── 🖼️ image_gen_0.png           # Initial generation
│       ├── 🖼️ image_gen_XXX.png         # Intermediate generations
│       ├── 🖼️ image_final.png           # Final enhanced image
│       └── 🖼️ imageConf_XXX.png         # Comparison images
├── 📁 image_basename_grayscale/          # Grayscale-only results (--both mode)
├── 📁 image_basename_color/              # Color-only results (--both mode)
└── 📁 image_basename_combined/           # Combined results (--both mode)
    └── 🖼️ comparison_frame.png           # Side-by-side comparison
```
## File Descriptions

- fitness: Tracks the best fitness value for each generation (lower is better)

- information: Documents all GA parameters and processing settings

matrixBest: Final enhanced image in tab-separated format (for analysis)

terms: Detailed breakdown of fitness function components

threshold: Optimal threshold evolution across generations

fitness_analysis.png: Visual plot of fitness improvement over time

```