"""
MedGA: Medical Image Enhancement using Genetic Algorithms

A novel evolutionary method based on Genetic Algorithms for the enhancement of
bimodal biomedical images. MedGA tackles the complexity of the enhancement problem
by exploiting Genetic Algorithms (GAs) to improve the appearance and visual quality
of images characterized by a bimodal gray level intensity histogram.

References:
    * L. Rundo, A. Tangherloni et al.: MedGA: a novel evolutionary method for image
      enhancement in medical imaging systems, Expert Systems with Applications, 119,
      387-399, 2019. doi: 10.1016/j.eswa.2018.11.013

    * L. Rundo, A. Tangherloni et al.: A novel framework for MR image segmentation
      and quantification by using MedGA, Computer Methods and Programs in Biomedicine,
      2019. doi: 10.1016/j.cmpb.2019.04.016

Copyright (C) 2019 - Andrea Tangherloni & Leonardo Rundo
Distributed under the terms of the GNU General Public License (GPL)
This file is part of MedGA.

MedGA is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.0 as published by
the Free Software Foundation.

MedGA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
"""

import getopt
import sys
import glob
import os
import time
import subprocess
import random as rnd
import numpy as np
import math
from datetime import datetime
import argparse
from typing import Optional, List
import cv2
from PIL import Image, ImageDraw, ImageFont
import fnmatch
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to import rich for beautiful CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.text import Text
    from rich import box
    from rich.markdown import Markdown
    from rich.layout import Layout
    from rich.live import Live
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("✨ Install 'rich' for enhanced CLI: pip install rich")

# Try to import MPI, but make it optional
MPI_AVAILABLE = False
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except (ImportError, RuntimeError):
    MPI_AVAILABLE = False


# ============================================================================
# IMAGE PROCESSING CLASS
# ============================================================================

class processing(object):
    """
    Image processing class for loading and preprocessing medical images.

    This class handles image loading, format conversion, histogram calculation,
    and optimal threshold computation for both grayscale and color images.

    Attributes:
        None (all methods are static or instance-based)
    """

    def loadImage(self, target_img_name: str, pathOut: str, process_color: bool = False) -> tuple:
        """
        Load and preprocess an input image for genetic algorithm processing.

        This method loads an image, converts it to the appropriate format (grayscale
        or color), calculates histograms, and determines optimal threshold values.
        It returns all necessary data for the genetic algorithm to process.

        Args:
            target_img_name (str): Path to the input image file.
            pathOut (str): Output directory path where original image will be saved.
            process_color (bool): If True, process as color image; if False, convert
                                 to grayscale. Default is False.

        Returns:
            tuple: A 7-element tuple containing:
                - original_image (np.ndarray): Original image as numpy array
                - numberGrayLevel (int): Number of non-zero histogram bins
                - hist (np.ndarray): Histogram of the image (256 bins)
                - posNoZeros (list): List of non-zero histogram positions
                - maxValueGray (int): Maximum gray level value in the image
                - T_k (int): Optimal threshold value computed using IOTS
                - processed_image (np.ndarray): Processed image (grayscale or color)

        Raises:
            Exception: If image loading fails or image format is unsupported.

        Note:
            - For color processing, the image is kept in color but histogram is
              calculated from luminance channel
            - For grayscale processing, color images are converted using standard
              RGB to grayscale conversion (0.299*R + 0.587*G + 0.114*B)
            - The first histogram bin (background) is always set to zero
        """
        try:
            # Load image using PIL
            image_pil = Image.open(target_img_name)

            # Convert to numpy array for processing - store as original image
            original_image = np.array(image_pil)

            # Save original image
            Image.fromarray(original_image).save(pathOut + os.sep + 'imageOriginal.png')

            # Determine if we should process as color
            is_color = process_color and len(original_image.shape) == 3

            if is_color:
                print("Processing color image with", original_image.shape[2], "channels")
                # For color processing, we'll use the original image
                processed_image = original_image.copy()

                # Calculate histogram from luminance for fitness calculation
                # Convert to grayscale using luminance formula for histogram analysis
                if original_image.shape[2] == 3:  # RGB
                    grayscale_for_hist = np.dot(original_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                elif original_image.shape[2] == 4:  # RGBA
                    rgb_image = original_image[...,:3]
                    grayscale_for_hist = np.dot(rgb_image, [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                print("Processing grayscale image")
                # For grayscale processing
                if len(original_image.shape) == 3:  # Color image but process as grayscale
                    if original_image.shape[2] == 3:  # RGB
                        processed_image = np.dot(original_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                    elif original_image.shape[2] == 4:  # RGBA
                        rgb_image = original_image[...,:3]
                        processed_image = np.dot(rgb_image, [0.299, 0.587, 0.114]).astype(np.uint8)
                else:  # Already grayscale
                    processed_image = original_image.copy()

                grayscale_for_hist = processed_image

            # Ensure processed image is 8-bit
            if processed_image.dtype != np.uint8:
                processed_image = self.__normalize_image(processed_image)

            # Use grayscale version for histogram analysis (for fitness calculation)
            maxValue = np.max(grayscale_for_hist)

            # Use fixed 256 bins for histogram
            hist, _ = np.histogram(grayscale_for_hist, bins=256, range=(0, 255))

            # Set first bin to zero (background)
            hist[0] = 0

            # Get non-zero positions
            posNoZeros = list(np.nonzero(hist)[0])

            # If no non-zero values, create a simple histogram
            if len(posNoZeros) == 0:
                hist[100] = 1000
                hist[200] = 1000
                posNoZeros = [100, 200]

            T_k = self.__optimalThreshold(hist, 0.001, 100)
            
            # Detect histogram modality (bimodal for medical images, general for normal images)
            histogram_modality = self.__detect_histogram_modality(hist)

            # Return 8 values (added modality):
            return original_image, len(posNoZeros), hist, posNoZeros, maxValue, T_k, processed_image, histogram_modality

        except Exception as e:
            print(f"Error loading image {target_img_name}: {str(e)}")
            raise

    def __normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 8-bit range (0-255).

        Handles various input formats including floating point (0-1 or 0-255),
        16-bit images, and other numeric types.

        Args:
            image (np.ndarray): Input image array of any numeric type.

        Returns:
            np.ndarray: Normalized 8-bit unsigned integer image array.
        """
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                return (image * 255).astype(np.uint8)
            else:
                return (image / (image.max() / 255.0)).astype(np.uint8)
        elif image.dtype == np.uint16:
            return (image / 256).astype(np.uint8)
        else:
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                return ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                return image.astype(np.uint8)

    def __optimalThreshold(self, hist_vals: np.ndarray, delta_T: float, max_it: int) -> int:
        """
        Efficient Iterative Optimal Threshold Selection (IOTS) algorithm.

        Computes the optimal threshold value that separates a bimodal histogram
        into two distinct distributions. This is a key component of the MedGA
        fitness function.

        Args:
            hist_vals (np.ndarray): Histogram values (256 bins for 8-bit images).
            delta_T (float): Convergence threshold - algorithm stops when threshold
                           change is less than this value.
            max_it (int): Maximum number of iterations to prevent infinite loops.

        Returns:
            int: Optimal threshold value (T_k) that best separates the histogram.

        Algorithm:
            1. Calculate initial threshold as histogram mean
            2. Iteratively refine threshold by:
               - Splitting histogram at current threshold
               - Calculating means of two halves (H1_mean, H2_mean)
               - Setting new threshold as average of means
            3. Stop when change is less than delta_T or max iterations reached
        """
        opt_T = 1
        h_dim = len(hist_vals)
        total_pixel_number = np.sum(hist_vals)

        if total_pixel_number == 0:
            return opt_T

        weighted_hist_sum = 0
        for i in range(0, h_dim):
            weighted_hist_sum = weighted_hist_sum + hist_vals[i] * i

        hist_mean = weighted_hist_sum / (total_pixel_number * 1.0)

        if hist_mean == 0:
            return opt_T

        T_k = 0
        T_k1 = int(math.floor(hist_mean))
        counter = 1

        while counter < max_it:
            if abs(T_k1 - T_k) <= delta_T:
                break

            T_k = T_k1
            H1_pixel_number = 0
            H2_pixel_number = 0
            weighted_H1_sum = 0
            weighted_H2_sum = 0

            for i in range(0, T_k):
                H1_pixel_number += hist_vals[i]
                weighted_H1_sum += hist_vals[i] * i

            for i in range(T_k, h_dim):
                H2_pixel_number += hist_vals[i]
                weighted_H2_sum += hist_vals[i] * i

            H1_mean = weighted_H1_sum / (H1_pixel_number * 1.0) if H1_pixel_number > 0 else 0
            H2_mean = weighted_H2_sum / (H2_pixel_number * 1.0) if H2_pixel_number > 0 else 0

            if H1_mean + H2_mean > 0:
                T_k1 = int(math.floor((H1_mean + H2_mean) / 2.0))
            else:
                break

            counter = counter + 1

        return T_k

    def __detect_histogram_modality(self, hist_vals: np.ndarray) -> str:
        """
        Detect if histogram is bimodal (suitable for medical images) or general.
        
        Uses peak detection to identify the number of modes in the histogram.
        
        Args:
            hist_vals (np.ndarray): Histogram values (256 bins).
            
        Returns:
            str: 'bimodal' if histogram has two distinct peaks, 'general' otherwise.
        """
        # Smooth histogram to reduce noise
        try:
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter1d(hist_vals.astype(float), sigma=2.0)
        except ImportError:
            # Fallback if scipy not available - simple moving average
            smoothed = hist_vals.astype(float)
            # Simple 3-point moving average
            for i in range(1, len(smoothed) - 1):
                smoothed[i] = (hist_vals[i-1] + hist_vals[i] + hist_vals[i+1]) / 3.0
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                # Only consider significant peaks (above 1% of max)
                if smoothed[i] > np.max(smoothed) * 0.01:
                    peaks.append(i)
        
        # Count distinct peaks (peaks separated by at least 30 intensity levels)
        distinct_peaks = []
        for peak in peaks:
            if len(distinct_peaks) == 0:
                distinct_peaks.append(peak)
            else:
                # Check if this peak is far enough from existing peaks
                if all(abs(peak - p) > 30 for p in distinct_peaks):
                    distinct_peaks.append(peak)
        
        # If we have exactly 2 distinct peaks, it's bimodal
        if len(distinct_peaks) == 2:
            return 'bimodal'
        else:
            return 'general'

    def __calculate_histogram_entropy(self, hist_vals: np.ndarray) -> float:
        """
        Calculate entropy of histogram (measure of information content).
        
        Higher entropy indicates better contrast and information distribution.
        
        Args:
            hist_vals (np.ndarray): Histogram values.
            
        Returns:
            float: Entropy value (bits).
        """
        # Normalize histogram to probabilities
        total = np.sum(hist_vals)
        if total == 0:
            return 0.0
        
        probs = hist_vals.astype(float) / total
        probs = probs[probs > 0]  # Remove zeros for log calculation
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def __calculate_histogram_contrast(self, hist_vals: np.ndarray) -> float:
        """
        Calculate contrast measure based on histogram spread.
        
        Args:
            hist_vals (np.ndarray): Histogram values.
            
        Returns:
            float: Contrast measure (higher is better).
        """
        total = np.sum(hist_vals)
        if total == 0:
            return 0.0
        
        # Calculate weighted mean
        weighted_sum = np.sum(hist_vals * np.arange(len(hist_vals)))
        mean = weighted_sum / total
        
        # Calculate standard deviation
        variance = np.sum(hist_vals * (np.arange(len(hist_vals)) - mean) ** 2) / total
        std_dev = np.sqrt(variance)
        
        # Normalize by max possible std (for 0-255 range, max std ~73)
        return std_dev / 73.0


# ============================================================================
# GENETIC ALGORITHM CLASSES
# ============================================================================

class gene(object):
    """
    Represents a single gene in a chromosome.

    A gene corresponds to a mapping from one gray level intensity to another.
    In the context of image enhancement, genes define how pixel intensities
    should be transformed to improve image quality.

    Attributes:
        position (int): The gray level value this gene maps to (0-255).
    """

    def __init__(self, pos: int = 0):
        """
        Initialize a gene with a position value.

        Args:
            pos (int): Initial gray level position (default: 0).
        """
        self.position = pos

    def mutatePosition(self, minGrayLevel: int, maxGrayLevel: int, opt_T: int, 
                       histogram_modality: str = 'bimodal'):
        """
        Mutate the gene's position based on optimal threshold.

        For bimodal images: The mutation is constrained by the optimal threshold to
        preserve bimodal structure. For general images: Mutation is less constrained
        to allow more exploration of the intensity space.

        Args:
            minGrayLevel (int): Minimum allowed gray level (typically 1).
            maxGrayLevel (int): Maximum allowed gray level (typically 255).
            opt_T (int): Optimal threshold value that separates the two modes.
            histogram_modality (str): 'bimodal' for constrained mutation, 'general' for flexible mutation.
        """
        # Ensure valid bounds
        if maxGrayLevel <= minGrayLevel:
            # Invalid range, use default
            self.position = minGrayLevel
            return
        
        # Ensure opt_T is within valid bounds
        opt_T = max(minGrayLevel, min(opt_T, maxGrayLevel))
        
        if histogram_modality == 'bimodal':
            # Constrained mutation for bimodal images
            if self.position <= opt_T:
                # Ensure valid range
                if opt_T > minGrayLevel:
                    value = rnd.randint(minGrayLevel, opt_T)
                else:
                    # If opt_T equals minGrayLevel, use full range
                    value = rnd.randint(minGrayLevel, maxGrayLevel)
            else:
                # Ensure valid range
                if maxGrayLevel > opt_T:
                    value = rnd.randint(opt_T, maxGrayLevel)
                else:
                    # If opt_T equals maxGrayLevel, use full range
                    value = rnd.randint(minGrayLevel, maxGrayLevel)
        else:
            # More flexible mutation for general images
            # Allow mutation anywhere in range, but slightly bias towards current region
            if self.position <= opt_T:
                # 70% chance to stay in lower region, 30% chance to explore upper region
                if rnd.uniform(0, 1) < 0.7:
                    # Ensure valid range for lower region
                    if opt_T > minGrayLevel:
                        value = rnd.randint(minGrayLevel, opt_T)
                    else:
                        value = rnd.randint(minGrayLevel, maxGrayLevel)
                else:
                    # Ensure valid range for upper region
                    if maxGrayLevel > opt_T:
                        value = rnd.randint(opt_T, maxGrayLevel)
                    else:
                        value = rnd.randint(minGrayLevel, maxGrayLevel)
            else:
                # 70% chance to stay in upper region, 30% chance to explore lower region
                if rnd.uniform(0, 1) < 0.7:
                    # Ensure valid range for upper region
                    if maxGrayLevel > opt_T:
                        value = rnd.randint(opt_T, maxGrayLevel)
                    else:
                        value = rnd.randint(minGrayLevel, maxGrayLevel)
                else:
                    # Ensure valid range for lower region
                    if opt_T > minGrayLevel:
                        value = rnd.randint(minGrayLevel, opt_T)
                    else:
                        value = rnd.randint(minGrayLevel, maxGrayLevel)

        # Final bounds check
        if value > maxGrayLevel:
            value = maxGrayLevel
        elif value < minGrayLevel:
            value = minGrayLevel

        self.position = value


class chromosome(object):
    """
    Represents a chromosome (individual) in the genetic algorithm population.

    A chromosome contains a set of genes that define a complete intensity mapping
    function for image enhancement. Each chromosome represents a potential solution
    to the image enhancement problem.

    Attributes:
        genes (list): List of gene objects defining the intensity mapping.
        crossPoint (int): Crossover point used during genetic operations.
        __fitness (float): Fitness value of this chromosome (lower is better).
        __opt_T (int): Optimal threshold for this chromosome's histogram.
        __hist (np.ndarray): Histogram of the enhanced image.
        __matrix (np.ndarray): Enhanced image matrix.
        __term1 (float): First term of fitness function.
        __term2 (float): Second term of fitness function.
        __term3 (float): Third term of fitness function.
        is_color (bool): Whether this chromosome processes color images.
    """

    def __init__(self, targetHist: np.ndarray, noZeroPosHist: list, numberOfGenes: int,
                 minGrayLevel: int, maxGrayLevel: int, mut_rate: float, T_k: int,
                 parent_1=None, parent_2=None, cross_point: Optional[int] = None,
                 is_color: bool = False, histogram_modality: str = 'bimodal'):
        """
        Initialize a chromosome either from scratch or from parents.

        Args:
            targetHist (np.ndarray): Target histogram of the original image.
            noZeroPosHist (list): List of non-zero histogram positions.
            numberOfGenes (int): Number of genes in the chromosome.
            minGrayLevel (int): Minimum gray level value (typically 1).
            maxGrayLevel (int): Maximum gray level value (typically 255).
            mut_rate (float): Mutation rate for genetic operations.
            T_k (int): Initial optimal threshold value.
            parent_1 (chromosome, optional): First parent for crossover.
            parent_2 (chromosome, optional): Second parent for crossover.
            cross_point (int, optional): Crossover point for uniform crossover.
            is_color (bool): Whether processing color images (default: False).
            histogram_modality (str): 'bimodal' for medical images, 'general' for normal images (default: 'bimodal').
        """
        self.genes = []
        self.crossPoint = None
        self.__fitness = None
        self.__opt_T = 0
        self.__hist = None
        self.__matrix = None
        self.__term1 = None
        self.__term2 = None
        self.__term3 = None
        self.is_color = is_color
        self.histogram_modality = histogram_modality

        if parent_1 and parent_2:
            op = geneticOperation()
            self.genes, self.crossPoint = op.crossoverUniform(parent_1, parent_2, cross_point)
            op.mutate(self.genes, minGrayLevel, maxGrayLevel, self.__opt_T, mut_rate, self.histogram_modality)
        else:
            dist = self.__generateUniformDistribution(noZeroPosHist, minGrayLevel, maxGrayLevel)
            for i in range(0, numberOfGenes):
                self.genes.append(gene(dist[i]))

        self.genes.sort(key=lambda x: x.position)
        self.__fitness, self.__opt_T, self.__hist = self.calculateFitness(
            targetHist, noZeroPosHist, maxGrayLevel, minGrayLevel)

    def __generateUniformDistribution(self, noZeroPosHist: list, minGrayLevel: int,
                                      maxGrayLevel: int) -> list:
        """
        Generate a uniform random distribution of gene positions.

        Creates initial random mapping values for a new chromosome.

        Args:
            noZeroPosHist (list): List of non-zero histogram positions.
            minGrayLevel (int): Minimum gray level.
            maxGrayLevel (int): Maximum gray level.

        Returns:
            list: Sorted list of random gray level values.
        """
        dist1 = np.random.uniform(minGrayLevel, maxGrayLevel, len(noZeroPosHist))
        dist = [int(round(j)) for j in dist1]
        return sorted(dist)

    def __calculateVariances(self, hist: np.ndarray, opt_T: int, mu1: float,
                            mu2: float) -> tuple:
        """
        Calculate variances and half-widths for the two histogram modes.

        Computes standard deviations and half-widths for the two distributions
        separated by the optimal threshold. These values are used in the fitness
        function to evaluate how well the bimodal structure is preserved.

        Args:
            hist (np.ndarray): Histogram array.
            opt_T (int): Optimal threshold value.
            mu1 (float): Mean of first distribution (below threshold).
            mu2 (float): Mean of second distribution (above threshold).

        Returns:
            tuple: (std1, std2, halfWidth1, halfWidth2) where:
                - std1: Standard deviation of first distribution
                - std2: Standard deviation of second distribution
                - halfWidth1: Half-width of first distribution
                - halfWidth2: Half-width of second distribution
        """
        noZeroPosHist = np.nonzero(hist)[0]

        val1 = noZeroPosHist[0]
        val2 = noZeroPosHist[0]
        pos = 0
        for i in range(1, len(noZeroPosHist)):
            if noZeroPosHist[i] <= opt_T:
                val2 = noZeroPosHist[i]
                pos = i
            else:
                break

        val3 = noZeroPosHist[pos+1]
        val4 = noZeroPosHist[-1]

        halfWidth1 = (val2 - val1) / 2.0
        halfWidth2 = (val4 - val3) / 2.0

        countOcc1 = 0
        acc1 = 0
        countOcc2 = 0
        acc2 = 0
        for i in range(len(noZeroPosHist)):
            greyLev = noZeroPosHist[i]
            if greyLev <= opt_T:
                countOcc1 += hist[greyLev]
                acc1 += (hist[greyLev] * (greyLev - mu1)**2)
            else:
                countOcc2 += hist[greyLev]
                acc2 += (hist[greyLev] * (greyLev - mu2)**2)

        std1 = math.sqrt((acc1 / (float(countOcc1)))) if countOcc1 > 0 else 0
        std2 = math.sqrt((acc2 / (float(countOcc2)))) if countOcc2 > 0 else 0

        return std1, std2, halfWidth1, halfWidth2

    def __optimalThreshold(self, hist_vals: np.ndarray, delta_T: float,
                          max_it: int) -> tuple:
        """
        Compute optimal threshold using iterative method.

        Similar to the processing class method but returns additional values
        needed for fitness calculation.

        Args:
            hist_vals (np.ndarray): Histogram values.
            delta_T (float): Convergence threshold.
            max_it (int): Maximum iterations.

        Returns:
            tuple: (T_k, H1_mean, H2_mean) where:
                - T_k: Optimal threshold
                - H1_mean: Mean of first distribution
                - H2_mean: Mean of second distribution
        """
        opt_T = 1
        h_dim = len(hist_vals)
        total_pixel_number = np.sum(hist_vals)
        weighted_hist_sum = 0
        for i in range(0, h_dim):
            weighted_hist_sum = weighted_hist_sum + hist_vals[i] * (i-1)

        hist_mean = weighted_hist_sum / (total_pixel_number*1.0)

        if hist_mean == 0:
            return opt_T, 0, 0

        T_k = 0
        T_k1 = int(math.floor(hist_mean))
        counter = 1

        while counter < max_it:
            if (T_k1 - T_k) <= delta_T:
                break

            T_k = T_k1
            H1_pixel_number = 0
            H2_pixel_number = 0
            for i in range(0, T_k):
                H1_pixel_number += hist_vals[i]
            for i in range((T_k+1), h_dim):
                H2_pixel_number += hist_vals[i]

            weighted_H1_sum = 0
            for i in range(0, T_k):
                weighted_H1_sum = weighted_H1_sum + hist_vals[i] * (i-1)

            weighted_H2_sum = 0
            for i in range((T_k+1), h_dim):
                weighted_H2_sum = weighted_H2_sum + hist_vals[i] * (i-1)

            H1_mean = weighted_H1_sum / (H1_pixel_number*1.0) if H1_pixel_number > 0 else 0
            H2_mean = weighted_H2_sum / (H2_pixel_number*1.0) if H2_pixel_number > 0 else 0

            T_k1 = int(math.floor((H1_mean + H2_mean) / 2.0))
            counter = counter + 1

        return T_k, H1_mean, H2_mean

    def saveCurrentImage(self, targetHist: np.ndarray, noZeroPosHist: list,
                        targetMatrix: np.ndarray, originalImage: np.ndarray,
                        f_name: str, f_nameConf: str):
        """
        Save the enhanced image and comparison plot.

        Applies the chromosome's gene mapping to create an enhanced image and
        saves both the enhanced image and a side-by-side comparison with the
        original image.

        Args:
            targetHist (np.ndarray): Target histogram.
            noZeroPosHist (list): Non-zero histogram positions.
            targetMatrix (np.ndarray): Original image matrix to enhance.
            originalImage (np.ndarray): Original image for comparison.
            f_name (str): Filename for saving enhanced image.
            f_nameConf (str): Filename for saving comparison plot.
        """
        self.__matrix = deepcopy(targetMatrix)
        newNoZeros = []
        for i in range(0, len(self.genes)):
            newNoZeros.append(self.genes[i].position)

        # Apply enhancement based on image type
        if len(targetMatrix.shape) == 3 and self.is_color:  # Color image processing
            enhanced_image = np.zeros_like(targetMatrix)
            for channel in range(targetMatrix.shape[2]):
                channel_data = targetMatrix[:, :, channel]
                enhanced_channel = channel_data.copy()

                # Create histogram for this channel to find intensity mapping
                channel_hist, _ = np.histogram(channel_data, bins=256, range=(0, 255))
                channel_hist[0] = 0
                channel_posNoZeros = list(np.nonzero(channel_hist)[0])

                # Apply enhancement to this channel
                for i in range(0, len(channel_posNoZeros)):
                    if i < len(newNoZeros):  # Ensure we don't go out of bounds
                        ind = channel_posNoZeros[i]
                        pos = np.where(channel_data == ind)
                        pos_x = pos[0]
                        pos_y = pos[1]
                        for j in range(0, len(pos_x)):
                            enhanced_channel[pos_x[j]][pos_y[j]] = newNoZeros[i]

                enhanced_image[:, :, channel] = enhanced_channel
            self.__matrix = enhanced_image
        else:  # Grayscale image processing
            for i in range(0, len(noZeroPosHist)):
                ind = noZeroPosHist[i]
                pos = np.where(targetMatrix == ind)
                pos_x = pos[0]
                pos_y = pos[1]
                for j in range(0, len(pos_x)):
                    self.__matrix[pos_x[j]][pos_y[j]] = newNoZeros[i]

        # Create comparison plot
        plt.figure(figsize=(12, 6))

        # Left: Original image
        plt.subplot(121)
        if len(originalImage.shape) == 3:
            plt.imshow(originalImage)
            plt.title('Original Color Image')
        else:
            plt.imshow(originalImage, cmap='Greys_r')
            plt.title('Original Image')

        # Right: Enhanced image
        plt.subplot(122)
        if len(self.__matrix.shape) == 3:
            plt.imshow(self.__matrix)
            plt.title('Enhanced Color Image')
        else:
            plt.imshow(self.__matrix, cmap='Greys_r')
            plt.title('Enhanced Grayscale Image')

        plt.tight_layout()
        plt.savefig(f_nameConf, dpi=150, bbox_inches='tight')
        plt.close()

        # Save enhanced image
        if self.__matrix.dtype != np.uint8:
            self.__matrix = np.clip(self.__matrix, 0, 255).astype(np.uint8)

        img = Image.fromarray(self.__matrix)
        img.save(f_name)

    def saveCombinedComparison(self, targetHist: np.ndarray, noZeroPosHist: list,
                              targetMatrix: np.ndarray, originalImage: np.ndarray,
                              grayscaleEnhanced: np.ndarray, colorEnhanced: np.ndarray,
                              f_nameConf: str, generation: int, fitness: float):
        """
        Save combined comparison showing original, grayscale enhanced, and color enhanced.

        Args:
            targetHist (np.ndarray): Target histogram.
            noZeroPosHist (list): Non-zero histogram positions.
            targetMatrix (np.ndarray): Original image matrix.
            originalImage (np.ndarray): Original image.
            grayscaleEnhanced (np.ndarray): Grayscale enhanced image.
            colorEnhanced (np.ndarray): Color enhanced image.
            f_nameConf (str): Output filename for comparison plot.
            generation (int): Current generation number.
            fitness (float): Current fitness value.
        """
        plt.figure(figsize=(18, 6))

        # Original image
        plt.subplot(131)
        if len(originalImage.shape) == 3:
            plt.imshow(originalImage)
            plt.title('Original Color Image')
        else:
            plt.imshow(originalImage, cmap='Greys_r')
            plt.title('Original Image')

        # Grayscale enhanced
        plt.subplot(132)
        if len(grayscaleEnhanced.shape) == 3:
            plt.imshow(grayscaleEnhanced)
            plt.title('Grayscale Enhanced')
        else:
            plt.imshow(grayscaleEnhanced, cmap='Greys_r')
            plt.title('Grayscale Enhanced')

        # Color enhanced
        plt.subplot(133)
        if len(colorEnhanced.shape) == 3:
            plt.imshow(colorEnhanced)
            plt.title('Color Enhanced')
        else:
            plt.imshow(colorEnhanced, cmap='Greys_r')
            plt.title('Color Enhanced')

        plt.suptitle(f'Generation: {generation}, Fitness: {fitness:.4f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f_nameConf, dpi=150, bbox_inches='tight')
        plt.close()

    def saveTermFitness(self, file: str, mod: str):
        """
        Save the three fitness terms to a file.

        Args:
            file (str): Output filename.
            mod (str): File mode ('w' for write, 'a' for append).
        """
        with open(file, mod) as fo:
            fo.write(str(self.__term1) + "\t")
            fo.write(str(self.__term2) + "\t")
            fo.write(str(self.__term3) + "\n")

    def calculateFitness(self, targetHist: np.ndarray, noZeroPosHist: list,
                         maxGrayLevel: int, minGrayLevel: int,
                         method: str = 'reverse') -> tuple:
        """
        Calculate fitness value for this chromosome.

        The fitness function evaluates how well the chromosome's intensity mapping
        preserves and enhances the bimodal structure of the histogram. Lower
        fitness values indicate better solutions.

        Fitness is computed as the sum of three terms:
        1. Term 1: Distance from optimal threshold to mean of two distributions
        2. Term 2: Difference between expected and actual variance of first mode
        3. Term 3: Difference between expected and actual variance of second mode

        Args:
            targetHist (np.ndarray): Target histogram.
            noZeroPosHist (list): Non-zero histogram positions.
            maxGrayLevel (int): Maximum gray level.
            minGrayLevel (int): Minimum gray level.
            method (str): 'reverse' or 'direct' mapping method (default: 'reverse').

        Returns:
            tuple: (fitness, opt_T, hist) where:
                - fitness: Total fitness value (lower is better)
                - opt_T: Optimal threshold
                - hist: Resulting histogram after mapping
        """
        hist = [0]*(maxGrayLevel+1)

        if method == 'reverse':
            oldIdx = self.genes[-1].position
            for i in range(len(noZeroPosHist)-1, -1, -1):
                idx = self.genes[i].position
                if idx < minGrayLevel or idx > maxGrayLevel:
                    print('idx', idx)
                    exit()
                ind = noZeroPosHist[i]
                if idx == oldIdx:
                    hist[idx] += targetHist[ind]
                else:
                    hist[idx] = targetHist[ind]
                    oldIdx = self.genes[i].position

        elif method == 'direct':
            oldIdx = self.genes[0].position
            for i in range(0, len(noZeroPosHist)):
                idx = self.genes[i].position
                if idx < minGrayLevel or idx > maxGrayLevel:
                    print('idx', idx)
                    exit()
                ind = noZeroPosHist[i]
                if idx == oldIdx:
                    hist[idx] += targetHist[ind]
                else:
                    hist[idx] = targetHist[ind]
                    oldIdx = self.genes[i].position

        # Use appropriate fitness function based on histogram modality
        if self.histogram_modality == 'bimodal':
            # Original bimodal fitness function for medical images
            opt_T, mu1, mu2 = self.__optimalThreshold(hist, 0.001, 100)
            sigma1, sigma2, halfWidth1, halfWidth2 = self.__calculateVariances(hist, opt_T, mu1, mu2)

            self.__term1 = abs(2*opt_T - mu1 - mu2)
            self.__term2 = abs(halfWidth1*0.33 - sigma1)
            self.__term3 = abs(halfWidth2*0.33 - sigma2)

            dist = self.__term1 + self.__term2 + self.__term3
            return dist, opt_T, hist
        else:
            # General image fitness function for normal blurred images
            return self.__calculate_general_fitness(hist, maxGrayLevel, minGrayLevel)

    def __calculate_general_fitness(self, hist: list, maxGrayLevel: int, minGrayLevel: int) -> tuple:
        """
        Calculate fitness for general (non-bimodal) images.
        
        This fitness function is designed for normal blurred color images and measures:
        1. Contrast enhancement (histogram spread)
        2. Entropy (information content)
        3. Histogram uniformity (avoiding over-concentration)
        
        Args:
            hist (list): Resulting histogram after intensity mapping.
            maxGrayLevel (int): Maximum gray level.
            minGrayLevel (int): Minimum gray level.
            
        Returns:
            tuple: (fitness, opt_T, hist) where fitness is minimized (lower is better).
        """
        hist_array = np.array(hist)
        
        # Calculate entropy (higher is better, so we want to maximize it)
        total = np.sum(hist_array)
        if total == 0:
            entropy = 0.0
        else:
            probs = hist_array.astype(float) / total
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                entropy = 0.0
        
        # Calculate contrast (standard deviation, higher is better)
        weighted_sum = np.sum(hist_array * np.arange(len(hist_array)))
        mean = weighted_sum / total if total > 0 else 128
        variance = np.sum(hist_array * (np.arange(len(hist_array)) - mean) ** 2) / total if total > 0 else 0
        std_dev = np.sqrt(variance)
        
        # Calculate histogram spread (range of used intensities)
        non_zero_indices = np.nonzero(hist_array)[0]
        if len(non_zero_indices) > 0:
            intensity_range = non_zero_indices[-1] - non_zero_indices[0]
        else:
            intensity_range = 0
        
        # Calculate histogram uniformity (penalize over-concentration)
        # We want histogram to be spread out, not concentrated in few bins
        max_bin_value = np.max(hist_array)
        concentration_penalty = max_bin_value / total if total > 0 else 1.0
        
        # Normalize metrics
        max_entropy = 8.0  # Maximum entropy for 256 bins
        max_std = 73.0     # Maximum std dev for 0-255 range
        max_range = 255.0  # Maximum intensity range
        
        # Fitness components (we want to minimize, so invert good metrics)
        # Lower fitness = better enhancement
        entropy_term = (max_entropy - entropy) / max_entropy  # Inverted: lower entropy = higher penalty
        contrast_term = (max_std - std_dev) / max_std        # Inverted: lower contrast = higher penalty
        range_term = (max_range - intensity_range) / max_range  # Inverted: smaller range = higher penalty
        
        # Combined fitness (weighted sum)
        # Weights: contrast is most important, then entropy, then range
        fitness = 0.5 * contrast_term + 0.3 * entropy_term + 0.15 * range_term + 0.05 * concentration_penalty
        
        # Use mean as pseudo-threshold for compatibility
        opt_T = int(mean)
        
        self.__term1 = contrast_term
        self.__term2 = entropy_term
        self.__term3 = range_term
        
        return fitness, opt_T, hist

    def getFitness(self) -> float:
        """
        Get the fitness value of this chromosome.

        Returns:
            float: Fitness value (lower is better).
        """
        return self.__fitness

    def getOpt_T(self) -> int:
        """
        Get the optimal threshold value.

        Returns:
            int: Optimal threshold.
        """
        return self.__opt_T

    def getMatrix(self) -> np.ndarray:
        """
        Get the enhanced image matrix.

        Returns:
            np.ndarray: Enhanced image array.
        """
        return self.__matrix


class geneticOperation(object):
    """
    Class containing genetic operators for crossover and mutation.

    This class implements the genetic operations that create new chromosomes
    from existing ones during the evolutionary process.
    """

    def mutate(self, genes: list, minGrayLevel: int, maxGrayLevel: int,
               opt_T: int, rate: float, histogram_modality: str = 'bimodal'):
        """
        Apply mutation to a list of genes.

        Each gene has a probability 'rate' of being mutated. Mutation strategy
        depends on histogram modality: constrained for bimodal, flexible for general.

        Args:
            genes (list): List of gene objects to potentially mutate.
            minGrayLevel (int): Minimum gray level.
            maxGrayLevel (int): Maximum gray level.
            opt_T (int): Optimal threshold.
            rate (float): Mutation probability (0.0 to 1.0).
            histogram_modality (str): 'bimodal' or 'general' (default: 'bimodal').
        """
        for i in range(0, len(genes)):
            if rnd.uniform(0, 1) < rate:
                genes[i].mutatePosition(minGrayLevel, maxGrayLevel, opt_T, histogram_modality)

    def crossoverSingle(self, parent_1: chromosome, parent_2: chromosome) -> list:
        """
        Perform single-point crossover between two parents.

        Args:
            parent_1 (chromosome): First parent chromosome.
            parent_2 (chromosome): Second parent chromosome.

        Returns:
            list: New chromosome genes created from crossover.
        """
        numberGenes = len(parent_1.genes)
        if numberGenes == 0:
            return []
        randNum = rnd.randint(0, numberGenes)
        list1 = deepcopy(parent_1.genes[0:randNum])
        list2 = deepcopy(parent_2.genes[randNum:numberGenes])
        return list1 + list2

    def crossoverUniform(self, parent_1: chromosome, parent_2: chromosome,
                        cross_point: Optional[int] = None) -> tuple:
        """
        Perform uniform crossover between two parents.

        This method creates a new chromosome by combining genes from both parents
        in a more complex pattern than single-point crossover.

        Args:
            parent_1 (chromosome): First parent chromosome.
            parent_2 (chromosome): Second parent chromosome.
            cross_point (int, optional): Specific crossover point. If None,
                                        randomly selected.

        Returns:
            tuple: (new_genes, cross_point) where:
                - new_genes: List of genes for new chromosome
                - cross_point: Crossover point used
        """
        if cross_point:
            numberGenes = len(parent_1.genes)
            list1 = []
            half = int(round(numberGenes / 2.0))
            if cross_point >= half:
                list1 = (deepcopy(parent_2.genes[0:cross_point-half]) +
                        deepcopy(parent_1.genes[cross_point-half:cross_point]) +
                        deepcopy(parent_2.genes[cross_point:numberGenes]))
            else:
                list1 = (deepcopy(parent_1.genes[0:cross_point]) +
                        deepcopy(parent_2.genes[cross_point:cross_point+half]) +
                        deepcopy(parent_1.genes[cross_point+half:numberGenes]))
            return list1, cross_point
        else:
            numberGenes = len(parent_1.genes)
            if numberGenes == 0:
                return [], 0
            if numberGenes == 1:
                randNum = 0
            else:
                randNum = rnd.randint(0, numberGenes-1)
            list1 = []
            half = int(round(numberGenes / 2.0))
            if randNum >= half:
                list1 = (deepcopy(parent_1.genes[0:randNum-half]) +
                        deepcopy(parent_2.genes[randNum-half:randNum]) +
                        deepcopy(parent_1.genes[randNum:numberGenes]))
            else:
                list1 = (deepcopy(parent_2.genes[0:randNum]) +
                        deepcopy(parent_1.genes[randNum:randNum+half]) +
                        deepcopy(parent_2.genes[randNum+half:numberGenes]))
            return list1, randNum


# ============================================================================
# MEDGA SEQUENTIAL CLASS
# ============================================================================

class MedGA(object):
    """
    Main MedGA class for sequential image enhancement using genetic algorithms.

    This class orchestrates the entire genetic algorithm process for enhancing
    medical images. It manages population initialization, evolution, selection,
    crossover, mutation, and fitness evaluation.

    Attributes:
        __pathIn (str): Input image path.
        __pathOut (str): Output directory path.
        __outputName (str): Base output directory for images.
        __outputNameFit (str): Fitness output file path.
        __outputNameThresh (str): Threshold output file path.
        __outputNameTerms (str): Fitness terms output file path.
        __outputNameInfo (str): Information output file path.
        __childrenPerGen (int): Number of children per generation.
        __numberOfGenes (int): Number of genes per chromosome.
        __minGrayLevel (int): Minimum gray level.
        __maxGrayLevel (int): Maximum gray level.
        __targetMatrix (np.ndarray): Target image matrix.
        __targetHist (np.ndarray): Target histogram.
        __noZeroPosHist (list): Non-zero histogram positions.
        __originalImage (np.ndarray): Original image.
        __grayscaleImage (np.ndarray): Grayscale version of image.
        __process_color (bool): Whether processing color images.
        __fitness_history (list): History of best fitness values.
        __generation_history (list): History of generation numbers.
    """

    def __init__(self, pathInput: str, pathOutput: str):
        """
        Initialize MedGA with input and output paths.

        Args:
            pathInput (str): Path to input image file.
            pathOutput (str): Path to output directory.
        """
        self.__pathIn = pathInput
        self.__pathOut = pathOutput

        self.__outputName = None
        self.__outputNameFit = None
        self.__outputNameThresh = None
        self.__outputNameTerms = None
        self.__outputNameInfo = None

        self.__childrenPerGen = None
        self.__numberOfGenes = None
        self.__minGrayLevel = None
        self.__maxGrayLevel = None
        self.__targetMatrix = None
        self.__targetHist = None
        self.__noZeroPosHist = None
        self.__originalImage = None
        self.__grayscaleImage = None
        self.__process_color = False
        self.__histogram_modality = 'bimodal'  # Default, will be detected from image

        # For fitness tracking
        self.__fitness_history = []
        self.__generation_history = []

    def startGA(self, pop_size: int, numGen: int, selection: str, cross_rate: float,
                mut_rate: float, elitism: int, numberIndTour: int, minGL: int = 1,
                process_color: bool = False):
        """
        Start the genetic algorithm evolution process.

        This is the main method that runs the complete genetic algorithm:
        1. Loads and preprocesses the image
        2. Initializes the population
        3. Evolves the population for specified generations
        4. Saves results and generates fitness plots

        Args:
            pop_size (int): Population size (number of chromosomes).
            numGen (int): Number of generations to evolve.
            selection (str): Selection method ('tournament', 'wheel', or 'ranking').
            cross_rate (float): Crossover rate (0.0 to 1.0).
            mut_rate (float): Mutation rate (0.0 to 1.0).
            elitism (int): Number of best individuals to preserve.
            numberIndTour (int): Number of individuals in tournament selection.
            minGL (int): Minimum gray level (default: 1).
            process_color (bool): Whether to process as color image (default: False).
        """
        self.__process_color = process_color

        # Image Processing object
        imPros = processing()

        # Paths used to save the images and other information
        self.__outputNameInfo = self.__pathOut + os.sep + 'information'
        self.__outputName = self.__pathOut + os.sep + 'images'

        if not os.path.exists(self.__outputName):
            os.makedirs(self.__outputName)

        self.__outputNameFit = self.__pathOut + os.sep + "fitness"
        self.__outputNameThresh = self.__pathOut + os.sep + "threshold"
        self.__outputNameTerms = self.__pathOut + os.sep + "terms"

        # Reading the input image
        (self.__originalImage, numberGrayLevel, self.__targetHist,
         self.__noZeroPosHist, maxValueGray, T_k, self.__grayscaleImage,
         self.__histogram_modality) = imPros.loadImage(
            self.__pathIn, self.__outputName, process_color)
        
        print(f"📊 Histogram modality detected: {self.__histogram_modality}")

        # Use appropriate image for processing
        if process_color and len(self.__originalImage.shape) == 3:
            self.__targetMatrix = self.__originalImage
            print(f"🎨 Processing COLOR image with {self.__originalImage.shape[2]} channels")
        else:
            self.__targetMatrix = self.__grayscaleImage
            print("⚫ Processing GRAYSCALE image")

        # GA settings
        self.__childrenPerGen = pop_size - elitism
        self.__numberOfGenes = numberGrayLevel
        self.__minGrayLevel = minGL
        self.__maxGrayLevel = maxValueGray

        # Initialize fitness tracking
        self.__fitness_history = []
        self.__generation_history = []

        # Saving the used GA settings
        with open(self.__outputNameInfo, "w") as fo:
            fo.write("******************************************************\n")
            fo.write("\t\t\t GA settings\n\n")
            fo.write("Processing mode: " + ("COLOR" if (process_color and len(self.__originalImage.shape) == 3) else "GRAYSCALE") + "\n")
            fo.write("Number of chromosome: " + str(pop_size) + "\n")
            fo.write("Number of elite chromosomes: " + str(elitism) + "\n")
            fo.write("Number of genes: " + str(self.__numberOfGenes) + "\n")
            fo.write("Number of generations: " + str(numGen) + "\n")
            fo.write("Crossover rate: " + str(cross_rate) + "\n")
            fo.write("Mutation rate:  " + str(mut_rate) + "\n")

        pop = []

        # Initialization of the GA instance
        pop = self.__initialize(pop, pop_size, mut_rate, T_k)

        # Evolution of the GA
        pop = self.__evolve(pop, cross_rate, mut_rate, numGen, elitism, T_k,
                           method=selection, numberInd=numberIndTour)

        # Generate fitness plot
        self.__generate_fitness_plot()

    def __initialize(self, pop: list, pop_size: int, mut_rate: float, T_k: int) -> list:
        """
        Initialize the genetic algorithm population.

        Creates an initial population of random chromosomes, sorts them by fitness,
        and saves the initial best individual.

        Args:
            pop (list): Empty population list (will be populated).
            pop_size (int): Size of population to create.
            mut_rate (float): Mutation rate (used in chromosome creation).
            T_k (int): Initial optimal threshold.

        Returns:
            list: Initialized and sorted population.
        """
        is_color = self.__process_color and len(self.__targetMatrix.shape) == 3

        for i in range(pop_size):
            pop.append(chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes,
                                self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k,
                                is_color=is_color, histogram_modality=self.__histogram_modality))

        # Sorting the population based on the fitness values
        pop.sort(key=lambda x: x.getFitness())

        # Record initial fitness
        initial_fitness = pop[0].getFitness()
        self.__fitness_history.append(initial_fitness)
        self.__generation_history.append(0)

        # Save initial enhanced image
        pop[0].saveCurrentImage(self.__targetHist, self.__noZeroPosHist, self.__targetMatrix,
                               self.__originalImage,
                               self.__outputName + os.sep + 'image_gen_0.png',
                               self.__outputName + os.sep + 'imageConf_gen_0.png')

        with open(self.__outputNameFit, "w") as fo:
            fo.write(str(pop[0].getFitness()) + "\n")

        with open(self.__outputNameThresh, "w") as fo:
            fo.write(str(pop[0].getOpt_T()) + "\n")

        pop[0].saveTermFitness(self.__outputNameTerms, "w")

        print(f"📊 Initial fitness: {initial_fitness:.4f}")
        return pop

    def __evolve(self, pop: list, cross_rate: float, mut_rate: float, numGen: int,
                elitism: int, T_k: int, method: str = 'wheel',
                numberInd: int = 10) -> list:
        """
        Evolve the population through genetic algorithm generations.

        Performs selection, crossover, mutation, and elitism for each generation.
        Tracks fitness progression and saves intermediate results.

        Args:
            pop (list): Initial population.
            cross_rate (float): Crossover rate.
            mut_rate (float): Mutation rate.
            numGen (int): Number of generations.
            elitism (int): Number of elite individuals to preserve.
            T_k (int): Optimal threshold.
            method (str): Selection method ('tournament', 'wheel', or 'ranking').
            numberInd (int): Number of individuals in tournament.

        Returns:
            list: Evolved population after all generations.
        """
        n = len(pop)
        is_color = self.__process_color and len(self.__targetMatrix.shape) == 3

        with open(self.__outputNameInfo, "a") as fo:
            if method == 'wheel':
                fo.write("Selection: wheel roulette\n")
            elif method == 'ranking':
                fo.write("Selection: ranking \n")
            else:
                fo.write("Selection: tournament with " + str(numberInd) + " individuals\n")

        op = geneticOperation()

        # Calculate intervals for saving images
        save_interval = max(50, numGen // 10)
        print(f"💾 Saving progress every {save_interval} generations")

        # The population evolves for (numGen-1) generations
        for i in range(1, numGen):
            # Selection methods
            if method == 'wheel':
                probabilities = []
                for j in range(0, n):
                    probabilities.append(pop[j].getFitness())
                sum_fit = np.sum(probabilities)

                for j in range(0, n):
                    probabilities[j] = (probabilities[j]) / (sum_fit * 1.0)

                probabilities = (1 - np.array(probabilities))
                probabilities /= np.sum(probabilities)

            elif method == 'ranking':
                probabilities = []
                rank = np.linspace(1, n, n)
                rank = rank[::-1]
                probabilities = rank / float(np.sum(rank))

            # New individuals
            countWhile = self.__childrenPerGen
            children = []

            while countWhile > 0:
                # Tournament selection
                if method == 'tournament':
                    dist1 = np.random.randint(0, n, numberInd)
                    dist2 = np.random.randint(0, n, numberInd)

                    individuals1 = []
                    individuals2 = []
                    for k in range(numberInd):
                        individuals1.append(pop[dist1[k]])
                        individuals2.append(pop[dist2[k]])

                    individuals1.sort(key=lambda x: x.getFitness())
                    individuals2.sort(key=lambda x: x.getFitness())

                    parent_1 = individuals1[0]
                    parent_2 = individuals2[0]

                # Roulette wheel or ranking selection
                else:
                    parent_1 = np.random.choice(pop, p=probabilities)
                    parent_2 = np.random.choice(pop, p=probabilities)

                # The latest individual is the best children
                if countWhile == 1:
                    child0 = chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes,
                                      self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k,
                                      parent_1=parent_1, parent_2=parent_2, is_color=is_color,
                                      histogram_modality=self.__histogram_modality)
                    child1 = deepcopy(chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes,
                                               self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k,
                                               parent_1=parent_1, parent_2=parent_2,
                                               cross_point=child0.crossPoint, is_color=is_color,
                                               histogram_modality=self.__histogram_modality))

                    if child0.getFitness() < child1.getFitness():
                        children.append(child0)
                    else:
                        children.append(child1)
                        countWhile = countWhile - 1

                # Crossover
                elif rnd.uniform(0, 1) < cross_rate:
                    child0 = chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes,
                                      self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k,
                                      parent_1=parent_1, parent_2=parent_2, is_color=is_color,
                                      histogram_modality=self.__histogram_modality)
                    children.append(child0)
                    child1 = deepcopy(chromosome(self.__targetHist, self.__noZeroPosHist, self.__numberOfGenes,
                                               self.__minGrayLevel, self.__maxGrayLevel, mut_rate, T_k,
                                               parent_1=parent_1, parent_2=parent_2,
                                               cross_point=child0.crossPoint, is_color=is_color,
                                               histogram_modality=self.__histogram_modality))
                    children.append(child1)
                    countWhile = countWhile - 2

                # Parents without crossover
                else:
                    op.mutate(parent_1.genes, self.__minGrayLevel, self.__maxGrayLevel,
                             parent_1.getOpt_T(), mut_rate, self.__histogram_modality)
                    op.mutate(parent_2.genes, self.__minGrayLevel, self.__maxGrayLevel,
                             parent_2.getOpt_T(), mut_rate, self.__histogram_modality)
                    parent_1.calculateFitness(self.__targetHist, self.__noZeroPosHist,
                                            self.__maxGrayLevel, self.__minGrayLevel)
                    parent_2.calculateFitness(self.__targetHist, self.__noZeroPosHist,
                                            self.__maxGrayLevel, self.__minGrayLevel)
                    children.append(parent_1)
                    children.append(parent_2)
                    countWhile = countWhile - 2

            # Elitism to keep the best individual(s) during the evolution
            pop[elitism:n] = deepcopy(children[0:self.__childrenPerGen])

            # Sorting the population based on the fitness values
            pop.sort(key=lambda x: x.getFitness())

            # Record fitness
            current_fitness = pop[0].getFitness()
            self.__fitness_history.append(current_fitness)
            self.__generation_history.append(i)

            with open(self.__outputNameFit, "a") as fo:
                fo.write(str(current_fitness) + "\n")

            with open(self.__outputNameThresh, "a") as fo:
                fo.write(str(pop[0].getOpt_T()) + "\n")

            pop[0].saveTermFitness(self.__outputNameTerms, "a")

            # Save best image at regular intervals
            if (i % save_interval == 0) or (i <= 10) or (i == numGen - 1):
                if i <= 10:
                    gen_suffix = f"gen_{i:02d}"
                else:
                    gen_suffix = f"gen_{i:04d}"

                pop[0].saveCurrentImage(self.__targetHist, self.__noZeroPosHist, self.__targetMatrix,
                                       self.__originalImage,
                                       self.__outputName + os.sep + f'image_{gen_suffix}.png',
                                       self.__outputName + os.sep + f'imageConf_{gen_suffix}.png')

                improvement = self.__fitness_history[0] - current_fitness
                improvement_pct = (improvement / self.__fitness_history[0]) * 100
                print(f"📈 Generation {i:4d}: Fitness = {current_fitness:.4f} "
                      f"(Improvement: {improvement_pct:6.2f}%)")

            if i == numGen - 1:
                # Save final comparison
                pop[0].saveCurrentImage(self.__targetHist, self.__noZeroPosHist, self.__targetMatrix,
                                       self.__originalImage,
                                       self.__outputName + os.sep + 'image_final.png',
                                       self.__outputName + os.sep + 'imageConf_final.png')

                np.savetxt(self.__pathOut + os.sep + 'matrixBest',
                          pop[0].getMatrix().reshape(-1, pop[0].getMatrix().shape[-1])
                          if len(pop[0].getMatrix().shape) == 3 else pop[0].getMatrix(),
                          fmt='%d')

                final_improvement = self.__fitness_history[0] - current_fitness
                final_improvement_pct = (final_improvement / self.__fitness_history[0]) * 100
                print(f"✅ Final generation {i}: Fitness = {current_fitness:.4f} "
                      f"(Total Improvement: {final_improvement_pct:.2f}%)")
        return pop

    def __generate_fitness_plot(self):
        """
        Generate fitness progression plot.

        Creates a two-panel plot showing:
        1. Fitness progression over generations
        2. Percentage improvement over time

        Saves the plot as 'fitness_analysis.png' in the output directory.
        """
        if len(self.__fitness_history) < 2:
            return

        plt.figure(figsize=(12, 6))

        # Fitness progression
        plt.subplot(1, 2, 1)
        plt.plot(self.__generation_history, self.__fitness_history, 'b-', linewidth=2,
                label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Progression')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Fitness improvement
        plt.subplot(1, 2, 2)
        initial_fitness = self.__fitness_history[0]
        improvements = [(initial_fitness - f) / initial_fitness * 100 for f in self.__fitness_history]
        plt.plot(self.__generation_history, improvements, 'g-', linewidth=2,
                label='Improvement %')
        plt.xlabel('Generation')
        plt.ylabel('Improvement (%)')
        plt.title('Fitness Improvement Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.__pathOut + os.sep + 'fitness_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Fitness analysis saved: {self.__pathOut + os.sep + 'fitness_analysis.png'}")


# ============================================================================
# MPI PARALLEL PROCESSING FUNCTIONS
# ============================================================================

WORKTAG = 0
DIETAG = 1

def master(toProcess: list, pathsOutput: list, population: int, generations: int,
          selection: str, cross_rate: float, mut_rate: float, elitism: int,
          pressure: int, verbose: bool) -> np.ndarray:
    """
    Master process for MPI parallel processing.

    Distributes images among slave processes and collects elapsed times.
    Implements a master-slave paradigm where the master coordinates work
    distribution and the slaves perform the actual image processing.

    Args:
        toProcess (list): List of image paths to process.
        pathsOutput (list): List of output directory paths.
        population (int): GA population size.
        generations (int): Number of GA generations.
        selection (str): Selection method.
        cross_rate (float): Crossover rate.
        mut_rate (float): Mutation rate.
        elitism (int): Elitism count.
        pressure (int): Tournament pressure.
        verbose (bool): Verbose output flag.

    Returns:
        np.ndarray: Array of elapsed times for each image.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    n = len(toProcess)
    status = MPI.Status()

    times = np.zeros(len(toProcess))
    idx = 0

    if len(toProcess) == 0:
        # The Master sends the DIETAG to the Slaves
        for i in range(1, size):
            comm.send(obj=None, dest=i, tag=DIETAG)
        return times

    # If the number of the images (n) is higher than the available Slaves (size-1),
    # (size-1) images are run in parallel
    if n > (size-1):
        for i in range(1, size):
            inp = [toProcess[i-1], pathsOutput[i-1], population, generations, selection,
                  cross_rate, mut_rate, elitism, pressure, verbose]
            comm.send(inp, dest=i, tag=WORKTAG)

        # As soon as a Slave is available, the Master assigns it a new image to process
        for i in range(size, n+1):
            im_free, elapsed = comm.recv(source=MPI.ANY_SOURCE, tag=10, status=status)
            times[idx] = elapsed
            idx += 1

            inp = [toProcess[i-1], pathsOutput[i-1], population, generations, selection,
                  cross_rate, mut_rate, elitism, pressure, verbose]
            comm.send(inp, dest=im_free, tag=WORKTAG)

        for i in range(size, n+1):
            im_free, elapsed = comm.recv(source=MPI.ANY_SOURCE, tag=10, status=status)
            times[idx] = elapsed
            idx += 1

    # If the number of images (n) is lower than the available cores (size-1),
    # only n Slaves are used
    else:
        for i in range(0, n):
            inp = [toProcess[i], pathsOutput[i], population, generations, selection,
                  cross_rate, mut_rate, elitism, pressure, verbose]
            comm.send(inp, dest=i+1, tag=WORKTAG)

        for i in range(0, n):
            im_free, elapsed = comm.recv(source=MPI.ANY_SOURCE, tag=10, status=status)
            times[idx] = elapsed
            idx += 1

    # The Master sends the DIETAG to the Slaves
    for i in range(1, size):
        comm.send(obj=None, dest=i, tag=DIETAG)

    return times

def slave():
    """
    Slave process for MPI parallel processing.

    Receives work assignments from the master, processes images using MedGA,
    and sends results back to the master. Continues until receiving DIETAG.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    while True:
        status = MPI.Status()

        # The Slave waits for an image to process
        inp = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        elapsed = 0

        if inp is not None:
            start = time.time()

            # MedGA execution on the input image by using the provided GA settings
            medga = MedGA(inp[0], inp[1])
            medga.startGA(inp[2], inp[3], inp[4], inp[5], inp[6], inp[7], inp[8])

            end = time.time()
            elapsed = end-start

            if inp[9]:
                sys.stdout.write(" * Analyzed image %s"%inp[0])
                sys.stdout.write(" -> Elapsed time %5.2fs on rank %d\n\n" % (elapsed, rank))

        if status.Get_tag():
            break

        # The Slave is free to process a new image
        comm.send([rank,elapsed], dest=0, tag=10)


# ============================================================================
# CLI INTERFACE CLASS
# ============================================================================

class MedGACLI:
    """
    Command-line interface class for MedGA with rich formatting support.

    Provides a beautiful, user-friendly CLI with progress tracking, parameter
    display, and interactive configuration. Falls back to simple text output
    if rich library is not available.

    Attributes:
        console (Console): Rich console object for formatted output (if available).
        start_time (float): Start time of processing.
        current_step (int): Current processing step.
        total_steps (int): Total number of steps.
    """

    def __init__(self):
        """Initialize the CLI interface."""
        self.console = Console() if RICH_AVAILABLE else None
        self.start_time = None
        self.current_step = 0
        self.total_steps = 0

    def print_header(self):
        """Print beautiful header with system information."""
        if self.console:
            header_text = Text()
            header_text.append("🧬 ", style="bold cyan")
            header_text.append("MedGA", style="bold magenta")
            header_text.append(" - Medical Image Enhancement using Genetic Algorithms", style="bold white")

            header_panel = Panel(
                header_text,
                box=box.DOUBLE_EDGE,
                style="bright_blue",
                padding=(1, 2)
            )
            self.console.print(header_panel)

            # Print info panel
            info_table = Table(show_header=False, box=box.ROUNDED, style="dim")
            info_table.add_column("", style="cyan")
            info_table.add_column("", style="white")

            info_table.add_row("📚 Reference", "L. Rundo, A. Tangherloni et al. (2019)")
            info_table.add_row("🔬 Method", "Evolutionary Image Enhancement")
            info_table.add_row("🎯 Purpose", "Medical Image Quality Improvement")
            info_table.add_row("⚙️  Version", "2.0 with Modern CLI & Analytics")

            self.console.print(Panel(info_table, title="📋 System Information", style="green"))
        else:
            print("=" * 80)
            print("🧬 MedGA - Medical Image Enhancement using Genetic Algorithms")
            print("=" * 80)
            print("📚 Reference: L. Rundo, A. Tangherloni et al. (2019)")
            print("🔬 Method: Evolutionary Image Enhancement")
            print("🎯 Purpose: Medical Image Quality Improvement")
            print("⚙️  Version: 2.0 with Modern CLI & Analytics")
            print("-" * 80)

    def print_parameters(self, args):
        """
        Print algorithm parameters in a beautiful table.

        Args:
            args: Argument namespace object containing all parameters.
        """
        if self.console:
            param_table = Table(title="⚙️  Algorithm Parameters", box=box.ROUNDED, show_header=True)
            param_table.add_column("Parameter", style="cyan", no_wrap=True)
            param_table.add_column("Value", style="white")
            param_table.add_column("Description", style="dim")

            param_table.add_row("Input", args.image or args.folder or "Interactive", "Source image or folder")
            param_table.add_row("Output", args.output, "Results directory")
            param_table.add_row("Output Format", args.format, "Output image format")
            param_table.add_row("Generations", str(args.generations), "GA evolution cycles")
            param_table.add_row("Population", str(args.population), "Chromosomes per generation")
            param_table.add_row("Selection", args.selection, "Parent selection method")
            param_table.add_row("Crossover Rate", f"{args.cross_rate:.2f}", "Chromosome mixing probability")
            param_table.add_row("Mutation Rate", f"{args.mut_rate:.2f}", "Gene alteration probability")
            param_table.add_row("Elitism", str(args.elitism), "Best chromosomes preserved")
            param_table.add_row("Tournament Pressure", str(args.pressure), "Tournament selection size")
            param_table.add_row("Mode", self._get_mode_display(args), "Processing approach")
            param_table.add_row("Distributed", "Yes" if args.distributed else "No", "MPI parallel processing")
            if args.distributed:
                param_table.add_row("Cores", str(args.cores), "MPI processes")
            param_table.add_row("Verbose", "Yes" if args.verbose else "No", "Detailed output")

            self.console.print(param_table)
        else:
            print("⚙️  ALGORITHM PARAMETERS:")
            print(f"  Input:          {args.image or args.folder or 'Interactive'}")
            print(f"  Output:         {args.output}")
            print(f"  Output Format:  {args.format}")
            print(f"  Generations:    {args.generations}")
            print(f"  Population:     {args.population}")
            print(f"  Selection:      {args.selection}")
            print(f"  Crossover Rate: {args.cross_rate:.2f}")
            print(f"  Mutation Rate:  {args.mut_rate:.2f}")
            print(f"  Elitism:        {args.elitism}")
            print(f"  Tournament:     {args.pressure}")
            print(f"  Mode:           {self._get_mode_display(args)}")
            print(f"  Distributed:    {'Yes' if args.distributed else 'No'}")
            if args.distributed:
                print(f"  Cores:          {args.cores}")
            print(f"  Verbose:        {'Yes' if args.verbose else 'No'}")
            print("-" * 80)

    def _get_mode_display(self, args) -> str:
        """
        Get display string for processing mode.

        Args:
            args: Argument namespace object.

        Returns:
            str: Formatted mode description string.
        """
        if args.both:
            return "🎨 Combined (Grayscale + Color)"
        elif args.color:
            return "🌈 Color Enhancement"
        else:
            return "⚫ Grayscale Enhancement"

    def create_progress_tracker(self, total_generations: int):
        """
        Create rich progress tracker.

        Args:
            total_generations (int): Total number of generations.

        Returns:
            Progress: Rich progress object or None if rich not available.
        """
        if self.console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
            return progress
        return None

    def print_fitness_improvement(self, initial_fitness: float, final_fitness: float,
                                 improvement: float):
        """
        Print fitness improvement results.

        Args:
            initial_fitness (float): Initial fitness value.
            final_fitness (float): Final fitness value.
            improvement (float): Improvement percentage.
        """
        if self.console:
            fitness_table = Table(title="📊 Fitness Improvement Analysis", box=box.ROUNDED)
            fitness_table.add_column("Metric", style="cyan")
            fitness_table.add_column("Value", style="white")
            fitness_table.add_column("Improvement", style="green")

            fitness_table.add_row("Initial Fitness", f"{initial_fitness:.4f}", "")
            fitness_table.add_row("Final Fitness", f"{final_fitness:.4f}", "")
            fitness_table.add_row("Total Improvement", f"{improvement:.2%}", f"{improvement:.2%}")

            self.console.print(fitness_table)
        else:
            print("📊 FITNESS IMPROVEMENT ANALYSIS:")
            print(f"  Initial Fitness: {initial_fitness:.4f}")
            print(f"  Final Fitness:   {final_fitness:.4f}")
            print(f"  Improvement:     {improvement:.2%}")

    def print_completion_summary(self, total_time: float, images_processed: int):
        """
        Print completion summary.

        Args:
            total_time (float): Total processing time in seconds.
            images_processed (int): Number of images processed.
        """
        if self.console:
            summary_table = Table(box=box.ROUNDED, show_header=False)
            summary_table.add_column("", style="cyan")
            summary_table.add_column("", style="white")

            summary_table.add_row("🕒 Total Time", f"{total_time:.2f} seconds")
            summary_table.add_row("📈 Images Processed", str(images_processed))
            summary_table.add_row("📁 Results Location", os.path.abspath("output"))
            summary_table.add_row("✅ Status", "COMPLETED SUCCESSFULLY")

            success_panel = Panel(
                summary_table,
                title="🎉 Processing Complete!",
                style="bold green",
                box=box.DOUBLE_EDGE
            )
            self.console.print(success_panel)
        else:
            print("🎉 PROCESSING COMPLETE!")
            print(f"🕒 Total Time:       {total_time:.2f} seconds")
            print(f"📈 Images Processed: {images_processed}")
            print(f"📁 Results Location: {os.path.abspath('output')}")
            print("=" * 80)

    def interactive_setup(self):
        """
        Interactive setup for parameters.

        Prompts user for all necessary parameters using rich prompts.

        Returns:
            Namespace: Argument namespace object with user-selected parameters.
        """
        if not self.console:
            print("Interactive mode requires rich. Using default parameters.")
            return self.get_default_args()

        self.console.print(Panel("🎛️  Interactive Configuration Mode", style="bold cyan"))

        class Args:
            """
            Lightweight namespace mirroring argparse output for interactive mode.

            Rich prompts populate attributes dynamically so the rest of the CLI
            can treat this object exactly like the one returned by argparse.
            """
            pass
        args = Args()

        # Input selection
        input_choice = Prompt.ask(
            "📁 Input type",
            choices=["single", "folder"],
            default="single"
        )

        if input_choice == "single":
            args.image = Prompt.ask("🖼️  Image file path")
            args.folder = None
        else:
            args.folder = Prompt.ask("📂 Folder path")
            args.image = None

        # Output directory
        args.output = Prompt.ask("💾 Output directory", default="output")

        # Output format
        args.format = Prompt.ask(
            "🖼️  Output format",
            choices=["jpg", "png", "tiff", "bmp"],
            default="jpg"
        )

        # GA Parameters
        args.population = IntPrompt.ask("👥 Population size", default=100)
        args.generations = IntPrompt.ask("🔄 Number of generations", default=100)
        args.selection = Prompt.ask(
            "🎯 Selection method",
            choices=["tournament", "wheel", "ranking"],
            default="tournament"
        )
        args.cross_rate = FloatPrompt.ask("🔀 Crossover rate", default=0.9)
        args.mut_rate = FloatPrompt.ask("🧬 Mutation rate", default=0.01)
        args.elitism = IntPrompt.ask("⭐ Elitism count", default=1)
        args.pressure = IntPrompt.ask("🏆 Tournament pressure", default=20)

        # Processing mode
        mode_choice = Prompt.ask(
            "🎨 Processing mode",
            choices=["grayscale", "color", "both"],
            default="grayscale"
        )
        args.color = (mode_choice == "color")
        args.both = (mode_choice == "both")

        # Advanced options
        args.distributed = Confirm.ask("⚡ Use distributed processing (MPI)", default=False)
        if args.distributed:
            args.cores = IntPrompt.ask("🖥️  Number of cores", default=4)
        else:
            args.cores = 4

        args.verbose = Confirm.ask("📢 Verbose output", default=True)

        return args

    def create_comparison_frame(self, original_path, grayscale_path, color_path, output_path, title="MedGA Enhancement Comparison"):
        """Create a combined comparison frame with all three images"""
        try:
            # Load images
            original_img = Image.open(original_path)
            grayscale_img = Image.open(grayscale_path)
            color_img = Image.open(color_path)

            # Resize images to have consistent display (comparison can be different size)
            target_height = min(original_img.height, grayscale_img.height, color_img.height, 600)

            def resize_to_height(img, height):
                """
                Resize a Pillow image to a given height while preserving aspect ratio.

                Args:
                    img (Image.Image): Input image to resize.
                    height (int): Target height in pixels.

                Returns:
                    Image.Image: Resized image using high-quality Lanczos resampling.
                """
                ratio = height / img.height
                new_width = int(img.width * ratio)
                return img.resize((new_width, height), Image.Resampling.LANCZOS)

            original_img = resize_to_height(original_img, target_height)
            grayscale_img = resize_to_height(grayscale_img, target_height)
            color_img = resize_to_height(color_img, target_height)

            # Create a new image with white background
            total_width = original_img.width + grayscale_img.width + color_img.width
            max_height = target_height + 100  # Extra space for labels

            combined_img = Image.new('RGB', (total_width, max_height), 'white')

            # Paste images
            x_offset = 0
            combined_img.paste(original_img, (x_offset, 50))
            x_offset += original_img.width

            combined_img.paste(grayscale_img, (x_offset, 50))
            x_offset += grayscale_img.width

            combined_img.paste(color_img, (x_offset, 50))

            # Add labels
            draw = ImageDraw.Draw(combined_img)

            # Try to use a nice font, fallback to default if not available
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                label_font = ImageFont.truetype("arial.ttf", 18)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()

            # Add title
            title_width = draw.textlength(title, font=title_font)
            title_x = (total_width - title_width) // 2
            draw.text((title_x, 10), title, fill='black', font=title_font)

            # Add image labels
            label_y = target_height + 60

            original_label = "Original Image"
            grayscale_label = "Grayscale Enhanced"
            color_label = "Color Enhanced"

            # Calculate label positions
            original_label_x = original_img.width // 2 - draw.textlength(original_label, font=label_font) // 2
            grayscale_label_x = original_img.width + grayscale_img.width // 2 - draw.textlength(grayscale_label, font=label_font) // 2
            color_label_x = original_img.width + grayscale_img.width + color_img.width // 2 - draw.textlength(color_label, font=label_font) // 2

            draw.text((original_label_x, label_y), original_label, fill='black', font=label_font)
            draw.text((grayscale_label_x, label_y), grayscale_label, fill='black', font=label_font)
            draw.text((color_label_x, label_y), color_label, fill='black', font=label_font)

            # Add separator lines
            line_y = target_height + 55
            draw.line([(original_img.width, 50), (original_img.width, line_y)], fill='gray', width=2)
            draw.line([(original_img.width + grayscale_img.width, 50), (original_img.width + grayscale_img.width, line_y)], fill='gray', width=2)

            # Save the combined image as PNG (always PNG for comparison frames)
            combined_img.save(output_path, quality=95)

            if self.console:
                self.console.print(f"✅ [green]Comparison frame saved: {output_path}[/green]")
            else:
                print(f"✅ Comparison frame saved: {output_path}")

            return True

        except Exception as e:
            if self.console:
                self.console.print(f"❌ [red]Error creating comparison frame: {str(e)}[/red]")
            else:
                print(f"❌ Error creating comparison frame: {str(e)}")
            return False

    def _find_enhanced_images_recursive(self, directory, file_format="jpg"):
        """Recursively search for enhanced images in directory and subdirectories"""
        enhanced_images = []

        if not os.path.exists(directory):
            return enhanced_images

        # Patterns to look for
        patterns = [
            f'image_*.{file_format}',
            f'image_*.png',
            f'enhanced_*.{file_format}',
            f'enhanced_*.png',
            f'image_gen_*.{file_format}',
            f'image_gen_*.png',
            f'image_final.{file_format}',
            f'image_final.png',
            f'image_best.{file_format}',
            f'image_best.png',
        ]

        # Exclude patterns (files we don't want)
        exclude_keywords = [
            'fitness', 'plot', 'analysis', 'config', 'comparison',
            'matrix', 'original', 'threshold', 'terms', 'information'
        ]

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_lower = file.lower()

                # Skip excluded files
                if any(keyword in file_lower for keyword in exclude_keywords):
                    continue

                # Check if file matches our patterns
                file_path = os.path.join(root, file)
                for pattern in patterns:
                    if fnmatch.fnmatch(file, pattern):
                        enhanced_images.append(file_path)
                        break

        return enhanced_images

    def _find_best_result_image(self, output_dir, file_format="jpg"):
        """Find the best ENHANCED result image (not plots) in an output directory"""
        # Search recursively for enhanced images
        enhanced_images = self._find_enhanced_images_recursive(output_dir, file_format)

        if enhanced_images:
            # Sort by generation number to get the final/best image
            def extract_gen_number(filename):
                """
                Derive a sortable numeric key from MedGA output filenames.

                Final/best images receive artificially large keys so they sort
                after intermediate generations; otherwise the generation or
                numeric suffix embedded in the filename is used.
                """
                try:
                    basename = os.path.basename(filename)
                    # Look for patterns like image_gen_100.jpg or image_final.jpg
                    if 'final' in basename.lower():
                        return 9999  # Final images get highest priority
                    elif 'best' in basename.lower():
                        return 9998  # Best images get second highest priority
                    elif 'gen_' in basename:
                        num_str = basename.split('gen_')[1].split('.')[0]
                        return int(num_str)
                    else:
                        # Extract number from filename like image_050.jpg
                        parts = basename.split('_')
                        for part in parts:
                            if part.isdigit():
                                return int(part)
                        return -1
                except:
                    return -1

            enhanced_images.sort(key=extract_gen_number, reverse=True)
            best_image = enhanced_images[0]

            if self.console:
                self.console.print(f"🔍 [green]Found enhanced image: {os.path.basename(best_image)}[/green]")
                self.console.print(f"   [dim]Path: {best_image}[/dim]")
            else:
                print(f"🔍 Found enhanced image: {os.path.basename(best_image)}")
                print(f"   Path: {best_image}")

            return best_image

        # If no enhanced images found, list what's available for debugging
        if self.console:
            self.console.print(f"🔍 [yellow]No enhanced images found in {output_dir}[/yellow]")
            # List all image files for debugging
            all_image_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        all_image_files.append(rel_path)

            if all_image_files:
                self.console.print(f"📁 [dim]All image files found: {', '.join(all_image_files)}[/dim]")
            else:
                self.console.print(f"📁 [dim]No image files found in {output_dir}[/dim]")
        else:
            print(f"🔍 No enhanced images found in {output_dir}")
            # List all image files for debugging
            all_image_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        all_image_files.append(rel_path)

            if all_image_files:
                print(f"📁 All image files found: {', '.join(all_image_files)}")
            else:
                print(f"📁 No image files found in {output_dir}")

        return None

    def _ensure_output_size(self, image_path, target_size=(512, 512)):
        """Ensure output image matches target size"""
        try:
            img = Image.open(image_path)
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(image_path)
                if self.console:
                    self.console.print(f"🔧 [dim]Resized {os.path.basename(image_path)} to {target_size}[/dim]")
            return True
        except Exception as e:
            if self.console:
                self.console.print(f"⚠️ [yellow]Could not resize {image_path}: {str(e)}[/yellow]")
            return False

    def _convert_output_format(self, output_dir, target_format="jpg"):
        """Convert all output images to the specified format, including subdirectories"""
        try:
            converted_files = []

            # Search through all subdirectories recursively
            for root, dirs, files in os.walk(output_dir):
                for filename in files:
                    # Only convert enhanced images, not plots or other files
                    if (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')) and
                        'fitness' not in filename.lower() and
                        'plot' not in filename.lower() and
                        'analysis' not in filename.lower() and
                        'comparison' not in filename.lower() and
                        'config' not in filename.lower()):

                        # Skip if already in target format
                        if filename.lower().endswith(f'.{target_format}'):
                            continue

                        input_path = os.path.join(root, filename)
                        name_without_ext = os.path.splitext(filename)[0]
                        output_path = os.path.join(root, f"{name_without_ext}.{target_format}")

                        # Convert image
                        img = Image.open(input_path)

                        # Save with appropriate settings for format
                        if target_format == "jpg":
                            img = img.convert("RGB")  # JPG doesn't support alpha
                            img.save(output_path, "JPEG", quality=95)
                        elif target_format == "png":
                            img.save(output_path, "PNG")
                        elif target_format == "tiff":
                            img.save(output_path, "TIFF")
                        elif target_format == "bmp":
                            img.save(output_path, "BMP")

                        # Remove original file
                        os.remove(input_path)
                        converted_files.append(output_path)

                        if self.console:
                            self.console.print(f"🔄 [dim]Converted {os.path.relpath(input_path, output_dir)} to {target_format}[/dim]")

            return len(converted_files) > 0
        except Exception as e:
            if self.console:
                self.console.print(f"⚠️ [yellow]Error converting format in {output_dir}: {str(e)}[/yellow]")
            return False

def run_enhancement(args):
    """Run the image enhancement process"""
    cli = MedGACLI()
    cli.start_time = time.time()

    # Print header and parameters
    cli.print_header()
    cli.print_parameters(args)

    # Validate MPI availability
    if hasattr(args, 'distributed') and args.distributed and not MPI_AVAILABLE:
        if cli.console:
            cli.console.print("❌ [red]MPI not available. Running in sequential mode.[/red]")
        else:
            print("❌ MPI not available. Running in sequential mode.")
        args.distributed = False

    # Find images to process
    to_process = []

    # Check if we have image or folder from interactive mode
    if hasattr(args, 'image') and args.image:
        if os.path.exists(args.image):
            to_process.append(args.image)
        else:
            print(f"❌ Error: Image file '{args.image}' not found.")
            return
    elif hasattr(args, 'folder') and args.folder:
        list_images = glob.glob(args.folder + os.sep + "*")
        for img_path in list_images:
            ext = img_path.split(".")[-1].lower()
            if ext in ["tiff", "tif", "png", "jpeg", "jpg", "bmp"] and os.path.exists(img_path):
                to_process.append(img_path)
    else:
        # If no image/folder specified, ask in interactive mode
        if cli.console:
            input_choice = Prompt.ask(
                "📁 Input type",
                choices=["single", "folder"],
                default="single"
            )

            if input_choice == "single":
                image_path = Prompt.ask("🖼️  Image file path")
                if os.path.exists(image_path):
                    to_process.append(image_path)
                else:
                    print(f"❌ Error: Image file '{image_path}' not found.")
                    return
            else:
                folder_path = Prompt.ask("📂 Folder path")
                list_images = glob.glob(folder_path + os.sep + "*")
                for img_path in list_images:
                    ext = img_path.split(".")[-1].lower()
                    if ext in ["tiff", "tif", "png", "jpeg", "jpg", "bmp"] and os.path.exists(img_path):
                        to_process.append(img_path)
        else:
            print("❌ Error: No input image or folder specified.")
            return

    if not to_process:
        print("❌ Error: No valid images found to process.")
        return

    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Process images
    total_times = []

    verbose = getattr(args, 'verbose', False)

    # For both mode, we need to track grayscale and color results separately
    if getattr(args, 'both', False):
        comparison_results = []

    if cli.console and verbose:
        # Use rich progress with live dashboard
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("[bold blue]{task.fields[image]}"),
            console=cli.console
        ) as progress:

            main_task = progress.add_task("🎨 Enhancing images...", total=len(to_process), image="")

            for i, image_path in enumerate(to_process):
                progress.update(main_task, advance=1, image=os.path.basename(image_path))

                # Determine output path
                filename = os.path.basename(image_path).split(".")[0]
                if getattr(args, 'both', False):
                    output_path = args.output + os.sep + filename + "_combined"
                else:
                    output_path = args.output + os.sep + filename

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # Process image
                start_time = time.time()

                if getattr(args, 'both', False):
                    # Process both modes and create comparison
                    grayscale_output = args.output + os.sep + filename + "_grayscale"
                    color_output = args.output + os.sep + filename + "_color"

                    if not os.path.exists(grayscale_output):
                        os.makedirs(grayscale_output)
                    if not os.path.exists(color_output):
                        os.makedirs(color_output)

                    # Process grayscale
                    progress.update(main_task, description=f"Processing {os.path.basename(image_path)} (Grayscale)")
                    medga_gray = MedGA(image_path, grayscale_output)
                    medga_gray.startGA(
                        args.population, args.generations, args.selection,
                        args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                        process_color=False
                    )

                    # Process color
                    progress.update(main_task, description=f"Processing {os.path.basename(image_path)} (Color)")
                    medga_color = MedGA(image_path, color_output)
                    medga_color.startGA(
                        args.population, args.generations, args.selection,
                        args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                        process_color=True
                    )

                    # Convert output formats if needed
                    if args.format != "png":  # MedGA defaults to PNG
                        cli._convert_output_format(grayscale_output, args.format)
                        cli._convert_output_format(color_output, args.format)

                    # Create comparison frame
                    progress.update(main_task, description=f"Creating comparison for {os.path.basename(image_path)}")

                    # Find the best ENHANCED images (not plots)
                    original_img = image_path
                    grayscale_best = cli._find_best_result_image(grayscale_output, args.format)
                    color_best = cli._find_best_result_image(color_output, args.format)

                    if original_img and grayscale_best and color_best:
                        comparison_path = output_path + os.sep + 'comparison_frame.png'
                        success = cli.create_comparison_frame(
                            original_img, grayscale_best, color_best, comparison_path,
                            f"MedGA Enhancement - {os.path.basename(image_path)}"
                        )
                        if success:
                            comparison_results.append(comparison_path)
                    else:
                        if cli.console:
                            cli.console.print(f"❌ [yellow]Could not find all enhanced images for comparison:[/yellow]")
                            cli.console.print(f"   [dim]Original: {os.path.basename(original_img) if original_img else 'MISSING'}[/dim]")
                            cli.console.print(f"   [dim]Grayscale: {os.path.basename(grayscale_best) if grayscale_best else 'MISSING'}[/dim]")
                            cli.console.print(f"   [dim]Color: {os.path.basename(color_best) if color_best else 'MISSING'}[/dim]")
                        else:
                            print(f"❌ Could not find all enhanced images for comparison:")
                            print(f"   Original: {os.path.basename(original_img) if original_img else 'MISSING'}")
                            print(f"   Grayscale: {os.path.basename(grayscale_best) if grayscale_best else 'MISSING'}")
                            print(f"   Color: {os.path.basename(color_best) if color_best else 'MISSING'}")

                else:
                    # Single mode processing
                    medga = MedGA(image_path, output_path)

                    if getattr(args, 'color', False):
                        medga.startGA(
                            args.population, args.generations, args.selection,
                            args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                            process_color=True
                        )
                    else:
                        medga.startGA(
                            args.population, args.generations, args.selection,
                            args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                            process_color=False
                        )

                    # Convert output formats if needed
                    if args.format != "png":  # MedGA defaults to PNG
                        cli._convert_output_format(output_path, args.format)

                elapsed = time.time() - start_time
                total_times.append(elapsed)
    else:
        # Simple processing without rich
        print("🔄 Processing images...")
        for i, image_path in enumerate(to_process):
            print(f"  📄 Processing {i+1}/{len(to_process)}: {os.path.basename(image_path)}")

            # Determine output path
            filename = os.path.basename(image_path).split(".")[0]
            if getattr(args, 'both', False):
                output_path = args.output + os.sep + filename + "_combined"
            else:
                output_path = args.output + os.sep + filename

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Process image
            start_time = time.time()

            if getattr(args, 'both', False):
                # Process both modes and create comparison
                grayscale_output = args.output + os.sep + filename + "_grayscale"
                color_output = args.output + os.sep + filename + "_color"

                if not os.path.exists(grayscale_output):
                    os.makedirs(grayscale_output)
                if not os.path.exists(color_output):
                    os.makedirs(color_output)

                # Process grayscale
                print(f"    ⚫ Processing grayscale version...")
                medga_gray = MedGA(image_path, grayscale_output)
                medga_gray.startGA(
                    args.population, args.generations, args.selection,
                    args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                    process_color=False
                )

                # Process color
                print(f"    🌈 Processing color version...")
                medga_color = MedGA(image_path, color_output)
                medga_color.startGA(
                    args.population, args.generations, args.selection,
                    args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                    process_color=True
                )

                # Convert output formats if needed
                if args.format != "png":  # MedGA defaults to PNG
                    print(f"    🔄 Converting grayscale output to {args.format}...")
                    cli._convert_output_format(grayscale_output, args.format)
                    print(f"    🔄 Converting color output to {args.format}...")
                    cli._convert_output_format(color_output, args.format)

                # Create comparison frame
                print(f"    🖼️  Creating comparison frame...")

                # Find the best ENHANCED images (not plots)
                original_img = image_path
                grayscale_best = cli._find_best_result_image(grayscale_output, args.format)
                color_best = cli._find_best_result_image(color_output, args.format)

                if original_img and grayscale_best and color_best:
                    comparison_path = output_path + os.sep + 'comparison_frame.png'
                    success = cli.create_comparison_frame(
                        original_img, grayscale_best, color_best, comparison_path,
                        f"MedGA Enhancement - {os.path.basename(image_path)}"
                    )
                    if success:
                        comparison_results.append(comparison_path)
                else:
                    print(f"    ❌ Could not find all enhanced images for comparison:")
                    print(f"       Original: {os.path.basename(original_img) if original_img else 'MISSING'}")
                    print(f"       Grayscale: {os.path.basename(grayscale_best) if grayscale_best else 'MISSING'}")
                    print(f"       Color: {os.path.basename(color_best) if color_best else 'MISSING'}")

            else:
                # Single mode processing
                medga = MedGA(image_path, output_path)

                if getattr(args, 'color', False):
                    medga.startGA(
                        args.population, args.generations, args.selection,
                        args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                        process_color=True
                    )
                else:
                    medga.startGA(
                        args.population, args.generations, args.selection,
                        args.cross_rate, args.mut_rate, args.elitism, args.pressure,
                        process_color=False
                    )

                # Convert output formats if needed
                if args.format != "png":  # MedGA defaults to PNG
                    print(f"    🔄 Converting output to {args.format}...")
                    cli._convert_output_format(output_path, args.format)

            elapsed = time.time() - start_time
            total_times.append(elapsed)
            print(f"  ✅ Completed in {elapsed:.2f}s")

    # Print completion summary
    total_time = time.time() - cli.start_time
    cli.print_completion_summary(total_time, len(to_process))

    # Show comparison results for both mode
    if getattr(args, 'both', False) and 'comparison_results' in locals() and comparison_results:
        if cli.console:
            cli.console.print("\n🎨 [bold cyan]COMPARISON FRAMES CREATED:[/bold cyan]")
            for result in comparison_results:
                cli.console.print(f"   📊 {result}")
        else:
            print("\n🎨 COMPARISON FRAMES CREATED:")
            for result in comparison_results:
                print(f"   📊 {result}")

def main():
    """
    Parse CLI arguments, validate them, and launch the MedGA workflow.

    The function wires together argparse-based input handling, optional
    interactive configuration, parameter validation, and the call into
    `run_enhancement`, acting as the canonical entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="🧬 MedGA - Medical Image Enhancement using Genetic Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 Examples:
  python MedGA.py -i input.jpg -g 1000 --both -v        # Enhance single image with comparison frame
  python MedGA.py -f images/ -g 500 --color --format png # Enhance folder in color as PNG
  python MedGA.py -i medical.png -g 2000 -p 200        # High population size
  python MedGA.py --interactive                        # Interactive mode

🎯 Processing Modes:
  --both    : Process both grayscale and color with comparison frame
  --color   : Process only color version
  (default) : Process only grayscale version

🖼️ Output Options:
  --format FORMAT : Output image format (jpg, png, tiff, bmp) - default: jpg

⚡ Distributed Processing (MPI):
  -d, --distributed : Enable MPI parallel processing
  -t, --cores       : Number of MPI processes (default: 4)

📊 Output:
  • Enhanced images at different generations (resized to match input)
  • Fitness progression plots
  • Comparison frames (with --both, always PNG)
  • Algorithm performance metrics
        """
    )

    # Input/Output options
    io_group = parser.add_argument_group('📁 Input/Output Options')
    io_mutex = io_group.add_mutually_exclusive_group(required=False)
    io_mutex.add_argument('-i', '--image', help='Input image file path')
    io_mutex.add_argument('-f', '--folder', help='Input folder containing images')
    io_group.add_argument('-o', '--output', default='output', help='Output directory (default: output)')
    io_group.add_argument('--format', choices=['jpg', 'png', 'tiff', 'bmp'], default='jpg',
                         help='Output image format (default: jpg)')

    # Genetic Algorithm parameters
    ga_group = parser.add_argument_group('⚙️ Genetic Algorithm Parameters')
    ga_group.add_argument('-p', '--population', type=int, default=100,
                         help='Population size (default: 100)')
    ga_group.add_argument('-g', '--generations', type=int, default=100,
                         help='Number of generations (default: 100)')
    ga_group.add_argument('-s', '--selection', choices=['tournament', 'wheel', 'ranking'],
                         default='tournament', help='Selection method (default: tournament)')
    ga_group.add_argument('-c', '--cross-rate', type=float, default=0.9,
                         help='Crossover rate (default: 0.9)', dest='cross_rate')
    ga_group.add_argument('-m', '--mut-rate', type=float, default=0.01,
                         help='Mutation rate (default: 0.01)', dest='mut_rate')
    ga_group.add_argument('-k', '--pressure', type=int, default=20,
                         help='Tournament pressure (default: 20)')
    ga_group.add_argument('-e', '--elitism', type=int, default=1,
                         help='Elitism count (default: 1)')

    # Processing modes
    mode_group = parser.add_argument_group('🎨 Processing Modes')
    mode_mutex = mode_group.add_mutually_exclusive_group()
    mode_mutex.add_argument('--color', action='store_true',
                           help='Process color images in color')
    mode_mutex.add_argument('--both', action='store_true',
                           help='Process both grayscale and color with comparison frame')

    # Distributed processing options
    dist_group = parser.add_argument_group('⚡ Distributed Processing (MPI)')
    dist_group.add_argument('-d', '--distributed', action='store_true',
                           help='Enable MPI distributed processing')
    dist_group.add_argument('-t', '--cores', type=int, default=4,
                           help='Number of MPI cores (default: 4)')

    # Additional options
    other_group = parser.add_argument_group('🔧 Additional Options')
    other_group.add_argument('-v', '--verbose', action='store_true',
                            help='Enable verbose output')
    other_group.add_argument('--interactive', action='store_true',
                           help='Launch interactive configuration mode')
    other_group.add_argument('--version', action='version',
                           version='MedGA 2.0 - Modern CLI Edition')

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        cli = MedGACLI()
        cli.print_header()
        interactive_args = cli.interactive_setup()

        # Merge interactive args with command line args
        for key, value in vars(interactive_args).items():
            setattr(args, key, value)

    # Validate input - only if not in interactive mode and no input provided
    if not args.interactive and not args.image and not args.folder:
        # If no input provided and not interactive, ask if user wants interactive mode
        if RICH_AVAILABLE:
            console = Console()
            if Confirm.ask("🎛️  No input specified. Launch interactive mode?"):
                cli = MedGACLI()
                cli.print_header()
                interactive_args = cli.interactive_setup()
                for key, value in vars(interactive_args).items():
                    setattr(args, key, value)
            else:
                parser.error("Either --image, --folder, or --interactive must be provided")
        else:
            parser.error("Either --image, --folder, or --interactive must be provided")

    # Validate parameters
    if args.population <= 0:
        parser.error("Population size must be positive")
    if args.generations <= 0:
        parser.error("Number of generations must be positive")
    if not (0 <= args.cross_rate <= 1):
        parser.error("Crossover rate must be between 0 and 1")
    if not (0 <= args.mut_rate <= 1):
        parser.error("Mutation rate must be between 0 and 1")
    if args.cores <= 0:
        parser.error("Number of cores must be positive")

    try:
        run_enhancement(args)
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {str(e)}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
