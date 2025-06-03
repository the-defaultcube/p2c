import numpy as np
from typing import Tuple, Dict, Any
from .utils import save_image, load_image

class BilinearInterpolator:
    def __init__(self):
        """
        Initialize the BilinearInterpolator for performing bilinear interpolation.
        """
        pass

    def interpolate(self, image: np.ndarray) -> np.ndarray:
        """
        Perform bilinear interpolation on the input image.

        Args:
            image: NumPy array representing the input image with Bayer pattern applied.

        Returns:
            NumPy array representing the interpolated image.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in RGB format.")

        height, width = image.shape[:2]
        interpolated = np.zeros_like(image)

        # Separate the channels
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        # Bilinear interpolation for each channel
        for channel in ['r', 'g', 'b']:
            if channel == 'r':
                ch = r
                interp = np.zeros_like(ch)
            elif channel == 'g':
                ch = g
                interp = np.zeros_like(ch)
            else:
                ch = b
                interp = np.zeros_like(ch)

            # Iterate over each pixel
            for i in range(height):
                for j in range(width):
                    if ch[i, j] == 0:  # Indicates missing value
                        # Find the nearest non-zero neighbors
                        neighbors = []
                        for di in [-1, 1]:
                            for dj in [-1, 1]:
                                if 0 <= i + di < height and 0 <= j + dj < width:
                                    if ch[i + di, j + dj] != 0:
                                        neighbors.append(ch[i + di, j + dj])
                        if neighbors:
                            interp[i, j] = np.mean(neighbors)
                        else:
                            interp[i, j] = ch[i, j]  # Keep as 0 if no neighbors
                    else:
                        interp[i, j] = ch[i, j]

            # Reassign the interpolated channel
            if channel == 'r':
                r = interp
            elif channel == 'g':
                g = interp
            else:
                b = interp

        # Reconstruct the image
        interpolated[:, :, 0] = r
        interpolated[:, :, 1] = g
        interpolated[:, :, 2] = b

        return interpolated

class GradientCorrectedInterpolator:
    def __init__(self, alpha: float = 0.5, beta: float = 0.625, gamma: float = 0.75):
        """
        Initialize the GradientCorrectedInterpolator with gain factors.

        Args:
            alpha: Gain factor for R channel correction.
            beta: Gain factor for G channel correction.
            gamma: Gain factor for B channel correction.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_gradients(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients for each color channel using surrounding pixels.

        Args:
            image: NumPy array representing the input image.

        Returns:
            Dictionary containing gradient arrays for R, G, B channels.
        """
        gradients = {}

        # Compute gradients for each channel
        for channel in ['r', 'g', 'b']:
            ch = image[:, :, ['r' == channel][0]]
            gradient = np.zeros_like(ch)

            # Use a 5x5 neighborhood for gradient calculation
            for i in range(ch.shape[0]):
                for j in range(ch.shape[1]):
                    sum_grad = 0.0
                    count = 0
                    for di in [-2, -1, 0, 1, 2]:
                        for dj in [-2, -1, 0, 1, 2]:
                            if di == 0 and dj == 0:
                                continue  # Skip the current pixel
                            ni, nj = i + di, j + dj
                            if 0 <= ni < ch.shape[0] and 0 <= nj < ch.shape[1]:
                                sum_grad += abs(ch[ni, nj] - ch[i, j])
                                count += 1
                    if count > 0:
                        gradient[i, j] = sum_grad / count
                    else:
                        gradient[i, j] = 0.0  # Handle edge cases

            gradients[channel] = gradient

        return gradients

    def interpolate(self, image: np.ndarray) -> np.ndarray:
        """
        Perform gradient-corrected interpolation on the input image.

        Args:
            image: NumPy array representing the input image with Bayer pattern applied.

        Returns:
            NumPy array representing the gradient-corrected interpolated image.
        """
        # Perform bilinear interpolation
        bilinear = BilinearInterpolator().interpolate(image)

        # Compute gradients
        gradients = self.compute_gradients(image)

        # Apply gradient correction
        corrected = np.zeros_like(bilinear)

        # Apply correction for each channel
        for channel in ['r', 'g', 'b']:
            bilinear_ch = bilinear[:, :, ['r' == channel][0]]
            gradient_ch = gradients[channel]
            gain = getattr(self, channel[0].upper() + 'Gain')  # Get alpha, beta, gamma

            # Apply correction
            correction = gain * gradient_ch
            corrected_ch = bilinear_ch + correction

            # Ensure values are within valid range
            np.clip(corrected_ch, 0.0, 1.0, out=corrected_ch)

            corrected[:, :, ['r' == channel][0]] = corrected_ch

        return corrected

    def __repr__(self) -> str:
        """
        Return a string representation of the interpolator.

        Returns:
            String with class name and gain factors.
        """
        return f"GradientCorrectedInterpolator(α={self.alpha}, β={self.beta}, γ={self.gamma})"
