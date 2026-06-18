from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sem_preprocess import SemImageAligner, TiltImage as CppTiltImage
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    logger.warning("C++ sem_preprocess module not available, using Python fallback")


@dataclass
class TiltImage:
    image: np.ndarray
    tilt_angle: float
    rotation_angle: float = 0.0
    pixel_size_nm: float = 1.0

    def __post_init__(self):
        if self.image.ndim == 2:
            pass
        elif self.image.ndim == 3 and self.image.shape[2] == 1:
            self.image = self.image.squeeze(axis=2)


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    drift_x_nm: float = 0.0
    drift_y_nm: float = 0.0
    correlation_score: float = 0.0
    transform_matrix: Optional[np.ndarray] = None


class _PythonAligner:
    def __init__(self):
        self.ref_image: Optional[np.ndarray] = None
        self.ref_tilt: float = 0.0
        self.ref_set: bool = False

    def set_reference_image(self, ref_image: np.ndarray, ref_tilt: float = 0.0) -> None:
        if ref_image.ndim == 3:
            import cv2
            gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = ref_image.copy()
        self.ref_image = gray.astype(np.float32)
        self.ref_tilt = ref_tilt
        self.ref_set = True

    def _ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)
        return float(np.mean(img1_norm * img2_norm))

    def _compute_affine(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        max_iterations: int = 500,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        import cv2

        warp_matrix = np.eye(2, 3, dtype=np.float64)
        height, width = dst.shape[:2]

        best_score = -1.0
        best_warp = warp_matrix.copy()
        step_size = 1.0
        prev_score = -1.0

        for _ in range(max_iterations):
            improved = False

            for param_idx in range(6):
                row = param_idx // 3
                col = param_idx % 3

                original = warp_matrix[row, col]

                warp_matrix[row, col] = original + step_size
                warped_plus = cv2.warpAffine(
                    src, warp_matrix, (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                score_plus = self._ncc(warped_plus, dst)

                warp_matrix[row, col] = original - step_size
                warped_minus = cv2.warpAffine(
                    src, warp_matrix, (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                score_minus = self._ncc(warped_minus, dst)

                if score_plus > best_score and score_plus > score_minus:
                    best_score = score_plus
                    best_warp = warp_matrix.copy()
                    best_warp[row, col] = original + step_size
                    improved = True
                elif score_minus > best_score:
                    best_score = score_minus
                    best_warp = warp_matrix.copy()
                    best_warp[row, col] = original - step_size
                    improved = True

                warp_matrix[row, col] = original

            if improved:
                warp_matrix = best_warp.copy()

            if abs(best_score - prev_score) < epsilon and not improved:
                step_size *= 0.5
                if step_size < 0.001:
                    break
            prev_score = best_score

        return warp_matrix

    def align_image(
        self,
        target_image: np.ndarray,
        target_tilt: float,
        max_iterations: int = 500,
        epsilon: float = 1e-6,
    ) -> AlignmentResult:
        import cv2

        if target_image.ndim == 3:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_image.copy()

        if not self.ref_set:
            self.set_reference_image(target_gray, target_tilt)
            return AlignmentResult(
                aligned_image=target_gray,
                drift_x_nm=0.0,
                drift_y_nm=0.0,
                correlation_score=1.0,
                transform_matrix=np.eye(2, 3, dtype=np.float64),
            )

        target_float = target_gray.astype(np.float32)

        warp_matrix = self._compute_affine(
            target_float, self.ref_image, max_iterations, epsilon
        )

        height, width = self.ref_image.shape[:2]
        aligned = cv2.warpAffine(
            target_gray, warp_matrix, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        aligned_float = aligned.astype(np.float32)
        score = self._ncc(aligned_float, self.ref_image)

        return AlignmentResult(
            aligned_image=aligned,
            drift_x_nm=float(warp_matrix[0, 2]),
            drift_y_nm=float(warp_matrix[1, 2]),
            correlation_score=score,
            transform_matrix=warp_matrix,
        )

    def align_tilt_series(self, tilt_series: List[TiltImage]) -> List[np.ndarray]:
        import cv2

        if not tilt_series:
            return []

        ref_idx = min(
            range(len(tilt_series)),
            key=lambda i: abs(tilt_series[i].tilt_angle),
        )

        self.set_reference_image(tilt_series[ref_idx].image, tilt_series[ref_idx].tilt_angle)

        results: List[np.ndarray] = []
        for i, ti in enumerate(tilt_series):
            if i == ref_idx:
                if ti.image.ndim == 3:
                    gray = cv2.cvtColor(ti.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = ti.image.copy()
                results.append(gray)
            else:
                result = self.align_image(ti.image, ti.tilt_angle)
                results.append(result.aligned_image)

        return results

    def build_multi_channel_tensor(
        self,
        aligned_images: List[np.ndarray],
        tilt_angles: List[float],
    ) -> np.ndarray:
        if not aligned_images:
            raise ValueError("No aligned images provided")

        rows, cols = aligned_images[0].shape[:2]
        channels = len(aligned_images)

        tensor = np.zeros((rows, cols, channels), dtype=np.float32)

        for c in range(channels):
            img = aligned_images[c].astype(np.float32)
            angle = tilt_angles[c]
            tilt_factor = 1.0 / np.cos(np.radians(angle))
            tensor[:, :, c] = img * tilt_factor

        return tensor

    def estimate_drift(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pixel_size_nm: float = 1.0,
    ) -> Tuple[float, float]:
        import cv2

        if img1.ndim == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1

        if img2.ndim == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2

        warp = self._compute_affine(
            img2_gray.astype(np.float32),
            img1_gray.astype(np.float32),
            max_iterations=200,
            epsilon=1e-5,
        )

        dx = warp[0, 2] * pixel_size_nm
        dy = warp[1, 2] * pixel_size_nm

        return float(dx), float(dy)


class ImageAligner:
    def __init__(self, use_cpp: Optional[bool] = None):
        if use_cpp is None:
            use_cpp = _CPP_AVAILABLE
        self.use_cpp = use_cpp and _CPP_AVAILABLE

        if self.use_cpp:
            self._aligner = SemImageAligner()
        else:
            self._aligner = _PythonAligner()

    @property
    def cpp_available(self) -> bool:
        return _CPP_AVAILABLE

    def set_reference_image(self, ref_image: np.ndarray, ref_tilt: float = 0.0) -> None:
        self._aligner.set_reference_image(ref_image, ref_tilt)

    def align_image(
        self,
        target_image: np.ndarray,
        target_tilt: float,
        max_iterations: int = 500,
        epsilon: float = 1e-6,
    ) -> AlignmentResult:
        result = self._aligner.align_image(
            target_image, target_tilt, max_iterations, epsilon
        )
        if self.use_cpp:
            aligned, dx, dy, score = result
            return AlignmentResult(
                aligned_image=aligned,
                drift_x_nm=dx,
                drift_y_nm=dy,
                correlation_score=score,
                transform_matrix=None,
            )
        return result

    def align_tilt_series(self, tilt_series: List[TiltImage]) -> List[np.ndarray]:
        if self.use_cpp:
            images = [ti.image for ti in tilt_series]
            angles = [ti.tilt_angle for ti in tilt_series]
            return self._aligner.align_tilt_series(images, angles)
        return self._aligner.align_tilt_series(tilt_series)

    def build_multi_channel_tensor(
        self,
        aligned_images: List[np.ndarray],
        tilt_angles: List[float],
    ) -> np.ndarray:
        if self.use_cpp:
            return self._aligner.build_multi_channel_tensor(aligned_images, tilt_angles)
        return self._aligner.build_multi_channel_tensor(aligned_images, tilt_angles)

    def estimate_drift(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pixel_size_nm: float = 1.0,
    ) -> Tuple[float, float]:
        if self.use_cpp:
            return self._aligner.estimate_drift(img1, img2, pixel_size_nm)
        return self._aligner.estimate_drift(img1, img2, pixel_size_nm)

    def apply_bilateral_filter(
        self,
        img: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> np.ndarray:
        import cv2
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    def adaptive_histogram_equalization(
        self,
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: int = 8,
    ) -> np.ndarray:
        import cv2
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        return clahe.apply(img)

    def remove_beam_drift_noise(
        self,
        img: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        import cv2
        smoothed = cv2.medianBlur(img, kernel_size)
        residual = img.astype(np.float32) - smoothed.astype(np.float32)
        threshold = 3.0 * np.std(residual)
        mask = np.abs(residual) > threshold
        result = img.astype(np.float32).copy()
        result[mask] = smoothed.astype(np.float32)[mask]
        return result.astype(np.uint8)
