#include "sem_preprocess.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <algorithm>

namespace euv {

SemImageAligner::SemImageAligner()
    : ref_tilt_(0.0), ref_set_(false) {}

SemImageAligner::~SemImageAligner() {}

void SemImageAligner::set_reference_image(const cv::Mat& ref_image, double ref_tilt) {
    ref_image_ = ref_image.clone();
    if (ref_image_.channels() > 1) {
        cv::cvtColor(ref_image_, ref_image_, cv::COLOR_BGR2GRAY);
    }
    ref_image_.convertTo(ref_image_, CV_32F);
    ref_tilt_ = ref_tilt;
    ref_set_ = true;
}

cv::Mat SemImageAligner::preprocess_for_alignment(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() > 1) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    gray.convertTo(gray, CV_32F);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    cv::Mat equalized;
    double min_val, max_val;
    cv::minMaxLoc(blurred, &min_val, &max_val);
    if (max_val > min_val) {
        equalized = (blurred - min_val) / (max_val - min_val);
    } else {
        equalized = blurred.clone();
    }

    return equalized;
}

double SemImageAligner::normalized_cross_correlation(
    const cv::Mat& img1,
    const cv::Mat& img2
) {
    cv::Mat mean1, mean2, std1, std2;
    cv::meanStdDev(img1, mean1, std1);
    cv::meanStdDev(img2, mean2, std2);

    double s1 = std1.at<double>(0);
    double s2 = std2.at<double>(0);

    if (s1 < 1e-8 || s2 < 1e-8) return 0.0;

    cv::Mat normalized1 = (img1 - mean1.at<double>(0)) / s1;
    cv::Mat normalized2 = (img2 - mean2.at<double>(0)) / s2;

    cv::Mat product = normalized1.mul(normalized2);
    double ncc = cv::mean(product).val[0];

    return std::max(-1.0, std::min(1.0, ncc));
}

cv::Mat SemImageAligner::compute_affine_transform(
    const cv::Mat& src,
    const cv::Mat& dst,
    int max_iterations,
    double epsilon
) {
    cv::Mat src_float = preprocess_for_alignment(src);
    cv::Mat dst_float = preprocess_for_alignment(dst);

    int width = dst.cols;
    int height = dst.rows;

    cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_64F);

    double prev_score = -1.0;
    double best_score = -1.0;
    cv::Mat best_warp = warp_matrix.clone();

    double step_size = 1.0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        bool improved = false;

        for (int param = 0; param < 6; ++param) {
            int row = param / 3;
            int col = param % 3;

            double original = warp_matrix.at<double>(row, col);

            warp_matrix.at<double>(row, col) = original + step_size;
            cv::Mat warped_plus;
            cv::warpAffine(src_float, warped_plus, warp_matrix, cv::Size(width, height),
                          cv::INTER_LINEAR, cv::BORDER_REFLECT);
            double score_plus = normalized_cross_correlation(warped_plus, dst_float);

            warp_matrix.at<double>(row, col) = original - step_size;
            cv::Mat warped_minus;
            cv::warpAffine(src_float, warped_minus, warp_matrix, cv::Size(width, height),
                          cv::INTER_LINEAR, cv::BORDER_REFLECT);
            double score_minus = normalized_cross_correlation(warped_minus, dst_float);

            if (score_plus > best_score && score_plus > score_minus) {
                best_score = score_plus;
                best_warp = warp_matrix.clone();
                best_warp.at<double>(row, col) = original + step_size;
                improved = true;
            } else if (score_minus > best_score) {
                best_score = score_minus;
                best_warp = warp_matrix.clone();
                best_warp.at<double>(row, col) = original - step_size;
                improved = true;
            }

            warp_matrix.at<double>(row, col) = original;
        }

        if (improved) {
            warp_matrix = best_warp.clone();
        }

        double current_score = best_score;
        if (std::abs(current_score - prev_score) < epsilon && !improved) {
            step_size *= 0.5;
            if (step_size < 0.001) break;
        }
        prev_score = current_score;
    }

    return warp_matrix;
}

AlignmentResult SemImageAligner::align_image(
    const cv::Mat& target_image,
    double target_tilt,
    int max_iterations,
    double epsilon
) {
    AlignmentResult result;

    if (!ref_set_) {
        set_reference_image(target_image, target_tilt);
        result.aligned_image = target_image.clone();
        result.transform_matrix = cv::Mat::eye(2, 3, CV_64F);
        result.drift_x_nm = 0.0;
        result.drift_y_nm = 0.0;
        result.correlation_score = 1.0;
        return result;
    }

    double tilt_ratio = std::cos(target_tilt * CV_PI / 180.0) /
                       std::cos(ref_tilt_ * CV_PI / 180.0);

    cv::Mat warp_matrix = compute_affine_transform(
        target_image, ref_image_, max_iterations, epsilon
    );

    cv::Mat target_gray;
    if (target_image.channels() > 1) {
        cv::cvtColor(target_image, target_gray, cv::COLOR_BGR2GRAY);
    } else {
        target_gray = target_image.clone();
    }

    int width = ref_image_.cols;
    int height = ref_image_.rows;

    cv::Mat aligned;
    cv::warpAffine(target_gray, aligned, warp_matrix, cv::Size(width, height),
                  cv::INTER_CUBIC, cv::BORDER_REFLECT);

    result.aligned_image = aligned;
    result.transform_matrix = warp_matrix.clone();
    result.drift_x_nm = warp_matrix.at<double>(0, 2);
    result.drift_y_nm = warp_matrix.at<double>(1, 2);

    cv::Mat aligned_float;
    aligned.convertTo(aligned_float, CV_32F);
    result.correlation_score = normalized_cross_correlation(aligned_float, ref_image_);

    return result;
}

std::vector<cv::Mat> SemImageAligner::align_tilt_series(
    const std::vector<TiltImage>& tilt_series
) {
    std::vector<cv::Mat> results;

    if (tilt_series.empty()) return results;

    size_t ref_idx = 0;
    double min_abs_tilt = std::abs(tilt_series[0].tilt_angle);
    for (size_t i = 1; i < tilt_series.size(); ++i) {
        if (std::abs(tilt_series[i].tilt_angle) < min_abs_tilt) {
            min_abs_tilt = std::abs(tilt_series[i].tilt_angle);
            ref_idx = i;
        }
    }

    set_reference_image(tilt_series[ref_idx].image, tilt_series[ref_idx].tilt_angle);

    for (size_t i = 0; i < tilt_series.size(); ++i) {
        if (i == ref_idx) {
            cv::Mat gray;
            if (tilt_series[i].image.channels() > 1) {
                cv::cvtColor(tilt_series[i].image, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = tilt_series[i].image.clone();
            }
            results.push_back(gray);
        } else {
            AlignmentResult r = align_image(
                tilt_series[i].image,
                tilt_series[i].tilt_angle
            );
            results.push_back(r.aligned_image);
        }
    }

    return results;
}

cv::Mat SemImageAligner::build_multi_channel_tensor(
    const std::vector<cv::Mat>& aligned_images,
    const std::vector<double>& tilt_angles
) {
    if (aligned_images.empty()) return cv::Mat();

    int rows = aligned_images[0].rows;
    int cols = aligned_images[0].cols;
    int channels = static_cast<int>(aligned_images.size());

    cv::Mat tensor(rows, cols, CV_32FC(channels));

    for (int c = 0; c < channels; ++c) {
        cv::Mat gray;
        if (aligned_images[c].depth() != CV_32F) {
            aligned_images[c].convertTo(gray, CV_32F);
        } else {
            gray = aligned_images[c].clone();
        }

        double angle = tilt_angles[c];
        double tilt_factor = 1.0 / std::cos(angle * CV_PI / 180.0);

        cv::Mat weighted = gray * tilt_factor;

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                tensor.ptr<cv::Vec<float, 1>>(y, x)[c] =
                    weighted.at<float>(y, x);
            }
        }
    }

    return tensor;
}

std::tuple<double, double> SemImageAligner::estimate_drift(
    const cv::Mat& img1,
    const cv::Mat& img2,
    double pixel_size_nm
) {
    cv::Mat warp = compute_affine_transform(img2, img1, 200, 1e-5);

    double dx_px = warp.at<double>(0, 2);
    double dy_px = warp.at<double>(1, 2);

    return std::make_tuple(dx_px * pixel_size_nm, dy_px * pixel_size_nm);
}

cv::Mat apply_bilateral_filter(const cv::Mat& img, int d, double sigma_color, double sigma_space) {
    cv::Mat result;
    cv::bilateralFilter(img, result, d, sigma_color, sigma_space);
    return result;
}

cv::Mat adaptive_histogram_equalization(const cv::Mat& img, double clip_limit, int tile_grid_size) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(tile_grid_size, tile_grid_size));
    cv::Mat result;
    clahe->apply(img, result);
    return result;
}

cv::Mat remove_beam_drift_noise(const cv::Mat& img, int kernel_size) {
    cv::Mat smoothed;
    cv::medianBlur(img, smoothed, kernel_size);

    cv::Mat float_img, float_smoothed;
    img.convertTo(float_img, CV_32F);
    smoothed.convertTo(float_smoothed, CV_32F);

    cv::Mat residual = float_img - float_smoothed;

    cv::Scalar mean_res, std_res;
    cv::meanStdDev(residual, mean_res, std_res);

    float threshold = static_cast<float>(3.0 * std_res.val[0]);

    cv::Mat mask = cv::abs(residual) > threshold;

    cv::Mat result = float_img.clone();
    float_smoothed.copyTo(result, mask);

    cv::Mat result_8u;
    result.convertTo(result_8u, CV_8U);

    return result_8u;
}

}
