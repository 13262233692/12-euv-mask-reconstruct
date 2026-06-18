#ifndef SEM_PREPROCESS_H
#define SEM_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>

namespace euv {

struct TiltImage {
    cv::Mat image;
    double tilt_angle;
    double rotation_angle;
    double pixel_size_nm;
};

struct AlignmentResult {
    cv::Mat aligned_image;
    cv::Mat transform_matrix;
    double drift_x_nm;
    double drift_y_nm;
    double correlation_score;
};

class SemImageAligner {
public:
    SemImageAligner();
    ~SemImageAligner();

    void set_reference_image(const cv::Mat& ref_image, double ref_tilt = 0.0);

    AlignmentResult align_image(
        const cv::Mat& target_image,
        double target_tilt,
        int max_iterations = 500,
        double epsilon = 1e-6
    );

    std::vector<cv::Mat> align_tilt_series(
        const std::vector<TiltImage>& tilt_series
    );

    cv::Mat build_multi_channel_tensor(
        const std::vector<cv::Mat>& aligned_images,
        const std::vector<double>& tilt_angles
    );

    std::tuple<double, double> estimate_drift(
        const cv::Mat& img1,
        const cv::Mat& img2,
        double pixel_size_nm = 1.0
    );

private:
    cv::Mat ref_image_;
    double ref_tilt_;
    bool ref_set_;

    cv::Mat compute_affine_transform(
        const cv::Mat& src,
        const cv::Mat& dst,
        int max_iterations,
        double epsilon
    );

    cv::Mat preprocess_for_alignment(const cv::Mat& img);

    double normalized_cross_correlation(
        const cv::Mat& img1,
        const cv::Mat& img2
    );
};

cv::Mat apply_bilateral_filter(const cv::Mat& img, int d = 9, double sigma_color = 75, double sigma_space = 75);
cv::Mat adaptive_histogram_equalization(const cv::Mat& img, double clip_limit = 2.0, int tile_grid_size = 8);
cv::Mat remove_beam_drift_noise(const cv::Mat& img, int kernel_size = 3);

}

#endif
