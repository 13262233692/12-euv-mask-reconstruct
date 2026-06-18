#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include "sem_preprocess.h"
#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace euv;

cv::Mat numpy_to_mat(const py::array_t<unsigned char>& array) {
    py::buffer_info buf = array.request();

    if (buf.ndim == 2) {
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        return cv::Mat(rows, cols, CV_8UC1, (unsigned char*)buf.ptr).clone();
    } else if (buf.ndim == 3) {
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int channels = buf.shape[2];
        int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
        return cv::Mat(rows, cols, type, (unsigned char*)buf.ptr).clone();
    }
    throw std::runtime_error("Unsupported array dimensions");
}

cv::Mat numpy_float_to_mat(const py::array_t<float>& array) {
    py::buffer_info buf = array.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    if (buf.ndim == 2) {
        return cv::Mat(rows, cols, CV_32FC1, (float*)buf.ptr).clone();
    } else if (buf.ndim == 3) {
        int channels = buf.shape[2];
        return cv::Mat(rows, cols, CV_32FC(channels), (float*)buf.ptr).clone();
    }
    throw std::runtime_error("Unsupported array dimensions");
}

py::array_t<unsigned char> mat_to_numpy_8u(const cv::Mat& mat) {
    cv::Mat temp;
    if (mat.depth() != CV_8U) {
        mat.convertTo(temp, CV_8U);
    } else {
        temp = mat;
    }

    if (temp.channels() == 1) {
        ssize_t rows = temp.rows;
        ssize_t cols = temp.cols;
        auto result = py::array_t<unsigned char>({rows, cols});
        py::buffer_info buf = result.request();
        unsigned char* ptr = static_cast<unsigned char*>(buf.ptr);
        for (int i = 0; i < rows; ++i) {
            std::memcpy(ptr + i * cols, temp.ptr<unsigned char>(i), cols);
        }
        return result;
    } else {
        ssize_t rows = temp.rows;
        ssize_t cols = temp.cols;
        ssize_t channels = temp.channels();
        auto result = py::array_t<unsigned char>({rows, cols, channels});
        py::buffer_info buf = result.request();
        unsigned char* ptr = static_cast<unsigned char*>(buf.ptr);
        for (int i = 0; i < rows; ++i) {
            std::memcpy(ptr + i * cols * channels, temp.ptr<unsigned char>(i), cols * channels);
        }
        return result;
    }
}

py::array_t<float> mat_to_numpy_float(const cv::Mat& mat) {
    cv::Mat temp;
    if (mat.depth() != CV_32F) {
        mat.convertTo(temp, CV_32F);
    } else {
        temp = mat;
    }

    if (temp.channels() == 1) {
        ssize_t rows = temp.rows;
        ssize_t cols = temp.cols;
        auto result = py::array_t<float>({rows, cols});
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        for (int i = 0; i < rows; ++i) {
            std::memcpy(ptr + i * cols, temp.ptr<float>(i), cols * sizeof(float));
        }
        return result;
    } else {
        ssize_t rows = temp.rows;
        ssize_t cols = temp.cols;
        ssize_t channels = temp.channels();
        auto result = py::array_t<float>({rows, cols, channels});
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        for (int i = 0; i < rows; ++i) {
            std::memcpy(ptr + i * cols * channels, temp.ptr<float>(i), cols * channels * sizeof(float));
        }
        return result;
    }
}

PYBIND11_MODULE(sem_preprocess, m) {
    m.doc() = "SEM image preprocessing module for EUV mask defect reconstruction";

    py::class_<TiltImage>(m, "TiltImage")
        .def(py::init<>())
        .def_readwrite("image", &TiltImage::image)
        .def_readwrite("tilt_angle", &TiltImage::tilt_angle)
        .def_readwrite("rotation_angle", &TiltImage::rotation_angle)
        .def_readwrite("pixel_size_nm", &TiltImage::pixel_size_nm);

    py::class_<AlignmentResult>(m, "AlignmentResult")
        .def(py::init<>())
        .def_readwrite("aligned_image", &AlignmentResult::aligned_image)
        .def_readwrite("transform_matrix", &AlignmentResult::transform_matrix)
        .def_readwrite("drift_x_nm", &AlignmentResult::drift_x_nm)
        .def_readwrite("drift_y_nm", &AlignmentResult::drift_y_nm)
        .def_readwrite("correlation_score", &AlignmentResult::correlation_score);

    py::class_<SemImageAligner>(m, "SemImageAligner")
        .def(py::init<>())
        .def("set_reference_image", [](SemImageAligner& self, const py::array_t<unsigned char>& ref_image, double ref_tilt) {
            cv::Mat mat = numpy_to_mat(ref_image);
            self.set_reference_image(mat, ref_tilt);
        }, py::arg("ref_image"), py::arg("ref_tilt") = 0.0)
        .def("align_image", [](SemImageAligner& self, const py::array_t<unsigned char>& target_image,
                                double target_tilt, int max_iterations, double epsilon) {
            cv::Mat target = numpy_to_mat(target_image);
            AlignmentResult result = self.align_image(target, target_tilt, max_iterations, epsilon);
            return py::make_tuple(
                mat_to_numpy_8u(result.aligned_image),
                result.drift_x_nm,
                result.drift_y_nm,
                result.correlation_score
            );
        }, py::arg("target_image"), py::arg("target_tilt"),
           py::arg("max_iterations") = 500, py::arg("epsilon") = 1e-6)
        .def("align_tilt_series", [](SemImageAligner& self,
                                      const std::vector<py::array_t<unsigned char>>& images,
                                      const std::vector<double>& tilt_angles) {
            std::vector<TiltImage> tilt_series;
            for (size_t i = 0; i < images.size(); ++i) {
                TiltImage ti;
                ti.image = numpy_to_mat(images[i]);
                ti.tilt_angle = tilt_angles[i];
                tilt_series.push_back(ti);
            }
            std::vector<cv::Mat> aligned = self.align_tilt_series(tilt_series);
            std::vector<py::array_t<unsigned char>> results;
            for (const auto& img : aligned) {
                results.push_back(mat_to_numpy_8u(img));
            }
            return results;
        })
        .def("build_multi_channel_tensor", [](SemImageAligner& self,
                                              const std::vector<py::array_t<unsigned char>>& aligned_images,
                                              const std::vector<double>& tilt_angles) {
            std::vector<cv::Mat> images;
            for (const auto& img : aligned_images) {
                images.push_back(numpy_to_mat(img));
            }
            cv::Mat tensor = self.build_multi_channel_tensor(images, tilt_angles);
            return mat_to_numpy_float(tensor);
        })
        .def("estimate_drift", [](SemImageAligner& self,
                                    const py::array_t<unsigned char>& img1,
                                    const py::array_t<unsigned char>& img2,
                                    double pixel_size_nm) {
            cv::Mat m1 = numpy_to_mat(img1);
            cv::Mat m2 = numpy_to_mat(img2);
            auto [dx, dy] = self.estimate_drift(m1, m2, pixel_size_nm);
            return py::make_tuple(dx, dy);
        }, py::arg("img1"), py::arg("img2"), py::arg("pixel_size_nm") = 1.0);

    m.def("apply_bilateral_filter", [](const py::array_t<unsigned char>& img,
                                       int d, double sigma_color, double sigma_space) {
        cv::Mat mat = numpy_to_mat(img);
        cv::Mat result = apply_bilateral_filter(mat, d, sigma_color, sigma_space);
        return mat_to_numpy_8u(result);
    }, py::arg("img"), py::arg("d") = 9, py::arg("sigma_color") = 75.0, py::arg("sigma_space") = 75.0);

    m.def("adaptive_histogram_equalization", [](const py::array_t<unsigned char>& img,
                                                double clip_limit, int tile_grid_size) {
        cv::Mat mat = numpy_to_mat(img);
        cv::Mat result = adaptive_histogram_equalization(mat, clip_limit, tile_grid_size);
        return mat_to_numpy_8u(result);
    }, py::arg("img"), py::arg("clip_limit") = 2.0, py::arg("tile_grid_size") = 8);

    m.def("remove_beam_drift_noise", [](const py::array_t<unsigned char>& img, int kernel_size) {
        cv::Mat mat = numpy_to_mat(img);
        cv::Mat result = remove_beam_drift_noise(mat, kernel_size);
        return mat_to_numpy_8u(result);
    }, py::arg("img"), py::arg("kernel_size") = 3);
}
