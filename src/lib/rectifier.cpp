#include "rectifier.hpp"
#include "camera_properties.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <NVX/nvx_opencv_interop.hpp>

void get_rectify_map(vx_matrix *xmap_left, vx_matrix *ymap_left, vx_matrix *xmap_right, vx_matrix *ymap_right)
{
    cv::Mat r1, r2, p1, p2, disparity_to_depth, left_m1, left_m2, right_m1, right_m2;

    auto halfsize = cv::Size(camera_properties::cam_width, camera_properties::cam_height);

    cv::Mat stereo_rot;
    cv::Rodrigues(cv::Vec<double, 3>(camera_properties::stereo_RX, camera_properties::stereo_RY, camera_properties::stereo_RZ), stereo_rot);

    auto stereo_translation = cv::Vec<double, 3>(camera_properties::stereo_TX, camera_properties::stereo_TY, camera_properties::stereo_TZ);

    cv::Mat const left_mat = (cv::Mat_<double>(3, 3) << camera_properties::left_fx, 0, camera_properties::left_cx,
                              0, camera_properties::left_fy, camera_properties::left_cy,
                              0, 0, 1);
    cv::Mat const left_dist = (cv::Mat_<double>(4, 1) << camera_properties::left_k1, camera_properties::left_k2, 0, 0);

    cv::Mat const rght_mat = (cv::Mat_<double>(3, 3) << camera_properties::rght_fx, 0, camera_properties::rght_cx,
                              0, camera_properties::rght_fy, camera_properties::rght_cy,
                              0, 0, 1);
    cv::Mat const rght_dist = (cv::Mat_<double>(4, 1) << camera_properties::rght_k1, camera_properties::rght_k2, 0, 0);

    // Initialise our stereo matrices.
    cv::stereoRectify(
        left_mat,
        left_dist,
        rght_mat,
        rght_dist,
        halfsize,
        stereo_rot,
        stereo_translation,
        r1,
        r2,
        p1,
        p2,
        disparity_to_depth);

    cv::initUndistortRectifyMap(
        left_mat,
        left_dist,
        r1,
        p1,
        halfsize,
        CV_32F,
        left_m1,
        left_m2);

    cv::initUndistortRectifyMap(
        rght_mat,
        rght_dist,
        r2,
        p2,
        halfsize,
        CV_32F,
        right_m1,
        right_m2);

    nvx_cv::copyCVMatToVXMatrix(left_m1, *xmap_left);
    nvx_cv::copyCVMatToVXMatrix(left_m2, *ymap_left);
    nvx_cv::copyCVMatToVXMatrix(right_m1, *xmap_right);
    nvx_cv::copyCVMatToVXMatrix(right_m2, *ymap_right);
}