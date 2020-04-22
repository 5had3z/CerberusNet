#ifndef __NVX__PIPELINE__
#define __NVX__PIPELINE__

#include <NVX/nvx.h>
#include <string>
#include <opencv2/core/core.hpp>

class PROCESSOR_NVX
{
private:
    vx_context m_context;
    vx_graph m_graph;
    vx_image m_raw_full, m_raw_left, m_raw_right;
    vx_image m_demo_left, m_demo_right;
    vx_image m_rectified_left, m_rectified_right;
    vx_matrix m_xmap_right, m_xmap_left, m_ymap_left, m_ymap_right;

    vx_rectangle_t m_fullRect;
    cv::Mat left_x, left_y, right_x, right_y;

public:
    PROCESSOR_NVX(int width, int height);
    ~PROCESSOR_NVX();
    void ProcessFrame(cv::Mat doubleProcessFrame);
};

#endif