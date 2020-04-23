#ifndef __NVX__PIPELINE__
#define __NVX__PIPELINE__

#include <NVX/nvx.h>
#include <string>
#include <opencv2/core/core.hpp>

class PROCESSOR_NVX
{
protected:
    vx_context m_context;
    vx_graph m_graph;
    vx_image m_raw_full, m_raw_left, m_raw_right;
    vx_image m_rectified_left, m_rectified_right;
    vx_matrix m_xmap_right, m_xmap_left, m_ymap_left, m_ymap_right;

    vx_rectangle_t m_fullRect;
    cv::Mat left_x, left_y, right_x, right_y;

    vx_node m_rectify_left_node, m_rectify_right_node;
public:
    PROCESSOR_NVX(int width, int height);
    ~PROCESSOR_NVX();
    
    void ProcessFrame(cv::Mat doubleProcessFrame);
    virtual void DisplayFrame();
};

class PROCESSOR_DISPARITY_FULL : public PROCESSOR_NVX
{
private:
    static const int pyr_levels = 4;
    vx_node left_cvt_color_node_;
    vx_node right_cvt_color_node_;

    vx_image disparity_short_[pyr_levels];
    vx_image aggregated_cost_[pyr_levels];
    vx_image cost_[pyr_levels];
    vx_image convolved_cost_[pyr_levels];

    vx_image m_left_gray;
    vx_image m_right_gray;
    vx_image m_disparity;

    vx_image full_aggregated_cost_;
    vx_image full_cost_;
    vx_image full_convolved_cost_;

public:
    PROCESSOR_DISPARITY_FULL(int width, int height);
    ~PROCESSOR_DISPARITY_FULL();

    void DisplayFrame() override;
};

#endif