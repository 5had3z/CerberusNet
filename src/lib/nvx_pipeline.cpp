#include "nvx_pipeline.hpp"
#include "rectifier.hpp"
#include "nvx_kernels/rectification_node.hpp"

#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui.hpp>
#include <NVX/nvx_opencv_interop.hpp>
#include <iostream>

#define VX_CHECK_STATUS(s)                                                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
    const auto status = (s);                                                                                           \
    if(status != VX_SUCCESS)                                                                                           \
    {                                                                                                                  \
      std::cout << "NVX STATUS ERROR: " << __FILE__ << "\tline: " << __LINE__ << "\tERROR: " <<  status;                                                                               \
    }                                                                                                                  \
    assert(status == VX_SUCCESS);                                                                                      \
  } while(false)

#define VX_CHECK_REFERENCE(s)  VX_CHECK_STATUS(vxGetStatus((vx_reference)(s)));

static void VX_CALLBACK NVXLogCallback(vx_context /*context*/, vx_reference /*ref*/, vx_status /*status*/, const vx_char string[])
{
    std::cout << "NVX ERROR: " << string;
}

PROCESSOR_NVX::PROCESSOR_NVX(int width, int height) :
    m_context(nullptr),
    m_graph(nullptr)
{    
    // Build Pipeline
    m_context = vxCreateContext();
    // Enables tracking of performance of each node
    // vxDirective((vx_reference)m_context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
    VX_CHECK_REFERENCE(m_context);

    m_graph = nvxCreateStreamGraph(m_context);
    VX_CHECK_REFERENCE(m_graph);

    const char* option = "-O3";
    VX_CHECK_STATUS( vxSetGraphAttribute(m_graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

    vxRegisterLogCallback(m_context, &NVXLogCallback, vx_false_e);

    // Create combined image
    m_raw_full = vxCreateImage(m_context, width * 2, height, VX_DF_IMAGE_RGB);
    VX_CHECK_REFERENCE(m_raw_full);

    // Extract ROI dims of Full Image
    vxGetValidRegionImage(m_raw_full, &m_fullRect);

    // Define ROI for LHS and RHS Images
    vx_rectangle_t leftRect;
    leftRect.start_x = m_fullRect.start_x;
    leftRect.start_y = m_fullRect.start_y;
    leftRect.end_x = m_fullRect.end_x / 2;
    leftRect.end_y = m_fullRect.end_y;

    vx_rectangle_t rightRect;
    rightRect.start_x = leftRect.end_x;
    rightRect.start_y = m_fullRect.start_y;
    rightRect.end_x = m_fullRect.end_x;
    rightRect.end_y = m_fullRect.end_y;
   
    // Create LHS and RHS Frames from Full Image
    m_raw_left = vxCreateImageFromROI(m_raw_full, &leftRect);
    VX_CHECK_REFERENCE(m_raw_left);
    m_raw_right = vxCreateImageFromROI(m_raw_full, &rightRect);
    VX_CHECK_REFERENCE(m_raw_right);

    // Rectified Images
    m_rectified_left = vxCreateImage(m_context, width, height, VX_DF_IMAGE_RGB);
    VX_CHECK_REFERENCE(m_rectified_left);
    m_rectified_right = vxCreateImage(m_context, width, height, VX_DF_IMAGE_RGB);
    VX_CHECK_REFERENCE(m_rectified_right);

    // Remapping Matricies
    m_xmap_left = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_REFERENCE(m_xmap_left);
    m_ymap_left = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_REFERENCE(m_ymap_left);

    m_xmap_right = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_REFERENCE(m_xmap_right);
    m_ymap_right = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_REFERENCE(m_ymap_right);

    // Register Custom Kernels
    registerRectificationKernel(m_context);

    // Rectify raw camera images
    m_rectify_left_node = rectificationNode(m_graph, m_raw_left, m_xmap_left, m_ymap_left, m_rectified_left);
    VX_CHECK_STATUS(vxVerifyGraph(m_graph));
    m_rectify_right_node = rectificationNode(m_graph, m_raw_right, m_xmap_right, m_ymap_right, m_rectified_right);
    VX_CHECK_STATUS(vxVerifyGraph(m_graph));

    // Add Rectification Maps to vx_matrix
    get_rectify_map(&m_xmap_left, &m_ymap_left, &m_xmap_right, &m_ymap_right);
}

PROCESSOR_NVX::~PROCESSOR_NVX()
{
    vxReleaseGraph(&m_graph);
    vxReleaseContext(&m_context);

    // Release Remapping Matricies
    vxReleaseMatrix(&m_xmap_right);
    vxReleaseMatrix(&m_ymap_right);
    vxReleaseMatrix(&m_xmap_left);
    vxReleaseMatrix(&m_ymap_left);

    // Release Images
    vxReleaseImage(&m_raw_full);
    vxReleaseImage(&m_raw_left);
    vxReleaseImage(&m_raw_right);
    vxReleaseImage(&m_rectified_left);
    vxReleaseImage(&m_rectified_right);

    // Release Nodes
    vxReleaseNode(&m_rectify_left_node);
    vxReleaseNode(&m_rectify_right_node);
}  

void PROCESSOR_NVX::ProcessFrame(cv::Mat doubleProcessFrame)
{
    // ROI Image Source
    vx_imagepatch_addressing_t src_addr;
    src_addr.dim_x = doubleProcessFrame.cols;
    src_addr.dim_y = doubleProcessFrame.rows;
    src_addr.stride_x = static_cast<vx_int32>(doubleProcessFrame.elemSize());
    src_addr.stride_y = static_cast<vx_int32>(doubleProcessFrame.step);
    
    VX_CHECK_STATUS( vxCopyImagePatch(m_raw_full, &m_fullRect, 0, &src_addr, doubleProcessFrame.data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );

    VX_CHECK_STATUS( vxProcessGraph(m_graph) );
}

void PROCESSOR_NVX::DisplayFrame()
{
    cv::Mat leftHost = nvx_cv::VXImageToCVMatMapper(m_rectified_left, 0, NULL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST).getMat();
    cv::Mat rightHost = nvx_cv::VXImageToCVMatMapper(m_rectified_right, 0, NULL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST).getMat();
    cv::Mat combined_frame = (0.25 * leftHost) + (0.75 * rightHost);

    cv::imshow("window", combined_frame);
    cv::waitKey(1);
}

PROCESSOR_DISPARITY_FULL::PROCESSOR_DISPARITY_FULL(int width, int height) :
  PROCESSOR_NVX(width, height)
{
  //  Disparity range to look for
  constexpr vx_uint32 min_disparity = 0;
  constexpr vx_uint32 max_disparity = 128;
  constexpr vx_uint32 full_D = max_disparity - min_disparity;

  //  Sum Absolute Difference Window
  constexpr vx_uint32 sad_win_size = 5;

  //  Hamming Cost Window
  constexpr vx_uint32 hc_win_size = 1;

  // Census Transform Window Size
  constexpr vx_int32 ct_win_size = 5;

  //  Divisors at each level
  const int D_divisors[pyr_levels] = { 4, 2, 1 };

  //  BT-cost clip value
  constexpr vx_int32 bt_clip_value = 31;

  // discontinuity penalties
  constexpr vx_int32 P1 = 8;
  constexpr vx_int32 P2 = 109;

  vx_enum scanlines_mask = 85;
  constexpr vx_int32 uniqueness_ratio = 0;
  constexpr vx_int32 max_diff = 320000;

  // Allocate full buffers
  {
      m_left_gray = vxCreateVirtualImage(m_graph, width, height, VX_DF_IMAGE_U8);
      VX_CHECK_REFERENCE(m_left_gray);

      m_right_gray = vxCreateVirtualImage(m_graph, width, height, VX_DF_IMAGE_U8);
      VX_CHECK_REFERENCE(m_right_gray);

      full_convolved_cost_ = vxCreateVirtualImage(m_graph, width * full_D / 4, height, VX_DF_IMAGE_U8);
      VX_CHECK_REFERENCE(full_convolved_cost_);

      if (sad_win_size > 1)
      {
          full_cost_ = vxCreateVirtualImage(m_graph, width * full_D / 4, height, VX_DF_IMAGE_U8);
          VX_CHECK_REFERENCE(full_cost_);
      }

      full_aggregated_cost_ = vxCreateVirtualImage(m_graph, width * full_D / 4, height, VX_DF_IMAGE_S16);
      VX_CHECK_REFERENCE(full_aggregated_cost_);

      for (int i = 0; i < pyr_levels; i++)
      {
          int divisor = 1 << i;
          disparity_short_[i] = vxCreateVirtualImage(m_graph,
                                                      width / divisor,
                                                      height / divisor,
                                                      VX_DF_IMAGE_S16);
          VX_CHECK_REFERENCE(disparity_short_[i]);
      }
  }

  left_cvt_color_node_ = vxColorConvertNode(m_graph, m_rectified_left, m_left_gray);
  VX_CHECK_REFERENCE(left_cvt_color_node_);

  right_cvt_color_node_ = vxColorConvertNode(m_graph, m_rectified_right, m_right_gray);
  VX_CHECK_REFERENCE(right_cvt_color_node_);


    for (int i = pyr_levels - 1; i >= 0; i--)
    {
        int divisor = 1 << i;

        int pyr_width = width / divisor;
        int pyr_height = height / divisor;
        int D = full_D / D_divisors[i];

        vx_image left_gray = vxCreateVirtualImage(m_graph, width, pyr_height, VX_DF_IMAGE_U8);
        VX_CHECK_REFERENCE(left_gray);

        vx_image right_gray = vxCreateVirtualImage(m_graph, pyr_width, pyr_height, VX_DF_IMAGE_U8);
        VX_CHECK_REFERENCE(right_gray);

        vx_node left_downscale_node = vxScaleImageNode(m_graph, m_left_gray, left_gray, VX_INTERPOLATION_TYPE_BILINEAR);
        VX_CHECK_REFERENCE(left_downscale_node);

        vx_node right_downscale_node = vxScaleImageNode(m_graph, m_right_gray, right_gray, VX_INTERPOLATION_TYPE_BILINEAR);
        VX_CHECK_REFERENCE(right_downscale_node);

        // apply census transform, if requested
        vx_image left_census = NULL, right_census = NULL;
        if (ct_win_size > 1)
        {
            left_census =  vxCreateVirtualImage(m_graph, pyr_width, pyr_height, VX_DF_IMAGE_U32);
            VX_CHECK_REFERENCE(left_census);
            right_census = vxCreateVirtualImage(m_graph, pyr_width, pyr_height, VX_DF_IMAGE_U32);
            VX_CHECK_REFERENCE(right_census);

            vx_node left_census_node = nvxCensusTransformNode(m_graph, left_gray, left_census, ct_win_size);
            VX_CHECK_REFERENCE(left_census_node);
            vx_node right_census_node = nvxCensusTransformNode(m_graph, right_gray, right_census, ct_win_size);
            VX_CHECK_REFERENCE(right_census_node);
        }

        vx_rectangle_t cost_rect { 0, 0, static_cast<vx_uint32>(pyr_width * D), static_cast<vx_uint32>(pyr_height) };
        convolved_cost_[i] = vxCreateImageFromROI(full_convolved_cost_, &cost_rect);
        VX_CHECK_REFERENCE(convolved_cost_[i]);

        if (sad_win_size > 1)
        {
            cost_[i] = vxCreateImageFromROI(full_cost_, &cost_rect);
            VX_CHECK_REFERENCE(cost_[i]);

            // census transformed images should be compared by hamming cost
            vx_node compute_cost_node = NULL;
            if (ct_win_size > 1)
            {
                compute_cost_node = nvxComputeCostHammingNode
                    (m_graph, left_census, right_census,
                      cost_[i],
                      min_disparity / D_divisors[i], max_disparity / D_divisors[i],
                      hc_win_size);
            }
            else
            {
                compute_cost_node = nvxComputeModifiedCostBTNode
                    (m_graph, left_gray, right_gray,
                      cost_[i],
                      min_disparity / D_divisors[i], max_disparity / D_divisors[i],
                      bt_clip_value);
            }
            VX_CHECK_REFERENCE(compute_cost_node);

            vx_node convolve_cost_node = nvxConvolveCostNode
                (m_graph,
                  cost_[i], convolved_cost_[i],
                  D, sad_win_size);
            VX_CHECK_REFERENCE(convolve_cost_node);
        }
        else
        {
            vx_node compute_cost_node = NULL;
            if (ct_win_size > 1)
            {
                compute_cost_node = nvxComputeCostHammingNode
                    (m_graph, left_census, right_census,
                      convolved_cost_[i],
                      min_disparity / D_divisors[i], max_disparity / D_divisors[i],
                      hc_win_size);
            }
            else
            {
                compute_cost_node = nvxComputeModifiedCostBTNode
                    (m_graph, left_gray, right_gray,
                      convolved_cost_[i],
                      min_disparity / D_divisors[i], max_disparity / D_divisors[i],
                      bt_clip_value);
            }
            VX_CHECK_REFERENCE(compute_cost_node);
        }

        if (i < pyr_levels - 1)
        {
            vx_node cost_prior_node = nvxPSGMCostPriorNode
                (m_graph, disparity_short_[i+1],
                  convolved_cost_[i],
                  D);
            VX_CHECK_REFERENCE(cost_prior_node);
        }

        aggregated_cost_[i] = vxCreateImageFromROI(full_aggregated_cost_, &cost_rect);
        VX_CHECK_REFERENCE(aggregated_cost_[i]);

        vx_node aggregate_cost_scanlines_node = nvxAggregateCostScanlinesNode
            (m_graph,
              convolved_cost_[i], aggregated_cost_[i],
              D, P1, P2, scanlines_mask);
        VX_CHECK_REFERENCE(aggregate_cost_scanlines_node);

        vx_node compute_disparity_node = nvxComputeDisparityNode
            (m_graph,
              aggregated_cost_[i],
              disparity_short_[i],
              min_disparity / D_divisors[i], max_disparity / D_divisors[i],
              uniqueness_ratio, max_diff);
        VX_CHECK_REFERENCE(compute_disparity_node);

        if (i < pyr_levels - 1)
        {
            vx_node disparity_merge_node = nvxPSGMDisparityMergeNode
                (m_graph,
                  disparity_short_[i+1],
                  disparity_short_[i], D);
            VX_CHECK_REFERENCE(disparity_merge_node);
        }
    }

    vx_int32 shift = 4;
    vx_scalar s_shift = vxCreateScalar(m_context, VX_TYPE_INT32, &shift);
    VX_CHECK_REFERENCE(s_shift);
    vx_node convert_depth_node = vxConvertDepthNode
        (m_graph, disparity_short_[0],
          m_disparity, VX_CONVERT_POLICY_SATURATE, s_shift);
    vxReleaseScalar(&s_shift);
    VX_CHECK_REFERENCE(convert_depth_node);

    VX_CHECK_STATUS( vxVerifyGraph(m_graph) );

}

PROCESSOR_DISPARITY_FULL::~PROCESSOR_DISPARITY_FULL()
{
  // Release Images
  vxReleaseImage(&m_disparity);
  vxReleaseImage(&m_left_gray);
  vxReleaseImage(&m_right_gray);

  vxReleaseImage(&full_aggregated_cost_);
  vxReleaseImage(&full_cost_);
  vxReleaseImage(&full_convolved_cost_);

  for (uint i = 0; i < pyr_levels; i++)
  {
    vxReleaseImage(&disparity_short_[i]);
    vxReleaseImage(&aggregated_cost_[i]);
    vxReleaseImage(&cost_[i]);
    vxReleaseImage(&convolved_cost_[i]);
  }

  vxReleaseNode(&left_cvt_color_node_);
  vxReleaseNode(&right_cvt_color_node_);
}

void PROCESSOR_DISPARITY_FULL::DisplayFrame()
{
    cv::imshow("window", nvx_cv::VXImageToCVMatMapper(m_disparity, 0, NULL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST).getMat());
    cv::waitKey(1);
}