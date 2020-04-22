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
    assert(status == VX_SUCCESS);                                                                                  \
  } while(false)

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
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_context));

    m_graph = vxCreateGraph(m_context);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_graph));

    const char* option = "-O3";
    VX_CHECK_STATUS( vxSetGraphAttribute(m_graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

    vxRegisterLogCallback(m_context, &NVXLogCallback, vx_false_e);

    // Create combined image
    m_raw_full = vxCreateImage(m_context, width * 2, height, VX_DF_IMAGE_RGB);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_raw_full));

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
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_raw_left));
    m_raw_right = vxCreateImageFromROI(m_raw_full, &rightRect);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_raw_right));

    // Rectified Images
    m_rectified_left = vxCreateImage(m_context, width, height, VX_DF_IMAGE_RGB);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_rectified_left));
    m_rectified_right = vxCreateImage(m_context, width, height, VX_DF_IMAGE_RGB);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_rectified_right));

    // Remapping Matricies
    m_xmap_left = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_xmap_left));
    m_ymap_left = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_ymap_left));

    m_xmap_right = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_xmap_right));
    m_ymap_right = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, width, height);
    VX_CHECK_STATUS(vxGetStatus((vx_reference)m_ymap_right));

    // Register Custom Kernels
    registerRectificationKernel(m_context);

    // Rectify raw camera images
    vx_node rectify_left_node = rectificationNode(m_graph, m_raw_left, m_xmap_left, m_ymap_left, m_rectified_left);
    VX_CHECK_STATUS(vxVerifyGraph(m_graph));
    vx_node rectify_right_node = rectificationNode(m_graph, m_raw_right, m_xmap_right, m_ymap_right, m_rectified_right);
    VX_CHECK_STATUS(vxVerifyGraph(m_graph));

    // Release Nodes
    vxReleaseNode(&rectify_left_node);
    vxReleaseNode(&rectify_right_node);

    // Add Rectification Maps to vx_matrix
    get_rectify_map(&m_xmap_left, &m_ymap_left, &m_xmap_right, &m_ymap_right);
}

PROCESSOR_NVX::~PROCESSOR_NVX()
{
    vxReleaseGraph(&m_graph);
    vxReleaseContext(&m_context);
    vxReleaseMatrix(&m_xmap_right);
    vxReleaseMatrix(&m_ymap_right);
    vxReleaseMatrix(&m_xmap_left);
    vxReleaseMatrix(&m_ymap_left);
    vxReleaseImage(&m_raw_full);
    vxReleaseImage(&m_raw_left);
    vxReleaseImage(&m_raw_right);
    vxReleaseImage(&m_demo_right);
    vxReleaseImage(&m_demo_left);
    vxReleaseImage(&m_rectified_left);
    vxReleaseImage(&m_rectified_right);
}  

void PROCESSOR_NVX::ProcessFrame(cv::Mat doubleProcessFrame)
{
    /* Main callback to do inference,stereo and any other processing on frame */

    // ROI Image Source
    vx_imagepatch_addressing_t src_addr;
    src_addr.dim_x = doubleProcessFrame.cols;
    src_addr.dim_y = doubleProcessFrame.rows;
    src_addr.stride_x = static_cast<vx_int32>(doubleProcessFrame.elemSize());
    src_addr.stride_y = static_cast<vx_int32>(doubleProcessFrame.step);
    
    VX_CHECK_STATUS( vxCopyImagePatch(m_raw_full, &m_fullRect, 0, &src_addr, doubleProcessFrame.data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );

    VX_CHECK_STATUS( vxProcessGraph(m_graph) );

    // Validate Pipeline is working
    cv::Mat leftHost = nvx_cv::VXImageToCVMatMapper(m_rectified_left, 0, NULL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST).getMat();
    cv::Mat rightHost = nvx_cv::VXImageToCVMatMapper(m_rectified_right, 0, NULL, VX_READ_ONLY, VX_MEMORY_TYPE_HOST).getMat();
    cv::Mat combined_frame = (0.25 * leftHost) + (0.75 * rightHost);

    cv::imshow("window", combined_frame);
    cv::waitKey(1);
}