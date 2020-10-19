#include "demosaic_node.hpp"

#include <cuda_runtime_api.h>
#include <nppcore.h>
#include <nppi.h>

#define KERNEL_DEMOSAIC_NAME "usr.nvx.demosaic"

// Kernel implementation
static vx_status VX_CALLBACK demosaic_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 2)
        return VX_FAILURE;

    vx_image src = (vx_image)parameters[0];
    vx_image dst = (vx_image)parameters[1];

    vx_status status = VX_SUCCESS;

    // Get CUDA stream, which is used for current node

    cudaStream_t stream = NULL;
    vxQueryNode(node, NVX_NODE_CUDA_STREAM, &stream, sizeof(stream));

    // Use this stream for NPP launch
    nppSetStream(stream);

    // Map OpenVX data objects into CUDA device memory

    vx_rectangle_t rect = {};
    vxGetValidRegionImage(src, &rect);

    vx_map_id src_map_id;
    vx_uint8* src_ptr;
    vx_imagepatch_addressing_t src_addr;
    status = vxMapImagePatch(src, &rect, 0, &src_map_id, &src_addr, (void **)&src_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)src, status, "[%s:%u] Failed to access \'src\' in Demosaic Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    vx_map_id dst_map_id;
    vx_uint8* dst_ptr;
    vx_imagepatch_addressing_t dst_addr;
    status = vxMapImagePatch(dst, &rect, 0, &dst_map_id, &dst_addr, (void **)&dst_ptr, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)dst, status, "[%s:%u] Failed to access \'dst\' in Demosaic Kernel", __FUNCTION__, __LINE__);
        vxUnmapImagePatch(src, src_map_id);
        return status;
    }

    // Call NPP function

    NppiSize oSizeROI;
    oSizeROI.width = src_addr.dim_x;
    oSizeROI.height = src_addr.dim_y;

    NppiRect oRectROI;
    oRectROI.x = 0;
    oRectROI.y = 0;
    oRectROI.width = src_addr.dim_x;
    oRectROI.height = src_addr.dim_y;

    //nppiCFAToRGB_8u_C1C3R_Ctx includes stream context but I think we're doing that with ln24 "nppSetStream(stream)" anyway?
    NppStatus npp_status = nppiCFAToRGB_8u_C1C3R(src_ptr, oRectROI.width*sizeof(uchar1), oSizeROI, oRectROI,
                            dst_ptr, oRectROI.width*sizeof(uchar3), NPPI_BAYER_RGGB, NPPI_INTER_LINEAR);

    if (npp_status != NPP_SUCCESS)
    {
        vxAddLogEntry((vx_reference)node, VX_FAILURE, "[%s:%u] nppiCFAToRGB_8u_C1C3R error", __FUNCTION__, __LINE__);
        status = VX_FAILURE;
    }

    // Unmap OpenVX data objects from CUDA device memory

    vxUnmapImagePatch(src, src_map_id);
    vxUnmapImagePatch(dst, dst_map_id);

    return status;
}

// Parameter validator
static vx_status VX_CALLBACK demosaic_validate(vx_node, const vx_reference parameters[],
                                                vx_uint32 num_params, vx_meta_format metas[])
{
    if (num_params != 2) return VX_ERROR_INVALID_PARAMETERS;

    vx_image src = (vx_image)parameters[0];

    vx_df_image src_format = 0;
    vxQueryImage(src, VX_IMAGE_ATTRIBUTE_FORMAT, &src_format, sizeof(src_format));

    vx_uint32 src_width = 0, src_height = 0;
    vxQueryImage(src, VX_IMAGE_ATTRIBUTE_WIDTH, &src_width, sizeof(src_width));
    vxQueryImage(src, VX_IMAGE_ATTRIBUTE_HEIGHT, &src_height, sizeof(src_height));

    vx_status status = VX_SUCCESS;

    if (src_format != VX_DF_IMAGE_U8)
    {
        status = VX_ERROR_INVALID_FORMAT;
        vxAddLogEntry((vx_reference)src, status, "[%s:%u] Invalid format for \'src\' in Demosaic Kernel, it should be VX_DF_IMAGE_U8", __FUNCTION__, __LINE__);
    }

    vx_meta_format dst_meta = metas[1];
    vx_df_image dst_format = VX_DF_IMAGE_RGB;
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_FORMAT, &dst_format, sizeof(dst_format));
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_WIDTH, &src_width, sizeof(src_width));
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &src_height, sizeof(src_height));

    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerdemosaicKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to allocate an ID for the Demosaic kernel");
        return status;
    }

    vx_kernel kernel = vxAddUserKernel(context, "gpu:" KERNEL_DEMOSAIC_NAME, id,
                                       demosaic_kernel,
                                       2,    // numParams
                                       demosaic_validate,
                                       NULL, // init
                                       NULL  // deinit
                                       );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create Demosaic Kernel");
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT , VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED); // src
    status |= vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED); // dst

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize Demosaic Kernel parameters");
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize Demosaic Kernel");
        return VX_FAILURE;
    }

    return status;
}



vx_node demosaicNode(vx_graph graph, vx_image src, vx_image dst)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByName(vxGetContext((vx_reference)graph), KERNEL_DEMOSAIC_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)src);
            vxSetParameterByIndex(node, 1, (vx_reference)dst);
        }
    }

    return node;
}