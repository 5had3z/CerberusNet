#include "rectification_node.hpp"

#include <cuda_runtime_api.h>
#include <nppcore.h>
#include <nppi.h>

#define KERNEL_RECTIFICATION_NAME "usr.nvx.rectification"

// Kernel implementation
static vx_status VX_CALLBACK rectification_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    if (num != 4)
        return VX_FAILURE;

    vx_image src = (vx_image)parameters[0];
    vx_matrix xmap = (vx_matrix)parameters[1];
    vx_matrix ymap = (vx_matrix)parameters[2];
    vx_image dst = (vx_image)parameters[3];

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
    vx_uint8* src_ptr = nullptr;
    vx_imagepatch_addressing_t src_addr;
    status = vxMapImagePatch(src, &rect, 0, &src_map_id, &src_addr, (void **)&src_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)src, status, "[%s:%u] Failed to access \'src\' in Rectificaton Kernel", __FUNCTION__, __LINE__);
        return status;
    }

    
    vx_map_id xmap_map_id;
    vx_float32* xmap_ptr = nullptr;
    status = nvxMapMatrix(xmap, &xmap_map_id, (void **)&xmap_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)xmap, status, "[%s:%u] Failed to access \'xmap\' in Rectificaton Kernel", __FUNCTION__, __LINE__);
        vxUnmapImagePatch(src, src_map_id);
        return status;
    }

    vx_map_id ymap_map_id;
    vx_float32* ymap_ptr = nullptr;
    status = nvxMapMatrix(ymap, &ymap_map_id, (void **)&ymap_ptr, VX_READ_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)ymap, status, "[%s:%u] Failed to access \'ymap\' in Rectificaton Kernel", __FUNCTION__, __LINE__);
        nvxUnmapMatrix(xmap, xmap_map_id);
        vxUnmapImagePatch(src, src_map_id);
        return status;
    }

    vx_map_id dst_map_id;
    vx_uint8* dst_ptr = nullptr;
    vx_imagepatch_addressing_t dst_addr;
    status = vxMapImagePatch(dst, &rect, 0, &dst_map_id, &dst_addr, (void **)&dst_ptr, VX_WRITE_ONLY, NVX_MEMORY_TYPE_CUDA, 0);

    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)dst, status, "[%s:%u] Failed to access \'dst\' in Rectificaton Kernel", __FUNCTION__, __LINE__);
        nvxUnmapMatrix(ymap, ymap_map_id);
        nvxUnmapMatrix(xmap, xmap_map_id);
        vxUnmapImagePatch(src, src_map_id);
        return status;
    }

    // Call NPP function

    NppiSize oSizeROI = { src_addr.dim_x, src_addr.dim_y };

    NppiRect oRectROI = { 0, 0, src_addr.dim_x, src_addr.dim_y };

    //nppiRemap_8u_C3R_Ctx includes stream context but I think we're doing that with ln24 "nppSetStream(stream)" anyway?
    NppStatus npp_status = nppiRemap_8u_C3R((Npp8u *)src_ptr, oSizeROI, src_addr.stride_y, oRectROI,
                                    (Npp32f *)xmap_ptr, oRectROI.width*sizeof(Npp32f),
                                    (Npp32f *)ymap_ptr, oRectROI.width*sizeof(Npp32f),
                                    (Npp8u *)dst_ptr, dst_addr.stride_y, oSizeROI, NPPI_INTER_LINEAR);

    if (npp_status != NPP_SUCCESS)
    {
        vxAddLogEntry((vx_reference)node, VX_FAILURE, "[%s:%u] nppiRemap_8u_C3R error", __FUNCTION__, __LINE__);
        status = VX_FAILURE;
    }

    // Unmap OpenVX data objects from CUDA device memory

    vxUnmapImagePatch(src, src_map_id);
    nvxUnmapMatrix(xmap, xmap_map_id);
    nvxUnmapMatrix(ymap, ymap_map_id);
    vxUnmapImagePatch(dst, dst_map_id);

    return status;
}

// Parameter validator
static vx_status VX_CALLBACK rectification_validate(vx_node, const vx_reference parameters[],
                                                vx_uint32 num_params, vx_meta_format metas[])
{
    if (num_params != 4) return VX_ERROR_INVALID_PARAMETERS;

    vx_image src = (vx_image)parameters[0];
    vx_matrix xmap = (vx_matrix)parameters[1];
    vx_matrix ymap = (vx_matrix)parameters[2];

    vx_df_image src_format = 0;
    vxQueryImage(src, VX_IMAGE_ATTRIBUTE_FORMAT, &src_format, sizeof(src_format));

    vx_uint32 src_width = 0, src_height = 0;
    vxQueryImage(src, VX_IMAGE_ATTRIBUTE_WIDTH, &src_width, sizeof(src_width));
    vxQueryImage(src, VX_IMAGE_ATTRIBUTE_HEIGHT, &src_height, sizeof(src_height));

    vx_enum xmap_format = 0;
    vx_size xmap_width = 0, xmap_height = 0;
    vxQueryMatrix(xmap, VX_MATRIX_TYPE, &xmap_format, sizeof(xmap_format));
    vxQueryMatrix(xmap, VX_MATRIX_COLUMNS, &xmap_width, sizeof(xmap_width));
    vxQueryMatrix(xmap, VX_MATRIX_ROWS, &xmap_height, sizeof(xmap_height));

    vx_enum ymap_format = 0;
    vx_size ymap_width = 0, ymap_height = 0;
    vxQueryMatrix(ymap, VX_MATRIX_TYPE, &ymap_format, sizeof(ymap_format));
    vxQueryMatrix(ymap, VX_MATRIX_COLUMNS, &ymap_width, sizeof(ymap_width));
    vxQueryMatrix(ymap, VX_MATRIX_ROWS, &ymap_height, sizeof(ymap_height));

    vx_status status = VX_SUCCESS;

    if (src_format != VX_DF_IMAGE_RGB)
    {
        status = VX_ERROR_INVALID_FORMAT;
        vxAddLogEntry((vx_reference)src, status, "[%s:%u] Invalid format for \'src\' in Rectificaton Kernel, it should be VX_DF_IMAGE_U8", __FUNCTION__, __LINE__);
    }

    if (xmap_format != VX_TYPE_FLOAT32)
    {
        status = VX_ERROR_INVALID_TYPE;
        vxAddLogEntry((vx_reference)xmap, status, "[%s:%u] Invalid format for \'xmap\' in Rectificaton Kernel, it should be VX_TYPE_FLOAT32", __FUNCTION__, __LINE__);
    }

    if (ymap_format != VX_TYPE_FLOAT32)
    {
        status = VX_ERROR_INVALID_TYPE;
        vxAddLogEntry((vx_reference)ymap, status, "[%s:%u] Invalid format for \'ymap\' in Rectificaton Kernel, it should be VX_TYPE_FLOAT32", __FUNCTION__, __LINE__);
    }

    if ( (src_width != xmap_width || src_width != ymap_width) || (src_height != xmap_height || src_height != ymap_height))
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        vxAddLogEntry((vx_reference)src, status, "[%s:%u]  \'src\', \'xmap\' or \'ymap\' have different size in Rectificaton Kernel", __FUNCTION__, __LINE__);
    }

    vx_meta_format dst_meta = metas[3];
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_FORMAT, &src_format, sizeof(src_format));
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_WIDTH,  &src_width,  sizeof(src_width));
    vxSetMetaFormatAttribute(dst_meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &src_height, sizeof(src_height));

    return status;
}

// Register user defined kernel in OpenVX context
vx_status registerRectificationKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_enum id;
    status = vxAllocateUserKernelId(context, &id);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to allocate an ID for the Rectification kernel");
        return status;
    }

    vx_kernel kernel = vxAddUserKernel(context, "gpu:" KERNEL_RECTIFICATION_NAME, id,
                                       rectification_kernel,
                                       4,    // numParams
                                       rectification_validate,
                                       NULL, // init
                                       NULL  // deinit
                                       );

    status = vxGetStatus((vx_reference)kernel);
    if (status != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Failed to create Rectification Kernel");
        return status;
    }

    status |= vxAddParameterToKernel(kernel, 0, VX_INPUT , VX_TYPE_IMAGE , VX_PARAMETER_STATE_REQUIRED); // src
    status |= vxAddParameterToKernel(kernel, 1, VX_INPUT , VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // xmap
    status |= vxAddParameterToKernel(kernel, 2, VX_INPUT , VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // ymap
    status |= vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE , VX_PARAMETER_STATE_REQUIRED); // dst

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to initialize Rectification Kernel parameters");
        return VX_FAILURE;
    }

    status = vxFinalizeKernel(kernel);

    if (status != VX_SUCCESS)
    {
        vxReleaseKernel(&kernel);
        vxAddLogEntry((vx_reference)context, status, "Failed to finalize Rectification Kernel");
        return VX_FAILURE;
    }

    return status;
}

// Actual Rectificaiton Node
vx_node rectificationNode(vx_graph graph, vx_image src, vx_matrix xmap, vx_matrix ymap, vx_image dst)
{
    vx_node node = NULL;

    vx_kernel kernel = vxGetKernelByName(vxGetContext((vx_reference)graph), KERNEL_RECTIFICATION_NAME);

    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        vxReleaseKernel(&kernel);

        if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
        {
            vxSetParameterByIndex(node, 0, (vx_reference)src);
            vxSetParameterByIndex(node, 1, (vx_reference)xmap);
            vxSetParameterByIndex(node, 2, (vx_reference)ymap);
            vxSetParameterByIndex(node, 3, (vx_reference)dst);
        }
    }

    return node;
}