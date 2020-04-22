// NppStatus nppiRemap_8u_C3R_Ctx	(	const Npp8u * 	pSrc,
// NppiSize 	oSrcSize,
// int 	nSrcStep,
// NppiRect 	oSrcROI,
// const Npp32f * 	pXMap,
// int 	nXMapStep,
// const Npp32f * 	pYMap,
// int 	nYMapStep,
// Npp8u * 	pDst,
// int 	nDstStep,
// NppiSize 	oDstSizeROI,
// int 	eInterpolation,
// NppStreamContext 	nppStreamCtx 
// )

#ifndef __NVX_RECT_NODE__
#define __NVX_RECT_NODE__

#include <NVX/nvx.h>

// Register AlphaComp kernel in OpenVX context
vx_status registerRectificationKernel(vx_context context);

// Create AlphaComp node
vx_node rectificationNode(vx_graph graph,
                      vx_image src, vx_matrix xmap,
                      vx_matrix ymap, vx_image dst);
                      
#endif