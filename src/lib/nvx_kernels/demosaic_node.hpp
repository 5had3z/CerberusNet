// NppStatus nppiCFAToRGB_8u_C1C3R_Ctx	(	const Npp8u * 	pSrc,
// int 	nSrcStep,
// NppiSize 	oSrcSize,
// NppiRect 	oSrcROI,
// Npp8u * 	pDst,
// int 	nDstStep,
// NppiBayerGridPosition 	eGrid,
// NppiInterpolationMode 	eInterpolation,
// NppStreamContext 	nppStreamCtx 
// )

#ifndef __NVX_DEMOSAIC_NODE__
#define __NVX_DEMOSAIC_NODE__

#include <NVX/nvx.h>

// Register AlphaComp kernel in OpenVX context
vx_status registerdemosaicKernel(vx_context context);

// Create AlphaComp node
vx_node demosaicNode(vx_graph graph,
                      vx_image src, vx_image dst);
                      
#endif