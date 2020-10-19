#ifndef _CAMERA_PROPERTIES
#define _CAMERA_PROPERTIES

namespace camera_properties{

// ---------------------------------------------
// -------------- STEREOLABS ZED ---------------
// ---------------------------------------------

/*  // ZED PRE REPAIR
    constexpr int cam_width = 1920;
    constexpr int cam_height = 1080;
    constexpr double cam_fps = 30;

    constexpr float focal = 2.8f;
    constexpr float sensor_height = 3.008f;

    constexpr float left_cx = 992.411f;
    constexpr float left_cy = 561.331f;
    constexpr float left_fx = 1399.68f;
    constexpr float left_fy = 1399.68f;
    constexpr float left_k1 = -0.173369f;
    constexpr float left_k2 = 0.0284801f;

    constexpr float rght_cx = 935.998f;
    constexpr float rght_cy = 608.05f;
    constexpr float rght_fx = 1399.72f;
    constexpr float rght_fy = 1399.72f;
    constexpr float rght_k1 = -0.172673f;
    constexpr float rght_k2 = 0.027555f;

#ifndef SKIP_RECTIFICATION
    constexpr float stereo_TX = -119.887f;
    constexpr float stereo_TY = 0.0f;
    constexpr float stereo_TZ = 0.0f;
    constexpr float stereo_RX = -0.00760284f;
    constexpr float stereo_RY = 0.0f;
    constexpr float stereo_RZ = -0.00464149f;
#endif
*/

   
/*  //   ZED POST REPAIR
    constexpr int cam_width = 1920;
    constexpr int cam_height = 1080;
    constexpr double cam_fps = 30;

    constexpr float focal = 2.8f;
    constexpr float sensor_height = 3.008f;

    constexpr float left_cx = 1113.9300f;
    constexpr float left_cy = 608.7609f;
    constexpr float left_fx = 1356.4093f;
    constexpr float left_fy = 1356.5698f;
    constexpr float left_k1 =  -0.1701f;
    constexpr float left_k2 =  0.0161f;

    constexpr float rght_cx = 1068.9616f;
    constexpr float rght_cy = 682.5528f;
    constexpr float rght_fx = 1351.2624f;
    constexpr float rght_fy = 1351.9398f;
    constexpr float rght_k1 = -0.1693f;
    constexpr float rght_k2 = 0.0199f;

#ifndef SKIP_RECTIFICATION
    constexpr float stereo_TX = -117.4407f;
    constexpr float stereo_TY = 0.0f;
    constexpr float stereo_TZ = 0.0f;
    constexpr float stereo_RX = -0.0037f;
    constexpr float stereo_RY = 0.0f;
    constexpr float stereo_RZ = 0.0003f;'
#endif
*/

// ---------------------------------------------
// ------------------ BASLERS ------------------
// ---------------------------------------------

/*
    // acA2440-75uc Full Res
    constexpr int cam_width = 2448;
    constexpr int cam_height = 2048;
    constexpr double cam_fps = 30;

    constexpr float focal = 5.0f;
    constexpr float sensor_height = 7.065f;

    constexpr float left_cx = 1071.7f;
    constexpr float left_cy = 897.23f;
    constexpr float left_fx = 1728.4f;
    constexpr float left_fy = 1711.9f;
    constexpr float left_k1 = 0;
    constexpr float left_k2 = 0;

    constexpr float rght_cx = 1307.3f;
    constexpr float rght_cy = 954.5f;
    constexpr float rght_fx = 1694.4f;
    constexpr float rght_fy = 1687.3f;
    constexpr float rght_k1 = 0;
    constexpr float rght_k2 = 0;

#ifndef SKIP_RECTIFICATION
    // Office Setup
    // constexpr float stereo_TX = -455.0f;
    // constexpr float stereo_TY = 0.0f;
    // constexpr float stereo_TZ = 0.0f;
    // constexpr float stereo_RX = 0.0f;
    // constexpr float stereo_RY = 0.0f;
    // constexpr float stereo_RZ = 0.0f;

    // Car
    constexpr float stereo_TX = -531.6f;
    constexpr float stereo_TY = 42.7914f;
    constexpr float stereo_TZ = 25.998f;
    constexpr float stereo_RX = -0.0576f;
    constexpr float stereo_RY = 0.1318f;
    constexpr float stereo_RZ = -0.1622f;
#endif
*/

/*
    // acA2440-75uc 2448x1080p

    constexpr int cam_width = 2448;
    constexpr int cam_height = 1080;
    constexpr double cam_fps = 30;

    constexpr float focal = 5.0f;
    constexpr float sensor_height = 3.726f;

    constexpr float left_cx = 1229.2f;
    constexpr float left_cy = 490.70f;
    constexpr float left_fx = 1456.7f;
    constexpr float left_fy = 1458.9f;
    constexpr float left_k1 =  -0.1209f;
    constexpr float left_k2 =  0.1237f;

    constexpr float rght_cx = 1175.5f;
    constexpr float rght_cy = 492.87f;
    constexpr float rght_fx = 1458.9f;
    constexpr float rght_fy = 1460.4f;
    constexpr float rght_k1 = -0.1347f;
    constexpr float rght_k2 = 0.1458f;

#ifndef SKIP_RECTIFICATION
    constexpr float stereo_TX = -441.19f;
    constexpr float stereo_TY = 0.728f;
    constexpr float stereo_TZ = 3.2331f;
    constexpr float stereo_RX = -0.0209f;
    constexpr float stereo_RY = 0.0254f;
    constexpr float stereo_RZ = -0.0007f;
#endif
*/


    // acA2440-75uc 1920x1080
    constexpr int cam_width = 1920;
    constexpr int cam_height = 1080;
    constexpr double cam_fps = 30;

    constexpr float focal = 5.0f;
    constexpr float sensor_height = 3.726f;

/*
    //Chat temp mounts
    constexpr float left_cx = 873.22f;
    constexpr float left_cy = 514.17f;
    constexpr float left_fx = 1425.1f;
    constexpr float left_fy = 1417.7f;
    constexpr float left_k1 = -0.1690f;
    constexpr float left_k2 = 0.2742f;

    constexpr float rght_cx = 942.98f;
    constexpr float rght_cy = 499.65f;
    constexpr float rght_fx = 1430.4f;
    constexpr float rght_fy = 1442.9f;
    constexpr float rght_k1 = -0.1083f;
    constexpr float rght_k2 = 0.1674f;
*/


    //Chat temp mounts v2
    constexpr float left_cx = 932.11f;
    constexpr float left_cy = 483.788f;
    constexpr float left_fx = 1541.5f;
    constexpr float left_fy = 1544.7f;
    constexpr float left_k1 = -0.0346f;
    constexpr float left_k2 = 0.0893f;

    constexpr float rght_cx = 880.3963f;
    constexpr float rght_cy = 501.9477f;
    constexpr float rght_fx = 1521.9f;
    constexpr float rght_fy = 1529.8f;
    constexpr float rght_k1 = -0.235f;
    constexpr float rght_k2 = 0.0893f; //-0.2005f

/*
    //Steel Mount
    constexpr float left_cx = 917.07f;
    constexpr float left_cy = 493.21f;
    constexpr float left_fx = 1541.3f;
    constexpr float left_fy = 1540.09f;
    constexpr float left_k1 = -0.1015f;
    constexpr float left_k2 = 0.2012f;

    constexpr float rght_cx = 934.75f;
    constexpr float rght_cy = 524.94f;
    constexpr float rght_fx = 1517.9f;
    constexpr float rght_fy = 1524.8f;
    constexpr float rght_k1 = -0.1129f;
    constexpr float rght_k2 = 0.1682f;

*/
/*
    //Steel Mount v2 - TRASH
    constexpr float left_cx = 954.37f;
    constexpr float left_cy = 492.44f;
    constexpr float left_fx = 1486.2f;
    constexpr float left_fy = 1483.5f;
    constexpr float left_k1 = -0.1539f;
    constexpr float left_k2 = 0.1212f;

    constexpr float rght_cx = 885.8f;
    constexpr float rght_cy = 513.7f;
    constexpr float rght_fx = 1493.6f;
    constexpr float rght_fy = 1486.8f;
    constexpr float rght_k1 = -0.1445f;
    constexpr float rght_k2 = 0.1221f;
*/
#ifndef SKIP_RECTIFICATION

/*
    // Steel Mount
    constexpr float stereo_TX = -451.6547f;
    constexpr float stereo_TY = 3.8422f;
    constexpr float stereo_TZ = 0.0f; //29.3138f;
    constexpr float stereo_RX = 0.0164f;
    constexpr float stereo_RY = -0.0084f;
    constexpr float stereo_RZ = 0.01f;
*/
/*
    // Steel Mount V2 - TRASH
    constexpr float stereo_TX = -437.21f;
    constexpr float stereo_TY = 1.4451f;
    constexpr float stereo_TZ = -11.5432f;
    constexpr float stereo_RX = 0.034f;
    constexpr float stereo_RY = 0.036f;
    constexpr float stereo_RZ = 0.0f;
*/  
/*
    // Chat Temp Mounts
    constexpr float stereo_TX = -497.1f;
    constexpr float stereo_TY = 24.077f;
    constexpr float stereo_TZ = -34.453f;
    constexpr float stereo_RX = -0.0128f;
    constexpr float stereo_RY = 0.0473f;
    constexpr float stereo_RZ = -0.1239f;
*/

    // Chat Temp Mounts v2
    constexpr float stereo_TX = -561.1981f;
    constexpr float stereo_TY = 0.6287f;
    constexpr float stereo_TZ = -60.9409f;
    constexpr float stereo_RX = 0.0331f; // -0.0331f
    constexpr float stereo_RY = 0.0419f; // -0.0219f
    constexpr float stereo_RZ = -0.0195f; // 0.0195f

#endif

#ifdef SKIP_RECTIFICATION
    constexpr float stereo_TX = -200.0f;
    constexpr float stereo_TY = 0.0f;
    constexpr float stereo_TZ = 0.0f;
    constexpr float stereo_RX = 0.0f;
    constexpr float stereo_RY = 0.0f;
    constexpr float stereo_RZ = 0.0f;
#endif
} // namespace camera_properties

#endif //_CAMERA_PROPERTIES