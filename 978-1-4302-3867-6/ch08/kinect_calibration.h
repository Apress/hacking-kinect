//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#pragma once

#include <opencv2/opencv.hpp>

namespace kinect_slam {

//////////////////////////////////////////////////////////////////
// CKinectCalibration: encapsulates the calibration parameters of 
//   the kinect and performs conversion between image coordinates
//   and world coordinates.
//////////////////////////////////////////////////////////////////
class CKinectCalibration
{
public:
    CKinectCalibration():
        calib_fx_d(586.16f), //These constants come from calibration,
        calib_fy_d(582.73f), //replace with your own
        calib_px_d(322.30f),
        calib_py_d(230.07),
        calib_dc1(-0.002851),
        calib_dc2(1093.57),
        calib_fx_rgb(530.11f),
        calib_fy_rgb(526.85f),
        calib_px_rgb(311.23f),
        calib_py_rgb(256.89f),
        calib_R( 0.99999f,   -0.0021409f,     0.004993f,
                    0.0022251f,      0.99985f,    -0.016911f,
                    -0.0049561f,     0.016922f,      0.99984f),
        calib_T(  -0.025985f,   0.00073534f,    -0.003411f)
    {}

    //disparity2point: converts a point in the depth image to a 3D point
    //in color camera coordinates.
    void disparity2point(int u,int v,short disp,cv::Matx31f &pc) {
        cv::Matx31f pd;

        pd(2) = 1.0f / (calib_dc1*(disp - calib_dc2));
        pd(0) = ((u-calib_px_d) / calib_fx_d) * pd(2);
        pd(1) = ((v-calib_py_d) / calib_fy_d) * pd(2);

        pc = calib_R*pd+calib_T;
    }

    //point2rgb: projects a 3D point in color camera coordinates onto the
    //image plane of the color camera and returns the pixel coordinates.
    void point2rgb(cv::Matx31f &pc, float &uc, float &vc)
    {
        uc = pc(0)*calib_fx_rgb/pc(2) + calib_px_rgb;
        vc = pc(1)*calib_fy_rgb/pc(2) + calib_py_rgb;
    }

private:
    const float calib_fx_d;
    const float calib_fy_d;
    const float calib_px_d;
    const float calib_py_d;
    const float calib_dc1;
    const float calib_dc2;
    const float calib_fx_rgb;
    const float calib_fy_rgb;
    const float calib_px_rgb;
    const float calib_py_rgb;
    cv::Matx33f calib_R;
    cv::Matx31f calib_T;
};

}