//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#pragma once

#include <assert.h>
#include <math.h>
#include <iostream>
using std::cout;
using std::endl;

#include <pthread.h>
#include <opencv2/opencv.hpp>

#include <GL/glut.h>
#include <GL/gl.h>

#include <boost/date_time.hpp>
using namespace boost::posix_time; 

#include "kinect_slam_freenect.h"
#include "kinect_slam_tracking.h"

namespace kinect_slam {

//////////////////////////////////////////////////////////////////
// CUIModule: Handles all communication with the user. Uses the GLUT framework 
//   to receive user input and renders the aggregated point cloud with OpenGL.
//////////////////////////////////////////////////////////////////
class CUIModule {
public:
    static CUIModule inst;

    void run(int argc, char **argv);

private:
    pthread_t freenect_thread,tracking_thread,gl_thread;
    kinect_slam::CFreenectModule freenect;
    kinect_slam::CTrackingModule tracking;

    int main_window;
    float aspect_ratio;
    float zoom;
    int mx,my;        // Prevous mouse coordinates
    int rotangles[2]; // Panning angles

    static const int rgb_tex_buffer_width=1024; //Total width of texture buffer
    static const int rgb_tex_buffer_height=1024;//Total height of texture buffer
    int rgb_tex_width; //Real width of texture
    int rgb_tex_height; //Real heightof texture
    cv::Mat3b rgb_tex; //Actual buffer
    
    GLuint gl_rgb_tex;
    GLuint gl_depth_tex;

    CUIModule();
    ~CUIModule();

    void init_gl(int argc, char **argv);
    void terminate();

    void refresh();

    //Glut callbacks
    void do_glutIdle();
    void do_glutTimer(int value);
    void do_glutDisplay();
    void do_glutReshape(int Width, int Height);
    void do_glutMotion(int x, int y);
    void do_glutMouse(int button, int state, int x, int y);
    void do_glutKeyboard(unsigned char key, int x, int y);

    static void sdo_glutIdle() {inst.do_glutIdle();}
    static void sdo_glutTimer(int value) {inst.do_glutTimer(value);}
    static void sdo_glutDisplay() {inst.do_glutDisplay();}
    static void sdo_glutReshape(int Width, int Height) {inst.do_glutReshape(Width, Height);}
    static void sdo_glutMotion(int x, int y) {inst.do_glutMotion(x,y);}
    static void sdo_glutMouse(int button, int state, int x, int y) {inst.do_glutMouse(button,state,x,y);}
    static void sdo_glutKeyboard(unsigned char key, int x, int y) {inst.do_glutKeyboard(key,x,y);}
};

}