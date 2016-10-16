//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#pragma once

#include <assert.h>
#include <math.h>
#include <iostream>

#include <pthread.h>
#include <libfreenect.h>
#include <opencv2/opencv.hpp>

namespace kinect_slam {

//////////////////////////////////////////////////////////////////
// CFreenectSharedData: data structure that contains shared
//   variables between the freenect module and other modules.
//////////////////////////////////////////////////////////////////
class CFreenectSharedData {
public:
    pthread_mutex_t mutex;
    pthread_cond_t data_ready_cond;
    cv::Mat1s *depth_mid; //Newest depth image
    cv::Mat3b *rgb_mid;   //Newest color image
    int got_depth, got_rgb; //Indicate the number of frames obtained (1 if no frames were dropped)

    CFreenectSharedData():
        mutex(PTHREAD_MUTEX_INITIALIZER),
        depth_mid(new cv::Mat1s(480,640)),
        rgb_mid(new cv::Mat3b(480,640)),
        got_depth(0), 
        got_rgb(0)
    {
        pthread_cond_init(&data_ready_cond, NULL);
    }

    ~CFreenectSharedData() {
        delete depth_mid;
        delete rgb_mid;
        pthread_cond_destroy(&data_ready_cond);
    }
};

//////////////////////////////////////////////////////////////////
// CFreenectModule: Handles all communication with the Freenect 
//   library. Acquires color and depth images from the Kinect and 
//   offers them to the other modules.
//////////////////////////////////////////////////////////////////
class CFreenectModule
{
public:
    volatile bool die; //Set to true to make the thread exit.
    CFreenectSharedData buffers;

    CFreenectModule();
    ~CFreenectModule();

    void run();
    static void *thread_entry(void *instance);

private:
    const freenect_frame_mode video_mode;
    const freenect_frame_mode depth_mode;
    freenect_context *f_ctx;
    freenect_device *f_dev;

    cv::Mat1s *depth_back; //Depth image buffers
    cv::Mat3b *rgb_back;   //Color image buffers

    void depth_callback(freenect_device *dev, void *depth, uint32_t timestamp);
    void rgb_callback(freenect_device *dev, void *rgb, uint32_t timestamp);
    static void static_depth_callback(freenect_device *dev, void *depth, uint32_t timestamp);
    static void static_rgb_callback(freenect_device *dev, void *rgb, uint32_t timestamp);
};

}