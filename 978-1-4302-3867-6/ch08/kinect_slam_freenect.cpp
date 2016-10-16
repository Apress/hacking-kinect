//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#include "kinect_slam_freenect.h"

namespace kinect_slam {

CFreenectModule::CFreenectModule():
    die(false),
    video_mode(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB)),
    depth_mode(freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT)),
    f_ctx(NULL),
    f_dev(NULL),
    depth_back(new cv::Mat1s(480,640)),
    rgb_back(new cv::Mat3b(480,640))
{
}

CFreenectModule::~CFreenectModule()
{
    delete depth_back;
    delete rgb_back;
}

void CFreenectModule::static_depth_callback(freenect_device *dev, void *depth, uint32_t timestamp) {
    CFreenectModule *module = (CFreenectModule*)freenect_get_user(dev);
    module->depth_callback(dev,depth,timestamp);
}
void CFreenectModule::static_rgb_callback(freenect_device *dev, void *rgb, uint32_t timestamp) {
    CFreenectModule *module = (CFreenectModule*)freenect_get_user(dev);
    module->rgb_callback(dev,rgb,timestamp);
}

void CFreenectModule::depth_callback(freenect_device *dev, void *depth, uint32_t timestamp) {
    assert(depth == depth_back->data);
    pthread_mutex_lock(&buffers.mutex);

    std::swap(depth_back, buffers.depth_mid); //Swap buffers

    freenect_set_depth_buffer(dev, depth_back->data);
    buffers.got_depth++;
    
    pthread_mutex_unlock(&buffers.mutex);
    pthread_cond_signal(&buffers.data_ready_cond);
}

void CFreenectModule::rgb_callback(freenect_device *dev, void *rgb, uint32_t timestamp) {
    assert(rgb == rgb_back->data);    
    pthread_mutex_lock(&buffers.mutex);

    std::swap(rgb_back, buffers.rgb_mid); //Swap buffers

    freenect_set_video_buffer(dev, rgb_back->data);
    buffers.got_rgb++;

    pthread_mutex_unlock(&buffers.mutex);
    pthread_cond_signal(&buffers.data_ready_cond);
}

void *CFreenectModule::thread_entry(void *instance) {
    CFreenectModule *p = (CFreenectModule*)instance;
    p->run();
    return NULL;
}

void CFreenectModule::run() {
    //Init freenect
    if (freenect_init(&f_ctx, NULL) < 0) {
        std::cout << "freenect_init() failed\n";
        die = 1;
        return;
    }
    freenect_set_log_level(f_ctx, FREENECT_LOG_WARNING);

    int nr_devices = freenect_num_devices(f_ctx);
    std::cout << "Number of devices found: " << nr_devices << std::endl;

    if (nr_devices < 1) {
        die = true;
        return;
    }

    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        std::cout << "Could not open device\n";
        die = true;
        return;
    }
    
    freenect_set_user(f_dev, this);
    freenect_set_led(f_dev,LED_GREEN);
    freenect_set_depth_callback(f_dev, static_depth_callback);
    freenect_set_depth_mode(f_dev, depth_mode);
    freenect_set_depth_buffer(f_dev, depth_back->data);    
    freenect_set_video_callback(f_dev, static_rgb_callback);
    freenect_set_video_mode(f_dev, video_mode);
    freenect_set_video_buffer(f_dev, rgb_back->data);

    freenect_start_depth(f_dev);
    freenect_start_video(f_dev);
    
    std::cout << "Kinect streams started\n";

    while (!die && freenect_process_events(f_ctx) >= 0) {
        //Let freenect process events
    }

    std::cout << "Shutting down Kinect...";

    freenect_stop_depth(f_dev);
    freenect_stop_video(f_dev);

    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);

    std::cout << "done!\n";
    return;
}

}