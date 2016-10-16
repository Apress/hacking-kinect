//////////////////////////////////////////////////////////////////
// Cloud in the Wind
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Includes and global variables
//////////////////////////////////////////////////////////////////
#include <assert.h>
#include <math.h>
#include <iostream>
using std::cout;
using std::endl;

#include <libfreenect.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pthread.h>
#include <opencv2/opencv.hpp>

#include <GL/gl.h>
#include <GL/glut.h>

#include <boost/date_time.hpp>
using namespace boost::posix_time; 

inline float randf() {return rand() / (float)RAND_MAX;}

volatile int die = 0;
pthread_mutex_t mid_buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t freenect_thread;


//////////////////////////////////////////////////////////////////
// Freenect code
//////////////////////////////////////////////////////////////////
freenect_context *f_ctx;
freenect_device *f_dev;
freenect_frame_mode video_mode = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB);
freenect_frame_mode depth_mode = freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT);

cv::Mat1s *depth_back, *depth_mid, *depth_front; //Depth image buffers
cv::Mat3b *rgb_back, *rgb_mid, *rgb_front;        //Color image buffers
int got_depth=0, got_rgb=0;

const float calib_fx_d=586.16f; //These constants come from calibration,
const float calib_fy_d=582.73f; //replace with your own
const float calib_px_d=322.30f;
const float calib_py_d=230.07;
const float calib_dc1=-0.002851;
const float calib_dc2=1093.57;
const float calib_fx_rgb=530.11f;
const float calib_fy_rgb=526.85f;
const float calib_px_rgb=311.23f;
const float calib_py_rgb=256.89f;
cv::Matx33f calib_R( 0.99999,   -0.0021409,     0.004993,
                    0.0022251,      0.99985,    -0.016911,
                    -0.0049561,     0.016922,      0.99984);
cv::Matx31f calib_T(  -0.025985,   0.00073534,    -0.003411);

void depth_cb(freenect_device *dev, void *depth, uint32_t timestamp) {
    assert(depth == depth_back->data);    
    pthread_mutex_lock(&mid_buffer_mutex);

    //Swap buffers
    cv::Mat1s *temp = depth_back;
    depth_back = depth_mid;
    depth_mid = temp;

    freenect_set_depth_buffer(dev, depth_back->data);
    got_depth++;
    
    pthread_mutex_unlock(&mid_buffer_mutex);
}

void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp) {
    assert(rgb == rgb_back->data);    
    pthread_mutex_lock(&mid_buffer_mutex);

    // swap buffers
    cv::Mat3b *temp = rgb_back;
    rgb_back = rgb_mid;
    rgb_mid = temp;

    freenect_set_video_buffer(dev, rgb_back->data);

    got_rgb++;

    pthread_mutex_unlock(&mid_buffer_mutex);
}

void *freenect_threadfunc(void *arg) {
    //Init freenect
    if (freenect_init(&f_ctx, NULL) < 0) {
        cout << "freenect_init() failed\n";
        die = 1;
        return NULL;
    }
    freenect_set_log_level(f_ctx, FREENECT_LOG_WARNING);

    int nr_devices = freenect_num_devices (f_ctx);
    cout << "Number of devices found: " << nr_devices << endl;

    if (nr_devices < 1) {
        die = 1;
        return NULL;
    }

    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        cout << "Could not open device\n";
        die = 1;
        return NULL;
    }

    freenect_set_led(f_dev,LED_GREEN);
    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_depth_mode(f_dev, depth_mode);
    freenect_set_depth_buffer(f_dev, depth_back->data);    
    freenect_set_video_callback(f_dev, rgb_cb);
    freenect_set_video_mode(f_dev, video_mode);
    freenect_set_video_buffer(f_dev, rgb_back->data);

    freenect_start_depth(f_dev);
    freenect_start_video(f_dev);
    
    cout << "Kinect streams started\n";

    while (!die && freenect_process_events(f_ctx) >= 0) {
        //Let freenect process events
    }

    cout << "Shutting down Kinect...";

    freenect_stop_depth(f_dev);
    freenect_stop_video(f_dev);

    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);

    cout << "done!\n";
    return NULL;
}

//////////////////////////////////////////////////////////////////
// Cloud building
//////////////////////////////////////////////////////////////////
typedef struct {
    PCL_ADD_POINT4D;    //PCL adds the x,y,z coordinates padded for SSE alginment
    union {
        struct {
            float tex_u;
            float tex_v;
        };
        float tex_uv[2];
    };
    float intensity;
    float weight;
    float velocity[3];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16 TMyPoint;
POINT_CLOUD_REGISTER_POINT_STRUCT (TMyPoint,           // Register our point type with the PCL
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, tex_u, tex_u)
                                    (float, tex_v, tex_v)
                                    (float, intensity, intensity)
                                    (float, weight, weight)
                                    (float[3], velocity, velocity)
 )

pcl::PointCloud<TMyPoint> point_cloud;
cv::Mat_<bool> is_frozen(480,640,false); //Indicates that this point has been touched by
                                        //the wind and will not be refreshed by Kinect
std::vector<unsigned int> valid_indices; //Indices of valid points to render

int cloud_count=0;
float point_cloud_min_x=1e10;    //Min and max values of point cloud
float point_cloud_max_x=-1e10;
float point_cloud_min_y=1e10;
float point_cloud_max_y=-1e10;

//create_point_cloud: creates the point cloud using Kinects color and depth images
void create_point_cloud() {
    float intensity_scale = point_cloud_max_y-point_cloud_min_y;
    valid_indices.clear();
    
    for(int v=0; v<depth_front->rows; v++)
        for(int u=0; u<depth_front->cols; u++) {
            unsigned int index = v*point_cloud.width+u;
            TMyPoint &p = point_cloud(u,v);
            bool is_valid;

            if(is_frozen(v,u))
                is_valid = true;
            else {
                const short d = depth_front->at<short>(v,u);
                if(d==2047) {
                    is_valid = false;
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                }
                else {
                    is_valid = true;
                    p.z = 1.0 / (calib_dc1*(d - calib_dc2));
                    p.x = ((u-calib_px_d) / calib_fx_d) * p.z;
                    p.y = ((v-calib_py_d) / calib_fy_d) * p.z;

                    //Project onto color image
                    cv::Matx31f &pm = *(cv::Matx31f*)p.data;
                    cv::Matx31f pc;

                    pc = calib_R*pm+calib_T;

                    float uc,vc;
                    uc = pc(0)*calib_fx_rgb/pc(2) + calib_px_rgb;
                    vc = pc(1)*calib_fy_rgb/pc(2) + calib_py_rgb;
                    p.tex_u = uc/(float)(rgb_front->cols-1);
                    p.tex_v = vc/(float)(rgb_front->rows-1);

                    p.z = -p.z; //Fix for opengl
                    p.x = p.x;
                    p.y = -p.y;
                    p.intensity = (p.y-point_cloud_min_y) / intensity_scale;
                }
            }

            if(is_valid)
                valid_indices.push_back(index);
        }
    //Calculate point cloud limits for the first 5 frames
    if(cloud_count++ < 5) {
        point_cloud_min_x=point_cloud_min_y=1e10;
        point_cloud_max_x=point_cloud_max_y=-1e10;
        for(unsigned int i=0; i<point_cloud.size(); i++) {
            TMyPoint &p = point_cloud[i];
            if(point_cloud_min_x > p.x)
                point_cloud_min_x = p.x;
            if(point_cloud_max_x < p.x)
                point_cloud_max_x = p.x;
            if(point_cloud_min_y > p.y)
                point_cloud_min_y = p.y;
            if(point_cloud_max_y < p.y)
                point_cloud_max_y = p.y;
        }
    }
}

void show_visualizer() {
    pcl::PointCloud<pcl::PointXYZRGB> cloud2;

    //Copy cloud
    for(unsigned int i=0; i<point_cloud.size(); i++) {
        TMyPoint &p0 = point_cloud[i];
        if(_isnan(p0.x))
            continue;
        pcl::PointXYZRGB p;
        p.x=p0.x; p.y=p0.y; p.z=p0.z;
        int u,v;
        u = (int)(p0.tex_u*639);
        v = (int)(p0.tex_v*479);
        p.r = (*rgb_front)(v,u)[0];
        p.g = (*rgb_front)(v,u)[1];
        p.b = (*rgb_front)(v,u)[2];
        cloud2.push_back(p);
    }

    //Show
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr(&cloud2));
    while(!viewer.wasStopped());
}

//////////////////////////////////////////////////////////////////
// Animation
//////////////////////////////////////////////////////////////////
ptime last_animation_time;
float wind_front_xr = 0.0f;
const float wind_front_velocity=0.03f;

void init_animation() {
    point_cloud.width = 640;
    point_cloud.height = 480;
    point_cloud.is_dense = false;
    point_cloud.resize(640*480);
    for(unsigned int i=0; i<point_cloud.size(); i++) {
        is_frozen[0][i] = false;
        point_cloud[i].x = point_cloud[i].y = point_cloud[i].z = std::numeric_limits<float>::quiet_NaN();
        point_cloud[i].weight = 1.0 + 10*randf();

        cv::Matx31f &velocity = *(cv::Matx31f*)point_cloud[i].velocity;
        velocity.zeros();
    }
}

cv::Matx31f get_wind_force(TMyPoint &p) {
    float u,v,w;
    float xr = (p.x-point_cloud_min_x) / (point_cloud_max_x-point_cloud_min_x); //Normalize x

    u = 1+(randf()-0.5f)*0.6;
    w = (randf()-0.5f)*0.2;
    v = (xr)*sinf(xr*50);
    return cv::Matx31f(u,v,w);
}

void animate() {
    ptime now = microsec_clock::local_time();
    time_duration span = now-last_animation_time;
    float time_ellapsed = span.ticks() / (float)span.ticks_per_second();
    last_animation_time = now;

	wind_front_xr += time_ellapsed*wind_front_velocity; //Wind wall moves from left to right
    float wind_front_x = wind_front_xr*(point_cloud_max_x-point_cloud_min_x) + point_cloud_min_x;

    for(unsigned int i=0; i<point_cloud.size(); i++) {
        TMyPoint &p = point_cloud[i];
        if(p.x < wind_front_x || is_frozen[0][i]) {
            is_frozen[0][i] = true;
            cv::Matx31f &velocity = *(cv::Matx31f*)p.velocity;
            cv::Matx31f force=get_wind_force(p);

            velocity += force*(time_ellapsed/p.weight);
            p.x += velocity(0)*time_ellapsed;
            p.y += velocity(1)*time_ellapsed;
            p.z += velocity(2)*time_ellapsed;
        }
    }
}

//////////////////////////////////////////////////////////////////
// OpenGL code
//////////////////////////////////////////////////////////////////
int main_window;
float zoom=1;
int mx=-1,my=-1;        // Prevous mouse coordinates
int rotangles[2] = {0}; // Panning angles

const int color_map_size = 256;
unsigned char color_map[3*color_map_size];

//Texture buffer must be power of 2 to avoit OpenGL problems
const int rgb_tex_buffer_width=1024; //Total width of texture buffer
const int rgb_tex_buffer_height=1024;//Total height of texture buffer
int rgb_tex_width; //Real width of texture
int rgb_tex_height; //Real heightof texture
cv::Mat3b rgb_tex(rgb_tex_buffer_height, rgb_tex_buffer_width); //Actual buffer

GLuint gl_rgb_tex;
GLuint gl_colormap_tex;
bool use_rgb_tex=true;

//create_color_map: creates the blue to red gradient 1D texture
void create_color_map() {
    int iteration_size=color_map_size/4;
    int step=color_map_size/iteration_size;
    int idx = 0;
    int r,g,b, rs,gs,bs;
    int step_count;

    r=0;g=0;b=128;
    for(int k=0; k<5; k++) {
        if(k==0 || k==4) step_count = iteration_size/2; else step_count = iteration_size;
        switch(k) {
        case 0: rs=0;gs=0;bs=step; break;
        case 1: rs=0;gs=step;bs=0; break;
        case 2:    rs=step;gs=0;bs=-step; break;
        case 3: rs=0;gs=-step;bs=0; break;
        case 4: rs=-step;gs=0;bs=0; break;
        }
        for(int i=0; i<step_count; i++) {
            r+=rs; g+=gs; b+=bs;
            color_map[3*idx+0] = cv::saturate_cast<uchar>(r);
            color_map[3*idx+1] = cv::saturate_cast<uchar>(g);
            color_map[3*idx+2] = cv::saturate_cast<uchar>(b);
            idx++;
        }
    }
}


void do_glutIdle() {
    //Process Kinect data
    if(got_depth > 0 || got_rgb > 0) {
        bool is_depth_new=false, is_rgb_new=false;

        pthread_mutex_lock(&mid_buffer_mutex);
        if(got_depth) {
            //Switch buffers
            cv::Mat1s *temp;

            temp = depth_mid;
            depth_mid = depth_front;
            depth_front = temp;

            got_depth = 0;
            is_depth_new = true;
        }
        if(got_rgb) {
            //Switch buffers
            cv::Mat3b *temp;

            temp = rgb_mid;
            rgb_mid = rgb_front;
            rgb_front = temp;

            got_rgb = 0;
            is_rgb_new = true;
        }

        pthread_mutex_unlock(&mid_buffer_mutex);

        if(is_depth_new)
            create_point_cloud(); //Create new point cloud from depth map
        if(is_rgb_new)
            cv::resize(*rgb_front, rgb_tex, cv::Size(rgb_tex_buffer_width,rgb_tex_buffer_height)); //Make rgb texture

        if ( glutGetWindow() != main_window ) 
            glutSetWindow(main_window);  
        glutPostRedisplay();
    }

    //Animate
    animate();

    //Die?
    if(die) {
        pthread_join(freenect_thread, NULL);
        glutDestroyWindow(main_window);
        pthread_exit(NULL);
    }
}

void do_glutDisplay() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Modelview matrix
    glLoadIdentity();
    glScalef(zoom,zoom,1);
    glTranslatef(0,0,-3.5);
    glRotatef(rotangles[0], 1,0,0);
    glRotatef(rotangles[1], 0,1,0);
    glTranslatef(0,0,1.5);

    //Show points
    if(valid_indices.size() > 0) {
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, sizeof(point_cloud[0]), &point_cloud[0].x);

        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        if(use_rgb_tex) {
            glEnable(GL_TEXTURE_2D);
            glTexImage2D(GL_TEXTURE_2D, 0, 3, rgb_tex_buffer_width, rgb_tex_buffer_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_tex.data);
            glTexCoordPointer(2, GL_FLOAT, sizeof(point_cloud[0]), &point_cloud[0].tex_uv);
        }
        else {
            glDisable(GL_TEXTURE_2D);
            glEnable(GL_TEXTURE_1D);
            glTexCoordPointer(1, GL_FLOAT, sizeof(point_cloud[0]), &point_cloud[0].intensity);
        }

        glPointSize(1.0f);
        glDrawElements(GL_POINTS,valid_indices.size(),GL_UNSIGNED_INT,&valid_indices.front());
    }

    glutSwapBuffers();
}

void do_glutKeyboard(unsigned char key, int x, int y) {
    switch(key) {
    case 27:    die = 1; break;
    case 'w':    zoom *= 1.1f; break;
    case 's':    zoom /= 1.1f; break;
    case 't':    use_rgb_tex = !use_rgb_tex; break;
    case 'v':  show_visualizer();
    }    
}

void do_glutReshape(int Width, int Height) {
    glViewport(0,0,Width,Height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, 4/3., 0.3, 200);
    glMatrixMode(GL_MODELVIEW);
}

void do_glutMotion(int x, int y) {
    if (mx>=0 && my>=0) {
        rotangles[0] += y-my;
        rotangles[1] += x-mx;
    }
    mx = x;
    my = y;
}

void do_glutMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        mx = x;
        my = y;
    }
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        mx = -1;
        my = -1;
    }
}

void init_gl(int Width, int Height) {
    //Create glut window
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);
    main_window = glutCreateWindow("Cloud in the Wind");

    //Glut callbacks
    glutDisplayFunc(do_glutDisplay);
    glutIdleFunc(do_glutIdle);
    glutReshapeFunc(do_glutReshape);
    glutKeyboardFunc(do_glutKeyboard);
    glutMotionFunc(do_glutMotion);
    glutMouseFunc(do_glutMouse);

    //RGB texture for opengl
    glGenTextures(1, &gl_colormap_tex);
    glBindTexture(GL_TEXTURE_1D, gl_colormap_tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    create_color_map();
    glTexImage1D(GL_TEXTURE_1D, 0, 3, color_map_size, 0, GL_RGB, GL_UNSIGNED_BYTE, color_map);

    rgb_tex_width = 640;
    rgb_tex_height = 640/(float)video_mode.width * video_mode.height;
    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //Default settings
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    do_glutReshape(Width, Height);

    last_animation_time = microsec_clock::local_time();
}

void *gl_threadfunc(void *arg) {
    cout << "GL thread started\n";
    cout << "'w' = zoom in, 's' = zoom out, 'v' = launch PCL cloud viewer, 't' = change texturing mode\n";
    init_gl(640, 480);
    init_animation();

    glutMainLoop();
    return NULL;
}

//////////////////////////////////////////////////////////////////
// Entry point
//////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cout << "Cloud viewer demo application\n";

    glutInit(&argc, argv);

    //Create buffers
    depth_back = new cv::Mat1s(depth_mode.height, depth_mode.width);
    depth_mid = new cv::Mat1s(depth_mode.height, depth_mode.width);
    depth_front = new cv::Mat1s(depth_mode.height, depth_mode.width);
    rgb_back = new cv::Mat3b(video_mode.height, video_mode.width);
    rgb_mid = new cv::Mat3b(video_mode.height, video_mode.width);
    rgb_front = new cv::Mat3b(video_mode.height, video_mode.width);

    //Init threads
    int res = pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
    if (res) 
    {
        cout << "pthread_create failed\n";
        return 1;
    }

    // Glut runs on main thread
    gl_threadfunc(NULL);

    return 0;
}