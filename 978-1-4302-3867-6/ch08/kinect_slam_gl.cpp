//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#include "kinect_slam_gl.h"

namespace kinect_slam {

CUIModule CUIModule::inst;

CUIModule::CUIModule():
    zoom(1),
    mx(-1),
    my(-1),
    rotangles(),
    rgb_tex(rgb_tex_buffer_height, rgb_tex_buffer_width)
{
    rotangles[0]=rotangles[1]=0;
}

CUIModule::~CUIModule() {
}

void CUIModule::init_gl(int argc, char **argv){
    glutInit(&argc, argv);

    //Create glut window
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);
    main_window = glutCreateWindow("Simple Kinect SLAM");

    //Glut callbacks
    glutDisplayFunc(sdo_glutDisplay);
    glutTimerFunc(5,sdo_glutTimer,0);
    //glutIdleFunc(sdo_glutIdle);
    glutReshapeFunc(sdo_glutReshape);
    glutKeyboardFunc(sdo_glutKeyboard);
    glutMotionFunc(sdo_glutMotion);
    glutMouseFunc(sdo_glutMouse);

    //RGB texture for opengl
    rgb_tex_width = 640;
    rgb_tex_height = 480;
    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_depth_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //Default settings
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    do_glutReshape(640, 480);
}

void CUIModule::refresh() {
    if ( glutGetWindow() != main_window) 
        glutSetWindow(main_window);  
    glutPostRedisplay();
}

void CUIModule::do_glutIdle() {
}

void CUIModule::do_glutTimer(int value) {
    if(tracking.shared.is_data_new)
        refresh();
    glutTimerFunc(5,sdo_glutTimer,0);
}

void CUIModule::do_glutReshape(int Width, int Height) {
    glViewport(0,0,Width,Height);
    aspect_ratio = Width/(float)Height;
}

void CUIModule::do_glutDisplay() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //Set up 3D projection matrices
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, 4/3., 0.3, 200);

    //Modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom,zoom,1);
    glTranslatef(0,0,-3.5);
    glRotatef(rotangles[0], 1,0,0);
    glRotatef(rotangles[1], 0,1,0);
    glTranslatef(0,0,1.5);

    float m[] = {1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1};
    glMultMatrixf(m);

    //Show points
    pthread_mutex_lock(&tracking.shared_mutex);

    if(!tracking.shared.views.empty()) {
        std::vector<CTrackedView>::iterator view_it;

        glDisable(GL_TEXTURE_2D);
        glPointSize(1.0f);
        glColor3ub(255,255,255);
        for(view_it=tracking.shared.views.begin(); view_it!=tracking.shared.views.end(); view_it++) {
            glPushMatrix();

            cv::Matx44f m;
            m(0,0) = view_it->R(0,0);
            m(1,0) = view_it->R(0,1);
            m(2,0) = view_it->R(0,2);
            m(0,1) = view_it->R(1,0);
            m(1,1) = view_it->R(1,1);
            m(2,1) = view_it->R(1,2);
            m(0,2) = view_it->R(2,0);
            m(1,2) = view_it->R(2,1);
            m(2,2) = view_it->R(2,2);
            m(3,0) = view_it->T(0,0);
            m(3,1) = view_it->T(1,0);
            m(3,2) = view_it->T(2,0);
            m(0,3) = 0;
            m(1,3) = 0;
            m(2,3) = 0;
            m(3,3) = 1;

            glMultMatrixf(m.val);

            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GL_FLOAT, sizeof(view_it->cloud[0]), &view_it->cloud[0].x);

            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(view_it->cloud[0]), &view_it->cloud[0].rgba);

            glDrawArrays(GL_POINTS,0,view_it->cloud.size());
            
            glPopMatrix();
        }
    }

    //Set up orthogonal matrices
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
  	glLoadIdentity();
	glOrtho (0, aspect_ratio, 1, 0, -1.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    float image_aspect_ratio = 640.0f/480.0f;
    float height=0.35f;
    glLoadIdentity();
    glTranslatef(0,1-height,0);
    glScalef(height/480.0f,height/480.0f,1);

    //Load textures
    cv::Mat3b resize_dst(rgb_tex, cv::Rect(0,0,640,480));
	cv::resize(*tracking.shared.active_rgb, resize_dst, cv::Size(resize_dst.cols, resize_dst.rows));
    
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, rgb_tex.cols, rgb_tex.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_tex.data);

    //Draw image
    glBegin(GL_TRIANGLE_FAN);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glTexCoord2f(0, 0); 
    glVertex3f(0,0,0);

	glTexCoord2f(rgb_tex_width/(float)rgb_tex.cols, 0); 
    glVertex3f(640,0,0);

	glTexCoord2f(rgb_tex_width/(float)rgb_tex.cols, rgb_tex_height/(float)rgb_tex.rows); 
    glVertex3f(640,480,0);

	glTexCoord2f(0, rgb_tex_height/(float)rgb_tex.rows); 
    glVertex3f(0,480,0);
	glEnd();

    //Draw points
    if(tracking.shared.tracks.size() > 0) {
        glPointSize(2.0f);
        glDisable(GL_TEXTURE_2D);

        glBegin(GL_POINTS);

        std::list<CFeatureTrack>::iterator track_it;
        for(track_it=tracking.shared.tracks.begin(); track_it!=tracking.shared.tracks.end(); track_it++) {
            if(track_it->missed_frames==0)
                glColor3ub(0,0,255);
            else
                glColor3ub(255,0,0);
            glVertex2f(track_it->active_position.x, track_it->active_position.y);
        }
        glEnd();
    }

    tracking.shared.is_data_new = false;
    pthread_mutex_unlock(&tracking.shared_mutex);

    glutSwapBuffers();
}

void CUIModule::do_glutKeyboard(unsigned char key, int x, int y) {
    switch(key) {
    case 27:    terminate(); break;
    case 'w':    zoom *= 1.1f; refresh(); break;
    case 's':    zoom /= 1.1f; refresh(); break;
    case 't':   tracking.shared.is_tracking_enabled = !tracking.shared.is_tracking_enabled;
    }    
}

void CUIModule::do_glutMotion(int x, int y) {
    if (mx>=0 && my>=0) {
        rotangles[0] += y-my;
        rotangles[1] += x-mx;
        refresh();
    }
    mx = x;
    my = y;
}

void CUIModule::do_glutMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        mx = x;
        my = y;
    }
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        mx = -1;
        my = -1;
    }
}

void CUIModule::terminate() {
    freenect.die = true;
    tracking.die = true;

    pthread_join(freenect_thread,NULL);
    pthread_join(tracking_thread,NULL);

    glutDestroyWindow(main_window);
    pthread_exit(NULL);
}

void CUIModule::run(int argc, char **argv) {
    cout << "Kinect slam demo application\n";
    cout << "'w'=zoom in, 's'=zoom out, 't'=enable/disable tracking\n";

    init_gl(argc, argv);

    tracking.freenect_data = &freenect.buffers;

    //Init threads
    int res;
    res = pthread_create(&freenect_thread, NULL, freenect.thread_entry, &freenect);
    if (res) 
    {
        cout << "Error creating freenect thread.\n";
        return;
    }
    res = pthread_create(&tracking_thread, NULL, tracking.thread_entry, &tracking);
    if (res) 
    {
        cout << "Error creating tracking thread.\n";
        return;
    }
 
    glutMainLoop();
}

}