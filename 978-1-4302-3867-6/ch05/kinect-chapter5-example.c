#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

#include <vector>
#include <iostream>

#include <stdio.h>
#include <string.h>
#include <libfreenect.h>

#include <pthread.h>

using namespace cv;
using namespace std;


struct location {
	Vec2f position;
	Vec2f speed;
	Vec2f start_pos;
	bool tracked;
	int id,size;
};


Mat rawdepth(Size(640,480),CV_16UC1);
Mat background(Size(640,480),CV_16UC1);
Mat difference(Size(640,480),CV_16UC1);

Mat depthf(Size(640,480),CV_8UC1);
Mat thresh(Size(640,480),CV_8UC1);
Mat denoise(Size(640,480),CV_8UC1);

//Mat rgb(Size(640,480),CV_8UC3,Scalar(0));


bool die = false;
int reset_bg = 5;

unsigned char lower_val =  8;
unsigned char upper_val = 20;

int min_size =  25;
int max_size = 150;

freenect_device *f_dev;
freenect_context *f_ctx;

pthread_t fnkt_thread;
pthread_mutex_t buf_mutex = PTHREAD_MUTEX_INITIALIZER;

void *freenect_threadfunc(void* arg) {
	printf("freenect thread started\n");
	while (!die && freenect_process_events(f_ctx) >= 0 ) {}
	printf("freenect thread finished\n");
	return NULL;
}

void depth_cb(freenect_device *dev, void* depth, uint32_t timestamp) {
		pthread_mutex_lock(&buf_mutex);
 
		// copy to OpenCV image buffer
		memcpy(rawdepth.data, depth, 640*480*2);
 
		pthread_mutex_unlock(&buf_mutex);
}
 
/* void rgb_cb(freenect_device *dev, void* rgb, uint32_t timestamp) {
		pthread_mutex_lock(&buf_mutex);
		//copy to ocv_buf..
		memcpy(rgb.data, rgb, 640*480*3);
		uint8_t* rgbdata = rgb.data;
		uint8_t tmp;
		for (int i = 0; i < 640*480; i++) {
			tmp = rgbdata[i*3];
			rgbdata[i*3] = rgbdata[i*3+2];
			rgbdata[i*3+2] = tmp;
		}
		pthread_mutex_unlock(&buf_mutex);
} */


int main(int argc, char **argv) {

	// tracking data
	vector< location > prev_touches;
	vector< location > touches;
	int next_id = 1;

	printf("\nHacking The Kinect, Chapter 5: Code Example\n\n");

	// setup, open device etc.
	if (freenect_init(&f_ctx, NULL) < 0) {
		printf("freenect_init failed\n");
		return 1;
	}

	freenect_set_log_level(f_ctx, FREENECT_LOG_INFO);

	int nr_devices = freenect_num_devices (f_ctx);
	printf("number of devices found: %d\n", nr_devices);

	int user_device_number = 0;
	if (argc > 1)
		user_device_number = atoi(argv[1]);

	if (nr_devices < 1)
		return 1;

	if (freenect_open_device(f_ctx, &f_dev, user_device_number) < 0) {
		printf("could not open device\n");
		return 1;
	}

	// indicate camera activity
	freenect_set_led(f_dev,LED_RED);

	// start the depth stream
	freenect_set_depth_callback(f_dev, depth_cb);
	freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM,FREENECT_DEPTH_REGISTERED));
	freenect_start_depth(f_dev);

	// start the video stream
	/*freenect_set_video_callback(f_dev, rgb_cb);
	freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM,FREENECT_VIDEO_RGB));
	freenect_start_video(f_dev);*/

	if (pthread_create(&fnkt_thread, NULL, freenect_threadfunc, NULL) ) {
		printf("pthread_create failed\n");
		return 1;
	}

	printf("entering mainloop. press space to reset, esc to quit.\n");


	while (!die) {

		/*************************************/
		/*      PART I: IMAGE PROCESSING     */
		/*************************************/

		// lock the buffer
		pthread_mutex_lock(&buf_mutex);

		// update background to current camera image
		if (reset_bg) {
			printf("resetting background...\n");
			rawdepth.copyTo( background );
			reset_bg--;
		}

		// subtract current image from background
		subtract( background, rawdepth, difference );

		// unlock the buffer
		pthread_mutex_unlock(&buf_mutex);

		// OpenCV doesn't support simultaneous upper & lower thresholds
		// => apply threshold in simple loop over all pixels
		uint16_t* diffdata = (uint16_t*)difference.data;
		uint8_t*  threshdata = thresh.data;
		for (int i = 0 ; i < 640*480; i++) {
			if ((diffdata[i] >= lower_val) && (diffdata[i] <= upper_val))
				threshdata[i] = 255;
			else
				threshdata[i] = 0;
		}

		// pre-remove noise from the image
		//erode( thresh, denoise, Mat() );

		// extract contours of connected components
		vector< vector<Point> > contours;
		findContours( thresh, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );

		// swap previous & current locations
		prev_touches = touches;
		touches.clear();

		// calculate moments and touch points
		for (vector< vector<Point> >::iterator vec = contours.begin(); vec != contours.end(); vec++) {

			// create Moments object
			Mat points = Mat(*vec);
			Moments img_moments = moments( points );
			double area = img_moments.m00;

			// area of component within range?
			if ((area > min_size) && (area < max_size)) {
				location temp_loc;
				temp_loc.position = Vec2f( img_moments.m10/area, img_moments.m01/area );
				temp_loc.size = area;
				touches.push_back( temp_loc );
			}
		}

		// assign new IDs to all new touch locations
		for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {
			loc->start_pos = loc->position;
			loc->speed = Vec2f(0,0);
			loc->tracked = false;
			loc->id = next_id;
			next_id++;
		}

		// see if we can match any old locations with the new ones
		for (vector<location>::iterator prev_loc = prev_touches.begin(); prev_loc != prev_touches.end(); prev_loc++) {

			// predict new position
			Vec2f predict = prev_loc->position + prev_loc->speed;
			double mindist = norm(prev_loc->speed) + 5; // minimum search radius: 5 px
			location* nearest = NULL;

			// search closest new touch location (that has not yet been assigned)
			for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {
				if (loc->tracked) continue;
				Vec2f delta = loc->position - predict;
				double dist = norm(delta);
				if (dist < mindist) {
					mindist = dist;
					nearest = &(*loc);
				}
			}

			// assign data from previous location
			if (nearest != NULL) {
				nearest->id = prev_loc->id;
				nearest->speed = nearest->position - prev_loc->position;
				nearest->start_pos = prev_loc->start_pos;
				nearest->tracked = true;
			}
		}
	
		// paint touch locations into image
		for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {
			char textbuffer[32]; snprintf(textbuffer,32,"%d",loc->id);
			Point v1 = Point(loc->position) - Point(5,5);
			Point v2 = Point(loc->position) + Point(5,5);
			rectangle( thresh, v1, v2, 255, CV_FILLED );
			putText( thresh, textbuffer, v2, FONT_HERSHEY_SIMPLEX, 1, 255 );
		}

		// display result
		imshow("Chapter 5 Example",thresh);				
		//imshow("rgb",rgb);

		/*************************************/
		/*   PART II: GESTURE RECOGNITION    */
		/*************************************/

		double touchcount = touches.size();

		// determine overall motion
		Vec2f motion(0,0);
		for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {
			Vec2f delta = loc->position - loc->start_pos;
			motion = motion + delta;
		}

		if (touchcount > 0) {
			motion[0] = motion[0] / touchcount;
			motion[1] = motion[1] / touchcount;
		}

		// determine initial/current centroid of touch points (for rotation & scale)
		Vec2f centroid_start(0,0);
		Vec2f centroid_current(0,0);

		for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {
			centroid_start = centroid_start + loc->start_pos;
			centroid_current = centroid_current + loc->start_pos;
		}

		if (touchcount > 0) {
			centroid_start[0] = centroid_start[0] / touchcount;
			centroid_start[1] = centroid_start[1] / touchcount;
			centroid_current[0] = centroid_current[0] / touchcount;
			centroid_current[1] = centroid_current[1] / touchcount;
		}

		// calculate rotation
		double angle = 0.0;
		for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {

			Vec2f p0 = loc->start_pos - centroid_start;
			Vec2f p1 = loc->position  - centroid_current;

			// normalize vectors
			double np0 = norm(p0), np1 = norm(p1);
			p0[0] = p0[0]/np0; p0[1] = p0[1]/np0;
			p1[0] = p1[0]/np1; p1[1] = p1[1]/np1;
	
			double current_angle = acos( p0.ddot(p1) ); // scalar product

			Mat m0 = ( Mat_<double>(3,1) << p0[0], p0[1], 0 );
			Mat m1 = ( Mat_<double>(3,1) << p1[0], p1[1], 0 );
			Mat cross = m0.cross(m1);
			if (cross.at<double>(2,0) < 0) // determine rotation direction
				current_angle = -current_angle;
			angle += current_angle;
		}
		
		if (touchcount > 0)
			angle = angle / touchcount;
	
		// calculate scale
		double scale = 0.0;
		for (vector<location>::iterator loc = touches.begin(); loc != touches.end(); loc++) {

			Vec2f p0 = loc->start_pos - centroid_start;
			Vec2f p1 = loc->position  - centroid_current;

			double relative_distance = norm(p0) / norm(p1);
			scale += relative_distance;
		}
		
		if (touchcount > 0)
			scale = scale / touchcount;

		// display results
		if (norm(motion) > 2.0)
			printf("motion: %f %f\n",motion[0],motion[1]);

		if (abs(angle) > 0.1)
			printf("rotation: %f\n",angle);

		if ((abs(scale-1.0) > 0.1) && (scale != 0.0))
			printf("scale: %f\n",scale);
	
		// check for keypress
		char k = cvWaitKey(5);
		if ( k == 27 ) die = true;   // ESC key
		if ( k == 32 ) reset_bg = 1; // space
	}

	printf("exiting mainloop.\n");

	//cvDestroyWindow("rgb");
	cvDestroyWindow("depth");
	freenect_set_led(f_dev,LED_OFF);

	pthread_join(fnkt_thread, NULL);
	pthread_exit(NULL);
}

