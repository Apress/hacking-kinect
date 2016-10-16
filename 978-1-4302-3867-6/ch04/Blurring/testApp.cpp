#include "testApp.h"

int mod = 4;

float threshold = 150;

//--------------------------------------------------------------
void testApp::setup() {
	kinect.init();
	kinect.setVerbose(true);
	kinect.open();
	
	resultImage.allocate(kinect.width/mod, kinect.height/mod, OF_IMAGE_GRAYSCALE);
	
	//	nearThreshold = 230;
	//	farThreshold  = 70;
	//	bThreshWithOpenCV = true;
	//	
	ofSetFrameRate(60);
	
	// zero the tilt on startup
	angle = 0;
	kinect.setCameraTiltAngle(angle);
}

//--------------------------------------------------------------
void testApp::update() {
	ofBackground(100, 100, 100);
	
	threshold = ofMap(mouseX, 0, ofGetViewportWidth(), 0, 255, 255);
	
//	cout << mouseX << ":" << threshold << endl;
	
	kinect.update();
	if(kinect.isFrameNew())	// there is a new frame and we are connected
	{
		
		greyImage.setFromPixels(kinect.getPixels(), kinect.getWidth(), kinect.getHeight(), OF_IMAGE_COLOR,true);
		greyImage.setFromPixels(kinect.getDepthPixels(), kinect.getWidth(), kinect.getHeight(), OF_IMAGE_GRAYSCALE,true);
		greyImage.setImageType(OF_IMAGE_GRAYSCALE);
		greyImage.resize(greyImage.getWidth()/mod, greyImage.getHeight()/mod);
		
		unsigned char * pixels = resultImage.getPixels();  
		
		for(int x = 0; x < greyImage.width; x++){
			for(int y = 0; y < greyImage.height; y++){
				int i = x + y * greyImage.width;
				
				int color = blur(&greyImage, x, y, 0);
				
				pixels[i] = color;
				
				if(color > threshold){
					pixels[i] = 255;
				} else {
					pixels[i] = 0;
				}
			}
		}
		
		resultImage.update();
	}
}

//--------------------------------------------------------------
void testApp::draw() {
	ofSetColor(255, 255, 255);
	greyImage.draw(0, 0, 640, 480);
	resultImage.draw(640, 0, 640, 480);
}


//--------------------------------------------------------------
void testApp::exit() {
	kinect.setCameraTiltAngle(0); // zero the tilt on exit
	kinect.close();
}


//--------------------------------------------------------------
float testApp::blur(ofImage* img, int x, int y, int blurSize){
	float greyLevel = 0;
	
	unsigned char* pixels = img->getPixels();
	
	int numPixels = 0;
	
    for(int dx = -blurSize; dx <= blurSize; dx++){
		for(int dy = -blurSize; dy <= blurSize; dy++){
			
			
			int newX = ofClamp((dx + x), 0, img->getWidth()  - 1);
			int newY = ofClamp((dy + y), 0, img->getHeight() - 1);
			
			numPixels++;
			
			int i =  (newX + newY * img->getWidth());
			
			greyLevel += pixels[i];
			
		}
    }
    
    greyLevel = greyLevel/numPixels;
    
    return greyLevel;
}

//--------------------------------------------------------------
void testApp::keyPressed (int key) {
	switch (key) {
		case OF_KEY_UP:
			angle++;
			if(angle>30) angle=30;
			kinect.setCameraTiltAngle(angle);
			break;
			
		case OF_KEY_DOWN:
			angle--;
			if(angle<-30) angle=-30;
			kinect.setCameraTiltAngle(angle);
			break;
	}
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y) {
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button)
{}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button)
{}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button)
{}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h)
{}

