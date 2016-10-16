//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#include "kinect_slam_tracking.h"

namespace kinect_slam {

template <class T1, class T2>
T1 round(T2 v) {return static_cast<T1>(v + static_cast<T2>(0.5));}

CTrackingSharedData::CTrackingSharedData():
    is_data_new(false),
    is_tracking_enabled(true),
    base_rgb(new cv::Mat3b(480,640)),
    base_pointmap(new cv::Mat3f(480,640)),
    active_rgb(new cv::Mat3b(480,640)),
    active_depth(new cv::Mat1s(480,640)),
    base_R(1,0,0, 0,1,0, 0,0,1),
    base_T(0,0,0)
{
}

CTrackingSharedData::~CTrackingSharedData() {
    delete base_rgb;
    delete base_pointmap;
    delete active_rgb;
    delete active_depth;
}

CTrackingModule::CTrackingModule():
    shared_mutex(PTHREAD_MUTEX_INITIALIZER),
    rgb_buffer(new cv::Mat3b(480,640)),
    depth_buffer(new cv::Mat1s(480,640))
{
}

CTrackingModule::~CTrackingModule() {
    delete depth_buffer;
    delete rgb_buffer;
}

void *CTrackingModule::thread_entry(void *instance) {
    CTrackingModule *p = (CTrackingModule*)instance;
    p->run();
    return NULL;
}

void CTrackingModule::compute_pointmap(const cv::Mat1s &depth, cv::Mat3f &pointmap) {
    pointmap = cv::Mat3f::zeros(480,640);

    for(int v=0; v<depth.rows; v++)
        for(int u=0; u<depth.cols; u++) {
            const short d = depth(v,u);
            if(d==2047) 
                continue;

            cv::Matx31f p;
            float uc,vc;
            int uci,vci;
            calib.disparity2point(u,v,d,p);
            calib.point2rgb(p,uc,vc);
            uci = (int)uc+0.5f;
            vci = (int)vc+0.5f;
            if(uci<0 || uci>=pointmap.cols || vci<0 || vci>=pointmap.rows)
                continue;
            
            cv::Vec3f &point = pointmap(vci,uci);
            if(point(2) == 0 || point(2) > p(2)) {
                point(0) = p(0);
                point(1) = p(1);
                point(2) = p(2);
            }
        }
}

void CTrackingModule::cloud_from_pointmap(const cv::Mat3b &rgb, const cv::Mat3f &pointmap, boost::container::vector<pcl::PointXYZRGB> &cloud) {
    for(int v=0; v<pointmap.rows; v++)
        for(int u=0; u<pointmap.cols; u++) {
            const cv::Vec3f &pm = pointmap(v,u);
            if(pm(2)==0) 
                continue;
            const cv::Vec3b &color = rgb(v,u);

            pcl::PointXYZRGB p;
            p.x = pm(0);
            p.y = pm(1);
            p.z = pm(2);
            p.r = color(2);
            p.g = color(1);
            p.b = color(0);

            cloud.push_back(p);
        }
}

void CTrackingModule::match_features(const cv::Mat1f &new_descriptors, std::vector<int> &match_idx) {
    cv::FlannBasedMatcher matcher;
    std::vector<cv::Mat> train_vector;
    std::vector<std::vector<cv::DMatch>> matches;

    train_vector.push_back(new_descriptors);
    matcher.add(train_vector);

    match_idx.resize(shared.tracks.size());

    std::list<CFeatureTrack>::iterator track_it;
    int i;
    for(i=0,track_it=shared.tracks.begin(); track_it!=shared.tracks.end(); i++,track_it++) {
        matcher.knnMatch(track_it->descriptor, matches, 2);
        float best_dist = matches[0][0].distance;
        float next_dist = matches[0][1].distance;

        if(best_dist < 0.6*next_dist)
            match_idx[i] = matches[0][0].trainIdx;
        else
            match_idx[i] = -1;
    }
}

bool CTrackingModule::is_track_stale(const CFeatureTrack &track) {
    return track.missed_frames > 10;
}

void CTrackingModule::update_tracks(const std::vector<cv::KeyPoint> &feature_points, const cv::Mat1f &feature_descriptors, const std::vector<int> &match_idx) {
    std::list<CFeatureTrack>::iterator track_it;
    int i;
    for(i=0,track_it=shared.tracks.begin(); track_it!=shared.tracks.end(); i++,track_it++) {
        int j = match_idx[i];
        if(j == -1)
            track_it->missed_frames++;
        else {
            track_it->missed_frames = 0;
            track_it->active_position = feature_points[j].pt;
            memcpy(track_it->descriptor.data, &feature_descriptors(j,0), sizeof(float)*feature_descriptors.cols);
        }
    }
    
    //Delete tracks
    shared.tracks.remove_if(is_track_stale);
}

float CTrackingModule::get_median_feature_movement() {
    std::vector<float> vals;
    float sum=0;
    int count=0;

    std::list<CFeatureTrack>::iterator track_it;
    for(track_it=shared.tracks.begin(); track_it!=shared.tracks.end(); track_it++) {
        if(track_it->missed_frames == 0) {
            vals.push_back(fabs(track_it->base_position.x - track_it->active_position.x) + 
                fabs(track_it->base_position.y - track_it->active_position.y));
        }
    }
    
    if(vals.empty())
        return 0;
    else {
        int n = vals.size()/2;
        std::nth_element(vals.begin(),vals.begin()+n,vals.end());
        return vals[n];
    }
}

void CTrackingModule::absolute_orientation(cv::Mat1f &X, cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T) {
    cv::Matx31f meanX(0,0,0),meanY(0,0,0);
    int point_count = X.rows;
    
    //Calculate mean
    for(int i=0; i<point_count; i++) {
        meanX(0) += X(i,0);
        meanX(1) += X(i,1);
        meanX(2) += X(i,2);
        meanY(0) += Y(i,0);
        meanY(1) += Y(i,1);
        meanY(2) += Y(i,2);
    }
    meanX *= 1.0f / point_count;
    meanY *= 1.0f / point_count;

    //Subtract mean
    for(int i=0; i<point_count; i++) {
        X(i,0) -= meanX(0);
        X(i,1) -= meanX(1);
        X(i,2) -= meanX(2);
        Y(i,0) -= meanY(0);
        Y(i,1) -= meanY(1);
        Y(i,2) -= meanY(2);
    }

    //Rotation
    cv::Mat1f A;
    A = Y.t() * X;

    cv::SVD svd(A);

    cv::Mat1f Rmat;
    Rmat = svd.vt.t() * svd.u.t();
    Rmat.copyTo(R);

    //Translation
    T = meanX - R*meanY;
}

void CTrackingModule::ransac_orientation(const cv::Mat1f &X, const cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T) {
    const int max_iterations = 200;
    const int min_support = 3;
    const float inlier_error_threshold = 0.2f;

    const int pcount = X.rows;
    cv::RNG rng;
    cv::Mat1f Xk(min_support,3), Yk(min_support,3);
    cv::Matx33f Rk;
    cv::Matx31f Tk;
    std::vector<int> best_inliers;

    for(int k=0; k<max_iterations; k++) {
        //Select random points
        for(int i=0; i<min_support; i++) {
            int idx = rng(pcount);
            Xk(i,0) = X(idx,0);
            Xk(i,1) = X(idx,1);
            Xk(i,2) = X(idx,2);
            Yk(i,0) = Y(idx,0);
            Yk(i,1) = Y(idx,1);
            Yk(i,2) = Y(idx,2);
        }

        //Get orientation
        absolute_orientation(Xk,Yk,Rk,Tk);

        //Get error
        std::vector<int> inliers;
        for(int i=0; i<pcount; i++) {
            float a,b,c,errori;
            cv::Matx31f py,pyy;
            py(0) = Y(i,0);
            py(1) = Y(i,1);
            py(2) = Y(i,2);
            pyy = Rk*py+T;
            a = pyy(0)-X(i,0);
            b = pyy(1)-X(i,1);
            c = pyy(2)-X(i,2);
            errori = sqrt(a*a+b*b+c*c);
            if(errori < inlier_error_threshold) {
                inliers.push_back(i);
            }
        }

        if(inliers.size() > best_inliers.size()) {
            best_inliers = inliers;
        }
    }
    std::cout << "Inlier count: " << best_inliers.size() << "/" << pcount << "\n";

    //Do final estimation with inliers
    Xk.resize(best_inliers.size());
    Yk.resize(best_inliers.size());
    for(unsigned int i=0; i<best_inliers.size(); i++) {
        int idx = best_inliers[i];
        Xk(i,0) = X(idx,0);
        Xk(i,1) = X(idx,1);
        Xk(i,2) = X(idx,2);
        Yk(i,0) = Y(idx,0);
        Yk(i,1) = Y(idx,1);
        Yk(i,2) = Y(idx,2);
    }
    absolute_orientation(Xk,Yk,R,T);
}

void CTrackingModule::transformation_from_tracks(const cv::Mat3f &active_pointmap, cv::Matx33f &R, cv::Matx31f &T) {
    std::list<CFeatureTrack>::iterator track_it;

    cv::Mat1f X(0,3), Y(0,3);
    X.reserve(shared.tracks.size());
    Y.reserve(shared.tracks.size());

    for(track_it=shared.tracks.begin(); track_it!=shared.tracks.end(); track_it++) {
        if(track_it->missed_frames!=0) 
            continue;
        
        int ub=round<int>(track_it->base_position.x),vb=round<int>(track_it->base_position.y);
        cv::Vec3f &base_point = (*shared.base_pointmap)(vb,ub);
        if(base_point(2)==0)
            continue;
        
        int ua=round<int>(track_it->active_position.x),va=round<int>(track_it->active_position.y);
        const cv::Vec3f &active_point = active_pointmap(va,ua);
        if(active_point(2)==0)
            continue;

        //Add to matrices
        int i=X.rows;
        X.resize(i+1);
        X(i,0) = base_point(0);
        X(i,1) = base_point(1);
        X(i,2) = base_point(2);
        
        Y.resize(i+1);
        Y(i,0) = active_point(0);
        Y(i,1) = active_point(1);
        Y(i,2) = active_point(2);
    }

    ransac_orientation(X,Y,R,T);
}

void CTrackingModule::run() {
    int frame_count=0;
    bool do_tracking;

    while(!die)
    {
        /////////////////////////////////////
        // Get data from freenect
        pthread_mutex_lock(&freenect_data->mutex);
        while(!die && (freenect_data->got_rgb == 0 || freenect_data->got_depth == 0))
            pthread_cond_wait(&freenect_data->data_ready_cond, &freenect_data->mutex);
        frame_count++;

        freenect_data->got_rgb = 0;
        freenect_data->got_depth = 0;

        std::swap(freenect_data->rgb_mid, rgb_buffer);
        std::swap(freenect_data->depth_mid, depth_buffer);

        do_tracking = frame_count > 10 && shared.is_tracking_enabled; //Skip first 10 frames

        pthread_mutex_unlock(&freenect_data->mutex);

        if(!do_tracking) {
            //Update only images and skip
            pthread_mutex_lock(&shared_mutex);
            std::swap(shared.active_depth, depth_buffer);
            std::swap(shared.active_rgb, rgb_buffer);
            shared.is_data_new = true;
            pthread_mutex_unlock(&shared_mutex);
            continue;
        }

        /////////////////////////////////////
        // Tracking
        //Extract surf
        std::vector<cv::KeyPoint> feature_points; //Extracted feature points
        std::vector<float> feature_descriptors;  //Descriptor data returned by SURF

        cv::Mat1b gray_img;
        cv::cvtColor(*rgb_buffer,gray_img,CV_RGB2GRAY,1); //Convert the image to gray level

        cv::SURF mysurf(100,4,1,false,false);
        mysurf(gray_img, cv::Mat(), feature_points, feature_descriptors); //Extract SURF features
        
        //Reshape the continuous descriptor vector into a matrix
        cv::Mat1f feature_descriptors_mat(feature_points.size(), mysurf.descriptorSize(), &feature_descriptors[0], sizeof(float)*mysurf.descriptorSize());

        //Match features
        std::vector<int> match_idx;
        match_features(feature_descriptors_mat, match_idx);

        //Update images & tracks
        pthread_mutex_lock(&shared_mutex);
        std::swap(shared.active_depth, depth_buffer);
        std::swap(shared.active_rgb, rgb_buffer);
        update_tracks(feature_points,feature_descriptors_mat,match_idx);
        pthread_mutex_unlock(&shared_mutex);

        //Create new view
        bool add_view=false;
        CTrackedView new_view;
        cv::Mat3f pointmap;
        if(shared.views.empty()) {
            new_view.R = shared.base_R;
            new_view.T = shared.base_T;
            compute_pointmap(*depth_buffer,pointmap);
            cloud_from_pointmap(*rgb_buffer,pointmap,new_view.cloud);
            add_view = true;
        }
        else {
            float movement = get_median_feature_movement();
            if(movement > 120) {
                std::cout << "Movement is " << movement << " do it!";
                
                compute_pointmap(*depth_buffer,pointmap);
                cloud_from_pointmap(*rgb_buffer,pointmap,new_view.cloud);

                cv::Matx33f stepR;
                cv::Matx31f stepT;
                transformation_from_tracks(pointmap,stepR,stepT);
                new_view.R = shared.base_R * stepR;
                new_view.T = shared.base_T + shared.base_R*stepT;
                add_view = true;
            }
        }

        //Update shared data
        pthread_mutex_lock(&shared_mutex);

        if(add_view) {
            shared.views.push_back(new_view);
            shared.active_rgb->copyTo(*shared.base_rgb);
            pointmap.copyTo(*shared.base_pointmap);

            shared.base_R = new_view.R;
            shared.base_T = new_view.T;

            shared.tracks.clear();
            for(unsigned int i=0; i<feature_points.size(); i++)
            {
                CFeatureTrack track;
                track.base_position = feature_points[i].pt;
                track.active_position = track.base_position;

                track.descriptor.create(1,feature_descriptors_mat.cols);
                memcpy(track.descriptor.data, &feature_descriptors_mat(i,0),sizeof(float)*feature_descriptors_mat.cols);

                track.missed_frames = 0;

                shared.tracks.push_back(track);
            }
        }

        shared.is_data_new = true;

        pthread_mutex_unlock(&shared_mutex);
    }
}

}