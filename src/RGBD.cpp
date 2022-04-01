#include<iostream>
#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>
#include<message_filters/subscriber.h>
#include<message_filters/time_synchronizer.h>
#include<message_filters/sync_policies/approximate_time.h>
#include<queue>

using namespace std;
class ImageGrabber{
public:
    ImageGrabber(){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD);

    cv::Mat RegionGrowing(const cv::Mat &im, int &x, int &y, const float &reg_maxdist);
    cv::Mat RegionGrowing_c(const cv::Mat &im, int &x, int &y, const float &reg_maxdist);
};

void toGray(cv::Mat& img, bool mbRGB){
    if(img.channels()==3){
        if(mbRGB)
            cv::cvtColor(img, img, CV_RGB2GRAY);
        else
            cv::cvtColor(img, img, CV_BGR2GRAY);
    }
    else if(img.channels()==4){
        if(mbRGB)
            cv::cvtColor(img, img, CV_RGBA2GRAY);
        else
            cv::cvtColor(img, img, CV_BGRA2GRAY);
    }
}

int main(int argc, char** argv){
    ros::init(argc, argv, "RGBD");
    ros::start();

    ros::NodeHandle nh;

    ImageGrabber igb;

    message_filters::Subscriber<sensor_msgs::Image> rgbImg(nh, "camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depthImg(nh, "camera/aligned_depth_to_color/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>
        syncPolice;
    message_filters::Synchronizer<syncPolice> sync(syncPolice(10), rgbImg, depthImg);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD, &igb, _1, _2));

    ros::spin();
    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD){
    cv_bridge::CvImageConstPtr cvPtrRGB;
    try
    {
        cvPtrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cvPtrD;
    try
    {
        cvPtrD = cv_bridge::toCvShare(msgD);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge expection: %s", e.what());
        return;
    }

    int x=320, y=240;
    float reg_maxdist=20;
    cv::Mat stored;
    cvPtrD->image.convertTo(stored, CV_8U);
    cv::imshow("Depth", stored);
    cv::waitKey(10);
    cv::Mat J = RegionGrowing(cvPtrD->image, x, y, reg_maxdist);
    cout << J.size() << endl;
    cv::Mat img = cvPtrRGB->image;
    toGray(img, false);
    cv::imshow("img", img);
    cv::waitKey(10);
}

cv::Mat ImageGrabber::RegionGrowing_c(const cv::Mat &im, int &x, int &y, const float &reg_maxdist){
    cv::Mat J = cv::Mat::zeros(im.size(), CV_32F);

    float reg_mean = im.at<float>(y, x);
    int reg_size = 1;

    int _neg_free = 10000;
    int neg_free = 10000;
    int neg_pos = -1;
    cv::Mat neg_list = cv::Mat::zeros(neg_free, 3, CV_32F);

    double pixdist=0;

    cv::Mat neigb(4, 2, CV_32F);

    neigb.at<float>(0,0) = -1;
    neigb.at<float>(0,1) = 0;
    neigb.at<float>(1,0) = 1;
    neigb.at<float>(1,1) = 0;
    neigb.at<float>(2,0) = 0;
    neigb.at<float>(2,1) = -1;
    neigb.at<float>(3,0) = 0;
    neigb.at<float>(3,1) = 1;

    while(pixdist < reg_maxdist && reg_size<im.total())
    {
        for(int j=0; j<4; j++)
        {
            int xn = x+neigb.at<float>(j, 0);
            int yn = y+neigb.at<float>(j, 1);
            bool ins=((xn >= 0) && (yn >= 0) && (xn < im.cols) && (yn < im.rows));

            if(ins && (J.at<float>(yn, xn)==0.) )
            {
                neg_pos++;
                neg_list.at<float>(neg_pos, 0) = xn;
                neg_list.at<float>(neg_pos, 1) = yn;
                neg_list.at<float>(neg_pos, 2) = im.at<float>(yn, xn);
                J.at<float>(yn, xn)=1;
            }
        }

        if((neg_pos+10) > neg_free){
            cout << "expand mat." << endl;
            cv::Mat _neg_list = cv::Mat::zeros(_neg_free, 3, CV_32F);
            neg_free += _neg_free;
            cout << neg_list.size() << endl;
            cv::vconcat(neg_list, _neg_list, neg_list);
            cout << neg_list.size() << endl;
        }

        cv::Mat dist;
        for(int i=0; i<neg_pos; i++)
        {
            double d = abs(neg_list.at<float>(i, 2) - reg_mean);
            dist.push_back(d);
        }

        double max;
        cv::Point ind, maxpos;
        cv::minMaxLoc(dist, &pixdist, &max, &ind, &maxpos);
        int index = ind.y;

        if(index != -1){
            J.at<float>(y,x) = -1;
            reg_size += 1;

            reg_mean = (reg_mean*reg_size +neg_list.at<float>(index, 2)/(reg_size+1));

            x = neg_list.at<float>(index, 0);
            y = neg_list.at<float>(index, 1);

            neg_list.at<float>(index, 0) = neg_list.at<float>(neg_pos, 0);
            neg_list.at<float>(index, 1) = neg_list.at<float>(neg_pos, 1);
            neg_list.at<float>(index, 2) = neg_list.at<float>(neg_pos, 2);
            neg_pos-=1;
        }
        else{
            pixdist = reg_maxdist;
        }
    }

    J = cv::abs(J);
    return J;
}

cv::Mat ImageGrabber::RegionGrowing(const cv::Mat &im, int &x, int &y, const float &reg_maxdist){
    cv::Mat J = cv::Mat::zeros(im.size(), CV_8U);

    ushort reg_mean = im.at<ushort>(y, x);
    if(reg_mean<=0) return cv::Mat(1, 1, CV_8U);
    cout << reg_mean << endl;
    int reg_size = 1;

    int _neg_free = 10000;
    int neg_free = 10000;
    int neg_pos = -1;
    cv::Mat neg_list = cv::Mat::zeros(neg_free, 3, CV_16U);

    double pixdist=0;

    cv::Mat neigb(4, 2, CV_16U);

    neigb.at<ushort>(0,0) = -1;
    neigb.at<ushort>(0,1) = 0;
    neigb.at<ushort>(1,0) = 1;
    neigb.at<ushort>(1,1) = 0;
    neigb.at<ushort>(2,0) = 0;
    neigb.at<ushort>(2,1) = -1;
    neigb.at<ushort>(3,0) = 0;
    neigb.at<ushort>(3,1) = 1;


    while(pixdist < reg_maxdist && reg_size<im.total())
    {
        for(int j=0; j<4; j++)
        {
            ushort xn = x+neigb.at<ushort>(j, 0);
            ushort yn = y+neigb.at<ushort>(j, 1);
            bool ins=((xn >= 0) && (yn >= 0) && (xn < im.cols) && (yn < im.rows));

            if(ins && (J.at<uchar>(yn, xn)==0) )
            {
                cout << xn << ", " << yn << ", " << im.at<ushort>(yn, xn) << endl;
                neg_pos++;
                neg_list.at<ushort>(neg_pos, 0) = xn;
                neg_list.at<ushort>(neg_pos, 1) = yn;
                neg_list.at<ushort>(neg_pos, 2) = im.at<ushort>(yn, xn);
                J.at<uchar>(yn, xn)=255;
            }
        }

        if((neg_pos+10) > neg_free){
            cout << "expand mat." << endl;
            cv::Mat _neg_list = cv::Mat::zeros(_neg_free, 3, CV_16U);
            neg_free += _neg_free;
            cout << neg_list.size() << endl;
            cv::vconcat(neg_list, _neg_list, neg_list);
            cout << neg_list.size() << endl;
        }

        cv::Mat dist;
        for(int i=0; i<neg_pos; i++)
        {
            int d = abs(neg_list.at<ushort>(i, 2) - reg_mean);
            cout << neg_list.at<ushort>(i, 2) << ", " << reg_mean << endl;
            cout << d << endl;
            dist.push_back(d);
        }

        double max;
        cv::Point ind, maxpos;
        cv::minMaxLoc(dist, &pixdist, &max, &ind, &maxpos);
        int index = ind.y;

        if(index != -1){
            reg_mean = (reg_mean*reg_size +neg_list.at<ushort>(index, 2))/(reg_size+1);
            reg_size += 1;

            x = neg_list.at<ushort>(index, 0);
            y = neg_list.at<ushort>(index, 1);

            neg_list.at<ushort>(index, 0) = neg_list.at<ushort>(neg_pos, 0);
            neg_list.at<ushort>(index, 1) = neg_list.at<ushort>(neg_pos, 1);
            neg_list.at<ushort>(index, 2) = neg_list.at<ushort>(neg_pos, 2);
            neg_pos-=1;
        }
        else{
            pixdist = reg_maxdist;
        }
    }
    cv::imshow("J", J);
    cv::waitKey(10);

    return J;
}