//
//  CVTRootViewController.m
//  OpenCVTest
//
//  Created by 赵常凯 on 14-8-15.
//  Copyright (c) 2014年 赵常凯. All rights reserved.
//
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif /* __ARM_NEON */

#import "CVTRootViewController.h"
#import <opencv2/nonfree/features2d.hpp>
//#import <opencv2/features2d/features2d.hpp>
#import <stdio.h>
#import <opencv2/core/core_c.h>
#import <UIKit/UIImage.h>
#import <CoreGraphics/CGImage.h>
#import <UIKit/UIImagePickerController.h>
#include "MatchPair.h"
#include <iostream>

const float reprojectionError = 2.0;
const float confidence = 0.995;
const int maxIteration = 1000;

@interface CVTRootViewController (){

}

@property int image_width;
@property int image_height;
@property (nonatomic, retain) CvVideoCamera* videoCamera;
@property cv::SiftFeatureDetector siftdetctor;
@property cv::ORB orb;

@property cv::SiftDescriptorExtractor siftdescriptor;

@property std::vector<cv::KeyPoint> keypts;
@property cv::Mat desc;
@property cv::Mat output;

@property (nonatomic, retain) UILabel* label;
@property (nonatomic,retain) NSMutableString *dur;

//检测时间
@property double duration;
//描述时间
@property double duration_desc;
//匹配时间
@property double duration_match;
//PROSAC time
@property double duration_PROSAC;

@property int num_matched_FP;


@property bool saved;
@property (nonatomic, strong)   UIImage* savedimg;


@property (nonatomic, strong)   UIImage* refUIimg;
@property std::vector<cv::KeyPoint> refkeypt;
@property std::vector<cv::DMatch> matched;
@property cv::Mat refDesc;


//Flann匹配
//@property cv::FlannBasedMatcher flannmatcher;
@property cv::flann::Index* flannIndex;

//匹配的特征点
@property std::vector<cv::KeyPoint> matchedrefkpt;
@property std::vector<cv::KeyPoint> matchedcamkpt;

//PROSAC
@property cv::Mat homo;
@property std::vector<MatchPair> matchedPoints;
@property std::vector<cv::DMatch> goodMatched;

@end

#pragma mark - implementation
@implementation CVTRootViewController

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        // Custom initialization
        self.saved = false;
    }
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    UIButton *btn = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    btn.frame = CGRectMake(50, 420, 100, 50);
    [btn setTitle:@"START" forState:UIControlStateNormal];
    [btn addTarget:self  action:@selector(pressed:) forControlEvents:UIControlEventTouchUpInside];
    

    self.label = [[UILabel alloc] initWithFrame:CGRectMake(10, 5, 250, 145)];
    //self.label.text = @"go";
    self.label.textColor = [UIColor greenColor];
    self.label.numberOfLines = 0;
    self.label.preferredMaxLayoutWidth = self.label.bounds.size.width;
    
    UIImageView *imageview = [[UIImageView alloc] initWithFrame:CGRectMake(-20, 0, 360, 480)];
    imageview.backgroundColor = [UIColor grayColor];
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:imageview];
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    //self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetLow;//192*144
    //self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;

    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    self.videoCamera.grayscaleMode = NO;
    self.videoCamera.delegate = self;

    //读ref img
    self.refUIimg = [UIImage imageNamed:@"chipslow.jpg"];
    cv::Mat reftemp = [self cvMatFromUIImage:self.refUIimg];

    _image_width = reftemp.cols;
    _image_height = reftemp.rows;
    //初始化一些设置
    //self.orb = cv::ORB(1000,2,8,31,0,2,cv::ORB::HARRIS_SCORE,31);
    self.siftdetctor.detect(reftemp, _refkeypt);
    self.siftdescriptor.compute(reftemp, _refkeypt, _refDesc);
   
    //const cv::Ptr<cv::flann::IndexParams>& indexParams = new cv::flann::KDTreeIndexParams(4);
    //const cv::Ptr<cv::flann::IndexParams>& indexParams = new cv::flann::KMeansIndexParams(15,5);
//    const cv::Ptr<cv::flann::IndexParams>& indexParams = new cv::flann::HierarchicalClusteringIndexParams;
//    const cv::Ptr<cv::flann::SearchParams>& searchParams = new cv::flann::SearchParams(128);
//    self.flannmatcher = cv::FlannBasedMatcher(indexParams,searchParams);
    
    //构造搜索索引树
    assert(_refDesc.cols == 128);
    self.flannIndex = new cv::flann::Index(_refDesc, cv::flann::KDTreeIndexParams(4));
    
    //装载视图
    [self.view addSubview:imageview];
    [self.view addSubview:btn];
    [self.view addSubview:self.label];

}

//改变label时间
- (void) changelabel :(NSMutableString *)str
{
    //检测时间
    self.dur = [[NSMutableString alloc] initWithString:@"Detecting time:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%.3f",self.duration ] mutableCopy];
    [self.dur appendString:@" ms\n"];
    //描述时间
    [self.dur appendString:@"Descripting time:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%.3f",self.duration_desc ] mutableCopy];
    [self.dur appendString:@" ms\n"];
    //匹配时间
    [self.dur appendString:@"Matching time:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%.3f",self.duration_match ] mutableCopy];
    [self.dur appendString:@" ms\n"];
    //PROSAC时间
    [self.dur appendString:@"PROSAC time:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%.3f",self.duration_PROSAC ] mutableCopy];
    [self.dur appendString:@" ms\n"];
    //特征点数量
    [self.dur appendString:@"REF_FP_No.:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%lu \n",_refkeypt.size() ] mutableCopy];
    [self.dur appendString:@"CAM_FP_No.:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%lu \n",_keypts.size() ] mutableCopy];
    [self.dur appendString:@"Matched_FP_No.:"];
    self.dur = [[self.dur stringByAppendingFormat:@"%d \n",self.num_matched_FP ] mutableCopy];
    
    //更新
    [self.label setText:self.dur];
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

#pragma mark - save photos

- (void)image: (UIImage *) image didFinishSavingWithError: (NSError *) error contextInfo: (void *) contextInfo
{
    NSString *msg = nil ;
    if(error != NULL){
        msg = @"保存图片失败" ;
    }else{
        msg = @"保存图片成功" ;
    }
    UIAlertView *alert = [[UIAlertView alloc] initWithTitle:@"保存图片结果提示"
                                                    message:msg
                                                   delegate:self
                                          cancelButtonTitle:@"确定"
                                          otherButtonTitles:nil];
    [alert show];
    
}
- (void)saveImageToPhotos
{
    UIImageWriteToSavedPhotosAlbum(self.savedimg, self, @selector(image:didFinishSavingWithError:contextInfo:), NULL);
}

#pragma mark - Matching
- (int) Matching:(cv::Mat) descriptor distance:(double &) distance
{
    int index = -1;
    
    if (descriptor.empty()) {
        return index;
    }
    
    cv::Mat resultIndex(1, 2, CV_32S);
    cv::Mat resultDistance(1, 2, CV_32FC1);
    
    double minDistance1 = 0;
    double minDistance2 = 0;
    
    self.flannIndex->knnSearch(descriptor, resultIndex, resultDistance, 2, cv::flann::SearchParams(20));
    
    minDistance1 = (double)(resultDistance.ptr<float>(0)[0]);
    minDistance2 = (double)(resultDistance.ptr<float>(0)[1]);
    
    
    int minIndex1 = resultIndex.ptr<int>(0)[0];
    int minIndex2 = resultIndex.ptr<int>(0)[1];
    
    index = minIndex1;
    if(minDistance2 < minDistance1)
    {
        double temp = minDistance1;
        minDistance1 = minDistance2;
        minDistance2 = temp;
        index = minIndex2;
    }
    
    
    if(minDistance1 > minDistance2 * 0.7) {  // 0.7 = nearestNeighbourhoodRatio
        return -1;
    }
    
    distance = minDistance1;
    
    return index;
}

#pragma mark - PROSAC funcs
double ComputeReprojError(cv::Point2f refPoint, cv::Point2f scePoint, double* homography)
{
    double ww = 1./(homography[6] * refPoint.x + homography[7] * refPoint.y + homography[8]);
    double x = (homography[0] * refPoint.x + homography[1] * refPoint.y +  homography[2]) * ww;
    double y = (homography[3] * refPoint.x + homography[4] * refPoint.y +  homography[5]) * ww;
    double error = sqrt((scePoint.x - x)*(scePoint.x - x)) + sqrt((scePoint.y - y)*(scePoint.y - y));
    return error;
}
int RANSACUpdateNumIters(double p, double ep, int model_points, int max_iters)
{
    // p stands for 1- ƞ0   ,ƞ0 is confidence default  = 5%
    // ep is the probability of outliers
    // model_points is sampling count = 4
				double num, denom;
				p = MAX(p, 0.);
				p = MIN(p, 1.);
				ep = MAX(ep, 0.);
				ep = MIN(ep, 1.);
    
				// avoid inf's & nan's
				num = MAX(1. - p, DBL_MIN);
				denom = 1. - pow(1. - ep,model_points);
				num = log(num);
				denom = log(denom);
    
				int result = denom >= 0 || -num >= max_iters*(-denom) ? max_iters : cvRound(num/denom);
				return result;
}
// for sort lambda func
typedef struct _CompareDistanceLess
{
    bool operator()(const MatchPair& p, const MatchPair& q) const
    {
        return p.matchpair.distance < q.matchpair.distance;
    }
}CompareDistanceLess;

- (bool) PROSAC_estimator:(std::vector<cv::KeyPoint> &)camFPs
                  refKpts:(std::vector<cv::KeyPoint> &)refFPs
              matchedkpts:(std::vector<cv::DMatch> &)matches
              matchedpair:(std::vector<MatchPair> &)matchedPoints
               homoMatrix:(cv::Mat &)homo
{
    
    int n = (int)refFPs.size();
    if(n != (int)camFPs.size())
        return NO;
    if(n < 5)
        return NO;
    
    std::cout<<"Prosac estimator~"<<std::endl;
    double h[9];
    double bestHomography[9];
    CvMat _h = cvMat(3, 3, CV_64F, h);
    
    std::vector<CvPoint2D64f> samplingObject;		samplingObject.resize(4);
    std::vector<CvPoint2D64f> samplingReference;	samplingReference.resize(4);
    CvMat samplingObjectPoints = cvMat(1, 4, CV_64FC2, &(samplingObject[0]));
    CvMat samplingReferencePoints = cvMat(1, 4, CV_64FC2, &(samplingReference[0]));
    
    CvRNG rng = cvRNG(cvGetTickCount());
    int bestCount = 0;
    int count = (int)refFPs.size();
    
    matchedPoints.clear();
    // copy matching point information
    for (int i = 0; i < count; i++) {
        matchedPoints.push_back(MatchPair(camFPs[i], refFPs[i],matches[i]));
    }
    
    sort(matchedPoints.begin(), matchedPoints.end(), CompareDistanceLess()); //不排序也不影响效果。⚠
    
    int samplingCount = 4;
    int maxIter = 1000; // default set maxIteration = 1000
    for(int i=0; i<maxIter; i++)
    {
        // reset
        for(int j=0; j<count; j++)
        {
            matchedPoints[j].isInlier = false;
        }
        
        // sampling
        double Tn1 = (double)samplingCount; //start from 4
        double Tn = Tn1 * (double)(count + 1) / (double)(count + 1 - samplingCount);
        samplingCount = samplingCount + (int)(Tn - Tn1 + 1.0);
        samplingCount = MIN(count-1, samplingCount);
        
        int index[4] = {-1, -1, -1, -1};
        for(int j=0; j<4; j++)
        {
            int tempIndex = cvRandInt(&rng) % samplingCount;
            while(index[0] == tempIndex || index[1] == tempIndex || index[2] == tempIndex)
            {
                tempIndex = cvRandInt(&rng) % samplingCount;
            }
            index[j] = tempIndex;
        }
        
        for(int j=0; j<4; j++)
        {
            int tempIndex = index[j];
            
            samplingObject[j].x = matchedPoints[tempIndex].cam_FP.pt.x;
            samplingObject[j].y = matchedPoints[tempIndex].cam_FP.pt.y;
            
            samplingReference[j].x = matchedPoints[tempIndex].ref_FP.pt.x;
            samplingReference[j].y = matchedPoints[tempIndex].ref_FP.pt.y;
        }
        
        // calculate homograpy
        cvFindHomography(&samplingReferencePoints, &samplingObjectPoints, &_h);
        
        int inlinerCount = 0;
        // calculate consensus set
        for(int j=0; j<count; j++)
        {
            double error = ComputeReprojError(matchedPoints[j].ref_FP.pt, matchedPoints[j].cam_FP.pt, h);
            if(error < reprojectionError)     //default reprojectionError = 2.0
            {
                matchedPoints[j].isInlier = true;
                inlinerCount++;
            }
        }
        
        if(inlinerCount > bestCount)
        {
            bestCount = inlinerCount;
            for(int k=0; k<9; k++)
                bestHomography[k] = h[k];
            
            if(confidence > 0) //default confidence = 0.995
                maxIter = RANSACUpdateNumIters(confidence, (double)(count - inlinerCount)/(double)count, 4, maxIteration);
        }
    }
    std::cout<<"terminate Prosac~"<<std::endl;
    // terminate
    if(bestCount >= 4)
    {
        //        LOGD("ProSACestimator::Calculate bestCount:%d \n", bestCount);
        
        for(int j=0; j<count; j++)
        {
            double error = ComputeReprojError(matchedPoints[j].ref_FP.pt, matchedPoints[j].cam_FP.pt, bestHomography);
            if(error < reprojectionError)
            {
                matchedPoints[j].isInlier = true;
            }
            else
            {
                matchedPoints[j].isInlier = false;
            }
        }
        
        std::vector<CvPoint2D64f> consensusReference;
        std::vector<CvPoint2D64f> consensusObject;
        
        for(int j=0; j<count; j++)
        {
            if(matchedPoints[j].isInlier)
            {
                consensusReference.push_back(cvPoint2D64f(matchedPoints[j].ref_FP.pt.x, matchedPoints[j].ref_FP.pt.y));
                consensusObject.push_back(cvPoint2D64f(matchedPoints[j].cam_FP.pt.x, matchedPoints[j].cam_FP.pt.y));
            }
        }
        
        CvMat consensusReferencePoints = cvMat(1, (int)consensusReference.size(), CV_64FC2, &(consensusReference[0]));
        CvMat consensusObjectPoints = cvMat(1, (int)consensusObject.size(), CV_64FC2, &(consensusObject[0]));
        
        cvFindHomography(&consensusReferencePoints, &consensusObjectPoints, &_h);
        // update
        //            for(int i=0; i<9; i++)
        //            this->homography.m1[i] = h[i];
        
        homo = (cv::Mat_<float>(3,3) << h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7],h[8]);
        //            this->DecomposeHomography(this->cameraParameter);
        std::cout<<"get a homo"<<std::endl;
        return YES;
    }
    return NO;
    
}

#pragma mark - draw rectangle
void drawRec(cv::Mat &img,int width,int height,cv::Mat homo){
    float h6 = homo.at<float>(2,0);
    float h7 = homo.at<float>(2,1);
    float h8 = homo.at<float>(2,2);
    
    float top_l_w = 1/h8;
    float bot_l_w = 1/(h7*height+h8);
    float top_r_w = 1/(h6*width+h8);
    float bot_r_w = 1/(h6*width+h7*height+h8);
    
    cv::Mat t_left = homo*(cv::Mat_<float>(3,1) << 0,0,1)*top_l_w;
    cv::Mat b_left = homo*(cv::Mat_<float>(3,1) << 0,height,1)*bot_l_w;
    cv::Mat t_right = homo*(cv::Mat_<float>(3,1) << width,0,1)*top_r_w;
    cv::Mat b_right = homo*(cv::Mat_<float>(3,1) << width,height,1)*bot_r_w;
    cv::line(img, cv::Point2f(t_left.at<float>(0,0),t_left.at<float>(1,0)), cv::Point2f(b_left.at<float>(0,0),b_left.at<float>(1,0)), CV_RGB(255, 255, 0),3);
    cv::line(img, cv::Point2f(t_left.at<float>(0,0),t_left.at<float>(1,0)), cv::Point2f(t_right.at<float>(0,0),t_right.at<float>(1,0)), CV_RGB(255, 255, 0),3);
    cv::line(img, cv::Point2f(b_right.at<float>(0,0),b_right.at<float>(1,0)), cv::Point2f(t_right.at<float>(0,0),t_right.at<float>(1,0)), CV_RGB(255, 255, 0),3);
    cv::line(img, cv::Point2f(b_right.at<float>(0,0),b_right.at<float>(1,0)), cv::Point2f(b_left.at<float>(0,0),b_left.at<float>(1,0)), CV_RGB(255, 255, 0),3);
}


#pragma mark - Protocol CvVideoCameraDelegate

#ifdef __cplusplus


- (void)processImage:(cv::Mat&)image
{
    
    //从摄像头读取的帧直接会显示出来
    self.num_matched_FP = 0;

    //做一个检测在这里
    double time_start=(double)cv::getTickCount();
    self.siftdetctor.detect(image, _keypts);
   // self.orb.detect(image, _keypts);
    self.duration = ((double)cv::getTickCount() - time_start)*1000/cv::getTickFrequency();
    
    //做描述
    double des_time_start=(double)cv::getTickCount();
    self.siftdescriptor.compute(image, _keypts, _desc);
    self.duration_desc = ((double)cv::getTickCount() - des_time_start)*1000/cv::getTickFrequency();

    //做匹配
    double match_time_start=(double)cv::getTickCount();
    
//    self.flannmatcher.match(_desc, _refDesc, _matched);
//    double min_dist = 100;
//    for( int i = 0; i < _matched.size(); i++ )
//    { double dist = _matched[i].distance;
//        if( dist < min_dist ) min_dist = dist;
//    }
    
    _matched.clear();
    _matchedrefkpt.clear();
    _matchedcamkpt.clear();
    _goodMatched.clear();
    if (!_desc.empty()) {
        
        for (int i = 0 ; i < _desc.rows; i++) {
            cv::DMatch dmatch;
            double distance;
            dmatch.queryIdx = i;
            dmatch.trainIdx = [self Matching:_desc.row(i) distance:distance];
            dmatch.distance = distance;
            if( dmatch.trainIdx != -1)
            {
                _matchedcamkpt.push_back(_keypts[i]);
                _matchedrefkpt.push_back(_refkeypt[dmatch.trainIdx]);
                _goodMatched.push_back(dmatch);
            }
            _matched.push_back(dmatch);
        }
    }
    
    
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    //-- PS.- radiusMatch can also be used here.
    // std::vector< cv::DMatch > good_matches;
    self.duration_match = ((double)cv::getTickCount() - match_time_start)*1000/cv::getTickFrequency();

    //做PROSAC
    double prosac_time_start=(double)cv::getTickCount();

    [self PROSAC_estimator:_matchedcamkpt refKpts:_matchedrefkpt matchedkpts:_goodMatched matchedpair:_matchedPoints homoMatrix:_homo];
    
    self.duration_PROSAC = ((double)cv::getTickCount() - prosac_time_start)*1000/cv::getTickFrequency();


    //绘制特征点
    cv::Scalar colorGREEN = cv::Scalar( 0, 255, 0 );
    cv::Scalar colorRED = cv::Scalar( 0, 0, 255 );

    //绘制方框
    if (!_homo.empty()) {
        drawRec(image, _image_width , _image_height, _homo);
    }

    //绘制特征点
    for( int i = 0; i < _matched.size(); i++ )
    {
        cv::Point center( _keypts[i].pt.x, _keypts[i].pt.y );
        cv::Size size( 1, 1 );
        if( _matched[i].trainIdx != -1)
        {
            //good_matches.push_back( matches[i]);
            cv::ellipse( image, center, size, 0, 0, 360, colorGREEN, 1, 8, 0 );
            self.num_matched_FP++;
        }else{
            cv::ellipse( image, center, size, 0, 0, 360, colorRED, 1, 8, 0 );

        }
        
        
    }
    
    //更新显示标签
    [self performSelectorOnMainThread:@selector(changelabel:) withObject:nil waitUntilDone:NO];
    
    //保存照片
//    if (!self.saved) {
//        //save a pic
//        self.savedimg = [self UIImageFromCVMat:image];
//        [self performSelectorOnMainThread:@selector(saveImageToPhotos) withObject:nil waitUntilDone:NO];
//        self.saved = true;
//    }

}


#endif

- (void)pressed:(UIButton *)btn
{
    [self.videoCamera start];

}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender
{
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
