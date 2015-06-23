//
//  MatchPair.h
//  OpenCVTest
//
//  Created by 赵常凯 on 15/2/27.
//  Copyright (c) 2015年 赵常凯. All rights reserved.
//

#ifndef OpenCVTest_MatchPair_h
#define OpenCVTest_MatchPair_h

#include <opencv2/core/core_c.h>

class MatchPair
{
public:
    cv::KeyPoint cam_FP;
    cv::KeyPoint ref_FP;
    cv::DMatch matchpair;
    bool isInlier;
    
    MatchPair(){}
    MatchPair(cv::KeyPoint cam,cv::KeyPoint ref,cv::DMatch dm){
        this->cam_FP = cam;
        this->ref_FP = ref;
        this->matchpair = dm;
        this->isInlier = false;
    }
    
};

#endif
