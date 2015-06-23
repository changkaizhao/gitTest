
INTRODUCTION
============

这个项目是AR实验项目
仅包含图片的特征点识别和描述，特征点FLANN匹配和PROSAC计算HOMO。
在计算outlier过滤时还需要增加两步（方向过滤和几何校验）详细见优化详细说明。

目前此项目中的CVTRootViewController中，特征点数据结构为opencv提供的cv::KeyPoint。描述结构为cv::Mat。
FLANN 匹配使用的是KDRANDOMTREES(4)。都适用opencv自带算法。需要提取算法进行优化。优化见优化方案。


