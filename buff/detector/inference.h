//
// Created by shaobing2 on 8/24/23.
//

#ifndef WIND_INFERENCE_INFERENCE_H
#define WIND_INFERENCE_INFERENCE_H

#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "../../debug.h"

using namespace std;


struct BuffObject
{
    int cls;
    int color;
    float prob;
    cv::Point2f apex[5];
    cv::Rect rect;
};


class BuffDetector
{
public:

    bool detect(cv::Mat &src,vector<BuffObject>& output);
    bool initModel(string path);
    cv::Mat scaledResize(cv::Mat& img);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    Eigen::Matrix<float,3,3> transfrom_matrix;

};


#endif //WIND_INFERENCE_INFERENCE_H
