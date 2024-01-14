//
// Created by shaobing2 on 8/24/23.
//

#include "inference.h"
static constexpr int INPUT_W = 640;
static constexpr int INPUT_H = 640;
static constexpr float SCORE_THRESHOLD = 0.2;
static constexpr float NMS_THRESHOLD = 0.4;
static constexpr float CONFIDENCE_THRESHOLD = 0.4;

static inline int argmax(const float *ptr, int len) {
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}


constexpr float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

template<class F, class T, class ...Ts>
T reduce(F &&func, T x, Ts... xs) {
    if constexpr (sizeof...(Ts) > 0){
        return func(x, reduce(std::forward<F>(func), xs...));
    } else {
        return x;
    }
}

template<class T, class ...Ts>
T reduce_max(T x, Ts... xs) {
    return reduce([](auto &&a, auto &&b){return std::max(a, b);}, x, xs...);
}

template<class T, class ...Ts>
T reduce_min(T x, Ts... xs) {
    return reduce([](auto &&a, auto &&b){return std::min(a, b);}, x, xs...);
}

inline cv::Mat BuffDetector::scaledResize(cv::Mat& img)
{
    float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));

    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;

    dw /= 2;
    dh /= 2;

    transfrom_matrix << 1.0 / r, 0, -dw / r,
            0, 1.0 / r, -dh / r,
            0, 0, 1;

    cv::Mat re;
    cv::resize(img, re, cv::Size(unpad_w,unpad_h));
    cv::Mat out;
    cv::copyMakeBorder(re, out, dh, dh, dw, dw, CV_HAL_BORDER_CONSTANT);

    return out;
}


bool BuffDetector::initModel(std::string path) {
    model = core.read_model(path);

    // Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    compiled_model = core.compile_model(model, "CPU");
    return true;
}
long long name_index = 0;
bool BuffDetector::detect(cv::Mat &src, vector<BuffObject> &output) {
    cv::Mat pr_img = scaledResize(src);
#ifdef SHOW_INPUT
    cv::namedWindow("network_input",0);
    imshow("network_input",pr_img);
#ifdef IMWRITE
    name_index++;
    if (name_index%100==0) {
        imwrite("/home/bubing2/buffdata/" + to_string(name_index/100) + ".jpg", pr_img);
    }
#endif
    cv::waitKey(1);
#endif //SHOW_INPUT
    auto *input_data = (float *) pr_img.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    auto *detections = output_tensor.data<float>();
    vector<int> cls;
    vector<int> color;
    vector<float> confidences;
    vector<cv::Point2f*> ppts;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < output_shape[1]; i++){
        float *detection = &detections[i * output_shape[2]];

        float confidence = detection[10];
        if (confidence >= CONFIDENCE_THRESHOLD){
            cv::Rect2f bbox1;
            bbox1.x = reduce_min(detection[0], detection[2], detection[4], detection[6], detection[8]);
            bbox1.y = reduce_min(detection[1], detection[3], detection[5], detection[7], detection[9]);
            bbox1.width = reduce_max(detection[0], detection[2], detection[4], detection[6], detection[8]) - bbox1.x;
            bbox1.height = reduce_max(detection[1], detection[3], detection[5], detection[7], detection[9]) - bbox1.y;
            color.push_back(argmax(detection + 11, 2));
            cls.push_back(argmax(detection + 13, 3));
            auto ppt = new cv::Point2f[5];
            ppt[0] = cv::Point2f(detection[0],detection[1]);
            ppt[1] = cv::Point2f(detection[2],detection[3]);
            ppt[2] = cv::Point2f(detection[4],detection[5]);
            ppt[3] = cv::Point2f(detection[6],detection[7]);
            ppt[4] = cv::Point2f(detection[8],detection[9]);
            ppts.push_back(ppt);
            boxes.push_back(bbox1);
            confidences.push_back(confidence);
        }
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int idx : nms_result)
    {
        if (cls[idx] == 0) continue;
        BuffObject result;
        Eigen::Matrix<float,3,5> apex_norm;
        Eigen::Matrix<float,3,5> apex_dst;

        apex_norm << ppts[idx][0].x,ppts[idx][1].x,ppts[idx][2].x,ppts[idx][3].x,ppts[idx][4].x,
                ppts[idx][0].y,ppts[idx][1].y,ppts[idx][2].y,ppts[idx][3].y,ppts[idx][4].y,
                1,1,1,1,1;

        apex_dst = transfrom_matrix * apex_norm;

        for (int i = 0; i < 5; i++)
        {
            result.apex[i] = cv::Point2f(apex_dst(0,i),apex_dst(1,i));
        }
        result.prob = confidences[idx];
        result.rect = boxes[idx];
        result.color = color[idx];
        if (cls[idx] == 1)
        {
            result.cls = 1;
        }
        else if (cls[idx] == 2)
        {
            result.cls = 0;
        }

        output.push_back(result);
    }
    for(auto & ppt : ppts) {
        delete[] ppt;
    }

    return true;
}