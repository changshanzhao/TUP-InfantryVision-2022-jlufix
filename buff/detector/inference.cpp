//
// Created by shaobing2 on 8/24/23.
//

#include "inference.h"
#define SHOW_BUFF
static constexpr int INPUT_W = 640;
static constexpr int INPUT_H = 640;
static constexpr float SCORE_THRESHOLD = 0.2;
static constexpr float NMS_THRESHOLD = 0.4;
static constexpr float CONFIDENCE_THRESHOLD = 0.4;

static string name[6] = {"BR","BU","BA","RR","RU","RA"};
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

Resize resize_and_pad(cv::Mat& img, cv::Size new_shape) {
    float width = img.cols;
    float height = img.rows;
    auto r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    cv::Scalar color = cv::Scalar(100, 100, 100);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

    return resize;
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
}

bool BuffDetector::detect(cv::Mat &src, vector<BuffObject> &output) {
    Resize res = resize_and_pad(src, cv::Size(INPUT_W, INPUT_H));
    auto *input_data = (float *) res.resized_image.data;
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
        BuffObject result;
        for(int i = 0; i < 5; i++) {
            result.apex[i] = ppts[idx][i];
        }
        result.prob = confidences[idx];
        result.rect = boxes[idx];
        result.color = color[idx];
        result.cls = cls[idx];
        output.push_back(result);
    }
    for(auto & ppt : ppts) {
        delete[] ppt;
    }

    for (auto detection : output) {
        auto ppt = detection.apex;
        auto cls = detection.cls;
        auto color = detection.color;
        auto classId = color * 3 + cls;
        float rx = (float) src.cols / (float) (res.resized_image.cols - res.dw);
        float ry = (float) src.rows / (float) (res.resized_image.rows - res.dh);
        for (int i = 0; i < 5; i++) {
            ppt[i].x = rx * ppt[i].x;
            ppt[i].y = ry * ppt[i].y;
        }
#ifdef SHOW_BUFF
        line(src, ppt[0], ppt[1], cv::Scalar(0, 255, 0), 2);
        line(src, ppt[1], ppt[2], cv::Scalar(0, 255, 0), 2);
        line(src, ppt[2], ppt[3], cv::Scalar(0, 255, 0), 2);
        line(src, ppt[3], ppt[4], cv::Scalar(0, 255, 0), 2);
        line(src, ppt[4], ppt[0], cv::Scalar(0, 255, 0), 2);
        putText(src, name[classId], ppt[0], 1, 1, cv::Scalar(0, 255, 0));
        cv::namedWindow("network_input",0);
        imshow("network_input",src);
#endif
    }

}