#pragma once
#include "Model.h"
#include <fstream>
#include <print>


class TableSeg:public yo::Model{

struct ParamsSeg{
    float bin = 0.5f;
    float score = 0.5f;
    float nms = 0.5f;
};
struct Boxinfo{
    float conf;
    cv::Rect box;
};

using SegResBox = std::vector<Boxinfo>;
using SegResMask = std::vector<cv::Mat>;

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;

    ParamsSeg params;
    std::vector<yo::Node> input_nodes;
    std::vector<yo::Node> output_nodes;
    std::vector<cv::Mat> input_images;

    Ort::Session* session = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolov10");
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image)override;
    void postprocess(std::vector<Ort::Value>& output_tensors)override;

    std::tuple<SegResBox,SegResMask> mask_decode(float* a,float* b,std::vector<int64_t>& a_shape,std::vector<int64_t>& b_shape);


public:
    TableSeg(){};
    TableSeg(const TableSeg&) = delete;// 删除拷贝构造函数
    TableSeg& operator=(const TableSeg&) = delete;// 删除赋值运算符
    ~TableSeg(){if(session != nullptr) delete session;};
    int setparms(ParamsSeg parms);
    std::variant<bool,std::string> initialize(std::vector<std::string>& onnx_paths, bool is_cuda) override;
    std::variant<bool,std::string> inference(cv::Mat &image) override;
    
};