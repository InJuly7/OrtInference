#include "TableSeg.h"

int TableSeg::setparms(ParamsSeg parms){
    this->params = std::move(parms);
    return 1;
}

std::variant<bool,std::string> TableSeg::initialize(std::vector<std::string>& onnx_paths, bool is_cuda){
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    std::string& onnx_path = onnx_paths[0];  
    if (!is_file(onnx_path)) {
        return std::format("Model file dose not exist.file:{}",onnx_path);
    }
    this->session_options.SetIntraOpNumThreads(4);
    if (is_cuda) {
        try {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic;
            options.do_copy_in_default_stream = 1;
            session_options.AppendExecutionProvider_CUDA(options);
            std::println("Using CUDA...");

            // const auto& api = Ort::GetApi();
            // OrtTensorRTProviderOptionsV2* tensorrt_options;
            // Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
            // std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
            // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(this->session_options),rel_trt_options.get()));
            // std::println("Using TensorRT...");
        }catch (const std::exception& e) {
            std::string error(e.what());
            return error;
        }
    }else {
        std::println("Using CPU...");
    }
    try {
#ifdef _WIN32
        unsigned len = onnx_path.size() * 2; // 预留字节数
        setlocale(LC_CTYPE, ""); //必须调用此函数,本地化
        wchar_t* p = new wchar_t[len]; // 申请一段内存存放转换后的字符串
        mbstowcs(p, onnx_path.c_str(), len); // 转换
        std::wstring wstr(p);
        delete[] p; // 释放申请的内存
        session = new Ort::Session(env, wstr.c_str(), this->session_options);
#else
        session = new Ort::Session(env, (const char*)onnx_path.c_str(), this->session_options);
#endif
    }catch (const std::exception& e) {
        return std::format("Failed to load model. Please check your onnx file!");
    }
    this->load_onnx_info(this->session,this->input_nodes,this->output_nodes);
    this->is_inited = true;
    std::println("initialize ok!!");
    return true;
}

std::variant<bool,std::string> TableSeg::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return "image can not empyt!";
    this->ori_img = &image;
    // 图片预处理
    // 1. 直接resize
    // 2. 图像padding
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        return "Image preprocess failed!";
    }

    // 创建模型输入张量
    // cv::Mat --> Ort::Value
    std::vector<Ort::Value> input_tensor;
    input_tensor.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        this->input_images[0].ptr<float>(),
        this->input_images[0].total(),
        this->input_nodes[0].dim.data(),
        this->input_nodes[0].dim.size())
    );

    // 推理
    std::vector<const char*> input_names,output_names;
    for(int i = 0;i<this->input_nodes.size();i++) input_names.push_back(this->input_nodes[i].name);
    for(int i = 0;i<this->output_nodes.size();i++) output_names.push_back(this->output_nodes[i].name);
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = this->session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),  // images
            input_tensor.data(), // 1*3*640*640
            input_tensor.size(), // 1
            output_names.data(), // output0
            output_names.size()); // 1
    }catch (const std::exception& e) {
        return "forward model failed!!";
    }
    // 输出后处理
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        return e.what();
    }
    return true;
}

void TableSeg::preprocess(cv::Mat &image){
    cv::Mat image_ = image.clone();
    auto net_w = (float)this->input_nodes[0].dim.at(3); // 640
    auto net_h = (float)this->input_nodes[0].dim.at(2); // 640
    float scale = std::min(net_w/image.cols,net_h/image.rows);
    cv::resize(image_,image_,cv::Size(int(image.cols*scale),int(image.rows*scale)));
    int top     = (net_h - image_.rows) / 2;
    int bottom  = (net_h - image_.rows) / 2 + int(net_h - image_.rows) % 2;
    int left    = (net_w - image_.cols) / 2;
    int right   = (net_w - image_.cols) / 2 + int(net_w - image_.cols) % 2; 
    cv::copyMakeBorder(image_, image_, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114)); // padding
    input_images.clear();
    std::vector<cv::Mat> mats{image_};
    cv::Mat blob = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(net_w, net_h), cv::Scalar(0, 0, 0), true, false);//1*3*640*640
    input_images.push_back(std::move(blob));
}


std::tuple<TableSeg::SegResBox,TableSeg::SegResMask> TableSeg::mask_decode(float* a,float* b,std::vector<int64_t>& a_shape,std::vector<int64_t>& b_shape){
    assert(a_shape.size() == 3 && b_shape.size() == 4);
    const int64_t rows = a_shape[1]; // 25200
    const int64_t cols = a_shape[2]; // 38 =  [cx,cy,w,h,conf,prob] + 32
    const int mask_h = b_shape[2]; // 160
    const int mask_w = b_shape[3]; // 160
    const int64_t nums = cols - 36;

    std::vector<float> scores(rows);
    std::vector<cv::Rect> bboxes(rows);
    // 可以并行
    for(size_t row = 0;row<rows;row++){
        size_t index = row * cols;
        bboxes[row] = cv::Rect(
            a[index] - 0.5f * a[index+2],  // x
            a[index + 1] - 0.5f * a[index+3],  // y
            a[index + 2],                 // w
            a[index + 3]                  // h
        );
        scores[row] = a[index+4];
    }
    // nms
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, params.score, params.nms, indices);
    if(indices.empty()){
        return {};
    }
    TableSeg::SegResMask masks(indices.size(),cv::Mat::zeros(cv::Size(mask_w,mask_h),CV_32FC1));
    TableSeg::SegResBox res_boxes(indices.size());
    cv::Mat keep_mask(indices.size(), 32, CV_32FC1); // [indices.size() ,32]

    // 也可以并行
    for (size_t i = 0; i < indices.size(); ++i) {
        int row_idx = indices[i];
        assert(row_idx < rows && "索引越界");
        float* src = a + row_idx * cols + 4 + nums;
        std::copy(src, src + 32, keep_mask.ptr<float>(i));
        res_boxes[i].conf = scores[indices[i]];
        res_boxes[i].box = bboxes[indices[i]];
    }
    // mul
    cv::Mat b_mask(32, mask_h * mask_w, CV_32FC1, b);
    cv::Mat T = keep_mask * b_mask;  // [indices.size(), mask_h*mask_w]
    // sigmoid
    cv::Mat T_sigmoid;  // [n,mask_h*mask_w] --> [n,mask_h,mask_w]
    cv::exp(-T, T_sigmoid);
    cv::divide(1.0f, 1.0f + T_sigmoid, T_sigmoid);  // 1/(1+exp(-x))

    cv::Size size = cv::Size(this->input_nodes[0].dim.at(3),this->input_nodes[0].dim.at(2));
    for(size_t row = 0; row<T_sigmoid.rows;row++){
        float* curr_row = T_sigmoid.ptr<float>(row); // [mask_h*mask_w]
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < mask_h; ++j) {
            for (int k = 0; k < mask_w; ++k) {
                // if(res_boxes[row].box.contains(cv::Point(k*4,j*4))){
                    masks[row].at<float>(j,k) = curr_row[j * mask_w + k];
                // }
            }
        }
        // 还原到原图参考系
        cv::resize(masks[row],masks[row],cv::Size(mask_w*4,mask_h*4));
        masks[row] = MT::UnpaddingImg(masks[row],size,ori_img->size());
        cv::threshold(masks[row],masks[row],params.bin,1,cv::THRESH_BINARY);
        masks[row].convertTo(masks[row],CV_8UC1,255);
        res_boxes[row].box = MT::unReferencePad(res_boxes[row].box,ori_img->size(),size);
    }
    return {res_boxes,masks};
}

void TableSeg::postprocess(std::vector<Ort::Value> &output_tensors){
    float* outdata1 = output_tensors[0].GetTensorMutableData<float>();  //[1,25200,38]
    float* outdata2 = output_tensors[1].GetTensorMutableData<float>();  //[1,32,160,160]
    auto outshape1 = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto outshape2 = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    // 解码
    auto[boxes,masks] = this->mask_decode(outdata1,outdata2,outshape1,outshape2);

    // 绘制
    for(auto&mask:masks){
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        int r = MT::GetRandom<int>(0,255);
        int g = MT::GetRandom<int>(0,255);
        int b = MT::GetRandom<int>(0,255);
        for(size_t i = 0;i<contours.size();i++){
            cv::drawContours(*ori_img, contours,i, cv::Scalar(b,g,r),2);
        }
    }
    // for(auto&[conf,box]:boxes){
    //     int r = MT::GetRandom<int>(0,255);
    //     int g = MT::GetRandom<int>(0,255);
    //     int b = MT::GetRandom<int>(0,255);
    //     cv::rectangle(*ori_img,box,cv::Scalar(b,g,r),1);
    // }
}
