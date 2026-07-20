#include "inference.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <vector>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "preprocess.cuh"
#include "postprocess.cuh"


Engine::Engine(const std::string& enginePath, const std::string& onnxPath) {
    modelInitialize(enginePath, onnxPath);
    allocateBuffers();
    cudaStreamCreate(&stream);
}

Engine::~Engine() {
    std::cout<<"Program kapatıldı."<<std::endl;
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    cudaStreamDestroy(stream);
    delete(context);
    delete(engine);
    delete(runtime);
    cudaFree(gpuResultsDevice); cudaFree(gpuResultCountDevice); cudaFreeHost(gpuResultsHostPinned);
}

void Engine::modelInitialize(const std::string& enginePath, const std::string& onnxPath){
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    if(file.good()){
        //Engine File has found
        size_t size = file.tellg();
        std::vector<char> buffer(size);
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), size);
        file.close();

        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
        context = engine->createExecutionContext();

        std::cout<<"Engine model başarıyla yüklendi."<<std::endl;

    } else {
        //Engine File has not found so we have to initialize ONNX model

        std::cout<<"Engine model bulunamadı, ONNX model üzerinden yeni bir Engine model oluşturuluyor. Bu işlem 2-3dk sürebilir."<<std::endl;
        auto builder = nvinfer1::createInferBuilder(logger);
        auto network = builder->createNetworkV2(0);
        auto config = builder->createBuilderConfig();
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

        auto parser = nvonnxparser::createParser(*network, logger);

        if(!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))){
            std::cerr<<"ONNX Parse has failed."<<std::endl;
        }

        network->getInput(0)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(0)->setType(nvinfer1::DataType::kHALF);

        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO); // binding noktalarinda gizli reformat/cast katmani eklenmesini engeller
            std::cout << "FP16 mod etkinlestirildi (I/O dahil)." << std::endl;
        }

	    nvinfer1::IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
	    if(!serialized) std::cerr<<"Engine dosyası oluşturulamadı."<<std::endl;

	    runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
        context = engine->createExecutionContext();

        std::ofstream e_file(enginePath, std::ios::binary);
        e_file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
        e_file.close();

	    delete(serialized);
        delete(parser);
        delete(network);
        delete(config);
        delete(builder);

        std::cout<<"Engine file has created and saved to directory."<<std::endl;
    }
}

void Engine::allocateBuffers(){
    //Bu adımda bellekten gerekli boyutlarda yer alıp tensorlara tahsis edeceğim
    auto inputDims = engine->getTensorShape("images");
    auto outputDims = engine->getTensorShape("output0");

    numAnchors = outputDims.d[2];
    numAttr = outputDims.d[1];

    boxesBuffer.reserve(numAnchors);
    confidencesBuffer.reserve(numAnchors);
    classIdsBuffer.reserve(numAnchors);
    inputBytes = sizeof(uint16_t);
    for(int i = 0; i<inputDims.nbDims; i++) inputBytes *= inputDims.d[i];
    inputSize = inputBytes / sizeof(uint16_t);

    outputBytes = sizeof(uint16_t);
    for(int i = 0; i<outputDims.nbDims; i++) outputBytes *= outputDims.d[i];
    outputSize = outputBytes / sizeof(uint16_t);

    cudaMalloc((void**) &inputDevice, inputBytes);
    cudaMalloc((void**) &outputDevice, outputBytes);
    cudaMalloc((void**)&gpuResultsDevice, maxGpuResults * sizeof(GpuDetection));
    cudaMalloc((void**)&gpuResultCountDevice, sizeof(int));
    cudaMallocHost((void**)&gpuResultsHostPinned, maxGpuResults * sizeof(GpuDetection));

    context->setTensorAddress("images", inputDevice);
    context->setTensorAddress("output0", outputDevice);



}

void Engine::preprocess(const cv::Mat& image) {
    //Bu adımda OpenCV aracılığıyla aldığımız Frame objelerinin NvInfer in anlayabileceği bir hale getireceğim

    /*
     *     OPENCV GÖRÜNTÜSÜ             NVINFER INPUT SHAPE
     *     1920*1080                    640*640
     *     BGR                          RGB
     *     uint8_t                      float32
     *     HWC (Height-Width-Color)     CHW (Color-Height-Width)
     *
     */

    // cv::Mat resized;
    // cv::resize(image, resized, cv::Size(640, 640));
    // cv::cvtColor(resized, resized, CV_BGR2RGB);
    // resized.convertTo(resized, CV_32FC3, 1.0f/255.0f);

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    gpuFrame.upload(image, cvStream);

    launchFusedPreprocess(
    gpuFrame.ptr<unsigned char>(), gpuFrame.cols, gpuFrame.rows, static_cast<int>(gpuFrame.step),
    reinterpret_cast<__half*>(inputDevice), 640, 640,
    stream);
    //Burada özellikle böyle karmaşık bir kod yazdım aksi takdirde 3*640*640 adet işlem yapmam gerekecekti HWC --> CHW dönüşümü için ancak CPU'mu seviyorum, GPU ile işlemek daha mantıklı

    if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess)
    {
        std::cout << "CUDA preprocess error: " << cudaGetErrorString(err) << std::endl;
    }
}

bool Engine::infer() const {
    if (!context->enqueueV3(stream)) {
        std::cerr<<"TensorRT inference failed!"<<std::endl;
        return false;
    }

    return true;
}

std::vector<Detection> Engine::postprocess(int cols, int rows) {
    const float scaleX = cols / 640.0f;
    const float scaleY = rows / 640.0f;

    // ESKI: CPU'da 8400 x numAttr tarama dongusu (~26ms) - kaldirildi.
    // YENI: GPU'da decode kernel'i taniyor, sadece esik gecen anchor'lari CPU'ya kopyaliyoruz.
    launchDecodeKernel(
        reinterpret_cast<__half*>(outputDevice), numAnchors, numAttr,
        0.25f, scaleX, scaleY,
        gpuResultsDevice, gpuResultCountDevice, maxGpuResults,
        stream);

    // Kac adet detection bulundugunu CPU'ya kopyala (4 byte, ucuz bir transfer)
    int hostResultCount = 0;
    cudaMemcpyAsync(&hostResultCount, gpuResultCountDevice, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // hostResultCount'u okumadan once senkron olmak zorundayiz

    int actualCount = std::min(hostResultCount, maxGpuResults);

    // Sadece bulunan detection sayisinda veri kopyala (8400 degil, genelde <200)
    if (actualCount > 0) {
        cudaMemcpyAsync(gpuResultsHostPinned, gpuResultsDevice,
                         actualCount * sizeof(GpuDetection),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // NMS icin CPU tarafinda format donusumu (artik kucuk bir liste uzerinde, ucuz)
    boxesBuffer.clear();
    confidencesBuffer.clear();
    classIdsBuffer.clear();

    for (int i = 0; i < actualCount; i++) {
        const GpuDetection& d = gpuResultsHostPinned[i];
        boxesBuffer.emplace_back(d.left, d.top, d.width, d.height);
        confidencesBuffer.push_back(d.score);
        classIdsBuffer.push_back(d.classId);
    }

    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxesBuffer, confidencesBuffer, 0.25f, 0.45f, nmsIndices);

    std::vector<Detection> detections;
    detections.reserve(nmsIndices.size());

    for (int idx : nmsIndices) {
        Detection det;
        det.box = boxesBuffer[idx];
        det.confidence = confidencesBuffer[idx];
        det.classId = classIdsBuffer[idx];
        detections.push_back(det);
    }
    return detections;
}
