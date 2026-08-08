#include <iostream>
#include <filesystem>
#include <climits>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "inference.hpp"
#include <CLI/CLI.hpp>

static std::filesystem::path getExecutableDir() {
	char buf[PATH_MAX];
	ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) -1);
	if(len == -1) {
		std::cerr<<"Mevcut çalışma dizini kullanılacak." <<std::endl;
		return std::filesystem::current_path();
	}
	buf[len] = '\0';
	return std::filesystem::path(buf).parent_path();
}

static std::string getCameraPipeline(int sensorId = 0) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensorId) + " ! "
           "video/x-raw(memory:NVMM), width=3280, height=2464, framerate=21/1 ! "
           "nvvidconv ! "
           "video/x-raw(memory:NVMM), width=1920, height=1080 ! "
           "nvvidconv ! "
           "video/x-raw, format=BGRx ! "
           "videoconvert ! "
           "video/x-raw, format=BGR ! "
           "appsink drop=true max-buffers=1";
}

int main(int argc, char** argv) {

    CLI::App app;

    const std::filesystem::path exeDir = getExecutableDir();

    std::string enginePath = (exeDir / "models" / "model.engine").string();
    std::string onnxPath = (exeDir / "models" / "model.onnx").string();
    std::string videoPath = (exeDir / "video.mkv").string();
    const std::filesystem::path outputDir = exeDir / "output";

    bool saveFrames = true;
    bool showFrames = false;
    bool useCam = false;
    bool verbose = false;

    app.add_flag("--show", showFrames, "Show frames (don't use if you using the CLI session)");
    app.add_flag("--save", saveFrames, "Save frames to output/ folder.");
    app.add_flag("-c, --cam", useCam, "Use the camera for source.");
    app.add_option("-s, --source", videoPath, "The directory of video source.");
    app.add_option("-o, --onnx", onnxPath, "ONNX model path.");
    app.add_option("-e, --engine", enginePath, "TensorRT engine model path.");
    app.add_flag("-v, --verbose", verbose, "Detailed logging.");
    CLI11_PARSE(app, argc, argv);

    cudaEvent_t evStart, evPreEnd, evInferEnd, evPostEnd;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evPreEnd);
    cudaEventCreate(&evInferEnd);
    cudaEventCreate(&evPostEnd);

    double totalPreMs = 0.0, totalInferMs = 0.0, totalPostMs = 0.0;
    int frameCount = 0;


    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if(cuda_devices <= 0) {
        std::cerr<< "Hata: Cihazınızda CUDA destekli bir GPU bulunamadı."<<std::endl;
        return -1;
    }

    Engine engine(enginePath, onnxPath);
    cv::VideoCapture cap;
    if(useCam) cap.open(getCameraPipeline(), cv::CAP_GSTREAMER);
    else cap.open(videoPath);

    if(!cap.isOpened()) {
        std::cerr<<"Hata: Video açılamadı. "<<std::endl;
        return 0;
    }

    cv::Mat frame;
    int cnt=0;
    while (true) {
        cap >> frame;
        // if(frame.empty()) break;
        cnt++;
        //İşlemlerin farklı şekillerde nasıl daha optimize çalışabileceğini ölçmek adına cudaEvent fonksiyonları kullanıldı.
        cudaEventRecord(evStart, engine.getStream());

        engine.preprocess(frame);
        cudaEventRecord(evPreEnd, engine.getStream());

        if (!engine.infer()) {
            std::cerr<<"Inference error."<<std::endl;
            break;
        }
        cudaEventRecord(evInferEnd, engine.getStream());

        auto detections = engine.postprocess(frame.cols, frame.rows);
        cudaEventRecord(evPostEnd, engine.getStream());

        cudaEventSynchronize(evPostEnd);

        float preMs, inferMs, postMs;
        cudaEventElapsedTime(&preMs, evStart, evPreEnd);
        cudaEventElapsedTime(&inferMs, evPreEnd, evInferEnd);
        cudaEventElapsedTime(&postMs, evInferEnd, evPostEnd);

        totalPreMs += preMs;
        totalInferMs += inferMs;
        totalPostMs += postMs;
        frameCount++;

        for (const auto& det: detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }
        double avgPre = totalPreMs / frameCount;
        double avgInfer = totalInferMs / frameCount;
        double avgPost = totalPostMs / frameCount;
        double avgTotal = avgPre + avgInfer + avgPost;

        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(1000.0/avgTotal));
        cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0, 255), 2);

        if (saveFrames) {
            std::string outputFile = (outputDir / ("file" + std::to_string(cnt) + ".jpg")).string();
            cv::imwrite(outputFile, frame);
        }

        if (showFrames) {
            cv::imshow("JetsonApp", frame);
            if(cv::waitKey(1) == 27) break;
        }

        //Hangi işlemin ne kadar sürdüğünü görüntülemek için konsola bastırıyorum, bu sayede Infer işleminin çalışıp çalışmadığını ve optimize etmem gerekirse hangi kısımları etmeliyim bunu anlayacağım.
        if (verbose && frameCount % 60 == 0) {
            std::cout << "=== Frame " << frameCount << " ===\n"
                      << "  Preprocess : " << avgPre   << " ms\n"
                      << "  Infer      : " << avgInfer << " ms\n"
                      << "  Postprocess: " << avgPost  << " ms\n"
                      << "  TOPLAM     : " << avgTotal << " ms  (~" << (1000.0 / avgTotal) << " FPS)\n"
                      << std::endl;
        }
    }

    if (useCam) cap.release();
    if (showFrames) cv::destroyAllWindows();
    cudaEventDestroy(evStart);
    cudaEventDestroy(evPreEnd);
    cudaEventDestroy(evInferEnd);
    cudaEventDestroy(evPostEnd);
    return 0;
}
