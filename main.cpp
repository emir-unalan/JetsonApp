#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "inference.hpp"



int main() {

    bool saveFrames = false;
    bool showFrames = true;

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

    Engine engine("../models/best.engine", "../models/yolo11s.onnx");

    cv::VideoCapture cap("../video.mkv");
    if(!cap.isOpened()) {
        std::cerr<<"Hata: Video açılamadı. "<<std::endl;
        return 0;
    }

    cv::Mat frame;
    int cnt=0;
    while (true) {
        cap >> frame;
        if(frame.empty()) break;
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
        cv::imshow("JetsonApp", frame);
        std::string outputFile = "file" +  std::to_string(cnt) + ".jpg";
//        cv::imwrite(outputFile, frame);

        if(cv::waitKey(1) == 27) break;

        //Hangi işlemin ne kadar sürdüğünü görüntülemek için konsola bastırıyorum, bu sayede Infer işleminin çalışıp çalışmadığını ve optimize etmem gerekirse hangi kısımları etmeliyim bunu anlayacağım.
        if (frameCount % 60 == 0) {
            std::cout << "=== Frame " << frameCount << " ===\n"
                      << "  Preprocess : " << avgPre   << " ms\n"
                      << "  Infer      : " << avgInfer << " ms\n"
                      << "  Postprocess: " << avgPost  << " ms\n"
                      << "  TOPLAM     : " << avgTotal << " ms  (~" << (1000.0 / avgTotal) << " FPS)\n"
                      << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    cudaEventDestroy(evStart);
    cudaEventDestroy(evPreEnd);
    cudaEventDestroy(evInferEnd);
    cudaEventDestroy(evPostEnd);
    return 0;
}
