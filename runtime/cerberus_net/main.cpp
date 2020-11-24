#include "src/cerberus.hpp"

#include <iostream>
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>

int main(int argc, char** argv)
{
    CERBERUS nnet;
    std::cout << "Init Success!" << std::endl;

    const std::string base_path = "/home/bryce/Documents/Cityscapes Data/leftImg8bit_sequence/train/aachen/";
    cv::Mat image1 = cv::imread(base_path+"aachen_000000_000020_leftImg8bit.png");
    cv::Mat image2 = cv::imread(base_path+"aachen_000001_000020_leftImg8bit.png");

    assert(!image1.empty());
    assert(!image2.empty());

    cv::Size net_input { nnet.getInputW(), nnet.getInputH() };
    cv::resize(image1, image1, net_input);
    cv::resize(image2, image2, net_input);

    std::cout << "Doing inference" << std::endl;

    std::chrono::high_resolution_clock timer;
    auto begin = timer.now();
    for (size_t i=0; i<10; i++)
    {
        nnet.doInference(image1, image2);
    }
    std::cout << "End Time: " << (timer.now() - begin).count() / 1e6 / 10. << " ms" << std::endl;

    std::cout << "Showing Images" << std::endl;
    cv::imshow("Sample Input", image1);
    cv::imshow("Sample Depth", nnet.get_depth());
    cv::Mat color_seg;
    cv::cvtColor(nnet.get_seg_image(), color_seg, cv::COLOR_RGB2BGR);
    cv::imshow("Sample Seg", color_seg);

    cv::waitKey(0);
}