#include "src/cerberus.hpp"

#include <iostream>
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <list>
#include <chrono>
#include <experimental/filesystem>

void single_image_example(CERBERUS& nnet)
{
    cv::Mat image1 = cv::imread("/home/bryce/Documents/Cityscapes Data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png");
    cv::Mat image2 = cv::imread("/home/bryce/Documents/Cityscapes Data/leftImg8bit_sequence/train/aachen/aachen_000000_000020_leftImg8bit.png");

    assert(!image1.empty());
    assert(!image2.empty());

    const cv::Size net_input { nnet.getInputW(), nnet.getInputH() };
    cv::resize(image1, image1, net_input);
    cv::resize(image2, image2, net_input);
    cv::imshow("Sample Input", image1);
    cv::cvtColor(image1, image1, cv::COLOR_BGR2RGB);
    cv::cvtColor(image2, image2, cv::COLOR_BGR2RGB);

    std::cout << "Doing inference" << std::endl;
    nnet.doInference(image1, image2);

    std::chrono::high_resolution_clock timer;
    auto begin = timer.now();
    for (size_t i=0; i<10; i++)
    {
        nnet.doInference(image1, image2);
    }
    std::cout << "End Time: " << (timer.now() - begin).count() / 1e6 / 10. << " ms" << std::endl;

    std::cout << "Showing Outputs" << std::endl;
    cv::imshow("Sample Depth", nnet.get_depth());
    cv::Mat color_seg;
    cv::cvtColor(nnet.get_seg_image(), color_seg, cv::COLOR_RGB2BGR);
    cv::imshow("Sample Seg", color_seg);

    cv::cvtColor(nnet.get_flow(), color_seg, cv::COLOR_RGB2BGR);
    cv::imshow("Sample Flow", color_seg);

    cv::waitKey(0);
}

std::list<std::string> get_images(std::string_view base_path)
{
    std::list<std::string> image_filenames;

    // Get all the png images from the folder
    for (const auto & entry : std::experimental::filesystem::directory_iterator(base_path))
    {
        if (entry.path().extension() == ".png") {
            image_filenames.emplace_back(entry.path().filename());
        }
    }

    // Sort them according to the cityscapes standard
    image_filenames.sort([](const std::string& str1, const std::string& str2)
        {
            return std::stoi(str1.substr(20, 6)) < std::stoi(str2.substr(20, 6));
        });

    // Optional printing to check
    // std::for_each(image_filenames.begin(), image_filenames.end(),
    //     [](const auto &filename){ std::cout << filename << std::endl; });

    return image_filenames;
}
void video_sequence_example(CERBERUS& nnet)
{
    std::string_view folder{"/media/bryce/4TB Seagate/Autonomous Vehicles Data/Cityscapes Data/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_01"};
    auto image_filenames = get_images(folder);

    const cv::Size net_input { nnet.getInputW(), nnet.getInputH() };
    for (auto filename = image_filenames.begin(); filename != --image_filenames.end(); /*no-op*/)
    {
        cv::Mat image1 = cv::imread(std::string(folder)+"/"+*filename);
        assert(!image1.empty());
        cv::resize(image1, image1, net_input);
        cv::imshow("Sample Input", image1);
        cv::cvtColor(image1, image1, cv::COLOR_BGR2RGB);

        cv::Mat image2 = cv::imread(std::string(folder)+"/"+*(++filename));
        assert(!image2.empty());
        cv::resize(image2, image2, net_input);
        cv::cvtColor(image2, image2, cv::COLOR_BGR2RGB);

        nnet.doInference(image1, image2);
        
        cv::imshow("Sample Depth", nnet.get_depth());
        cv::Mat color_seg;
        cv::cvtColor(nnet.get_seg_image(), color_seg, cv::COLOR_RGB2BGR);
        cv::imshow("Sample Seg", color_seg);

        cv::cvtColor(nnet.get_flow(), color_seg, cv::COLOR_RGB2BGR);
        cv::imshow("Sample Flow", color_seg);
        cv::waitKey(0);
    }
}

int main(int argc, char** argv)
{
    CERBERUS nnet;
    std::cout << "Init Success!" << std::endl;

    single_image_example(nnet);
}