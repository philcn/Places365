#pragma once

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;

class PlacesClassifier {
public:
	PlacesClassifier( const std::string &modelFile, const std::string &trainedFile );
	std::vector<std::pair<int, float>> Run( const cv::Mat &img );
	
private:
	void WrapInputLayer( std::vector<cv::Mat> *inputChannels );
	void Preprocess( const cv::Mat &img, std::vector<cv::Mat> *inputChannels );
    std::vector<std::pair<int, float>> ConvertOutput( Blob<float> *outputLayer );
	
	std::shared_ptr<Net<float>> mNet;
	cv::Size mInputGeometry;
	int mNumChannels;
	cv::Mat mMean;
	
	std::vector<cv::Mat> mWrappedInputChannels;
};
