#include "PlacesClassifier.h"

using namespace caffe;
using namespace std;

#define CPU_ONLY

PlacesClassifier::PlacesClassifier( const string &modelFile, const string &trainedFile )
{
#ifdef CPU_ONLY
	Caffe::set_mode( Caffe::CPU );
#else
	Caffe::set_mode( Caffe::GPU );
	Caffe::SetDevice( 0 );
#endif
	
	mNet.reset( new Net<float>( modelFile, TEST ) );
	mNet->CopyTrainedLayersFrom( trainedFile );
	
	Blob<float>* inputLayer = mNet->input_blobs()[0];
	mNumChannels = inputLayer->channels();
	CHECK( mNumChannels == 3 || mNumChannels == 1 ) << "Input layer should have 1 or 3 channels.";
	
	mInputGeometry = cv::Size( inputLayer->width(), inputLayer->height() );
	
	inputLayer->Reshape( 1, mNumChannels, mInputGeometry.height, mInputGeometry.width );
	mNet->Reshape();
}

std::vector<std::pair<int, float>> PlacesClassifier::Run( const cv::Mat& img )
{
	mWrappedInputChannels.clear();
	
	WrapInputLayer( &mWrappedInputChannels );
	
	Preprocess( img, &mWrappedInputChannels );
	
	mNet->Forward();
	
	Blob<float> *output = mNet->output_blobs()[0];
	
	return ConvertOutput( output );
}

std::vector<std::pair<int, float>> PlacesClassifier::ConvertOutput( Blob<float> *outputLayer )
{
	const float *outputData = outputLayer->cpu_data();
	std::vector<std::pair<int, float>> result;
	
	for( int i = 0; i < 365; ++i ) {
		result.push_back( std::make_pair( i, outputData[i] ) );
	}
	
	std::sort( result.begin(), result.end(), [] ( std::pair<int, float> lhs, std::pair<int, float> rhs ) {
		return lhs.second > rhs.second;
	} );
	
	result.resize( 5 );
	
	return result;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void PlacesClassifier::WrapInputLayer( vector<cv::Mat> *inputChannels )
{
	Blob<float>* inputLayer = mNet->input_blobs()[0];
	
	int width = inputLayer->width();
	int height = inputLayer->height();
	float *inputData = inputLayer->mutable_cpu_data();
	for( int i = 0; i < inputLayer->channels(); ++i ) {
		cv::Mat channel( height, width, CV_32FC1, inputData );
		inputChannels->push_back( channel );
		inputData += width * height;
	}
	
	CHECK(reinterpret_cast<float*>(mWrappedInputChannels.at(0).data) == mNet->input_blobs()[0]->cpu_data())
    	<< "Input channels are not wrapping the input layer of the network.";
}

void PlacesClassifier::Preprocess( const cv::Mat &img, vector<cv::Mat> *inputChannels )
{
	cv::Mat sample;
	if( img.channels() == 4 && mNumChannels == 3 )
		cv::cvtColor( img, sample, cv::COLOR_RGBA2BGR );
	else if( img.channels() == 1 && mNumChannels == 3 )
		cv::cvtColor( img, sample, cv::COLOR_GRAY2BGR );
	else if( img.channels() == 3 && mNumChannels == 3 )
		cv::cvtColor( img, sample, cv::COLOR_RGB2BGR );
	else
		sample = img;
	
	cv::Mat sampleResized;
	if( sample.size() != mInputGeometry )
		cv::resize( sample, sampleResized, mInputGeometry );
	else
		sampleResized = sample;
	
	cv::Mat sampleFloat;
	sampleResized.convertTo( sampleFloat, CV_32FC3 );
	
	cv::Mat sampleScaled;
	sampleScaled = sampleFloat * 255.0f;
	
//	cv::Mat sampleNormalized;
//	cv::subtract( sampleScaled, mMean, sampleNormalized );
	
	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split( sampleFloat, *inputChannels );
	
	CHECK(reinterpret_cast<float*>(inputChannels->at(0).data) == mNet->input_blobs()[0]->cpu_data())
    	<< "Input channels are not wrapping the input layer of the network.";
}
