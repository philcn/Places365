#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ip/Resize.h"
#include "cinder/Utilities.h"
#include "cinder/qtime/QuickTime.h"

#include <caffe/caffe.hpp>
#include <caffe/data_transformer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>

#include "PlacesClassifier.h"
#include "CinderOpenCV.h"
#include "CinderImGui.h"

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace caffe;

#define PRETRAINED "alexnet_places365.caffemodel"
#define MODEL_FILE "deploy_alexnet_places365.prototxt"
#define LABEL_FILE "categories_places365.txt"

class Places365App : public App {
  public:
	void setup() override;
	void fileDrop( FileDropEvent event ) override;
	void update() override;
	void draw() override;
	
	void parseLabelFile();
	void classify( SurfaceRef image );
	
	unique_ptr<PlacesClassifier> mClassifier;
	
	std::map<int, std::string> mLabels;
	
	SurfaceRef mTestImage;
	gl::TextureRef mTestImageTexture;
	
	qtime::MovieSurfaceRef	mTestMovie;
	
	std::vector<std::pair<std::string, float>> mCurrentResults;
};

void Places365App::setup()
{
	// initialize caffe stuff
	string modelFile = getAssetPath( MODEL_FILE ).string();
	string trainedFile = getAssetPath( PRETRAINED ).string();
	mClassifier = make_unique<PlacesClassifier>( modelFile, trainedFile );
	
	parseLabelFile();
	
	ui::initialize();
}

void Places365App::parseLabelFile()
{
	auto labelFile = loadString( loadAsset( LABEL_FILE ) );

	std::vector<std::string> tokens;
	boost::split_iterator<std::string::iterator> lineItr, endItr;
	for( lineItr = boost::make_split_iterator( labelFile, boost::token_finder( boost::is_any_of( "\n\r" ) ) ); lineItr != endItr; ++lineItr ) {
		// retrieve a single, trimmed line
		std::string line = boost::algorithm::trim_copy( boost::copy_range<std::string>( *lineItr ) );
		if( line.empty() )
			continue;

		// split into tokens
		boost::algorithm::split( tokens, line, boost::is_any_of( " " ), boost::token_compress_off );

		// skip if data was incomplete
		if( tokens.size() < 2 )
			continue;

		try {
			std::string label = boost::trim_copy( tokens[0] );
			int id = atoi( boost::trim_copy( tokens[1] ).c_str() );
			mLabels[id] = label.substr( 3 );
		}
		catch( ... ) {
			continue;
		}
	}
}

void Places365App::classify( SurfaceRef image )
{
	cv::Mat img = toOcv( *mTestImage );
	auto result = mClassifier->Run( img );
	
	mCurrentResults.clear();
	
	for( int i = 0; i < result.size(); ++i ) {
    	mCurrentResults.push_back( make_pair( mLabels[result[i].first], result[i].second ) );
	}
}

void Places365App::fileDrop( FileDropEvent event )
{
	auto extension = event.getFile( 0 ).extension().string();
	
	mTestMovie.reset();
	mTestImage.reset();
	mTestImageTexture.reset();
	
	if( extension == ".png" || extension == ".jpg" ) {
    	mTestImage = Surface::create( loadImage( loadFile( event.getFile( 0 ) ) ) );
    	mTestImageTexture = gl::Texture::create( *mTestImage );
    	
    	classify( mTestImage );
	}
	else if( extension == ".mov" || extension == ".mp4" ) {
		mTestMovie = qtime::MovieSurface::create( event.getFile( 0 ) );
		mTestMovie->play();
		mTestMovie->setLoop();
	}
}

void Places365App::update()
{
	if( mTestMovie && mTestMovie->isPlaying() ) {
		mTestImage = mTestMovie->getSurface();
		
		if( mTestImage ) {
        	mTestImageTexture = gl::Texture::create( *mTestImage );
        		
    		if( getElapsedFrames() % 30 == 0 ) {
            	classify( mTestImage );
    		}
		}
	}
}

void Places365App::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
	gl::ScopedMatrices mtx;
	gl::setMatricesWindow( getWindowSize() );
	
	if( mTestImageTexture ) {
    	auto drawArea = Area::proportionalFit( mTestImage->getBounds(), Area( vec2( 0 ), getWindowSize() ), true, true );
    	gl::draw( mTestImageTexture, drawArea );
	}
	
	ui::ScopedWindow window( "Prediction" );
	if( mCurrentResults.size() >= 5 ) {
		for( int i = 0; i < 5; ++i ) {
        	ui::Text( "Prediction %d: %s: %03f", i + 1, mCurrentResults[i].first.c_str(), mCurrentResults[i].second );
		}
		
		static gl::TextureFontRef texFont = gl::TextureFont::create( ci::Font( "Arial", 96 ) );
		gl::ScopedGlslProg glsl_( gl::getStockShader( gl::ShaderDef().color().texture() ) );
    	gl::ScopedColor col_( 1.0, 1.0, 1.0, 0.3 );
		
		auto size = texFont->measureString( mCurrentResults[0].first );
		texFont->drawString( mCurrentResults[0].first, vec2( getWindowSize() ) / 2.0f - size / 2.0f );
	}
	else {
		ui::Text( "No input" );
	}
}

CINDER_APP( Places365App, RendererGl )
