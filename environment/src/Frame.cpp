#include "Frame.h"
#include <string>


Frame::Frame() : Frame(Framenumber(0), "")
{
	// empty
}


Frame::Frame(Framenumber framenumber, const std::string timestamp) :
	framenumber(framenumber),
	timestamp(timestamp),
	actors()
{
	// empty
}
