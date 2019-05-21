#pragma once

#include <map>

#include "named_type.h"

#include "Actor.h"


struct FrameTag {};
using Framenumber = NamedType<int, FrameTag, Comparable, Addable, Subtractable, Incrementable, PreIncrementable>;

using Trajectory = std::map<Framenumber, Actor::State>;


struct Frame
{
	Frame();
	Frame(Framenumber framenumber, const std::string timestamp);

	Framenumber framenumber;
	std::string timestamp;

	std::vector<StatefulActor> actors;
};