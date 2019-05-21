#include "Actor.h"

Actor::State::State() :
	position(make_named<WorldVector>(0.f, 0.f, 0.f))
{
	// empty
}


Actor::State::State(const WorldVector & position, const WorldRotation & rotation) :
	position(position),
	rotation(rotation)
{
	// empty
}


Actor::Actor() :
	identifier(""),
	size(make_named<WorldVector>(0.f, 0.f, 0.f))
{
	// empty
}


Actor::Actor(const std::string & identifier, const WorldVector & size) :
	identifier(identifier),
	size(size)
{
	// empty
}


bool Actor::operator==(const Actor & other) const
{
	return identifier == other.identifier;
}


StatefulActor::StatefulActor(const Actor & actor, const Actor::State state) :
	actor(actor),
	state(state)
{
	// empty	
}