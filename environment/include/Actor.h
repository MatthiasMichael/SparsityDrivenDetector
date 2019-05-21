#pragma once

#include "Rotation.h"
#include "WorldCoordinateSystem_SDD.h"
#include <optional>


struct Actor
{
	struct State
	{
		using WorldRotation = Rotation<WorldCoordinateSystem>;

		State();
		State(const WorldVector & position, const WorldRotation & rotation);
		
		WorldVector position;
		WorldRotation rotation;
	};


	Actor();
	Actor(const std::string & identifier, const WorldVector & size);

	bool operator==(const Actor & other) const;

	std::string identifier;
	WorldVector size;
};


namespace std
{
	template <>
	struct hash<Actor>
	{
		std::size_t operator()(const Actor & actor) const
		{
			using std::size_t;
			using std::hash;
			using std::string;

			return hash<string>()(actor.identifier);
		}
	};
}


struct StatefulActor
{
	StatefulActor(const Actor & actor, const Actor::State state);

	Actor actor;
	Actor::State state;
};