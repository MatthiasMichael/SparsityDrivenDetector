#include "Scene.h"

#include "UnrealCoordinateSystem.h"
#include "CoordinateTransform.h"




Scene::Scene() : actors(),
                 trajectories(),
                 framenumbers(),
                 timestamps()
{
	// empty
}


Scene::Scene(const SceneInfo & sceneInfo) : Scene()
{
	using UnrealToWorld = CoordinateTransform<UnrealCoordinateSystem, WorldCoordinateSystem>;

	using Degrees = NamedScalarTypes<ScalarType>::Degrees;

	std::map<int, Actor> actorToId;

	for (const auto & info : sceneInfo.actorTypeInfo)
	{
		const auto size = make_named<WorldVector>(info.size_x, info.size_y, info.size_z);

		const Actor actor(info.name, size);

		actors.insert(actor);
		trajectories[actor] = Trajectory();

		actorToId[info.id] = actor;
	}

	for (const auto & info : sceneInfo.actorPositionInfo)
	{
		const Framenumber framenumber(info.framenumber);
		if (framenumbers.find(framenumber) == framenumbers.end())
		{
			framenumbers.insert(framenumber);
			timestamps[framenumber] = info.timestamp;
		}

		const auto ue_pos = make_named<UnrealVector>(info.pos_x, info.pos_y, info.pos_z);
		const auto w_pos = UnrealToWorld::sourceToTarget(ue_pos);

		const Rotation<UnrealCoordinateSystem> ue_rotation(Degrees(info.yaw), Degrees(info.pitch), Degrees(info.roll));
		const Rotation<WorldCoordinateSystem> w_rotation(ue_rotation);
		
		const Actor::State state(w_pos, w_rotation);

		if (actorToId.find(info.actor_id) == actorToId.end())
		{
			throw std::runtime_error("Scene contains position info for unkown actor!");
		}

		const Actor & actor = actorToId[info.actor_id];

		trajectories[actor][framenumber] = state;
	}
}


Frame Scene::getFrame(Framenumber framenumber) const
{
	try
	{
		Frame f(framenumber, timestamps.at(framenumber));

		for (const auto & entry : trajectories)
		{
			const Actor & actor = entry.first;
			const Trajectory & trajectory = entry.second;

			const auto it = trajectory.find(framenumber);
			if (it != trajectory.end())
			{
				f.actors.push_back(StatefulActor(actor, it->second));
			}
		}

		return f;
	}
	catch (...)
	{
		return Frame(framenumber, "invalid");
	}
}


Trajectory Scene::getTrajectory(const Actor & actor)
{
	return trajectories.at(actor);
}


std::vector<Frame> Scene::getAllFrames() const
{
	std::vector<Frame> frames;
	for (const auto & framenumber : framenumbers)
	{
		frames.push_back(getFrame(framenumber));
	}
	return frames;
}


Framenumber Scene::getMaxFramenumber() const
{
	return *framenumbers.rbegin();
}
