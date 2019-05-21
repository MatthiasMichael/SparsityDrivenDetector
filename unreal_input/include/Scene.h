#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "named_type.h"

#include "Camera.h"

#include "Actor.h"
#include "Frame.h"

#include "WorldCoordinateSystem_SDD.h"
#include "SceneInfo.h"


class Scene
{
public:
	Scene();
	Scene(const SceneInfo & sceneInfo);

	const std::set<Framenumber> & getFramenumbers() const { return framenumbers; }

	const std::unordered_set<Actor> & getActors() const { return actors; }

	Frame getFrame(Framenumber framenumber) const;
	std::vector<Frame> getAllFrames() const;
	Framenumber getMaxFramenumber() const;

	Trajectory getTrajectory(const Actor & actor);

private:
	std::unordered_set<Actor> actors;
	std::unordered_map<Actor, Trajectory> trajectories;

	std::set<Framenumber> framenumbers;
	std::map<Framenumber, std::string> timestamps;
};