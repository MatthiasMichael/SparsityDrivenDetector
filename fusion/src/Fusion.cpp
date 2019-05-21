#include "Fusion.h"

#include <boost/qvm/all.hpp>

#include "qvm_eigen.h"
#include "qvm_cuda.h"


std::vector<const MergedSolutionActor *> getActorPointer(const MergedSolution & solution)
{
	std::vector<const MergedSolutionActor *> actors;
	actors.reserve(solution.actors.size());
	std::transform(solution.actors.begin(), solution.actors.end(), std::back_inserter(actors),
	               [](const MergedSolutionActor & a) { return &a; });
	return actors;
}


bool isInside(const MergedSolutionActor & actor, const sfs::VoxelCluster & volume)
{
	const auto & r = volume.getPreciseBoundingBox();
	const auto & p = actor.position.get();

	return
		p(0) >= r.x1 && p(0) <= r.x2 &&
		p(1) >= r.y1 && p(1) <= r.y2;
}


float dist(const sfs::Voxel * pVoxel, const MergedSolutionActor & a)
{
	auto d = pVoxel->center - boost::qvm::convert_to<float3>(a.position.get());
	d.z = 0;
	return length(d);
}


std::vector<FusedSolutionActor> splitMultiAssignedVolume(const sfs::VoxelCluster & volume,
                                                         const std::vector<MergedSolutionActor> & actors)
{
	std::vector<std::vector<const sfs::Voxel *>> voxelsAssignedToActor(actors.size(),
	                                                                   std::vector<const sfs::Voxel *>{ });

	for (const auto pVoxel : volume.getVoxel())
	{
		const auto closestActorIt = std::min_element(actors.begin(), actors.end(),
		                                             [pVoxel](const MergedSolutionActor & a1,
		                                                      const MergedSolutionActor a2)
		                                             {
			                                             return dist(pVoxel, a1) < dist(pVoxel, a2);
		                                             });

		const auto closestActorIdx = closestActorIt - actors.begin();

		voxelsAssignedToActor[closestActorIdx].push_back(pVoxel);
	}

	std::vector<FusedSolutionActor> ret;
	ret.reserve(actors.size());

	for (size_t idx_actor = 0; idx_actor < actors.size(); ++idx_actor)
	{
		if(voxelsAssignedToActor[idx_actor].empty())
		{
			continue;
		}

		FusedSolutionActor a{ actors[idx_actor], sfs::VoxelCluster(voxelsAssignedToActor[idx_actor]) };
		ret.push_back(a);
	}

	return ret;
}


std::vector<MergedSolutionActor> getActorsByIndices(const std::vector<MergedSolutionActor> & actors,
                                                    const std::vector<size_t> & vec_idx)
{
	std::vector<MergedSolutionActor> ret;
	ret.reserve(vec_idx.size());
	std::transform(vec_idx.begin(), vec_idx.end(), std::back_inserter(ret),
	               [&actors](size_t idx) { return actors[idx]; });
	return ret;
}


void removeTooFarAwayVoxels(const MergedSolutionActor & actor, sfs::VoxelCluster & volume)
{
	auto & voxels = volume.getChangeableVoxel();

	// erase remove leads to really strange results... so lets just build a new vector

	std::vector<const sfs::Voxel *> closeEnoughVoxels;
	closeEnoughVoxels.reserve(voxels.size());

	const auto sqrMaxDist = actor.info.maxSize.get()(0) * actor.info.maxSize.get()(0);

	for (auto v : voxels)
	{
		const auto dx = v->center.x - actor.position.get()(0);
		const auto dy = v->center.y - actor.position.get()(1);
		if (dx * dx + dy * dy < sqrMaxDist)
		{
			closeEnoughVoxels.push_back(v);
		}
	}

	volume = sfs::VoxelCluster(closeEnoughVoxels);
}


FusedSolution fuse(const MergedSolution & solution, std::vector<sfs::VoxelCluster> volumes)
{
	// Currently we assume there are no actors on top of each other

	const auto & actors = solution.actors;

	const auto numActors = actors.size();
	const auto numVolumes = volumes.size();

	std::map<size_t, std::vector<size_t>> volume_assignedActors;
	std::map<size_t, size_t> actor_assignedVolume;

	for (size_t idx_actor = 0; idx_actor < numActors; ++idx_actor)
	{
		const MergedSolutionActor & actor = actors[idx_actor];

		for (size_t idx_volume = 0; idx_volume < numVolumes; ++idx_volume)
		{
			const sfs::VoxelCluster volume = volumes[idx_volume];

			if (isInside(actor, volume))
			{
				const auto [actor_assignedVolume_it, actor_assignedVolume_inserted] =
					actor_assignedVolume.emplace(idx_actor, idx_volume);

				assert(actor_assignedVolume_inserted);

				const auto [volume_assignedActors_it, volume_assignedActors_inserted] =
					volume_assignedActors.emplace(idx_volume, std::vector<size_t>{ });

				volume_assignedActors_it->second.push_back(idx_actor);
				break;
			}
		}
	}

	std::vector<FusedSolutionActor> fusedActors;
	
	for (const auto & [idx_volume, vec_idx_actor] : volume_assignedActors)
	{
		if (vec_idx_actor.size() == 1)
		{
			FusedSolutionActor a{ actors[vec_idx_actor.front()], volumes[idx_volume] };
			fusedActors.push_back(a);
		}
		else
		{
			std::vector<MergedSolutionActor> actorsForSplitting = getActorsByIndices(actors, vec_idx_actor);

			const auto splitVolumes = splitMultiAssignedVolume(volumes[idx_volume], actorsForSplitting);

			fusedActors.insert(fusedActors.end(), splitVolumes.begin(), splitVolumes.end());
		}
	}

	FusedSolution ret{ solution.framenumber, solution.timestamp, { } };

	for (auto & actor : fusedActors)
	{
		removeTooFarAwayVoxels(actor.actor, actor.volume);
		if(!actor.volume.empty())
		{
			ret.actors.push_back(actor);
		}
	}

	return ret;
}
