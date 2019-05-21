#pragma once

#include <map>
#include <set>
#include <functional>

#include "VoxelCluster.h"


namespace sfs
{
	namespace detail
	{
		bool d_eq(const float & a, const float & b);
		bool d_geq(const float & a, const float & b);
		bool d_leq(const float & a, const float & b);

		std::vector<float3> getVoxelsToLookAt(float3 center, Roi3DF area, float3 voxelSize);
		std::vector<float3> getVoxelsToLookAt(int dist, float3 center, Roi3DF area, float3 voxelSize);
	}

	template<typename VoxelContainer>
	std::vector<VoxelCluster> sequentialFill
	(
		const VoxelContainer & voxel,
		const Roi3DF & area,
		const float3 & voxelSize,
		const int maxDist,
		std::function<size_t(const float3 &)> getLinearOffset
	)
	{
		using namespace detail;

		std::vector<VoxelCluster> clusters;
		std::map<const Voxel *, size_t> regionIndex;
		std::map<size_t, std::set<size_t>> clashes;

		const size_t numVoxel = voxel.size();
		for (int i = 0; i < numVoxel; ++i)
		{
			const Voxel * pV = &voxel[i];

			if (!pV->isActive)
				continue;

			std::vector<float3> voxelsToLookAt = maxDist <= 1 ? 
				getVoxelsToLookAt(maxDist, pV->center, area, voxelSize) :
				getVoxelsToLookAt(pV->center, area, voxelSize);

			std::set<size_t> observedIndices;
			for (const auto & v : voxelsToLookAt)
			{
				const Voxel * pLookAt = &voxel[getLinearOffset(v)];

				if (pLookAt->isActive)
				{
					observedIndices.insert(regionIndex[pLookAt]);
				}
			}

			if (observedIndices.empty()) // New Cluster
			{
				VoxelCluster newCluster(pV);
				clusters.push_back(newCluster);
				regionIndex[pV] = clusters.size() - 1;
			}
			else if (observedIndices.size() == 1) // no clash
			{
				size_t idx = *(observedIndices.begin());
				clusters[idx].addVoxelWithoutBoundingBoxChange(pV);
				regionIndex[pV] = idx;
			}
			else // Clash
			{
				size_t minIndex = *(observedIndices.begin());

				clusters[minIndex].addVoxelWithoutBoundingBoxChange(pV);
				regionIndex[pV] = minIndex;

				for (std::set<size_t>::const_iterator it = observedIndices.begin(); it != observedIndices.end(); ++it)
				{
					if (it == observedIndices.begin())
						continue;

					if (clashes.find(minIndex) == clashes.end())
					{
						clashes[minIndex] = std::set<size_t>();
					}
					clashes[minIndex].insert(*it);
				}
			}

		}

		// Clashes beseitigen
		bool clashesUnified = false;
		while (!clashesUnified)
		{
			// Herausfinden ob es clashes der Form
			// 0 -> 123
			// 1 -> 24
			// gibt. Diese werden zurechtgerückt bevor sie aufgelöst werden
			clashesUnified = true;
			for (auto entry = clashes.begin(); entry != clashes.end(); ++entry)
			{
				for (size_t c2 : entry->second)
				{
					const auto c3 = clashes.find(c2);
					if (c3 != clashes.end())
					{
						clashesUnified = false;
						entry->second.insert(c3->second.begin(), c3->second.end());
						clashes.erase(c3);
						break;
					}
				}
				if (!clashesUnified)
					break;
			}
		}

		for (auto & entry : clashes)
		{
			for (size_t idx : entry.second)
			{
				clusters[entry.first].unite(clusters[idx]);
				clusters[idx].reset();
			}
		}

		std::vector<VoxelCluster> finishedCluster;
		for (const VoxelCluster & vc : clusters)
		{
			if (!vc.empty())
			{
				finishedCluster.push_back(vc);
				finishedCluster.back().resetBoundingBox();
			}

		}

		return finishedCluster;
	}

}