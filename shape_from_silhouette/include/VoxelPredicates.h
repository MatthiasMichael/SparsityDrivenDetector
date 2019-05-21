#pragma once

#include "ExtendedVoxel.h"

namespace sfs
{

	inline std::vector<ExtendedVoxel::GeometryPredicate> getIndividualSegmentationPredicates(size_t numCameras)
	{
		using SegmentationVector = std::vector<ExtendedVoxel::SegmentationStatus>;
		using VisibilityVector = std::vector<ExtendedVoxel::VisibilityStatus>;

		std::vector<ExtendedVoxel::GeometryPredicate> predicates;

		for (size_t i = 0; i < numCameras; ++i)
		{
			predicates.push_back([i](const SegmentationVector & segmentation, const VisibilityVector & visibility)
			{
				return segmentation[i] == ExtendedVoxel::Marked;
			});
		}

		return predicates;
	}


	inline std::vector<ExtendedVoxel::GeometryPredicate> getIndividualVisibilityPredicates(size_t numCameras)
	{
		using SegmentationVector = std::vector<ExtendedVoxel::SegmentationStatus>;
		using VisibilityVector = std::vector<ExtendedVoxel::VisibilityStatus>;

		std::vector<ExtendedVoxel::GeometryPredicate> predicates;

		for (size_t i = 0; i < numCameras; ++i)
		{
			predicates.push_back([i](const SegmentationVector & segmentation, const VisibilityVector & visibility)
			{
				return visibility[i] == ExtendedVoxel::Visible;
			});
		}

		return predicates;
	}


	inline ExtendedVoxel::GeometryPredicate defaultPredicate()
	{
		return [](const std::vector<ExtendedVoxel::SegmentationStatus> & segmentation,
			const std::vector<ExtendedVoxel::VisibilityStatus> & visibility)
		{
			int countSegmented = 0, countVisible = 0;
			for (int i = 0; i < visibility.size(); ++i)
			{
				if (segmentation[i] == ExtendedVoxel::Marked)
				{
					++countSegmented;
				}

				if (visibility[i] == ExtendedVoxel::Visible)
				{
					++countVisible;
				}
			}
			return countSegmented > 1 && countSegmented == countVisible;
		};
	}

}