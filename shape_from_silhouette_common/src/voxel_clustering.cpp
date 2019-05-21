#include "voxel_clustering.h"

#include <algorithm>


namespace sfs
{
	// An sich nur eine Kopie des rtcvClustering, aber so angepasst, dass es mit den Zeigern und dem Rest hier funktioniert
	// Sehr langsam für eine große Anzahl an segmentierten Voxeln
	size_t cluster(std::vector<VoxelCluster> & vpCluster,
	               const std::vector<const Voxel *> & vpObject,
	               double maxClusterDist,
	               size_t minNoCluster,
	               size_t maxNoCluster
	)
	{
		const size_t noClusterAtBegin = vpObject.size();
		size_t noCluster = noClusterAtBegin;

		// Initialisierung: Jedes Cluster enthält genau ein Objekt
		vpCluster.clear();
		for (size_t i = 0; i < noClusterAtBegin; ++i)
		{
			vpCluster.push_back(VoxelCluster(vpObject[i]));
		}

		// Initiale Distanzen zwischen Clustern berechnen
		std::vector<double> arrDist(noClusterAtBegin * noClusterAtBegin, 0.);
		for (size_t i = 0; i < noClusterAtBegin; ++i)
		{
			for (size_t j = 0; j < i; ++j)
			{
				arrDist[i + j * noClusterAtBegin] = vpCluster[i].distTo(vpCluster[j]);
			}
		}

		// Schleife für Clustering
		while (true)
		{
			double minDist = std::numeric_limits<double>::max();
			int minDistIdx1 = -1;
			int minDistIdx2 = -1;

			// 1. Die zwei Cluster finden, die kleinste Distanz zueinander haben
			for (int i = 0; i < noClusterAtBegin; ++i)
			{
				for (int j = 0; j < i; ++j)
				{
					const double & actDist = arrDist[i + j * noClusterAtBegin];
					if (actDist < minDist)
					{
						minDist = actDist;
						minDistIdx1 = j;
						minDistIdx2 = i;
					}
				}
			}

			if ((minDistIdx1 < 0) || (minDist > maxClusterDist))
				break;

			// 2. Diese beiden Cluster verschmelzen, wenn nah genug beeinander
			vpCluster[minDistIdx1].unite(vpCluster[minDistIdx2]);
			vpCluster[minDistIdx2].reset();

			// 3. Dist-Array updaten I: Altes Cluster deaktivieren
			for (int i = 0; i < minDistIdx2; ++i)
			{
				arrDist[minDistIdx2 + i * noClusterAtBegin] = std::numeric_limits<double>::max();
			}
			for (int i = minDistIdx2 + 1; i < noClusterAtBegin; ++i)
			{
				arrDist[i + minDistIdx2 * noClusterAtBegin] = std::numeric_limits<double>::max();
			}

			// 4. Dist-Array updaten II: Abstände zu neu geschaffenem Cluster updaten
			//#pragma omp parallel for schedule(guided)
			for (int i = 0; i < minDistIdx1; ++i)
			{
				arrDist[minDistIdx1 + i * noClusterAtBegin] = vpCluster[minDistIdx1].distTo(vpCluster[i]);
			}

			//#pragma omp parallel for schedule(guided)
			for (int i = minDistIdx1 + 1; i < noClusterAtBegin; ++i)
			{
				arrDist[i + minDistIdx1 * noClusterAtBegin] = vpCluster[i].distTo(vpCluster[minDistIdx1]);
			}

			--noCluster;

			if (noCluster <= minNoCluster)
				break;
		}

		std::sort(vpCluster.begin(), vpCluster.end(), [](const VoxelCluster & a, const VoxelCluster & b)
		{
			return a.size() > b.size();
		});
		vpCluster.resize(noCluster);

		return noCluster;
	}
}
