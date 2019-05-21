#include "Space.h"

#include "vectorOperations.h"

#include "Roi3DF_cuda.h"

#include "voxel_clustering.h"
#include "space_sequentialFill.h"


namespace std
{

	bool operator==(const std::vector<std::shared_ptr<sfs::ExtendedVoxel>> & left, const std::vector<std::shared_ptr<sfs::ExtendedVoxel>> & right)
	{
		if (left.size() != right.size())
		{
			return false;
		}
		
		for (size_t i = 0; i < left.size(); ++i)
		{
			if (!(*left[i] == *right[i]))
			{
				return false;
			}
		}
		
		return true;
	}


	bool operator!=(const std::vector<std::shared_ptr<sfs::ExtendedVoxel>> & left, const std::vector<std::shared_ptr<sfs::ExtendedVoxel>> & right)
	{
		return !(left == right);
	}

}

namespace sfs
{
	bool operator==(const Space & left, const Space & right)
	{
		return left.m_voxel == right.m_voxel &&
			left.m_sizeVoxels == right.m_sizeVoxels &&
			left.m_numVoxels == right.m_numVoxels &&
			left.m_area == right.m_area &&
			left.m_models == right.m_models;
	}


	void writeBinary(std::ostream & os, const Space & s)
	{
		//writeBinary(os, s.m_voxel);
		writeBinary(os, s.m_sizeVoxels);
		writeBinary(os, s.m_numVoxels);
		writeBinary(os, s.m_area);
		//writeBinaryTrivial(os, s.m_models);
	}


	void readBinary(std::istream & is, Space & s)
	{
		//readBinary(is, s.m_voxel);
		readBinary(is, s.m_sizeVoxels);
		readBinary(is, s.m_numVoxels);
		readBinary(is, s.m_area);
		//readBinaryTrivial(is, s.m_models);
	}


	Space::Space(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels) : 
		Space(area, voxelSize, cameraModels, Mesh{})
	{
		// empty
	}


	Space::Space(const Roi3DF & area, const float3 & voxelSize, const CameraSet & cameraModels, const Mesh & staticMesh) :
		m_area(area), m_models(cameraModels)
	{
		
		init(voxelSize, staticMesh);
	}


	void Space::init(const float3 & voxelSize, const Mesh & walls)
	{
		m_sizeVoxels = voxelSize;
		m_numVoxels = ceil(m_area.size<float3>() / voxelSize);
		
		const unsigned int numVoxelsLinear = static_cast<unsigned int>(prod(m_numVoxels));

		m_voxel.reserve(numVoxelsLinear);
		m_rawVoxel.reserve(numVoxelsLinear);

		std::cout << "Need to create " << m_numVoxels.x << " * " << m_numVoxels.y << " * " << m_numVoxels.z;
		std::cout << " = " << numVoxelsLinear << " Voxels." << std::endl;

		const float3 initialOffset = voxelSize / 2;
		const float3 start = m_area.start<float3>();

		for (unsigned x = 0; x < m_numVoxels.x; ++x)
		{
			const float posX = start.x + initialOffset.x + x * voxelSize.x;
			for (unsigned y = 0; y < m_numVoxels.y; ++y)
			{
				const float posY = start.y + initialOffset.y + y * voxelSize.y;
				for (unsigned z = 0; z < m_numVoxels.z; ++z)
				{
					const float posZ = start.z + initialOffset.x + z * voxelSize.z; //TODO: InitialOffset.x correct?
					
					m_voxel.emplace_back(make_float3(posX, posY, posZ), voxelSize, m_models, walls);
					m_rawVoxel.push_back(m_voxel.back().getVoxelPointer());
					
				} // end for z
			} // end for y
			const double progress = double(x) / m_numVoxels.x;

			std::cout << "\rCreating Space: " << progress * 100 << "%      " << std::flush;
		} // end for x

		std::cout << "\rCreating Space: " << 100 << "%      \n\n" << std::endl;

		//debugDrawAllConvexHulls();
		//debugCheckVoxelAccess();
	}


	void Space::update(const std::vector<cv::Mat> & imagesSegmentation, double minPercentSegmentedPixel, ExtendedVoxel::GeometryPredicate p)
	{
#pragma omp parallel for
		for (int i = 0; i < m_voxel.size(); ++i)
		{
			m_voxel[i].updateStatus(imagesSegmentation, minPercentSegmentedPixel, p);
		}
	}


	void Space::update(ExtendedVoxel::GeometryPredicate p)
	{
#pragma omp parallel for
		for (int i = 0; i < m_voxel.size(); ++i)
		{
			m_voxel[i].updateStatus(p);
		}
	}


	std::vector<VoxelCluster> Space::clusterVoxels(double maxClusterDistance)
	{
		std::vector<const Voxel *> clusterObjects;
		clusterObjects.reserve(m_voxel.size());

		for (const auto v : m_voxel)
		{
			if (v.isActive())
				clusterObjects.push_back(v.getVoxelPointer());
		}

		std::vector<VoxelCluster> clusters;
		cluster(clusters, clusterObjects, maxClusterDistance, 1);

		return clusters;
	}


	std::vector<VoxelCluster> Space::sequentialFill(int maxDist) const
	{
		class VoxelWrapper
		{
		public:
			VoxelWrapper(const std::vector<const Voxel *> * v) : v(v) {}

			size_t size() const { return v->size(); }
			const Voxel & operator[](size_t i) const { return *(*v)[i]; }

		private:
			const std::vector<const Voxel *> * v;
		};


		return ::sfs::sequentialFill
		(
			VoxelWrapper{ &m_rawVoxel },
			m_area, 
			m_sizeVoxels, 
			maxDist, 
			[this](const float3 & v) { return this->getLinearOffset(v); }
		);
	}


	const ExtendedVoxel & Space::getVoxel(size_t idx) const
	{
		if (idx >= m_voxel.size())
		{
			throw std::out_of_range("Space does not contain this voxel!");
		}
		return m_voxel[idx];
	}


	const ExtendedVoxel & Space::getVoxel(double x, double y, double z) const
	{
		return getVoxel(getLinearOffset(x, y, z));
	}


	std::vector<const Voxel *> Space::getVoxel(Roi3DF area) const
	{
		area.intersect(area);
		std::vector<const Voxel *> v;

		for (auto x = area.x1; x < area.x2; x += m_sizeVoxels.x)
		{
			for (auto y = area.y1; x < area.y2; y += m_sizeVoxels.y)
			{
				for (auto z = area.z1; z < area.z2; z += m_sizeVoxels.z)
				{
					v.push_back(getVoxel(x, y, z).getVoxelPointer());
				}
			}
		}
		return v;
	}


	std::vector<const Voxel *> Space::getActiveVoxels() const
	{
		std::vector<const Voxel *> activeVoxels;
		activeVoxels.reserve(getNumVoxelsLinear());

		for (const auto & v : m_voxel)
		{
			if (v.isActive())
			{
				activeVoxels.push_back(v.getVoxelPointer());
			}
		}

		activeVoxels.shrink_to_fit();
		return activeVoxels;
	}


	std::vector<const ExtendedVoxel *> Space::getExtendedVoxel(Roi3DF area) const
	{
		area.intersect(area);
		std::vector<const ExtendedVoxel *> v;

		for (double x = area.x1; x < area.x2; x += m_sizeVoxels.x)
		{
			for (double y = area.y1; x < area.y2; y += m_sizeVoxels.y)
			{
				for (double z = area.z1; z < area.z2; z += m_sizeVoxels.z)
				{
					v.push_back(&getVoxel(x, y, z));
				}
			}
		}
		return v;
	}


	std::vector<const ExtendedVoxel *> Space::getActiveExtendedVoxels() const
	{
		std::vector<const ExtendedVoxel *> activeVoxels;
		activeVoxels.reserve(getNumVoxelsLinear());

		for (unsigned int i = 0; i < getNumVoxelsLinear(); ++i)
		{
			const ExtendedVoxel * pVoxel = &m_voxel[i];
			if (pVoxel->isActive())
			{
				activeVoxels.push_back(pVoxel);
			}
		}

		activeVoxels.shrink_to_fit();
		return activeVoxels;
	}


	void Space::setVoxelActive(double x, double y, double z, bool b)
	{
		/*if(!m_area.contains(x, y, z))
		{
			throw std::out_of_range("Space does not contain this voxel!");
		}*/

		x -= m_area.x1;
		y -= m_area.y1;
		z -= m_area.z1;

		x /= m_sizeVoxels.x;
		y /= m_sizeVoxels.y;
		z /= m_sizeVoxels.z;

		assert(x >= 0 && y >= 0 && z >= 0);

		const size_t idx = static_cast<size_t>(x) * m_numVoxels.y * m_numVoxels.z + static_cast<size_t>(y) * m_numVoxels.z + static_cast<size_t>(z);

		m_voxel[idx].setActive(b);
	}


	void Space::setVoxelActive(size_t idx, bool b)
	{
		if (idx >= m_voxel.size())
		{
			throw std::out_of_range("Space does not contain this voxel!");
		}

		m_voxel[idx].setActive(b);
	}


	size_t Space::getLinearOffset(double x, double y, double z) const
	{
		assert(m_area.contains(x, y, z));

		if (!m_area.contains(x, y, z))
		{
			std::cout << "Fail2" << std::endl;
			throw std::out_of_range("Space does not contain a Voxel with these coordinates!");
		}

		x -= m_area.x1;
		y -= m_area.y1;
		z -= m_area.z1;

		x /= m_sizeVoxels.x;
		y /= m_sizeVoxels.y;
		z /= m_sizeVoxels.z;

		assert(x >= 0 && y >= 0 && z >= 0);

		return static_cast<size_t>(x) * m_numVoxels.y * m_numVoxels.z + static_cast<size_t>(y) * m_numVoxels.z + static_cast<size_t>(z);
	}


	void Space::debugCheckVoxelAccess()
	{
		int idX = 0;
		for (double x = m_area.x1 + m_sizeVoxels.x / 2; x < m_area.x2 - m_sizeVoxels.x / 2; x += m_sizeVoxels.x, ++idX)
		{
			int idY = 0;
			for (double y = m_area.y1 + m_sizeVoxels.y / 2; y < m_area.y2 - m_sizeVoxels.y / 2; y += m_sizeVoxels.y, ++idY)
			{
				int idZ = 0;
				for (double z = m_area.z1 + m_sizeVoxels.z / 2; z < m_area.z2 - m_sizeVoxels.z / 2; z += m_sizeVoxels.z, ++idZ)
				{
					const size_t linearIdx = idX * m_numVoxels.y * m_numVoxels.z + idY * m_numVoxels.z + idZ;

					const ExtendedVoxel & fromOffset = getVoxel(linearIdx);
					const ExtendedVoxel & fromCoords = getVoxel(x, y, z);

					if (fromCoords != fromOffset)
					{
						throw("Access Check Failed!");
					}
				}
			}
		}
		std::cout << "Access Check Succeeded!" << std::endl;
	}


	// TODO: Use cv::Mat images
	//void Space::debugDrawAllConvexHulls(const std::string & path) const
	//{
	//	std::vector<rtcvImage32> images;
	//	for (const auto & cam : m_models)
	//	{
	//		images.push_back(rtcvImage32(cam.getImageSize().get()(0), cam.getImageSize().get()(1)));
	//		images.back().fill(0);
	//	}

	//	for (int i = 0; i < m_voxel.size(); ++i)
	//	{
	//		/*const auto & convexHulls = m_voxel[i]->getConvexHull();
	//		for(int j = 0; j < convexHulls.size(); ++j)
	//		{
	//			drawPolygon(&images[j], convexHulls[j], 255);
	//		}*/

	//		for (int j = 0; j < images.size(); ++j)
	//		{
	//			m_voxel[i].loopOverImgPoints(j, [&images, j](int x, int y) { images[j].setValue(x, y, images[j].getValue(x, y) + 1); });

	//		}
	//	}

	//	for(size_t i = 0; i < images.size(); ++i)
	//	{
	//		std::stringstream ss;
	//		ss << path << "/cam_" << i << ".pgm";
	//		images[i].writeToFile(ss.str());
	//	}
	//}


	const ExtendedVoxel & Space::getVoxel(float3 centerCoords) const
	{
		return getVoxel(centerCoords.x, centerCoords.y, centerCoords.z);
	}


	size_t Space::getLinearOffset(float3 coords) const { return getLinearOffset(coords.x, coords.y, coords.z); }

}