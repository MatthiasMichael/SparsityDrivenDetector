#include "space_sequentialFill.h"

#include <algorithm>


namespace sfs
{
	namespace detail
	{
		std::vector<float3> getVoxelsToLookAt(float3 center, Roi3DF area, float3 voxelSize)
		{
			std::vector<float3> v;

			float dx = voxelSize.x;
			float dy = voxelSize.y;
			float dz = voxelSize.z;

			v.push_back(make_float3(center.x - dx, center.y - dy, center.z - dz));
			v.push_back(make_float3(center.x - dx, center.y - dy, center.z));
			v.push_back(make_float3(center.x - dx, center.y - dy, center.z + dz));

			v.push_back(make_float3(center.x - dx, center.y, center.z - dz));
			v.push_back(make_float3(center.x - dx, center.y, center.z));
			v.push_back(make_float3(center.x - dx, center.y, center.z + dz));

			v.push_back(make_float3(center.x - dx, center.y + dy, center.z - dz));
			v.push_back(make_float3(center.x - dx, center.y + dy, center.z));
			v.push_back(make_float3(center.x - dx, center.y + dy, center.z + dz));

			v.push_back(make_float3(center.x, center.y - dy, center.z - dz));
			v.push_back(make_float3(center.x, center.y - dy, center.z));
			v.push_back(make_float3(center.x, center.y - dy, center.z + dz));

			v.push_back(make_float3(center.x, center.y, center.z - dz));

			v.erase(std::remove_if(v.begin(), v.end(), [&area](float3 & t) { return !area.contains(t.x, t.y, t.z); }), v.end());
			return v;
		}


		bool d_eq(const float & a, const float & b)
		{
			return std::abs(a - b) < 0.0001f;
		}


		bool d_geq(const float & a, const float & b)
		{
			return a > b || d_eq(a, b);
		}


		bool d_leq(const float & a, const float & b)
		{
			return a < b || d_eq(a, b);
		}


		// Helper für sequentialFill
		std::vector<float3> getVoxelsToLookAt(int dist, float3 center, Roi3DF area, float3 voxelSize)
		{
			std::vector<float3> v;

			float e = 0.0001f;

			float dx = dist * voxelSize.x;
			float dy = dist * voxelSize.y;
			float dz = dist * voxelSize.z;

			Roi3DF areaLookAt(center.x - dx, center.y - dy, center.z - dz, center.x, center.y + dy, center.z + dz);
			/*areaLookAt.intersect(area);*/ // Macht nur Probleme, da der initiale Offset verändert wird. Leiber im Nachhinein Voxel entfernen

			for (float x = areaLookAt.x1; x <= areaLookAt.x2 + e; x += voxelSize.x)
			{
				for (float y = areaLookAt.y1; y <= areaLookAt.y2 + e; y += voxelSize.y)
				{
					for (float z = areaLookAt.z1; z <= areaLookAt.z2 + e; z += voxelSize.z)
					{
						v.push_back(make_float3(x, y, z));
					}
				}
			}

			v.erase(std::remove_if(v.begin(), v.end(), [&area](float3 & t) { return !area.contains(t.x, t.y, t.z); }), v.end());
			v.erase(std::remove_if(v.begin(), v.end(), [&center, &e](float3 & t) {
				return t.x > center.x || d_eq(t.x, center.x) && (t.y >= center.y || t.z >= center.z); }),
				v.end());
			return v;
		}


		float dist(float3 a, float3 b)
		{
			return length(a - b);
		}
	}
}