#pragma once

#include <vector>

#include "Roi3DF_eigen.h"

#include "IdentifiableCamera.h"
#include "Mesh.h"


class Environment
{
public:
	Environment() = default;
	Environment(const Mesh & staticMesh, const Mesh & navMeshMesh, const CameraSet & cameras);

	void save(const std::string & filename) const;

	const Mesh & getStaticMesh() const { return m_staticMesh; }
	const Mesh & getNavMesh() const { return m_navMesh; }

	const CameraSet & getCameras() const { return m_cameras; }

	static Environment load(const std::string & filename);

	friend bool operator==(const Environment & lhs, const Environment & rhs);

private:
	Mesh m_staticMesh;
	Mesh m_navMesh;

	CameraSet m_cameras;
};