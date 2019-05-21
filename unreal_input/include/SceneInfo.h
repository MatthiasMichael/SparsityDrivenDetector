#pragma once

#include <memory>

#include <QTemporaryDir>

#include "ActorPositionSqlSerializer.h"
#include "ActorTypeSqlSerializer.h"
#include "CameraSqlSerializer.h"
#include "MapSqlSerializer.h"
#include "NavMeshSqlSerializer.h"


class SceneInfo
{
public:
	SceneInfo(const SceneInfo & other);
	SceneInfo(SceneInfo && other) = default;

	SceneInfo & operator=(const SceneInfo & other);
	SceneInfo & operator=(SceneInfo && other) = default;

	~SceneInfo() = default;

	std::vector<ActorPositionInfo> actorPositionInfo;
	std::vector<ActorTypeInfo> actorTypeInfo;
	std::vector<CameraInfo> cameraInfo;
	std::vector<MapPolygon> map;
	std::vector<NavMeshPolygon> navMesh;

	std::map<int, ActorTypeInfo*> typeInfoToId;
	std::map<int, std::vector<ActorPositionInfo*>> positionsToFramenum;

	std::map<int, std::string> filenameToCameraId;

	int framenumStart;
	int framenumEnd;

	static SceneInfo importScene(const std::string & sceneFile);

private:
	static void unzipScene(const std::string & sceneFile, const QTemporaryDir & dbDir, const QTemporaryDir & videoDir);
	static bool hasEnding(const std::string & str, const std::string & ending);

	static const std::string s_dbFilename;

	SceneInfo(const std::vector<ActorPositionInfo> & _actorPositionInfo,
		const std::vector<ActorTypeInfo> & _actorTypeInfo,
		const std::vector<CameraInfo> & _cameraInfo,
		const std::vector<MapPolygon> & _map,
		const std::vector<NavMeshPolygon> & _navMesh,
		std::shared_ptr<QTemporaryDir> videoDir);

	void buildMaps();
	void buildVideoFilenames();

	std::shared_ptr<QTemporaryDir> m_videoDir;
};
