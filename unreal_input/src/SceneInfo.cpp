#include "SceneInfo.h"

#include <iostream>
#include <fstream>

#include "zip.h"


const std::string SceneInfo::s_dbFilename = "scene.db";


void SceneInfo::buildMaps()
{
	for (auto & info : actorTypeInfo)
	{
		typeInfoToId[info.id] = &info;
	}

	for (auto & info : actorPositionInfo)
	{
		framenumStart = std::min(framenumStart, info.framenumber);
		framenumEnd = std::max(framenumEnd, info.framenumber);

		if (positionsToFramenum.find(info.framenumber) == positionsToFramenum.end())
		{
			positionsToFramenum[info.framenumber] = std::vector<ActorPositionInfo *>();
		}

		positionsToFramenum[info.framenumber].push_back(&info);
	}
}


void SceneInfo::buildVideoFilenames()
{
	for (auto & info : cameraInfo)
	{
		filenameToCameraId[info.id] = m_videoDir->filePath(info.videoFile.c_str()).toStdString();
	}
}


SceneInfo::SceneInfo(const std::vector<ActorPositionInfo> & _actorPositionInfo,
                     const std::vector<ActorTypeInfo> & _actorTypeInfo,
                     const std::vector<CameraInfo> & _cameraInfo,
                     const std::vector<MapPolygon> & _map,
                     const std::vector<NavMeshPolygon> & _navMesh,
                     std::shared_ptr<QTemporaryDir> videoDir) :
	actorPositionInfo(_actorPositionInfo),
	actorTypeInfo(_actorTypeInfo),
	cameraInfo(_cameraInfo),
	map(_map),
	navMesh(_navMesh),
	typeInfoToId(),
	positionsToFramenum(),
	filenameToCameraId(),
	framenumStart(std::numeric_limits<int>::max()),
	framenumEnd(std::numeric_limits<int>::lowest()),
	m_videoDir(videoDir)
{
	//cameraInfo = { cameraInfo[0] };
	buildMaps();
	buildVideoFilenames();
}


SceneInfo::SceneInfo(const SceneInfo & other) :
	SceneInfo(
		other.actorPositionInfo,
		other.actorTypeInfo,
		other.cameraInfo,
		other.map,
		other.navMesh,
		other.m_videoDir)
{
	// empty
}


SceneInfo & SceneInfo::operator=(const SceneInfo & other)
{
	if (&other == this)
		return *this;

	actorPositionInfo = other.actorPositionInfo;
	actorTypeInfo = other.actorTypeInfo;
	cameraInfo = other.cameraInfo;

	map = other.map;
	navMesh = other.navMesh;

	framenumStart = other.framenumStart;
	framenumEnd = other.framenumEnd;

	m_videoDir = other.m_videoDir;

	buildMaps();
	buildVideoFilenames();

	return *this;
}


SceneInfo SceneInfo::importScene(const std::string & SceneInfoFile)
{
	QTemporaryDir dbDir;
	auto videoDir = std::make_shared<QTemporaryDir>();

	if (!dbDir.isValid())
	{
		throw std::runtime_error("Cannot make temporary directory.");
	}

	/*std::cout << "Temp DB Dir: " << dbDir.path().toStdString() << std::endl;
	std::cout << "Temp Video Dir: " << videoDir->path().toStdString() << std::endl;*/

	unzipScene(SceneInfoFile, dbDir, *videoDir);

	const std::string dbFile = dbDir.filePath(s_dbFilename.c_str()).toStdString();

	return SceneInfo(
		ActorPositionSqlSerializer(dbFile).readAll(),
		ActorTypeSqlSerializer(dbFile).readAll(),
		CameraSqlSerializer(dbFile).readAll(),
		MapSqlSerializer(dbFile).readAll(),
		NavMeshSqlSerializer(dbFile).readAll(),
		videoDir
	);
}


bool SceneInfo::hasEnding(const std::string & str, const std::string & ending)
{
	if (str.length() < ending.length())
	{
		return false;
	}
	return str.compare(str.length() - ending.length(), ending.length(), ending) == 0;
}


void SceneInfo::unzipScene(const std::string & sceneFile, const QTemporaryDir & dbDir, const QTemporaryDir & videoDir)
{
	int err;
	zip * archive = zip_open(sceneFile.c_str(), ZIP_RDONLY, &err);

	if (!archive)
	{
		throw std::runtime_error("Error during Zip Open!");
	}

	zip_stat_t info;

	for (int i = 0; i < zip_get_num_files(archive); ++i)
	{
		zip_stat_index(archive, i, 0, &info);
		zip_file_t * file = zip_fopen_index(archive, i, 0);

		if (!file)
		{
			throw std::runtime_error("Error during Zip Unzip!");
		}

		std::vector<char> fileBuffer(info.size, 0);
		int len = zip_fread(file, fileBuffer.data(), info.size);

		if (len < 0)
		{
			throw std::runtime_error("Error during Zip File Read!");
		}

		const QTemporaryDir & targetDir = hasEnding(info.name, ".db") ? dbDir : videoDir;
		const std::string absolutePath = targetDir.filePath(info.name).toStdString();

		std::ofstream unzippedFile(absolutePath, std::ios::out | std::ios::binary | std::ios::trunc);
		unzippedFile.write(fileBuffer.data(), info.size);

		zip_fclose(file);
	}
	zip_close(archive);
}
