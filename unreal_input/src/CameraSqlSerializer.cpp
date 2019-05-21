#include "CameraSqlSerializer.h"

#include <sstream>


int callback_cameraInfo(void * instance_void, int argc, char ** argv, char ** columnNames)
{
	if (argc != CameraInfo::NumElements)
	{
		throw std::runtime_error("Invalid number of results for Actor Position!");
	}

	auto * instance = static_cast<CameraSqlSerializer *>(instance_void);

	CameraInfo info = { };

	for (int i = 0; i < CameraInfo::NumElements; ++i)
	{
		std::stringstream ss;
		ss << argv[i];

		switch (i)
		{
			case 0: ss >> info.id;
				break;
			case 1: ss >> info.pos_x;
				break;
			case 2: ss >> info.pos_y;
				break;
			case 3: ss >> info.pos_z;
				break;
			case 4: ss >> info.yaw;
				break;
			case 5: ss >> info.pitch;
				break;
			case 6: ss >> info.roll;
				break;
			case 7: ss >> info.focalLength;
				break;
			case 8: ss >> info.imageWidth;
				break;
			case 9: ss >> info.imageHeight;
				break;
			case 10: ss >> info.videoFile;
				break;
			default: break;
		}
	}

	instance->addCameraInfo(info);

	return 0;
}


CameraSqlSerializer::CameraSqlSerializer(const std::string & dbName) : SqlSerializer(dbName)
{
	// empty
}


void CameraSqlSerializer::resetTables()
{
	const std::string queryCreateTable(
		"CREATE TABLE IF NOT EXISTS 'Cameras' ("
		"'ID' INT NOT NULL UNIQUE PRIMARY KEY, "
		"'pos_x' REAL NOT NULL,"
		"'pos_y' REAL NOT NULL,"
		"'pos_z' REAL NOT NULL,"
		"'yaw' REAL NOT NULL,"
		"'pitch' REAL NOT NULL,"
		"'roll' REAL NOT NULL,"
		"'focalLength' REAL NOT NULL,"
		"'imageWidth' INT NOT NULL,"
		"'imageHeight' INT NOT NULL,"
		"'videoFile' TEXT NOT NULL);"
	);

	executeQuery(queryCreateTable);

	executeQuery("DELETE FROM Cameras;");
}


void CameraSqlSerializer::write(const std::vector<CameraInfo> & cameraInfo)
{
	for (const auto & c : cameraInfo)
	{
		std::stringstream ss;
		ss << "INSERT INTO Cameras (ID, pos_x, pos_y, pos_z, yaw, pitch, roll, focalLength, imageWidth, imageHeight, videoFile) VALUES (";

		ss << c.id << ", ";

		ss << c.pos_x << ", ";
		ss << c.pos_y << ", ";
		ss << c.pos_z << ", ";

		ss << c.yaw << ", ";
		ss << c.pitch << ", ";
		ss << c.roll << ", ";

		ss << c.focalLength << ", ";

		ss << c.imageWidth << ", ";
		ss << c.imageHeight << ", ";

		ss << "\"" << c.videoFile << "\");";

		executeQuery(ss.str());
	}
}


std::vector<CameraInfo> CameraSqlSerializer::readAll()
{
	m_currentDbRead.clear();

	executeQuery("SELECT * FROM Cameras;", &callback_cameraInfo, static_cast<void*>(this));

	return m_currentDbRead;

}

void CameraSqlSerializer::addCameraInfo(const CameraInfo & info)
{
	m_currentDbRead.push_back(info);
}


