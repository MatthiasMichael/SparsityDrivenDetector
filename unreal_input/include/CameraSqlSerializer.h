#pragma once

#include <string>
#include <vector>

#include "SqlSerializer.h"


struct CameraInfo
{
	int id;
	float pos_x, pos_y, pos_z;
	float yaw, pitch, roll;
	float focalLength;
	int imageWidth, imageHeight;
	std::string videoFile;

	constexpr static const int NumElements = 11;
};


class CameraSqlSerializer : private SqlSerializer
{
public:
	CameraSqlSerializer(const std::string & dbName);

	void resetTables();
	void write(const std::vector<CameraInfo> & cameraInfo);
	std::vector<CameraInfo> readAll();

	friend int callback_cameraInfo(void *, int argc, char ** argv, char ** columnNames);

private:
	void addCameraInfo(const CameraInfo & info);

private:
	std::vector<CameraInfo> m_currentDbRead;
};
