#pragma once

#include "Environment.h"
#include "SceneInfo.h"

Environment environmentFromSceneInfo(const SceneInfo & sceneInfo);
IdentifiableCamera cameraFromCameraInfo(const CameraInfo & info);