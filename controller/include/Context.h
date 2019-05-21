#pragma once

#include "Configuration.h"

#include "Scene.h"
#include "Environment.h"
#include "SparsityDrivenDetector.h"
#include "SparsityDrivenDetectorPostProcessing.h"
#include "Fusion.h"
#include "VideoInput.h"
#include "ShapeFromSilhouetteBridge.h"


class Context
{
public:
	struct StringConstants;

	struct Elements
	{
		SceneInfo m_sceneInfo;

		Scene m_scene;
		Environment m_environment;
		SparsityDrivenDetector m_detector;
		SparsityDrivenDetectorPostProcessing m_postProcessing;

		std::unique_ptr<sfs::ShapeFromSilhouette_Impl> m_sfs;

		std::unique_ptr<VideoInput> m_input;
	};

	Context(Elements && elements);
	Context(const Configuration & config);

	friend class Controller;

private:
	static Elements buildContext(const Configuration & config);

	// Holds the smart pointer to the temporary video directory so we need to store it somewhere
	SceneInfo m_sceneInfo; 

	Scene m_scene;
	Environment m_environment;
	SparsityDrivenDetector m_detector;
	SparsityDrivenDetectorPostProcessing m_postProcessing;

	std::unique_ptr<sfs::ShapeFromSilhouette_Impl> m_sfs;

	std::unique_ptr<VideoInput> m_input;
};
