#pragma once

#include "boost/qvm/all.hpp"

#include "qvm_osg.h"
#include "qvm_eigen.h"

#include "FunctionalOsgDisplayDescriptor.h"

#include "DisplayableAreaLights.h"
#include "DisplayableCameras.h"
#include "DisplayableCoordinateSystem.h"
#include "DisplayableFrame.h"
#include "DisplayableWalls.h"

#include "Camera.h"
#include "Environment.h"

#include "WorldCoordinateSystem.h"
#include "ImageCoordinateSystem.h"
#include "DisplayableMesh.h"


inline auto areaLightsDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Area Lights",
	[](Parametrizable &) { },
	[](const Parametrizable &, const Roi3DF & roi)
	{
		return std::make_unique<DisplayableAreaLight>(roi);
	}
);


inline auto coordinateSystemDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Coordinate System",
	[](Parametrizable & d)
	{
		d.addDoubleParameter("Pos X", -1000, 1000, 0, 1);
		d.addDoubleParameter("Pos Y", -1000, 1000, 0, 1);
		d.addDoubleParameter("Pos Z", -1000, 1000, 0, 1);
		 
		d.addDoubleParameter("Size", 0, 100, 0.2, 1);
		 
		d.addDoubleParameter("Beam Length", 0, 100, 10, 1);
	},
	[](const Parametrizable & d)
	{
		return std::make_unique<DisplayableCoordinateSystem>
		(
			osg::Vec3
			(
				static_cast<float>(d.getDoubleParameter("Pos X")),
				static_cast<float>(d.getDoubleParameter("Pos Y")),
				static_cast<float>(d.getDoubleParameter("Pos Z"))
			),
			static_cast<float>(d.getDoubleParameter("Size")),
			static_cast<float>(d.getDoubleParameter("Beam Length"))
		);
	}
);


inline auto camerasDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Cameras",
	[](Parametrizable & d)
	{
		d.addDoubleParameter("Camera Size", 5, 100, 10, 1);
		d.addDoubleParameter("Ray Scaling", 10, 1000, 50, 10);
	},
	[](const Parametrizable & d, const CameraSet & cameras)
	{
		using Size = IdentifiableCamera::Size;

		std::vector<CameraGeometry::Info> camInfos;

		for (const auto & cam : cameras)
		{
			const auto & o = cam.getOrigin().get();
			const osg::Vec3 origin = boost::qvm::convert_to<osg::Vec3>(o);

			const Size imageSize = cam.getImageSize();

			const ImagePoint topLeft = make_named<ImagePoint>(0, 0);
			const ImagePoint topRight = make_named<ImagePoint>(imageSize(0) - 1, 0);
			const ImagePoint bottomLeft = make_named<ImagePoint>(0, imageSize(1) - 1);
			const ImagePoint bottomRight = make_named<ImagePoint>(imageSize(0) - 1, imageSize(1) - 1);

			const ViewRay<WorldCoordinateSystem> dir_topLeft = cam.imageToWorld(topLeft);
			const ViewRay<WorldCoordinateSystem> dir_topRight = cam.imageToWorld(topRight);
			const ViewRay<WorldCoordinateSystem> dir_bottomLeft = cam.imageToWorld(bottomLeft);
			const ViewRay<WorldCoordinateSystem> dir_bottomRight = cam.imageToWorld(bottomRight);

			camInfos.push_back(CameraGeometry::Info
				{
					origin,
					boost::qvm::convert_to<osg::Vec3>(dir_topLeft.direction.get()),
					boost::qvm::convert_to<osg::Vec3>(dir_topRight.direction.get()),
					boost::qvm::convert_to<osg::Vec3>(dir_bottomLeft.direction.get()),
					boost::qvm::convert_to<osg::Vec3>(dir_bottomRight.direction.get())
				});
		}

		return std::make_unique<DisplayableCameras>
		(
			camInfos,
			static_cast<float>(d.getDoubleParameter("Camera Size")),
			static_cast<float>(d.getDoubleParameter("Ray Scaling"))
		);
	}
);


inline auto staticMeshDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Static Mesh",
	[](Parametrizable & d)
	{
		d.addBoolParameter("Show Wire Frame", false);
	},
	[](const Parametrizable & d, const Mesh & navMesh)
	{
		const osg::Vec4 color(0.5f, 0.5f, 0.5f, 1.0f);
		const bool showWireFrame = d.getBoolParameter("Show Wire Frame");
		return std::make_unique<DisplayableMesh>(navMesh, color, showWireFrame);
	}
);


inline auto navMeshDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Nav Mesh",
	[](Parametrizable & d)
	{
		d.addBoolParameter("Show Wire Frame", false);
	},
	[](const Parametrizable & d, const Mesh & navMesh)
	{
		const osg::Vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
		const bool showWireFrame = d.getBoolParameter("Show Wire Frame");
		return std::make_unique<DisplayableMesh>(navMesh, color, showWireFrame);
	}
);


inline auto groundTruthDescriptor = makeFunctionalOsgDisplayDescriptor
(
	"Ground Truth",
	[](Parametrizable &) { },
	[](const Parametrizable &, const Frame & frame)
	{
		return std::make_unique<DisplayableFrame>(frame);
	}
);