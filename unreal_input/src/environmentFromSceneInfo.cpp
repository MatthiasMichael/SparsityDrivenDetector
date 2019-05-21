#include "environmentFromSceneInfo.h"

#include "UnrealCoordinateSystem.h"
#include "ImageCoordinateSystem.h"
#include "IdentifiableCamera.h"
#include "Mesh.h"


IdentifiableCamera cameraFromCameraInfo(const CameraInfo & info)
{
	using Degrees = NamedScalarTypes<ScalarType>::Degrees;
	using Vector3 = GeometricTypes<ScalarType, 3>::Vector;
	using Translation = GeometricTypes<ScalarType, 3>::Transform::Translation;

	IdentifiableCamera camera(make_named<IdentifiableCamera::ID>(info.id));

	using FocalLength = decltype(camera)::FocalLength;
	using Size = decltype(camera)::Size;

	const Rotation<UnrealCoordinateSystem> ue_rotation(Degrees(info.yaw), Degrees(info.pitch), Degrees(info.roll));
	const Rotation<WorldCoordinateSystem> w_rotation(ue_rotation);

	const UnrealVector ue_camPos(Vector3(info.pos_x, info.pos_y, info.pos_z));
	const WorldVector w_camPos = convertTo<WorldCoordinateSystem>(ue_camPos);

	// Important! Camera looks in x direction
	camera.setBaseDirections(
		{
			make_named<WorldVector>(0.f, -1.f, 0.f),
			make_named<WorldVector>(0.f, 0.f, -1.f),
			make_named<WorldVector>(1.f, 0.f, 0.f)
		}
	);

	camera.extrinsicTransform(w_rotation);
	camera.extrinsicTransform(make_named<WorldTransform>(Translation(w_camPos.get())));

	camera.setFocalLength(FocalLength(info.focalLength));
	camera.setOpticalCenter(make_named<ImagePoint>(info.imageWidth / 2, info.imageHeight / 2));
	camera.setImageSize(make_named<Size>(info.imageWidth, info.imageHeight));

	return camera;
}


Environment environmentFromSceneInfo(const SceneInfo & sceneInfo)
{
	CameraSet cameras;
	for (const auto & info : sceneInfo.cameraInfo)
	{
		cameras.insert(cameraFromCameraInfo(info));
	}

	using ItMap = decltype(sceneInfo.map.begin());
	using ItNavMesh = decltype(sceneInfo.navMesh.begin());

	const auto map_face = Mesh::fromTypedPolygonVector<ItMap, UnrealCoordinateSystem>(sceneInfo.map.begin(), sceneInfo.map.end());
	const auto navMesh = Mesh::fromTypedPolygonVector<ItNavMesh, UnrealCoordinateSystem>(sceneInfo.navMesh.begin(), sceneInfo.navMesh.end());

	return Environment(map_face, navMesh, cameras);
}
