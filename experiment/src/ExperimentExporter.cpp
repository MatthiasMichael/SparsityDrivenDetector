#include "ExperimentExporter.h"

#include "enumerate.h"


ScalarType getPos(const StatefulActor & a, int i)
{
	return a.state.position.get()(i);
}


ScalarType getSize(const StatefulActor & a, int i)
{
	return a.actor.size.get()(i);
}


ScalarType getPos(const SolutionActor & a, int i)
{
	return a.position.get()(i);
}


ScalarType getSize(const SolutionActor & a, int i)
{
	if (i == 0 && i == 1)
	{
		return a.info.targetSize.get()(0);
	}

	return a.info.targetSize.get()(1);
}


ScalarType getPos(const MergedSolutionActor & a, int i)
{
	return a.position.get()(i);
}


ScalarType getSize(const MergedSolutionActor & a, int i)
{
	if (i == 0 && i == 1)
	{
		return a.info.targetSize.get()(0);
	}

	return a.info.targetSize.get()(1);
}


ScalarType getPos(const FusedSolutionActor & a, int i)
{
	return a.actor.position.get()(i);
}


ScalarType getSize(const FusedSolutionActor & a, int i)
{
	return a.volume.getPreciseBoundingBox().size<Vector3>()(i);
}


template <typename ActorContainer>
std::list<QString> makeInsertQueries(const ActorContainer & container, const QString table)
{
	QString insertQuery
	(
		"INSERT INTO %1 (ID_FRAME, ID_ACTOR, pos_x, pos_y, pos_z, size_x, size_y, size_z) VALUES"
		"(%2, %3, %4, %5, %6, %7, %8, %9);"
	);

	std::list<QString> ret;

	for (const auto & [i, a] : enumerate(container.actors))
	{
		ret.push_back
		(
			insertQuery
			.arg(table)
			.arg(container.framenumber.get()) // ID_FRAME
			.arg(i) // ID_Actor // TODO: Maybe make this the string identifier from actor?
			.arg(getPos(a, 0))
			.arg(getPos(a, 1))
			.arg(getPos(a, 2))
			.arg(getSize(a, 0))
			.arg(getSize(a, 1))
			.arg(getSize(a, 2))
		);
	}

	return ret;
}


template <typename ActorContainer>
void addContainer(const ActorContainer & container, const QString & table,
                  std::function<void(const std::string &)> executeQuery)
{
	for (const auto q : makeInsertQueries(container, table))
	{
		executeQuery(q.toStdString());
	}
}


struct ExperimentExporter::StringConstants
{
	inline static const auto TableGroundTruth = "GroundTruth";
	inline static const auto TableInitialSolution = "InitialSolution";
	inline static const auto TableMergedSolution = "MergedSolution";
	inline static const auto TableFusedSolution = "FusedSolution";

	inline static const auto TableFusedVolume = "FusedVolume";
};


ExperimentExporter::ExperimentExporter(const std::string & dbName) : SqlSerializer(dbName)
{
	resetTables();
}


void ExperimentExporter::resetTables()
{
	const QString queryCreateTable(
		"CREATE TABLE IF NOT EXISTS '%1' ("
		"'ID_FRAME' INT NOT NULL, "
		"'ID_ACTOR' INT NOT NULL, "
		"'pos_x' REAL NOT NULL, "
		"'pos_y' REAL NOT NULL, "
		"'pos_z' REAL NOT NULL, "
		"'size_x' REAL NOT NULL, "
		"'size_y' REAL NOT NULL, "
		"'size_z' REAL NOT NULL, "
		"PRIMARY KEY('ID_FRAME', 'ID_ACTOR'));"
	);

	const QString queryClearTable(
		"DELETE FROM '%1';"
	);

	const std::vector<QString> allTables =
	{
		StringConstants::TableGroundTruth,
		StringConstants::TableInitialSolution,
		StringConstants::TableMergedSolution,
		StringConstants::TableFusedSolution,
	};

	for (const auto & s : allTables)
	{
		executeQuery(queryCreateTable.arg(s).toStdString());
		executeQuery(queryClearTable.arg(s).toStdString());
	}

	const QString queryCreateTableFusedVolume(
		"CREATE TABLE IF NOT EXISTS '%1' ("
		"'ID_FRAME' INT NOT NULL, "
		"'ID_ACTOR' INT NOT NULL, "
		"'voxelCenter_x' REAL NOT NULL, "
		"'voxelCenter_y' REAL NOT NULL, "
		"'voxelCenter_z' REAL NOT NULL, "
		"'boundingBox_start_x' REAL NOT NULL, "
		"'boundingBox_start_y' REAL NOT NULL, "
		"'boundingBox_start_z' REAL NOT NULL, "
		"'boundingBox_end_x' REAL NOT NULL, "
		"'boundingBox_end_y' REAL NOT NULL, "
		"'boundingBox_end_z' REAL NOT NULL, "
		"PRIMARY KEY('ID_FRAME', 'ID_ACTOR'));"
	);

	executeQuery(queryCreateTableFusedVolume.arg(StringConstants::TableFusedVolume).toStdString());
	executeQuery(queryClearTable.arg(StringConstants::TableFusedVolume).toStdString());
}


void ExperimentExporter::begin()
{
	executeQuery("BEGIN;");
}


void ExperimentExporter::commit()
{
	executeQuery("COMMIT;");
}


void ExperimentExporter::newGroundTruth(const Frame & f)
{
	addContainer(f, StringConstants::TableGroundTruth, [this](const std::string & s) { executeQuery(s); });
}


void ExperimentExporter::newSolution(const Solution & s)
{
	addContainer(s, StringConstants::TableInitialSolution, [this](const std::string & s) { executeQuery(s); });
}


void ExperimentExporter::newMergedSolution(const MergedSolution & s)
{
	addContainer(s, StringConstants::TableMergedSolution, [this](const std::string & s) { executeQuery(s); });
}


void ExperimentExporter::newFusedSolution(const FusedSolution & s)
{
	addContainer(s, StringConstants::TableFusedSolution, [this](const std::string & s) { executeQuery(s); });
}


void ExperimentExporter::newFusedVolume(const FusedSolution & s)
{
	QString insertQuery
	(
		"INSERT INTO %1 (ID_FRAME, ID_ACTOR,"
		"voxelCenter_x, voxelCenter_y, voxelCenter_z, "
		"boundingBox_start_x, boundingBox_start_y, boundingBox_start_z, "
		"boundingBox_end_x, boundingBox_end_y, boundingBox_end_z"
		") VALUES (%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12);"
	);


	for (const auto &[i, a] : enumerate(s.actors))
	{
		float3 voxelCenter = make_float3(0);
		for(const auto v : a.volume.getVoxel())
		{
			voxelCenter += v->center;
		}
		voxelCenter /= a.volume.getVoxel().size();
		
		const QString q =
			insertQuery
			.arg(StringConstants::TableFusedVolume)
			.arg(s.framenumber.get())
			.arg(i)
			.arg(voxelCenter.x)
			.arg(voxelCenter.y)
			.arg(voxelCenter.z)
			.arg(a.volume.getPreciseBoundingBox().x1)
			.arg(a.volume.getPreciseBoundingBox().y1)
			.arg(a.volume.getPreciseBoundingBox().z1)
			.arg(a.volume.getPreciseBoundingBox().x2)
			.arg(a.volume.getPreciseBoundingBox().y2)
			.arg(a.volume.getPreciseBoundingBox().z2);

		executeQuery(q.toStdString());
	}
}
