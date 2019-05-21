#include "ConfigurationExporter.h"
#include <QString>
#include <enumerate.h>


struct ConfigurationExporter::StringConstants
{
	inline static const auto TableConfig = "Configuration";
	inline static const auto TableTemplates = "Configuration_Templates";
};

ConfigurationExporter::ConfigurationExporter(const std::string & dbName) : SqlSerializer(dbName)
{
	resetTables();
}

void ConfigurationExporter::resetTables()
{
	const QString queryCreateTableConfig(
		"CREATE TABLE IF NOT EXISTS '%1' ("
		"'ID' INT NOT NULL, "
		"'scene' TEXT NOT NULL, "
		"'grid.distX' REAL NOT NULL, "
		"'grid.distY' REAL NOT NULL, "
		"'grid.height' REAL NOT NULL, "
		"'image.width' INT NOT NULL, "
		"'image.height' INT NOT NULL, "
		"'input.firstFrame' INT NOT NULL, "
		"'detector.optimizationType' TEXT NOT NULL, "
		"'detector.maxMergingDistance' REAL NOT NULL, "
		"'optimization.advancedInitialization' INT NOT NULL, "
		"'optimization.rootAlg' TEXT NOT NULL, "
		"'sfs.init.computeLocation' TEXT NOT NULL, "
		"'sfs.init.voxelSize' REAL NOT NULL, "
		"'sfs.param.minPartSegmentation' REAL NOT NULL, "
		"'sfs.param.maxClusterDistance' INT NOT NULL, "
		"PRIMARY KEY('ID'));"
	);

	const QString queryCreateTableTemplates(
		"CREATE TABLE IF NOT EXISTS '%1' ("
		"'ID' INT NOT NULL, "
		"'file' TEXT NOT NULL, "
		"'class' INT NOT NULL, "
		"'width' REAL NOT NULL, "
		"'height' REAL NOT NULL, "
		"'maxWidth' REAL NOT NULL, "
		"'maxHeight' REAL NOT NULL, "
		"PRIMARY KEY('ID'));"
	);

	const QString queryClearTable(
		"DELETE FROM '%1';"
	);

	executeQuery(queryCreateTableConfig.arg(StringConstants::TableConfig).toStdString());
	executeQuery(queryCreateTableTemplates.arg(StringConstants::TableTemplates).toStdString());

	executeQuery(queryClearTable.arg(StringConstants::TableConfig).toStdString());
	executeQuery(queryClearTable.arg(StringConstants::TableTemplates).toStdString());
}

void ConfigurationExporter::exportConfiguration(const Configuration & c)
{
	const QString insertConfigQuery(
		"INSERT INTO '%1' ("
		"ID, "
		"scene, "
		"'grid.distX', 'grid.distY', 'grid.height', "
		"'image.width', 'image.height', "
		"'input.firstFrame', "
		"'detector.optimizationType', 'detector.maxMergingDistance', "
		"'optimization.advancedInitialization', 'optimization.rootAlg', "
		"'sfs.init.computeLocation', 'sfs.init.voxelSize', "
		"'sfs.param.minPartSegmentation', 'sfs.param.maxClusterDistance'"
		") VALUES "
		"(%2, '%3', %4, %5, %6, %7, %8, %9, '%10', %11, %12, '%13', '%14', %15, %16, %17);"
	);

	const QString insertTemplateQuery(
		"INSERT INTO '%1' (ID, file, class, width, height, maxWidth, maxHeight) "
		"VALUES (%2, '%3', %4, %5, %6, %7, %8);"
	);

	const std::map<IloCplex::Algorithm, QString> algorithmToIdentifier =
	{
		{ IloCplex::AutoAlg   , "AutoAlg",  },
		{ IloCplex::Dual      , "Dual",     },
		{ IloCplex::Primal    , "Primal",   },
		{ IloCplex::Barrier   , "Barrier",  },
		{ IloCplex::Network   , "Network",  },
		{ IloCplex::Sifting   , "Sifting",  },
		{ IloCplex::Concurrent, "Concurrent"}
	};

	const std::map<sfs::ComputeLocation, QString> computeLocationToIdentifier =
	{
		{sfs::Host, "Host"},
		{sfs::Device, "Device"}
	};

	const QString q1 = insertConfigQuery
		.arg(StringConstants::TableConfig)
		.arg(0)
		.arg(QString::fromStdString(c.getSceneFile()))
		.arg(c.getGridParameters().distancePoints.get()(0))
		.arg(c.getGridParameters().distancePoints.get()(1))
		.arg(c.getGridParameters().groundPlaneHeight)
		.arg(c.getTargetImageSize().get()(0))
		.arg(c.getTargetImageSize().get()(1))
		.arg(c.getFirstFrame().get())
		.arg(QString::fromStdString(c.getOptimizationType()))
		.arg(c.getMaxMergingDistance())
		.arg(c.getAdvancedInitialization())
		.arg(algorithmToIdentifier.at(c.getRootAlgorithm()))
		.arg(computeLocationToIdentifier.at(c.getComputeLocation()))
		.arg(c.getVoxelSize()(0))
		.arg(c.getSfsParameters().minSegmentation)
		.arg(c.getSfsParameters().maxClusterDistance);

	executeQuery(q1.toStdString());

	for(const auto & [i, file] : enumerate(c.getTemplateFiles()))
	{
		const QString q2 = insertTemplateQuery
			.arg(StringConstants::TableTemplates)
			.arg(i)
			.arg(QString::fromStdString(file))
			.arg(c.getTemplateClasses()[i])
			.arg(c.getTemplateTargetSizes()[i].get()(0))
			.arg(c.getTemplateTargetSizes()[i].get()(1))
			.arg(c.getTemplateMaxSizes()[i].get()(0))
			.arg(c.getTemplateMaxSizes()[i].get()(1));

		executeQuery(q2.toStdString());
	}

}
