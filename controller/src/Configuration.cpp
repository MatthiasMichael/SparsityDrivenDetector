#include "Configuration.h"

#include <fstream>

#include <boost/program_options.hpp>

const std::map<std::string, IloCplex::Algorithm> algorithmToIdentifier =
{
	{ "AutoAlg", IloCplex::AutoAlg },
	{ "Dual", IloCplex::Dual },
	{ "Primal", IloCplex::Primal },
	{ "Barrier", IloCplex::Barrier },
	{ "Network", IloCplex::Network },
	{ "Sifting", IloCplex::Sifting },
	{ "Concurrent", IloCplex::Concurrent }
};


boost::program_options::options_description getCommandLineOptions()
{
	using namespace boost::program_options;

	options_description commandLineOptions("Command Line Options");
	commandLineOptions.add_options()
		("help,h", "Show help")
		("config,c", value<std::string>()->default_value("res/settings.ini"), "Name of the configuration file");

	return commandLineOptions;
}


boost::program_options::options_description getConfigFileOptions()
{
	using namespace boost::program_options;

	options_description configuration("Configuration");
	configuration.add_options()
		("scene", value<std::string>(), "The scene to be processed")
		("grid.distX", value<ScalarType>(), "Distance of grid points in X direction")
		("grid.distY", value<ScalarType>(), "Distance of grid points in Y direction")
		("grid.height", value<ScalarType>(), "Height of grid points in Z direction")
		("image.width", value<int>(), "Image width used for optimization")
		("image.height", value<int>(), "Image width used for optimization")
		("template.files", value<std::vector<std::string>>()->multitoken(), "List of template files")
		("template.classes", value<std::vector<int>>()->multitoken(), "List of template classes. Used to determine which templates represent the same class and can be merged.")
		("template.height", value<std::vector<ScalarType>>()->multitoken(), "List of template heights. Used for projection during dictionary creation.")
		("template.width", value<std::vector<ScalarType>>()->multitoken(), "List of template widths. Used for projection during dictionary creation.")
		("template.maxHeight", value<std::vector<ScalarType>>()->multitoken(), "List of maximal heights a template can have. Used for voxel assignment.")
		("template.maxWidth", value<std::vector<ScalarType>>()->multitoken(), "List of maximal widths a template can have. Used for voxel assignment." )
		("input.firstFrame", value<int>(), "First frame that should be considered for processing")
		("detector.optimizationType", value<std::string>(), "Optimization that should be used for the detector")
		("detector.maxMergingDistance", value<ScalarType>(), "Maximum distance for which detections can be merged")
		("optimization.advancedInitialization", value<int>(), "Advanced initialization (0-2)")
		("optimization.rootAlg", value<std::string>(), "Optimization Algorithm to be used")
		("sfs.init.computeLocation", value<std::string>(), "'Host' oder 'Device' to decide whether to use Shape from Silhouette on the CPU oder GPU")
		("sfs.init.voxelSize", value<ScalarType>(), "Length of all sides of a voxel. Currently only voxels with equal side length are supported.")
		("sfs.param.minPartSegmentation", value<ScalarType>(), "Minimum amount of a voxel's projection that needs to be segmented so that the voxel is marked by the respective camera.")
		("sfs.param.maxClusterDistance", value<int>(), "Maximal distance in multiples of voxel side length between two voxels so that they can still belong to the same cluster. ")
		("debug.skipSDDBuild", value<int>(), "!DEBUG! SDD won't build any algorithm info and won't be usable")
		("debug.skipSFSBuild", value<int>(), "!DEBUG! SFS won't build voxel info and won't be usable");
		
	return configuration;
}


Configuration::Configuration() : configAvailable(false),
                                 maxMergingDistance(0),
                                 advancedInitialization(0),
                                 rootAlgorithm(IloCplex::NoAlg),
                                 computeLocation(sfs::Host),
                                 sfsParameters(),
                                 debug_skipSDDBuildStep(false),
                                 debug_skipSFSBuildStep(false)
{
	// empty
}


Configuration::Configuration(int argc, char ** argv)
{
	using namespace boost::program_options;

	variables_map vars_cmd;
	store(parse_command_line(argc, argv, getCommandLineOptions()), vars_cmd);
	notify(vars_cmd);

	if (vars_cmd.count("help"))
	{
		std::cout << options_description("All Options").
		             add(getCommandLineOptions()).
		             add(getConfigFileOptions()) <<
			std::endl;
		return;
	}

	initFromFilename(vars_cmd["config"].as<std::string>());
}


Configuration::Configuration(const std::string configFile)
{
	initFromFilename(configFile);
}


void Configuration::initFromFilename(const std::string & configFile)
{
	using namespace boost::program_options;

	variables_map vars_cfg;

	{
		std::ifstream is(configFile);
		store(parse_config_file(is, getConfigFileOptions()), vars_cfg);
	}

	notify(vars_cfg);

	sceneFile = vars_cfg["scene"].as<std::string>();

	gridParameters = GridPoints::Parameters
	{
		make_named<GridPoints::Distance>
		(
			vars_cfg["grid.distX"].as<ScalarType>(),
			vars_cfg["grid.distY"].as<ScalarType>()
		),
		vars_cfg["grid.height"].as<ScalarType>()
	};

	targetImageSize = make_named<TemplateTransformer::ImageSize>
	(
		vars_cfg["image.width"].as<int>(),
		vars_cfg["image.height"].as<int>()
	);

	templateFiles = vars_cfg["template.files"].as<std::vector<std::string>>();

	templateClasses = vars_cfg["template.classes"].as<std::vector<int>>();

	const auto templateTargetWidths = vars_cfg["template.width"].as<std::vector<ScalarType>>();
	const auto templateTargetHeights = vars_cfg["template.height"].as<std::vector<ScalarType>>();

	const auto templateMaxWidths = vars_cfg["template.maxWidth"].as<std::vector<ScalarType>>();
	const auto templateMaxHeights = vars_cfg["template.maxHeight"].as<std::vector<ScalarType>>();

	if (
		templateFiles.size() != templateClasses.size() ||
		templateFiles.size() != templateTargetWidths.size() ||
		templateFiles.size() != templateTargetHeights.size() ||
		templateFiles.size() != templateMaxWidths.size() ||
		templateFiles.size() != templateMaxHeights.size()
	   )
	{
		throw std::runtime_error("Length of template infos not consistent.");
	}

	for (size_t i = 0; i < templateFiles.size(); ++i)
	{
		templateTargetSizes.emplace_back(make_named<Template::Size>(templateTargetWidths[i], templateTargetHeights[i]));
		templateMaxSizes.emplace_back(make_named<Template::Size>(templateMaxWidths[i], templateMaxHeights[i]));
	}

	firstFrame = make_named<Framenumber>(vars_cfg["input.firstFrame"].as<int>());

	optimizationType = vars_cfg["detector.optimizationType"].as<std::string>();
	maxMergingDistance = vars_cfg["detector.maxMergingDistance"].as<ScalarType>();

	advancedInitialization = vars_cfg["optimization.advancedInitialization"].as<int>();
	if (advancedInitialization < 0 || advancedInitialization > 2)
	{
		std::cout << "Advanced initialization parameter out of range." << std::endl;
		std::cout << "Fallback to 1." << std::endl << std::endl;
		advancedInitialization = 1;
	}

	try
	{
		rootAlgorithm = algorithmToIdentifier.at(vars_cfg["optimization.rootAlg"].as<std::string>());
	}
	catch (const std::out_of_range &)
	{
		std::cout << "Invalid algorithm identifier found." << std::endl;
		std::cout << "Fallback to AutoAlg." << std::endl << std::endl;
		rootAlgorithm = IloCplex::AutoAlg;
	}

	const auto computeLocation_s = vars_cfg["sfs.init.computeLocation"].as<std::string>();
	computeLocation = sfs::tryStringToComputeLocation(computeLocation_s); // propagate Exception

	const auto voxelSizeValue = vars_cfg["sfs.init.voxelSize"].as<ScalarType>();
	voxelSize = Vector3(voxelSizeValue, voxelSizeValue, voxelSizeValue);

	sfsParameters.minSegmentation = vars_cfg["sfs.param.minPartSegmentation"].as<ScalarType>();
	sfsParameters.maxClusterDistance = vars_cfg["sfs.param.maxClusterDistance"].as<int>();

	if(vars_cfg.count("debug.skipSDDBuild") != 0)
	{
		debug_skipSDDBuildStep = vars_cfg["debug.skipSDDBuild"].as<int>() != 0;
	}
	else
	{
		debug_skipSDDBuildStep = false;
	}

	if (vars_cfg.count("debug.skipSFSBuild") != 0) 
	{
		debug_skipSFSBuildStep = vars_cfg["debug.skipSFSBuild"].as<int>() != 0;
	}
	else
	{
		debug_skipSFSBuildStep = false;
	}

	configAvailable = true;
}


const std::string & Configuration::getSceneFile() const
{
	ensureConfigAvailable();
	return sceneFile;
}


const GridPoints::Parameters & Configuration::getGridParameters() const
{
	ensureConfigAvailable();
	return gridParameters;
}


const TemplateTransformer::ImageSize & Configuration::getTargetImageSize() const
{
	ensureConfigAvailable();
	return targetImageSize;
}


const std::vector<std::string> & Configuration::getTemplateFiles() const
{
	ensureConfigAvailable();
	return templateFiles;
}


const std::vector<int> & Configuration::getTemplateClasses() const
{
	ensureConfigAvailable();
	return templateClasses;
}


const std::vector<Template::Size> & Configuration::getTemplateTargetSizes() const
{
	ensureConfigAvailable();
	return templateTargetSizes;
}


const std::vector<Template::Size> & Configuration::getTemplateMaxSizes() const
{
	ensureConfigAvailable();
	return templateMaxSizes;
}


const Framenumber & Configuration::getFirstFrame() const
{
	ensureConfigAvailable();
	return firstFrame;
}


const std::string & Configuration::getOptimizationType() const
{
	ensureConfigAvailable();
	return optimizationType;
}


ScalarType Configuration::getMaxMergingDistance() const
{
	ensureConfigAvailable();
	return maxMergingDistance;
}


CPXINT Configuration::getAdvancedInitialization() const
{
	ensureConfigAvailable();
	return advancedInitialization;
}


IloCplex::Algorithm Configuration::getRootAlgorithm() const
{
	ensureConfigAvailable();
	return rootAlgorithm;
}


sfs::ComputeLocation Configuration::getComputeLocation() const
{
	ensureConfigAvailable();
	return computeLocation;
}


Vector3 Configuration::getVoxelSize() const
{
	ensureConfigAvailable();
	return voxelSize;
}


const sfs::ShapeFromSilhouette_Impl::Parameters & Configuration::getSfsParameters() const
{
	ensureConfigAvailable();
	return sfsParameters;
}


bool Configuration::getSkipSDD() const
{
	ensureConfigAvailable();
	return debug_skipSDDBuildStep;
}


bool Configuration::getSkipSFS() const
{
	ensureConfigAvailable();
	return debug_skipSFSBuildStep;
}


Configuration Configuration::tryMakeConfiguration(int argc, char ** argv)
{
	Configuration config;

	try
	{
		config = Configuration(argc, argv);
	}
	catch (std::exception & e)
	{
		std::cout << "Error during Configuration initialization:\n";
		std::cout << e.what();
		std::cout << "\n\n" << std::endl;
	}

	return config;
}


void Configuration::ensureConfigAvailable() const
{
	assert(configAvailable);

	if (!configAvailable)
	{
		throw std::runtime_error("No config available");
	}
}
