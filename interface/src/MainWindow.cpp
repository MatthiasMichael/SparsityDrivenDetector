#include "MainWindow.h"

#include "ui_MainWindow.h"

#include "DisplayDescriptors_Environment.h"
#include "DisplayDescriptors_SparsityDrivenDetector.h"
#include "DisplayDescriptors_ShapeFromSilhouette.h"
#include "DisplayDescriptors_Fusion.h"


MainWindow::MainWindow(QWidget * pParent /*= nullptr*/) :
	QWidget(pParent),
	m_ui(std::make_unique<Ui::MainWindow>())
{
	m_ui->setupUi(this);

	osg::StateSet * rootStateSet = m_ui->widget_osg->getOsgWidget()->getRoot()->getOrCreateStateSet();
	rootStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
	rootStateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);

	id_environment_lights = m_ui->widget_osg->registerVisualization(areaLightsDescriptor);
	id_environment_coordinates = m_ui->widget_osg->registerVisualization(coordinateSystemDescriptor);
	id_environment_cameras = m_ui->widget_osg->registerVisualization(camerasDescriptor);
	id_environment_staticMesh = m_ui->widget_osg->registerVisualization(staticMeshDescriptor);
	id_environment_navMesh = m_ui->widget_osg->registerVisualization(navMeshDescriptor);

	id_grid = m_ui->widget_osg->registerVisualization(gridDescriptor);

	id_groundTruth = m_ui->widget_osg->registerVisualization(groundTruthDescriptor);

	id_sfs = m_ui->widget_osg->registerVisualization(objectsDescriptor);

	id_solution = m_ui->widget_osg->registerVisualization(solutionDescriptor);
	id_mergedSolution = m_ui->widget_osg->registerVisualization(mergedSolutionDescriptor);
	id_fusedSolution = m_ui->widget_osg->registerVisualization(fusedSolutionDescriptor);
	
	m_segmentationLabels = 
	{
		m_ui->labelImageSegm1,
		m_ui->labelImageSegm2,
		m_ui->labelImageSegm3,
		m_ui->labelImageSegm4,
	};

	m_reconstructedLabels =
	{
		m_ui->labelImageReconst1,
		m_ui->labelImageReconst2,
		m_ui->labelImageReconst3,
		m_ui->labelImageReconst4
	};

}


MainWindow::~MainWindow()
{
	// empty but needed to delete a unique_ptr to incomplete type
}


void MainWindow::showStaticElements(const Environment & env) const
{
	using Vector = GeometricTypes<ScalarType, 3>::Vector;

	const auto roi = env.getStaticMesh().getBoundingBox();

	m_ui->widget_osg->setDataToVisualize(id_environment_lights, roi);
	m_ui->widget_osg->setDataToVisualize(id_environment_coordinates);
	m_ui->widget_osg->setDataToVisualize(id_environment_cameras, env.getCameras());
	m_ui->widget_osg->setDataToVisualize(id_environment_staticMesh, env.getStaticMesh());
	m_ui->widget_osg->setDataToVisualize(id_environment_navMesh, env.getNavMesh());

	m_ui->widget_osg->updateUi();

	const osg::Vec3 center = roi.center<osg::Vec3>();
	const osg::Vec3 size = roi.size<osg::Vec3>();

	m_ui->widget_osg->getOsgWidget()->setLookAt(osg::Vec3(size.x(), size.y(), 3 * size.z()), center);
}


void MainWindow::showGrid(const GridPoints & grid) const
{
	m_ui->widget_osg->setDataToVisualize(id_grid, grid);
	m_ui->widget_osg->updateUi();
}


void MainWindow::showGroundTruth(const Frame & f) const
{
	m_ui->widget_osg->setDataToVisualize(id_groundTruth, f);
	m_ui->widget_osg->updateUi();
}


void MainWindow::showSolution(const Solution & s) const
{
	m_ui->widget_osg->setDataToVisualize(id_solution, s);
	m_ui->widget_osg->updateUi();
}


void MainWindow::showMergedSolution(const MergedSolution & s) const
{
	m_ui->widget_osg->setDataToVisualize(id_mergedSolution, s);
	m_ui->widget_osg->updateUi();
}


void MainWindow::showFusedSolution(const FusedSolution & s) const
{
	m_ui->widget_osg->setDataToVisualize(id_fusedSolution, s);
	m_ui->widget_osg->updateUi();
}


void MainWindow::showSfsObjects(const std::vector<const sfs::Voxel *> & voxel,
	const std::vector<sfs::VoxelCluster> & cluster) const
{
	m_ui->widget_osg->setDataToVisualize(id_sfs, voxel, cluster);
	m_ui->widget_osg->updateUi();
}


void MainWindow::showSegmentationImages(const std::vector<cv::Mat> & images) const
{
	const size_t maxIndex = std::min(images.size(), m_segmentationLabels.size());

	for (size_t i = 0; i < maxIndex; ++i)
	{
		QImage img(images[i].data, images[i].cols, images[i].rows, QImage::Format_Indexed8);
		m_segmentationLabels[i]->setPixmap(QPixmap::fromImage(img));
	}
}


void MainWindow::showReconstructedImages(const std::vector<cv::Mat> & images) const
{
	const size_t maxIndex = std::min(images.size(), m_reconstructedLabels.size());

	for (size_t i = 0; i < maxIndex; ++i)
	{
		QImage img(images[i].data, images[i].cols, images[i].rows, QImage::Format_Indexed8);
		m_reconstructedLabels[i]->setPixmap(QPixmap::fromImage(img));
	}
}


void MainWindow::clearSolution() const
{
	m_ui->widget_osg->clearDataToVisualize(id_solution);
	m_ui->widget_osg->updateUi();
}


void MainWindow::clearMergedSolution() const
{
	m_ui->widget_osg->clearDataToVisualize(id_mergedSolution);
	m_ui->widget_osg->updateUi();
}


void MainWindow::clearFusedSolution() const
{
	m_ui->widget_osg->clearDataToVisualize(id_fusedSolution);
	m_ui->widget_osg->updateUi();
}


void MainWindow::clearReconstructedImages() const
{
	for(auto l : m_reconstructedLabels)
	{
		l->clear();
	}
}
