#pragma once

#include "VoxelSegmentationInfo.cuh"

/**
 * Frage: WARUM IST DAS HIER SO LEER
 `*
 * Antwort:
 * Im Moment ist nur eine Funktion als Predicate unterstützt und in die Kernel fest integriert. 
 * Es ist schwierig, alle Funktionen, die in der CPU Implementierung zur Verfügung standen in die 
 * GPU Version zu integrieren, da hierfür Informationen benötigt werden, die auf der GPU nicht zur Verfügung stehen
 * Z.B. welche Kameras berücksichtigt werden sollen (alles was in der CPU-GUI anklickbar war).
 * Die einzige Möglichkeit die ich hier sehe ist eine Templatisierung der Kernel. Diese werden dann entweder
 * mit einem Device-Funktionsobjekt oder einem Lambda-Ausdruck aufgerufen werden. 
 * Scheinbar MUSS ein Kernel, der ein Lambda entgegennimmt immer mit dem Typ dieses Lambdas templatisiert werden
 * Daran führt also kein Weg dran vorbei. 
 * In jedem Fall könnte es Probleme geben, die bool-Vektoren mit den Infos welche Kameras angeschaut werden sollen
 * angemessen schnell auf das Device zu bringen.
 * Die gesamte Aktion würde also zu einer erheblichen Erhöhung der Laufzeit führen, was aktuell nicht mit ihrem
 * Nutzen aufgewogen werden kann.
 * Das Feature wird also hinzugefügt, sobald es doch nötig wird ggf. mit einem anderen Ausführungspfad um die 
 * aktuelle Geschwindigkeit beibehalten zu können wenn es nicht benötigt wird.
 */

namespace sfs
{
	namespace cuda
	{

		typedef bool(*GeometryPredicate)(const VoxelSegmentationInfo & voxel);

		__device__ inline bool voxelPredicate_maximumVisibleActive(const VoxelSegmentationInfo & voxel)
		{
			uint countSegmented = 0, countVisible = 0;
			for (uint i = 0; i < voxel.m_numImages; ++i)
			{
				if (voxel.m_segmentationStats[i] == Marked)
				{
					++countSegmented;
				}

				if (voxel.m_visibilityStats[i] == Visible)
				{

					++countVisible;
				}
			}

			return countSegmented > 1 && countSegmented == countVisible;
		}
	}
}