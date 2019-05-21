#pragma once

#include "VoxelSegmentationInfo.cuh"

/**
 * Frage: WARUM IST DAS HIER SO LEER
 `*
 * Antwort:
 * Im Moment ist nur eine Funktion als Predicate unterst�tzt und in die Kernel fest integriert. 
 * Es ist schwierig, alle Funktionen, die in der CPU Implementierung zur Verf�gung standen in die 
 * GPU Version zu integrieren, da hierf�r Informationen ben�tigt werden, die auf der GPU nicht zur Verf�gung stehen
 * Z.B. welche Kameras ber�cksichtigt werden sollen (alles was in der CPU-GUI anklickbar war).
 * Die einzige M�glichkeit die ich hier sehe ist eine Templatisierung der Kernel. Diese werden dann entweder
 * mit einem Device-Funktionsobjekt oder einem Lambda-Ausdruck aufgerufen werden. 
 * Scheinbar MUSS ein Kernel, der ein Lambda entgegennimmt immer mit dem Typ dieses Lambdas templatisiert werden
 * Daran f�hrt also kein Weg dran vorbei. 
 * In jedem Fall k�nnte es Probleme geben, die bool-Vektoren mit den Infos welche Kameras angeschaut werden sollen
 * angemessen schnell auf das Device zu bringen.
 * Die gesamte Aktion w�rde also zu einer erheblichen Erh�hung der Laufzeit f�hren, was aktuell nicht mit ihrem
 * Nutzen aufgewogen werden kann.
 * Das Feature wird also hinzugef�gt, sobald es doch n�tig wird ggf. mit einem anderen Ausf�hrungspfad um die 
 * aktuelle Geschwindigkeit beibehalten zu k�nnen wenn es nicht ben�tigt wird.
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