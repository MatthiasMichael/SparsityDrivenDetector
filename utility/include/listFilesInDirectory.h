#pragma once

#ifdef _WIN32
	#include <windows.h>
#else
	#include "dirent.h"
	#include <sys/stat.h>
#endif

#include <vector>
#include <string>
#include <iostream>

#pragma comment(lib, "User32.lib")


static bool endsWith( const std::string & filename, const std::string & ending)
{
	if (filename.length() >= ending.length())
	{
		return (filename.compare(filename.length() - ending.length(), ending.length(), ending) == 0);
	}
	return false;
}


static bool isValidDirectory(const std::string & path)
{
	#ifdef _WIN32
		DWORD dwAttrib = GetFileAttributes(path.c_str());

		return (dwAttrib != INVALID_FILE_ATTRIBUTES && 
			  (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
		   
	// Ubuntu
	#else
		struct stat sb;
		return (stat(&path[0], &sb) == 0 && S_ISDIR(sb.st_mode));

	#endif
}


static std::vector<std::string> listFilesInDirectory(const std::string & directory)
{
	#ifdef _WIN32
		std::vector<std::string> files;

		if (directory.length() > (MAX_PATH - 3))
		{
			std::cout << "Directory path is too long." << std::endl;
			return files;
		}

		std::string dirAppended = directory;
		if(endsWith(dirAppended, "\\"))
		{
			dirAppended.append("*");
		}
		else
		{
			dirAppended.append("\\*");
		}

		WIN32_FIND_DATA ffd;
		HANDLE hFind = FindFirstFile(dirAppended.c_str(), &ffd);

		if (INVALID_HANDLE_VALUE == hFind)
		{
			std::cout << "Error Handle" << std::endl;
			return files;
		}

		do
		{
			if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) //file is no directory
			{
				std::string filename = directory;
				if(!endsWith(filename, "\\"))
				{
					filename.append("\\");
				}
				filename.append(ffd.cFileName);
				files.push_back(filename);
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		if (GetLastError() != ERROR_NO_MORE_FILES)
		{
			std::cout << "Unknown error" << std::endl;
		}

		FindClose(hFind);
		return files;

	// Ubuntu
	#else
		DIR *dirHandle;
		struct stat sb;
		struct dirent *entry;

		std::vector<std::string> files;
		std::string dir = directory;

		if (dir.length() > (255 - 3))
		{
			std::cout << "Directory path is too long." << std::endl;
			return files;
		}

		if(!endsWith(dir, "/"))
		{
			dir.append("/");
		}

		if((dirHandle = opendir (&dir[0])) != NULL)
		{
 			/* print all the files and directories within directory */
  			while ((entry = readdir (dirHandle)) != NULL)
 			{
				std::string file = dir + entry->d_name;


				if (stat(&file[0], &sb) == 0 && S_ISREG(sb.st_mode))
				{
    			  		files.push_back(file);
 				}
  			}
  			closedir(dirHandle);
		} 
		else 
		{
 			/* could not open directory */
  			std::cout << "Directory Open Error" << std::endl;
		}
  	
		return files;

	#endif
}


static std::vector<std::string> filterFileList(const std::vector<std::string> & fileList, const std::string & ending)
{
	std::vector<std::string> filteredList;
	for (size_t i = 0; i < fileList.size(); ++i)
	{
		if (endsWith(fileList[i], ending))
		{
			filteredList.push_back(fileList[i]);
		}
	}
	return filteredList;
}


static std::vector<std::string> filterFileList(const std::vector<std::string> & fileList, const std::vector<std::string> & endings)
{
	std::vector<std::string> filteredList;
	for (size_t i = 0; i < fileList.size(); ++i)
	{
		for (size_t j = 0; j < endings.size(); ++j)
		{
			if (endsWith(fileList[i], endings[j]))
			{
				filteredList.push_back(fileList[i]);
				break;
			}
		}
	}
	return filteredList;
}