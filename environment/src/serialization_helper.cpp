#include "serialization_helper.h"

#include "zip.h"


path makeTempDir(const path & parentDir)
{
	// TODO: Ensure that the temp dir name is unique
	const path pathTempFolder = parentDir / "___temp";

	if (is_directory(pathTempFolder))
	{
		remove_all(pathTempFolder);
	}

	create_directory(pathTempFolder);

	return pathTempFolder;
}


void zipDir(const path & dir, path outfile)
{
	using namespace boost::filesystem;

	outfile.replace_extension(".zip");

	if (is_regular_file(outfile))
	{
		remove(outfile);
	}

	int err = 0;

	zip * archive = zip_open(outfile.string().c_str(), ZIP_CREATE, &err);

	if (err)
	{
		throw std::runtime_error("Error during zip open");
	}

	zip_source * source = nullptr;

	for (directory_entry & e : recursive_directory_iterator(dir))
	{
		const auto relativePath = e.path().generic_string().substr(dir.string().size() + 1);

		if (is_directory(e))
		{
			zip_dir_add(archive, relativePath.c_str(), ZIP_FL_ENC_GUESS);
		}
		else if (is_regular_file(e))
		{
			source = zip_source_file(archive, e.path().string().c_str(), 0, 0);
			zip_file_add(archive, relativePath.c_str(), source, ZIP_FL_OVERWRITE);
		}
	}

	zip_close(archive);
}


void unzipDir(path zipFile, const path & outdir)
{
	using namespace boost::filesystem;

	zipFile.replace_extension(".zip");

	if (!is_directory(outdir))
	{
		create_directory(outdir);
	}

	int err = 0;
	zip * archive = zip_open(zipFile.string().c_str(), ZIP_RDONLY, &err);

	if (!archive)
	{
		throw std::runtime_error("Error during Zip Open!");
	}

	zip_stat_t info;

	for (int i = 0; i < zip_get_num_entries(archive, 0); i++)
	{
		zip_stat_index(archive, i, 0, &info);

		if (info.name[strlen(info.name) - 1] == '/')
		{
			path dir = outdir / std::string(info.name);
			create_directories(dir);
			continue;
		}

		zip_file_t * file = zip_fopen_index(archive, i, 0);

		if (!file)
		{
			throw std::runtime_error("Error during Zip Open!");
		}

		std::vector<char> fileBuffer(info.size, 0);

		if (zip_fread(file, fileBuffer.data(), info.size) < 0)
		{
			throw std::runtime_error("Error during Zip File Read!");
		}

		path targetLocation = outdir / std::string(info.name);

		create_directories(targetLocation.parent_path());

		std::ofstream unzippedFile(targetLocation.string(), std::ios::out | std::ios::binary | std::ios::trunc);
		unzippedFile.write(fileBuffer.data(), info.size);

		zip_fclose(file);
	}

	zip_close(archive);
}


NoNewlineOutStream::NoNewlineOutStream(std::ostream & o):
	mp_os(&o),
	m_owning(false)
{
	// empty
}


NoNewlineOutStream::NoNewlineOutStream(const std::string & filename):
	mp_os(new std::ofstream(filename)),
	m_owning(true)
{
	// empty
}


NoNewlineOutStream::~NoNewlineOutStream()
{
	if (m_owning)
	{
		delete mp_os;
	}
}


NoNewlineOutStream & NoNewlineOutStream::operator<<(std::ostream &(* f)(std::ostream &))
{
	if (f == &std::endl)
	{
		*mp_os << " ";
	}
	else
	{
		f(*mp_os);
	}
	return *this;
}
