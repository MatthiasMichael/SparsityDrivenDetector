#pragma once

#include <sstream>

#include <boost/filesystem.hpp>


using boost::filesystem::path;

path makeTempDir(const path & parentDir);

void zipDir(const path & dir, path outfile);
void unzipDir(path zipFile, const path & outdir);


class NoNewlineOutStream
{
public:
	explicit NoNewlineOutStream(std::ostream & o);
	explicit NoNewlineOutStream(const std::string & filename);

	~NoNewlineOutStream();

	NoNewlineOutStream(NoNewlineOutStream &&) = delete;
	NoNewlineOutStream(const NoNewlineOutStream &) = delete;

	NoNewlineOutStream & operator=(NoNewlineOutStream &&) = delete;
	NoNewlineOutStream & operator=(const NoNewlineOutStream &) = delete;

	NoNewlineOutStream & operator<<(std::ostream & (*f)(std::ostream &));

	template <typename T>
	NoNewlineOutStream & operator<<(const T & v)
	{
		m_ss.str("");
		m_ss << v;
		auto x = m_ss.str();
		std::replace(x.begin(), x.end(), '\n', ' ');
		while(x.back() == ' ')
		{
			x.pop_back();
		}
		*mp_os << x;
		return *this;
	}

private:
	std::ostream * mp_os;
	const bool m_owning;

	std::stringstream m_ss;
};
