#include "IdentifiableCamera.h"


IdentifiableCamera::IdentifiableCamera() :
	IdentifiableCamera(make_named<ID>(-1))
{
	// empty
}

IdentifiableCamera::IdentifiableCamera(const ID id) :
	Camera<WorldCoordinateSystem>(),
	m_id(id)
{
	// empty
}


bool operator<(const IdentifiableCamera & lhs, const IdentifiableCamera & rhs) 
{
	return (lhs.m_id < rhs.m_id);
}


std::ostream & operator<<(std::ostream & os, const IdentifiableCamera & c) 
{
	os << c.m_id.get() << std::endl;
	os << *(static_cast<const IdentifiableCamera::WorldCamera *>(&c));
	return os;
}


std::istream & operator>>(std::istream & is, IdentifiableCamera & c) 
{
	is >> c.m_id.get();
	is >> *(static_cast<IdentifiableCamera::WorldCamera *>(&c));
	return is;
}


std::ostream & operator<<(std::ostream & os, const CameraSet & s) 
{
	os << s.size() << std::endl;
	std::for_each(s.begin(), --s.end(), [&os](const IdentifiableCamera & c) { os << c << std::endl; });
	os << *s.rbegin();
	return os;
}


std::istream & operator>>(std::istream & is, CameraSet & s) 
{
	s.clear();

	size_t size = 0;
	is >> size;

	for(size_t i = 0; i < size; ++i)
	{
		IdentifiableCamera c;
		is >> c;
		s.insert(c);
	}

	return is;
}
