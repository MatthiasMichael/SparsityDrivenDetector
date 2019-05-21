#pragma once

#include <set>

#include "named_type.h"

#include "Camera.h"
#include "WorldCoordinateSystem_SDD.h"



/**
 * When images are fetched from the video input it has to be ensured, that
 * these images are always stored in the same order.
 * When read from the DB each camera gets an ID and an associated video file.
 * IdentifiableCamera stores the ID and uses it to create an ordering in CameraSet.
 */
class IdentifiableCamera : public Camera<WorldCoordinateSystem>
{
private:
	template <typename Derived>
	struct IDSkills : public Comparable<Derived>, Hashable<Derived> { };

	struct IDTag {};

public:
	using ID = NamedType<int, IDTag, IDSkills>;
	using WorldCamera = Camera<WorldCoordinateSystem>;

	explicit IdentifiableCamera(const ID id);

	ID getID() const { return m_id; }
	bool valid() const { return m_id.get() > 0; }

	friend bool operator<(const IdentifiableCamera & lhs, const IdentifiableCamera & rhs);

	friend std::ostream & operator<<(std::ostream & os, const IdentifiableCamera & c);
	friend std::istream & operator>>(std::istream & is, IdentifiableCamera & c);

	friend std::ostream & operator<<(std::ostream & os, const std::set<IdentifiableCamera> & s);
	friend std::istream & operator>>(std::istream & is, std::set<IdentifiableCamera> & s);

private:
	IdentifiableCamera(); // Only to be used by operator>>

private:
	ID m_id;
};

using CameraSet = std::set<IdentifiableCamera>;