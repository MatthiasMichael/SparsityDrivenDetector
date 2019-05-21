#pragma once

#include <cassert>
#include <exception>
#include <numeric>
#include <vector>

template <typename Iterator>
class TemporaryJoinedVector
{
public:
	struct IteratorPair
	{
		IteratorPair(Iterator _begin, Iterator _end) :
			begin(_begin),
			end(_end),
			size(end - begin)
		{
			// empty
		}


		const Iterator begin;
		const Iterator end;
		const size_t size;
	};


	TemporaryJoinedVector(std::vector<IteratorPair> ranges) :
		m_ranges(ranges),
		m_size(std::accumulate(m_ranges.begin(), m_ranges.end(), size_t(0),
		                       [](const size_t & a, const IteratorPair & b)
		                       {
			                       return a + b.size;
		                       }))
	{
		// empty
	}


	typename std::iterator_traits<Iterator>::value_type const & operator[](size_t idx) const // TODO: Const reference return?
	{
		assert(idx < m_size);

		if (idx >= m_size)
			throw std::logic_error("Index out of bounds!");

		for (const auto & r : m_ranges)
		{
			if (idx >= r.size)
			{
				idx -= r.size;
			}
			else
			{
				return *(r.begin + idx);
			}
		}

		throw std::logic_error("Index out of bounds!");
	}


	size_t size() const { return m_size; }

private:
	const std::vector<IteratorPair> m_ranges;
	const size_t m_size;
};