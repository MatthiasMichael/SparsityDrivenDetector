#pragma once

#include <cassert>
#include <vector>

/**
* Temporärer Wrapper um aus einer Range eines Vektors einen eigens ansprechbaren Vektor zu machen.
* Dadurch können alle Beobachtungen für ein Pixel einfach und ohne Kopieraufwand zurückgegeben werden.
**/
template <typename Iterator>
class TemporaryVector
{
public:
	TemporaryVector() : m_begin(), m_end(), m_size(0)
	{
		// empty
	}

	TemporaryVector(const Iterator begin, const Iterator end) : 
	  m_begin(begin), m_end(end), m_size(end - begin)
	{
		// empty
	}

	typename std::iterator_traits<Iterator>::reference operator[](size_t idx) 
	{ 
		assert(idx < m_size);

		if(idx >= m_size)
			throw std::logic_error("Index out of bounds!");

		return *(m_begin + idx); 
	}

	typename std::iterator_traits<Iterator>::value_type const & operator[](size_t idx) const // TODO: Const reference return?
	{ 
		assert(idx < m_size);

		if(idx >= m_size)
			throw std::logic_error("Index out of bounds!");

		return *(m_begin + idx); 		
	}

	Iterator begin() const { return m_begin; }
	Iterator end() const { return m_end; }

	const Iterator cbegin() const { return m_begin; }
	const Iterator cend() const { return m_end; }

	size_t size() const { return m_size; }

private:
	Iterator m_begin;
	Iterator m_end;

	size_t m_size;
};


/**
* Wie oben, kann zwei Vektoren zusammenfassen. Auf den Inhalt kann jedoch nur lesend zugegriffen werden.
*/
template <typename Iterator>
class TemporaryJoinedVector
{
public:
	TemporaryJoinedVector(const Iterator beginFirst, const Iterator endFirst, 
		const Iterator beginSecond, const Iterator endSecond) : 
	m_beginFirst(beginFirst), m_endFirst(endFirst), m_beginSecond(beginSecond), m_endSecond(endSecond),
		m_sizeFirst(endFirst - beginFirst), m_sizeSecond(endSecond - beginSecond), m_size(m_sizeFirst + m_sizeSecond)
	{
		// empty
	}
 
	typename std::iterator_traits<Iterator>::value_type const & operator[](size_t idx) const // TODO: Const reference return?
	{ 
		assert(idx < m_size);

		if(idx >= m_size)
			throw std::logic_error("Index out of bounds!");

		if(idx < m_sizeFirst)
		{
			return *(m_beginFirst + idx); 
		}
		else
		{
			idx -= m_sizeFirst;
			return *(m_beginSecond + idx);
		}
	}

	size_t size() const { return m_size; }

private:
	const Iterator m_beginFirst;
	const Iterator m_endFirst;

	const Iterator m_beginSecond;
	const Iterator m_endSecond;

	const size_t m_sizeFirst;
	const size_t m_sizeSecond;
	const size_t m_size;
};