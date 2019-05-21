#pragma once

#include <iostream>
#include <list>
#include <string>
#include <sstream>
#include <vector>


template<typename T>
class SparseMatrix
{
public:
	struct Entry
	{
		size_t col; T val;

		friend bool operator==(const Entry & lhs, const Entry & rhs)
		{
			return lhs.col == rhs.col && lhs.val == rhs.val;
		}
	};
	using Row = std::list<Entry>;

	SparseMatrix() : SparseMatrix(0, 0) { }
	SparseMatrix(size_t rows, size_t cols);

	void clear();
	void addEntry(size_t row, size_t col, T val);
	const Row & getRow(size_t row) const;

	void write(std::ostream & os) const;
	void read(std::istream & is);

	friend bool operator==(const SparseMatrix & lhs, const SparseMatrix & rhs)
	{
		if (lhs.m_maxRows != rhs.m_maxRows || lhs.m_maxCols != rhs.m_maxCols)
		{
			return false;
		}

		if (lhs.m_rows.size() != rhs.m_rows.size())
		{
			return false;
		}

		return lhs.m_rows == rhs.m_rows;
	}


private:
	size_t m_maxRows;
	size_t m_maxCols;

	std::vector<Row> m_rows;
};


template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols) : m_maxRows(rows), m_maxCols(cols), m_rows(m_maxRows, Row{})
{
	
}


template <typename T>
void SparseMatrix<T>::clear()
{
	m_rows = std::vector<Row>(m_maxRows, Row{});
}


template <typename T>
void SparseMatrix<T>::addEntry(size_t row, size_t col, T val)
{
	if(row > m_maxRows || col > m_maxCols)
	{
		throw std::out_of_range("Row or Col too big...");
	}

	std::list<Entry> & r = m_rows[row];

	auto it = r.begin();
	while(it != r.end() && it->col <= col)
	{
		++it;
	}

	if(it == r.end())
	{
		r.push_back(Entry{col, val});
	}
	else if (it->col == col)
	{
		it->val = val;
	}
	else if(it->col > col)
	{
		r.insert(it, Entry{ col, val });
	}
}


template <typename T>
const typename SparseMatrix<T>::Row & SparseMatrix<T>::getRow(size_t row) const
{
	if (row > m_maxRows)
	{
		throw std::out_of_range("row too big!");
	}

	return m_rows[row];
}


template<typename T>
void SparseMatrix<T>::write(std::ostream & os) const
{
	for(size_t r = 0; r < m_rows.size(); ++r)
	{
		for(const Entry & e : m_rows[r])
		{
			os << r + 1 << " " << e.col + 1 << " " << e.val << std::endl;
		}
	}
}


template <typename T>
void SparseMatrix<T>::read(std::istream & is)
{
	clear();

	int row, col;
	T val;

	std::string line;
	while (std::getline(is, line))
	{
		std::istringstream ss(line);
		ss >> row >> col >> val;

		addEntry(row - 1, col - 1, val);
	}
}


