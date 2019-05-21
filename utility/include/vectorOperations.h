#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>


// Für das Sortieren benötigt. Speichert einen Wert (hier in der Regel ein double) zusammen mit dem Index ab, welchen er 
// ursprünglich in einem Vektor hatte.
template<typename T>
struct ValueWithIndex
{
	ValueWithIndex(T val, const unsigned int i) : value(val), index(i) { }
	T value;
	unsigned int index;
};


// Vergleichs-Funktionsobjekt, welches ValueWithIndex Objekte anhand ihres values vergleichen kann
template<typename T>
struct ValueWithIndexComparator
{
	bool operator()(const ValueWithIndex<T> & first, const ValueWithIndex<T> & second)
	{
		return first.value < second.value;
	}
};


// Sortiert den Vektor values und gibt ein Array indices zurück wobei indices[i] die Position ist, die values[i] vor dem
// Soriteren hatte.
//
// TODO: Wenn das umkopieren zu lange dauert, dann muss die optimize-Methode im Weak Classifier umgebaut werden um direkt 
// mit std::vector<ValueWithIndex> zu arbeiten.
template<typename T>
std::vector<unsigned int> sortWithIndex(std::vector<T>& values)
{
	const size_t size = values.size();
	std::vector<ValueWithIndex<T>> toSort;
	toSort.reserve(size);
	for (unsigned int i = 0; i < size; ++i)
		toSort.push_back(ValueWithIndex<T>(values[i], i));

	std::sort(toSort.begin(), toSort.end(), ValueWithIndexComparator<T>());

	std::vector<unsigned int> indices(size);
	for (size_t i = 0; i < size; ++i)
	{
		indices[i] = toSort[i].index;
		values[i] = toSort[i].value;
	}

	return indices;
}


template<typename T, typename U>
inline T scalarProduct(const std::vector<T>& v, const std::vector<U>& w) //-> decltype(T * U)
{
	assert(v.size() == w.size());

	T t(0);
	//for (unsigned i = v.size(); i--; t += v[i] * w[i]);
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		t += v[i] * w[i];
	}
	return t;
}


inline double scalarProduct(const std::vector<double> & v, const std::vector<bool> & w)
{
	assert(v.size() == w.size());

	double t(0);
	for (size_t i = 0; i < v.size(); ++i)
	{
		t += v[i] * (w[i] ? 1 : 0);
	}
	return t;
}


inline double scalarProduct(const std::vector<bool> & v, const std::vector<double> & w)
{
	return scalarProduct(w, v);
}


template<typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> & input)
{
	std::vector<std::vector<T>> transposedVector;
	for (unsigned int i = 0; i < input[0].size(); ++i) // TODO: input[0].size() ein Hack?
	{
		std::vector<T> valueIForAllVectors(input.size());
		for (unsigned int j = 0; j < input.size(); ++j)
			valueIForAllVectors[j] = input[j][i];
		transposedVector.push_back(valueIForAllVectors);
	}
	return transposedVector;
}


/** 
 * Alle Operatorüberladungen befinden sich im std-namespace, damit diese beim Argument Dependent Lookup
 * auch korrekt gefunden werden. Im globalen namespace werden diese in manchen Fällen (abhängig von der Zahl
 * der Template-Argumente) nicht gefunden, da erst im std-namespace nach geeigneten Instaziierungen gesucht wird.
 * Hier wird dann oft der Default-Fall gefunden, welcher in einem Kompilierungsfehler resultiert. Zumindest ist das
 * bei den Stream-Operatoren der Fall. 
 */
namespace std
{

	template<typename T>
	std::vector<T> & operator*=(std::vector<T> & a, const std::vector<T> & b)
	{
		assert(a.size() == b.size());

		std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::multiplies<T>());
		return a;
	}


	template<typename T>
	std::vector<T> & operator/=(std::vector<T> & a, const std::vector<T> & b)
	{
		assert(a.size() == b.size());

		std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::divides<T>());
		return a;
	}


	template<typename T>
	std::vector<T> & operator+=(std::vector<T> & a, const std::vector<T> & b)
	{
		assert(a.size() == b.size());

		std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
		return a;
	}


	template<typename T>
	std::vector<T> & operator-=(std::vector<T> & a, const std::vector<T> & b)
	{
		assert(a.size() == b.size());

		std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::minus<T>());
		return a;
	}


	template<typename T>
	std::vector<T> operator*(std::vector<T> a, const std::vector<T> & b)
	{
		return a *= b;
	}


	template<typename T>
	std::vector<T> operator/(std::vector<T> a, const std::vector<T> & b)
	{
		return a /= b;
	}



	template<typename T>
	std::vector<T> operator+(std::vector<T> a, const std::vector<T> & b)
	{
		return a += b;
	}


	template<typename T>
	std::vector<T> operator-(std::vector<T> a, const std::vector<T> & b)
	{
		return a -= b;
	}

#ifndef __CUDACC__ // It seems CUDA / NVCC has Problems with the iterator construct

	template<typename T>
	std::vector<T> operator*=(std::vector<T> & b, const double scalar)
	{
		for (std::vector<T>::iterator i = b.begin(); i != b.end(); ++i)
			*i *= scalar;
		return b;
	}


	template<typename T>
	std::vector<T> operator/=(std::vector<T> & b, const double scalar)
	{
		for (std::vector<T>::iterator i = b.begin(); i != b.end(); ++i)
			*i /= scalar;
		return b;
	}


	template<typename T>
	std::vector<T> operator+=(std::vector<T> & b, const double scalar)
	{
		for (std::vector<T>::iterator i = b.begin(); i != b.end(); ++i)
			*i += scalar;
		return b;
	}



	template<typename T>
	std::vector<T> operator-=(std::vector<T> & b, const double scalar)
	{
		for (std::vector<T>::iterator i = b.begin(); i != b.end(); ++i)
			*i -= scalar;
		return b;
	}


	template<typename T>
	std::vector<T> operator*(const double scalar, std::vector<T> b)
	{
		return b *= scalar;
	}

	template<typename T>
	std::vector<T> operator+(const double scalar, std::vector<T> b)
	{
		return b += scalar;
	}

	template<typename T>
	std::vector<T> operator-(const double scalar, std::vector<T> b)
	{
		return b -= scalar;
	}


	template<typename T>
	std::vector<T> operator*(std::vector<T> b, const double scalar)
	{
		return b *= scalar;
	}


	template<typename T>
	std::vector<T> operator/(std::vector<T> b, const double scalar)
	{
		return b /= scalar;
	}


	template<typename T>
	std::vector<T> operator+(std::vector<T> b, const double scalar)
	{
		return b += scalar;
	}


	template<typename T>
	std::vector<T> operator-(std::vector<T> b, const double scalar)
	{
		return b -= scalar;
	}

#endif __CUDACC__


	template<typename T>
	std::ostream & operator<<(std::ostream & os, const std::vector<T> & v)
	{
		const size_t numElements = v.size();
		os << numElements;

		for (size_t i = 0; i < numElements; ++i)
		{
			os << std::endl << v[i];
		}

		return os;
	}


	template < typename T >
	std::istream & operator>>(std::istream & is, std::vector<T> & v)
	{
		size_t numElements = 0;
		is >> numElements;

		if(is.peek() == '\n') is.get(); // endl überlesen

		v.clear();
		v.reserve(numElements);

		for (size_t i = 0; i < numElements; ++i)
		{
			T t;
			is >> t;
			v.push_back(t);
		}

		return is;
	}


	template<typename T>
	bool operator==(const std::vector<T> & first, const std::vector<T> & second)
	{
		if (first.size() != second.size())
		{
			return false;
		}
		else
		{
			for (size_t i = 0; i < first.size(); ++i)
			{
				if (!(first[i] == second[i]))
				{
					return false;
				}
			}
		}
		return true;
	}


	template<typename T>
	bool operator!=(const std::vector<T> & first, const std::vector<T> & second)
	{
		return !(first == second);
	}

}


// Generelle triviale Serialisierung. Sollte vielleicht aus diesem Header entfernt werden und in einen eigenen binary_serialization header
template<typename T>
static void writeBinary(std::ostream & os, const T & t)
{
	os.write(reinterpret_cast<const char*>(&t), sizeof(T));
}


template<typename T>
static void readBinary(std::istream & is, T & t)
{
	is.read(reinterpret_cast<char*>(&t), sizeof(T));
}


// Funktionen für binäre Serialisierung von Vektoren. write/readBinary sollte angewendet werden, wenn T nicht trivial
// serialisierbar ist (mindestens einen Pointer besitzt oder Member hat die Pointer haben oder auf andere weise speziell
// auseinandergebaut und wieder zusammengesetzt werden können). Dann muss T wieder eigene read/writeBinary Funktionen 
// anbieten. Die Funktionen die es hier drüber gibt serialisieren ebenfalls nur trivial.
template<typename T>
void writeBinary(std::ostream & os, const std::vector<T> & v)
{
	const size_t s = v.size();
	os.write(reinterpret_cast<const char*>(&s), sizeof(s));

	for (size_t i = 0; i < s; ++i)
	{
		writeBinary(os, v[i]);
	}
}


template<typename T>
void readBinary(std::istream & is, std::vector<T> & v)
{
	size_t s;
	is.read(reinterpret_cast<char*>(&s), sizeof(s));
	v.resize(s);

	for (size_t i = 0; i < s; ++i)
	{
		readBinary(is, v[i]);
	}
}


// Binary serialization und deserialization. Warning: Unsafe. Only works if T is trivially serializable!!!
template<typename T>
void writeBinaryTrivial(std::ostream & os, const std::vector<T> & v)
{
	const size_t s = v.size();
	os.write(reinterpret_cast<const char*>(&s), sizeof(s));
	os.write(reinterpret_cast<const char*>(&v[0]), s * sizeof(T));
}


template<typename T>
void readBinaryTrivial(std::istream & is, std::vector<T> & v)
{
	size_t s;
	is.read(reinterpret_cast<char*>(&s), sizeof(s));
	v.resize(s);
	is.read(reinterpret_cast<char*>(&v[0]), s * sizeof(T));
}


template<typename T>
void writeBinary(std::ostream & os, const std::shared_ptr<T> & p)
{
	writeBinary(os, *p);
}


template<typename T>
void readBinary(std::istream & is, std::shared_ptr<T> & p)
{
	p = std::make_shared<T>();
	readBinary(is, *p);
}


template<typename T>
std::vector<const T*> vectorOfPointersTo(const std::vector<T> & v)
{
	std::vector<const T*> ret;
	ret.reserve(v.size());
	for(int i = 0; i < v.size(); ++i)
	{
		ret.push_back(&v[i]);
	}
	return ret;
}