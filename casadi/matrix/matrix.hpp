/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include "../casadi_exception.hpp"
#include "../printable_object.hpp"
#include "element.hpp"
#include "crs_sparsity.hpp"

namespace CasADi{

  
/** \brief General sparse matrix class
  \author Joel Andersson 
  \date 2010	
*/
template<class T>
class Matrix : public std::vector<T>, public PrintableObject{

  public:
    /** \brief  constructors */
    /// empty 0-by-0 matrix constructor
    Matrix();
    
    /// empty n-by-m matrix constructor
    Matrix(int n, int m);
    
    /// dense n-by-m matrix filled with val constructor
    Matrix(int n, int m, const T& val);

    /// This constructor enables implicit type conversion from a scalar type
    Matrix(const T &val);

    /** \brief  Create an expression from an stl vector  */
    template<typename A>
    Matrix(const std::vector<A>& x){
      sparsity_ = CRSSparsity(x.size(),1,true);
      std::vector<T>::resize(x.size());
      copy(x.begin(),x.end(),std::vector<T>::begin());
    }

    /** \brief  Create a non-vector expression from an stl vector */
    template<typename A>
    Matrix(const std::vector<A>& x,  int n, int m){
      if(x.size() != n*m) throw CasadiException("Matrix::Matrix(const std::vector<double>& x,  int n, int m): dimension mismatch");
      sparsity_ = CRSSparsity(n,m,true);
      std::vector<T>::resize(x.size());
      copy(x.begin(),x.end(),std::vector<T>::begin());
    }
    
    // get the number of non-zeros
    //int size() const;        

    /// get the number of elements
    int numel() const;

    /// get the first dimension
    int size1() const;       
    
    /// get the second dimension
    int size2() const;

    //@{
    /// Check type of matrix
    bool empty() const; // is the matrix empty
    bool scalar() const; // is the matrix scalar
    bool vector() const; // is the matrix a vector
    //@}

    /// get an element
    const T getElement(int i=0, int j=0) const;
    
    /// get a reference to an element
    T& getElementRef(int i=0, int j=0);
  
    /// Access an element 
    Element<Matrix<T>,T> operator()(int i, int j=0){ return Element<Matrix<T>,T>(*this,i,j); }

    /// Const access an element 
    const T operator()(int i, int j=0) const{ return getElement(i,j); }

    /** \brief  Make the matrix an dense n-by-m matrix */
    void makeDense(int n, int m, const T& val);

    /** \brief  Make the matrix an empty n-by-m matrix */
    void makeEmpty(int n, int m);

    //@{
    /// Printing
    virtual void print(std::ostream &stream=std::cout) const; // print default style
    void printScalar(std::ostream &stream=std::cout) const; // print scalar
    void printVector(std::ostream &stream=std::cout) const; // print vector-style
    void printMatrix(std::ostream &stream=std::cout) const; // print matrix-style
    //@}

    // Get the sparsity pattern
    const std::vector<int>& col() const;
    const std::vector<int>& rowind() const;
    std::vector<int>& col();
    std::vector<int>& rowind();
    int col(int el) const;
    int rowind(int row) const;
    
    
    void clear();
    void resize(int n, int m);
    void reserve(int nnz);

    
  private:
    /// Sparsity of the matrix in a compressed row storage (CRS) format
    CRSSparsity sparsity_;
};

// Implementations
template<class T>
const T Matrix<T>::getElement(int i, int j) const{
  int ind = sparsity_.getNZ(i,j);
  if(ind==-1)
    return 0;
  else
    return std::vector<T>::at(ind);
}

template<class T>
T& Matrix<T>::getElementRef(int i, int j){
  int oldsize = sparsity_.size();
  int ind = sparsity_.getNZ(i,j);
  if(oldsize != sparsity_.size())
    std::vector<T>::insert(std::vector<T>::begin()+ind,0);
  return std::vector<T>::at(ind);
}

template<class T>
int Matrix<T>::size1() const{
  return sparsity_.size1();
}

template<class T>
int Matrix<T>::size2() const{
  return sparsity_.size2();
}

template<class T>
int Matrix<T>::numel() const{
  return size1()*size2();
}

        
template<class T>
void Matrix<T>::makeDense(int n, int m, const T& val){
  if(n*m != numel()){
    sparsity_ = CRSSparsity(n,m,true);
    std::vector<T>::clear();
    std::vector<T>::resize(n*m, val);
  }
}

template<class T>
bool Matrix<T>::empty() const{
  return numel()==0;
}

template<class T>
bool Matrix<T>::scalar() const{
  return size1()==1 && size2()==1;
}

template<class T>
bool Matrix<T>::vector() const{
  return size2()==1;
}

template<class T>
Matrix<T>::Matrix(){
  sparsity_ = CRSSparsity(0,0,false);
}

template<class T>
Matrix<T>::Matrix(int n, int m){
  sparsity_ = CRSSparsity(n,m,false);
}

template<class T>
Matrix<T>::Matrix(int n, int m, const T& val){
  sparsity_ = CRSSparsity(n,m,true);
  std::vector<T>::resize(n*m, val);
}

template<class T>
void Matrix<T>::makeEmpty(int n, int m){
  sparsity_ = CRSSparsity(n,m,false);
  std::vector<T>::clear();
}

template<class T>
void Matrix<T>::printScalar(std::ostream &stream) const {
  if(numel()!=1) throw CasadiException("operator=: argument not scalar");
  stream << (*this)(0);
}
  
template<class T>
void Matrix<T>::printVector(std::ostream &stream) const {
  // print dimension and first element
  stream << "[" << size1() << "](";
  stream << (*this)(0);
  for(int i=1; i<size1(); i++){
    stream << "," << (*this)(i);
  }
  stream << ")";  
}

template<class T>
void Matrix<T>::printMatrix(std::ostream &stream) const{ 
 // print dimension
 stream << "[" << size1() << "," << size2() << "]";
 // print the first row
 stream << "((";
  stream << (*this)(0,0);
  for(int j=1; j<size2(); j++){
    stream << "," << (*this)(0,j);
 }
 stream << ")";
 // print the rest of the rows
 for(int i=1; i<size1(); i++){  
    stream << ",("; 
    stream << (*this)(i,0);
    for(int j=1; j<size2(); j++){
      stream << "," << (*this)(i,j);
    }
    stream << ")";
  }
  stream << ")";  
}

template<class T>
void Matrix<T>::print(std::ostream &stream) const{
  if (empty())
    stream << "<empty expression>";
  else if (numel()==1)
    printScalar(stream);
  else if (size2()==1)
    printVector(stream);
  else
    printMatrix(stream);
}

template<class T>
const std::vector<int>& Matrix<T>::col() const{
  return sparsity_.col();
}

template<class T>
std::vector<int>& Matrix<T>::col(){
  return sparsity_.col();
}

template<class T>
const std::vector<int>& Matrix<T>::rowind() const{
  return sparsity_.rowind();
}

template<class T>
std::vector<int>& Matrix<T>::rowind(){
  return sparsity_.rowind();
}

template<class T>
int Matrix<T>::col(int el) const{
  return sparsity_.col(el);
}

template<class T>
int Matrix<T>::rowind(int row) const{
  return sparsity_.rowind(row);
}

template<class T>
void Matrix<T>::reserve(int nnz){
  std::vector<T>::reserve(nnz);
  col().reserve(nnz);
}

template<class T>
void Matrix<T>::resize(int n_, int m_){
  sparsity_.resize(n_,m_);
}

template<class T>
void Matrix<T>::clear(){
  sparsity_ = CRSSparsity(0,0,false);
  std::vector<T>::clear();
}

template<class T>
Matrix<T>::Matrix(const T &val){
  sparsity_ = CRSSparsity(1,1,true);
  std::vector<T>::resize(1,val);
}

} // namespace CasADi


#endif // MATRIX_HPP

