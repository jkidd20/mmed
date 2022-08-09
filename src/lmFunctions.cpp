// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]

#include <RcppArmadillo.h>
#include <RcppNumerical.h>

using namespace Rcpp;
using namespace Numer;

// [[Rcpp::export]]
List fastLm(const arma::vec & y, const arma::mat & X) {
  
  int n = X.n_rows, k = X.n_cols;
  
  arma::colvec coef = arma::solve(X, y); 
  arma::colvec resid = y - X*coef; 
  
  double sig2 = arma::as_scalar(arma::trans(resid)*resid/(n-k));
  arma::Mat<double> varMat = 
    sig2 * arma::inv(arma::trans(X)*X);

  return List::create(Named("coef") = coef,
                      Named("covMat")       = varMat);
}


// Function to return significance value of first variable in X
// [[Rcpp::export]]
double fastLmSig(const arma::vec & y, const arma::mat & X) {
  int n = X.n_rows, k = X.n_cols;
  arma::colvec coef = arma::solve(X, y); 
  arma::colvec resid = y - X*coef; 
  double sig2 = arma::as_scalar(arma::trans(resid)*resid/(n-k));
  arma::Mat<double> xtxInv;
  if(arma::inv(xtxInv, arma::trans(X)*X)){
    arma::Mat<double> varMat = sig2 * xtxInv;
    return 2*R::pt(std::abs(coef(1)) / std::sqrt(varMat(1, 1)), n-k, false, false);
  }
  return 1;
}


// Function to return significance value with interaction
// [[Rcpp::export]]
double fastLmSig_int(const arma::vec & y, const arma::mat & X, const arma::vec & inter) {
  int n = X.n_rows, k = X.n_cols;
  
  arma::mat newX = arma::zeros(n, k + 1);
  newX.submat(0, 0, n-1, 1) = X.submat(0, 0, n-1, 1);
  newX.submat(0, 2, n-1, 2) = inter;
  newX.submat(0, 3, n-1, k) = X.submat(0, 2, n-1, k-1);
  
  arma::colvec coef = arma::solve(newX, y); 
  arma::colvec resid = y - newX*coef; 
  double sig2 = arma::as_scalar(arma::trans(resid)*resid/(n-k-1));
  arma::Mat<double> xtxInv;
  if(arma::inv(xtxInv, arma::trans(newX)*newX)){
    arma::Mat<double> varMat = sig2 * xtxInv;
    double newNumerator = std::abs(coef(1) + coef(2));
    double newSD = std::sqrt(varMat(1, 1) + varMat(2, 2) + 2.0*varMat(1, 2));
    return 2*R::pt(newNumerator / newSD, n-k - 1, false, false);
  }
  return 1;
}
