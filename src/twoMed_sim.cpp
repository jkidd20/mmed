// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(BH)]]

#include <RcppDist.h>
#include <RcppNumerical.h>
#include <boost/math/special_functions/bessel.hpp>
#include <math.h>
#include <Rcpp.h>

//#include <vector>
//#include <deque>
//#include <map>
//#include <string>
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <cfloat>
//#include <algorithm>    // std::sort, std::count
//#include <random>
//#include <eigen3/Eigen/Dense>
//#include <boost/math/special_functions/factorials.hpp>
//#include <boost/math/special_functions/gamma.hpp>
//#include <boost/math/distributions/normal.hpp> // for normal,pdf,cdf
#include <math.h>

using namespace Eigen;
using namespace Rcpp;
using namespace Numer;


// Terms: _x denotes for xth mediator, _obs indicates the observed mediator
//        gamma - mediator to response effect
//        ksi - parameter vector for independent to mediator
//        beta- parameter vector for the mediators to the response
//        psi - independent-mediator interaction - which mediator denoted by _x
//        G   - independent variable value (also included in X and Y, but specially passed for ease)
//        Z   - matrix (or row vector) for mediator to response regression
//        X   - matrix (or row vector) for independent to mediator regression
//        h1  - mediator-mediator interaction term
//        h2  - independent-mediator-mediator interaction term
//        sig2- variance of mediator to response regression
//        tau2- variance of independent to mediator regression
//        rho - parameter for correlation between mediators
//        r   - correlation between mediators in proportional distributions


// Update December 2020 to handle interaction terms
// makes each a and b value dependent upon observation
//[[Rcpp::export]]
double calc_a_val_twosim(const double & gamma, const double & psi, const double & G,
                         const double & S_obs, const double & h1, const double & h2,
                         const double & sig2, const double & tau2, const double & rho){

  return( 1 / (
      std::pow(gamma + psi*G + S_obs*(h1 + G*h2), 2.0) / sig2  +
        1 / (( 1 - std::pow(rho, 2.0) ) * tau2)    )
  ) ;
}

/// b-value - don't include a term here - multiply it in other functions;
//[[Rcpp::export]]
double calc_b_val_twosim(const double & gamma_same, const double & gamma_diff,
                         const double & psi_same, const double & psi_diff,
                         const double & h1, const double & h2, const double & rho,
                         const double & tau2_same,  const double & tau2_diff, const double & sig2,
                         const double & Y, const double & G, const arma::rowvec & Z, const arma::vec & beta,
                         const double & S_obs, const arma::rowvec & X, const arma::vec & ksi_same,
                         const arma::vec & ksi_diff){

  double yPart = Y - as_scalar(Z*beta) - (gamma_diff + G*psi_diff)*S_obs;
  double p1 = ((gamma_same + psi_same*G + S_obs*(h1 + h2*G))*yPart) / sig2;

  double p2 = rho*(S_obs - as_scalar(X*ksi_diff)) / ( (1.0 - std::pow(rho, 2.0)) * std::sqrt(tau2_diff*tau2_same));

  double p3 = as_scalar(X*ksi_same) / ( (1.0 - std::pow(rho, 2.0)) * tau2_same);
  // Rcout << p1 << ", " << p2 << ", " << p3 << "\n";
  return (p1 + p2 + p3);
}



//[[Rcpp::export]]
double calc_c_val_twosim(const double & gamma_same, const double & gamma_diff,
                         const double & psi_same, const double & psi_diff, const double & G,
                         const double & tau2_same,  const double & tau2_diff,
                         const double & sig2, const double & rho){
  return ( tau2_same*( std::pow(gamma_diff + psi_diff*G, 2.0)*tau2_diff * ( 1.0 - std::pow(rho, 2.0)) + sig2) ) /
    ( 2.0*(gamma_same + psi_same*G)*(gamma_diff + psi_diff*G)*std::sqrt(tau2_same*tau2_diff)*rho +
      std::pow(gamma_same + psi_same*G, 2.0)*tau2_same +
      std::pow(gamma_diff + psi_diff*G, 2.0)*tau2_diff + sig2 );
}

//[[Rcpp::export]]
double calc_d_val_twosim(const double & gamma_same, const double & gamma_diff,
                         const double & psi_same, const double & psi_diff,
                         const double & rho, const double & r,
                         const double & tau2_same,  const double & tau2_diff, const double & sig2,
                         const double & Y, const double & G, const arma::rowvec & Z, const arma::rowvec & X,
                         const arma::vec & beta, const arma::vec & ksi_same, const arma::vec & ksi_diff,
                         const double & c_same, const double c_diff){
  double p1 = (as_scalar(X*ksi_diff)*(std::sqrt(c_diff*tau2_same) * r - std::sqrt(c_same*tau2_diff)*rho)) /
    (std::sqrt(tau2_same)*tau2_diff*( 1.0 - rho*rho ));
  double p2 = (as_scalar(X*ksi_same)*( std::sqrt(c_same*tau2_diff)  - std::sqrt(c_diff*tau2_same)*rho*r )) /
    (std::sqrt(tau2_diff)*tau2_same*( 1.0 - rho*rho ));
  double p3 = ( ( Y - as_scalar(Z*beta) ) * ( std::sqrt(c_same)*(gamma_same+psi_same*G) + std::sqrt(c_diff)*(gamma_diff+psi_diff*G)*r )) /
    sig2;
  return std::sqrt(c_same)*(p1 + p2 + p3);
}


// multiple by a outside of function
//[[Rcpp::export]]
double calc_corr_b_twosim(const double & gamma_1, const double & gamma_2,
                          const double & psi_1, const double & psi_2, const double & G,
                          const double & tau2_1, const double & tau2_2,
                          const double & c1, const double & c2,
                          const double & sig2, const double & rho){
  return ( ( 1.0 - rho*rho )*std::sqrt(tau2_1*tau2_2)*sig2 ) /
    (
        std::sqrt(c1*c2) * (
            rho*sig2 - (gamma_1 + psi_1*G)*(gamma_2 + psi_2*G)*std::sqrt(tau2_1*tau2_2)*( 1.0 - std::pow(rho, 2.0) )
        )
    );
}

//[[Rcpp::export]]
double calc_corr_r_twosim(const double & b){
  double r1 = (-b + std::sqrt(b*b + 4.0)) / 2.0;
  double r2 = (-b - std::sqrt(b*b + 4.0)) / 2.0;

  if(r1 >= -1 & r1 <= 1){return r1;}
  if(r2 >= -1 & r2 <= 1){return r2;}

  return 0;
}


/* Functions to find expected values in the case of two missing and mediator-mediator interaction
 * Uses integration over space of s values
 * December 2020 - initial try to integrate over area larger than obsereved s values
 */
/* Class needed according to https://cran.r-project.org/web/packages/RcppNumerical/vignettes/introduction.html */
/* Class already in Numeric namespace apparently
 class MFunc
 {
 public:
 virtual double operator()(Constvec& x) = 0;
 virtual ~MFunc() {}
 };
 */

// Class for Multidimensional integration
class s1s2Int_unord: public MFunc
{
private:
  const double sig2;          // sigma squared (variance term)
  const double rho;           // rho correlation term
  double a;             // (Y-beta*X)/sigma;
  double a1X;           // (alpha1*X)
  double a2X;           // (alpha2*X)
  double tau_1;         // tau_1 term (SD not Var)
  double tau_2;         // tau_2 term (SD not Var)
  double gS1;           // (gamma_1 + psi_1*G) / sigma  (sigma is sqrt(sig2) )
  double gS2;           // (gamma_2 + psi_2*G) / sigma
  double ht;            // (h1 + h2*G) / sigma
  int ps1;              // power of s1 in expectation
  int ps2;              // power of s2 in expectation
public:
  s1s2Int_unord(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2_,
                const double & gamma_1, const double & psi_1,
                const double & gamma_2, const double & psi_2,
                const double & h1, const double & h2,
                arma::vec ksi_1, arma::vec ksi_2, arma::rowvec X,
                const double & tau2_1, const double & tau2_2,
                const double & rho_, const int & ps1_, const int & ps2_) : sig2(sig2_), rho(rho_), ps1(ps1_), ps2(ps2_)
  {
    a     = (Y - as_scalar(Z*beta));
    a1X   = as_scalar(X*ksi_1);
    a2X   = as_scalar(X*ksi_2);
    tau_1 = std::sqrt(tau2_1);
    tau_2 = std::sqrt(tau2_2);
    gS1   = (gamma_1 + psi_1*G);
    gS2   = (gamma_2 + psi_2*G);
    ht    = (h1 + h2*G);
  }

  // PDF of bivariate normal
  double operator()(Constvec& x)
  {
    double b = x[0]*gS1 + x[1]*gS2 + x[0]*x[1]*ht;
    double c = (x[0] - a1X)/tau_1;
    double d = (x[1] - a2X)/tau_2;
    double den1 = std::exp(-std::pow(a - b, 2.0)/(2*sig2)) / std::sqrt(2.0*M_PI*sig2);
    double den2 = std::exp(-(1 / (2.0*(1-std::pow(rho, 2.0))))*( std::pow(c, 2.0) + std::pow(d, 2.0) - 2.0*rho*c*d )) /
      (2*M_PI*tau_1*tau_2*std::sqrt(1 - rho*rho));

    return std::pow(x[0], ps1)*std::pow(x[1], ps2) * den1*den2;
    // double baseInt = std::exp(-std::pow(a - b, 2.0)/(2*sig2) -
    //        (1 / 2.0*(1-std::pow(rho, 2.0)))*( std::pow(c, 2.0) + std::pow(d, 2.0) - 2.0*rho*c*d ));
    //
    // return std::pow(x[0], ps1)*std::pow(x[1], ps2) * baseInt;
  }
};


// [[Rcpp::export]]
double bothMissInt_unord(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
                         const double & gamma_1, const double & psi_1,
                         const double & gamma_2, const double & psi_2,
                         const double & h1, const double & h2,
                         arma::vec ksi_1, arma::vec ksi_2, arma::rowvec X,
                         const double & tau2_1, const double & tau2_2, const double & rho,
                         const int & ps1, const int & ps2,
                         const double & lowS1, const double & lowS2,
                         const double & highS1, const double & highS2,
                         const int & nDivisions = 5){

  s1s2Int_unord f(Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X, tau2_1, tau2_2, rho,
                  ps1, ps2);
  double err_est;
  int err_code;
  long double res = 0;
  Eigen::VectorXd lower(2);
  Eigen::VectorXd upper(2);
  double step1 = ( highS1 - lowS1 ) / nDivisions;
  double step2 = ( highS2 - lowS2 ) / nDivisions;
  double step1b = ( highS1 - lowS1 ) / (nDivisions+1);
  double step2b = ( highS2 - lowS2 ) / (nDivisions+1);
  lower << lowS1, lowS2;
  upper << highS1, highS2;
  
  if(nDivisions == 1){
    res = integrate(f, lower, upper, err_est, err_code, 1e8);
  }else{
// number of integrations must be a perfect square to cover all area
// i corresponds to s1, j to s2;
    for(int i = 0; i < nDivisions; i++){
      lower << lowS1 + (step1 * i), lowS2;
      upper << lowS1 + (step1 * (i + 1)), lowS2 + step2;
      for(int j = 0; j < nDivisions; j ++){
        lower(1) = lowS2 + (step2 * j);
        upper(1) =  lowS2 + (step2 * (j + 1));
        res += integrate(f, lower, upper, err_est, err_code, 1e8);
      }
    }
  /// second loop with 1 more division
    for(int i = 0; i < (nDivisions+1); i++){
      lower << lowS1 + (step1b * i), lowS2;
      upper << lowS1 + (step1b * (i + 1)), lowS2 + step2b;
      for(int j = 0; j < (nDivisions+1); j ++){
        lower(1) = lowS2 + (step2b * j);
        upper(1) =  lowS2 + (step2b * (j + 1));
        res += integrate(f, lower, upper, err_est, err_code, 1e8);
      }
    }
    res /=2.0;
  }
//Rcout << res<< "\n";
  return res;
  
/* Old Code used to find lower and upper from integrals
 * Now find these values in outside function (no powers) and use those
 */  
  // long double newres = res + 1e-8;
  // double myStep = 10.0;
  // while(myStep > 1e-2){
  //   while(newres - res > 1e-12){
  //     res = newres;
  //     if(missKey1 >= 0){
  //       lower(0) -= myStep/2.0;
  //     }
  //     if(missKey2 >= 0){
  //       lower(1) -= myStep/2.0;
  //     }
  //     if(missKey1 <= 0){
  //       upper(0) += myStep/2.0;
  //     }
  //     if(missKey2 <= 0){
  //       upper(1) += myStep/2.0;
  //     }
  //     newres = integrate(f, lower, upper, err_est, err_code, 10000);
  //   }
  //   // after hit bad part put upper and lower limits back one step and update step
  //   if(missKey1 >= 0){
  //     lower(0) += myStep/2.0;
  //   }
  //   if(missKey2 >= 0){
  //     lower(1) += myStep/2.0;
  //   }
  //   if(missKey1 <= 0){
  //     upper(0) -= myStep/2.0;
  //   }
  //   if(missKey2 <= 0){
  //     upper(1) -= myStep/2.0;
  //   }
  //   newres = integrate(f, lower, upper, err_est, err_code, 10000);
  //   myStep /= 10.0;
  // }
  // return newres;
}

//[[Rcpp::export]]
double bothMissPDF_unord(arma::vec x,
                         const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
                         const double & gamma_1, const double & psi_1,
                         const double & gamma_2, const double & psi_2,
                         const double & h1, const double & h2,
                         arma::vec ksi_1, arma::vec ksi_2, arma::rowvec X,
                         const double & tau2_1, const double & tau2_2, const double & rho,
                         const int & ps1, const int & ps2){
  double a = Y - as_scalar(Z*beta);
  double a1X   = as_scalar(X*ksi_1);
  double a2X   = as_scalar(X*ksi_2);
  double tau_1 = std::sqrt(tau2_1);
  double tau_2 = std::sqrt(tau2_2);
  double gS1   = (gamma_1 + psi_1*G);
  double gS2   = (gamma_2 + psi_2*G);
  double ht    = (h1 + h2*G);
  
  double b = x(0)*gS1 + x(1)*gS2 + x(0)*x(1)*ht;
  double c = (x(0) - a1X)/tau_1;
  double d = (x(1) - a2X)/tau_2;
  double den1 = std::exp(-std::pow(a - b, 2.0)/(2*sig2)) / std::sqrt(2*M_PI*sig2);
  double den2 = std::exp(-(1 / (2.0*(1-std::pow(rho, 2.0))))*( std::pow(c, 2.0) + std::pow(d, 2.0) - 2.0*rho*c*d )) /
    (2*M_PI*tau_1*tau_2*std::sqrt(1 - rho*rho));
  
  return std::pow(x[0], ps1)*std::pow(x[1], ps2) * den1*den2;
}

// [[Rcpp::export]]
arma::mat bothMissInt_unord_limits(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
                                   const double & gamma_1, const double & psi_1,
                                   const double & gamma_2, const double & psi_2,
                                   const double & h1, const double & h2,
                                   arma::vec ksi_1, arma::vec ksi_2, arma::rowvec X,
                                   const double & tau2_1, const double & tau2_2, const double & rho,
                                   const double & lowS1, const double & lowS2,
                                   const double & highS1, const double & highS2,
                                   const int missKey1 = 0, const int missKey2 = 0, 
                                   const int nSteps = 1e4, const double limit = 1e-14, 
                                   const double stepCorrect = 1000.0){
  s1s2Int_unord f(Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X, tau2_1, tau2_2, rho,
                  0, 0);
  double err_est;
  int err_code;
  double S1range = highS1 - lowS1;
  double S2range = highS2 - lowS2;
  int largeStep = 100;
  Eigen::VectorXd lower(2);
  Eigen::VectorXd upper(2);
  lower << lowS1, lowS2;
  upper << highS1, highS2;
  arma::mat bounds = arma::zeros(2, 2);

  long double res = integrate(f, lower, upper, err_est, err_code, 10000);

  // While loop to make sure integration isn't starting too wide;
  while(res < 1e-5 & largeStep > 2){
    lower << lowS1 + S1range / largeStep, lowS2 + S2range / largeStep;
    upper << highS1 - S1range / largeStep, highS2 - S2range / largeStep;
    res = integrate(f, lower, upper, err_est, err_code, 10000);
    largeStep --;
  }
  
  long double newres = res + 1e-8;
  double myStep = 10.0;
  while(myStep > 1e-5){
    while(newres - res > 1e-13 & newres > 1e-5){
      res = newres;
      if(missKey1 >= 0){
        lower(0) -= myStep/2.0;
      }
      if(missKey2 >= 0){
        lower(1) -= myStep/2.0;
      }
      if(missKey1 <= 0){
        upper(0) += myStep/2.0;
      }
      if(missKey2 <= 0){
        upper(1) += myStep/2.0;
      }
      newres = integrate(f, lower, upper, err_est, err_code, 10000);
    }
    // after hit bad part put upper and lower limits back one step and update step
    if(missKey1 >= 0){
      lower(0) += myStep/2.0;
    }
    if(missKey2 >= 0){
      lower(1) += myStep/2.0;
    }
    if(missKey1 <= 0){
      upper(0) -= myStep/2.0;
    }
    if(missKey2 <= 0){
      upper(1) -= myStep/2.0;
    }
    newres = integrate(f, lower, upper, err_est, err_code, 10000);
    myStep /= 10.0;
  }


// Addition Jan 11, 2021 to widen search area
// Able to do so due to splitting integrals 
  if(missKey1 >= 0){
    upper(0) += myStep*stepCorrect;
  }
  if(missKey2 >= 0){
    upper(1) += myStep*stepCorrect;
  }
  if(missKey1 <= 0){
    lower(0) -= myStep*stepCorrect;
  }
  if(missKey2 <= 0){
    lower(1) -= myStep*stepCorrect;
  }
  
  bounds(0, 0) = lower(0);
  bounds(1, 0) = lower(1);
  bounds(0, 1) = upper(0);
  bounds(1, 1) = upper(1);

  return bounds;
}

// 
// // [[Rcpp::export]]
// arma::mat bothMissInt_unord_limits2(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
//                                    const double & gamma_1, const double & psi_1,
//                                    const double & gamma_2, const double & psi_2,
//                                    const double & h1, const double & h2,
//                                    arma::vec ksi_1, arma::vec ksi_2, arma::rowvec X,
//                                    const double & tau2_1, const double & tau2_2, const double & rho,
//                                    const double & lowS1, const double & lowS2,
//                                    const double & highS1, const double & highS2,
//                                    const int missKey1 = 0, const int missKey2 = 0, 
//                                    const int nSteps = 1e4, const double limit = 1e-14){
//   double low1 = highS1;
//   double low2 = highS2; 
//   double high1 = lowS1; 
//   double high2 = lowS2;
//   double stepVal1 = (highS1 - lowS1) / nSteps;
//   double stepVal2 = (highS2 - lowS2) / nSteps;
//   int findIt1 = 0;
//   double pdfV = 0;
//   bool foundStart = false;
//   arma::mat bounds = arma::zeros(2, 2);
//   bounds(0, 0) = low1;
//   bounds(1, 0) = low2;
//   bounds(0, 1) = high1;
//   bounds(1, 1) = high2;
//   
//   arma::vec curVec = {lowS1, lowS2};
//   
//   /// Find starting location;
//   while(!foundStart){
//     curVec(1) = lowS2;
//     int findIt = 0;
//     while(findIt < nSteps){
//       foundStart = (bothMissPDF_unord(curVec,
//                              Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_2, psi_2,
//                              h1,  h2, ksi_1, ksi_2, X, tau2_1, tau2_2, rho,
//                              0, 0) > limit*100.0);
//       curVec(1) += stepVal2;
//       findIt++;
//     }
//     curVec(0) += stepVal1; 
//   }
//   
//   arma::vec holdVec = curVec;
//   Rcout << holdVec << "\n";
//   
//   // Strategy: start at holdVec - for a given S1 value, go out each direction on S2 until no longer above limit
//   //           For S1 values, start at holdVec, and go up and down 
//   
//   // 
//   // // start low, go high, do grid search;
//   // while(findIt1 < nSteps){
//   //   int findIt2 = 0;
//   //   curVec(1) = lowS2;
//   //   //Rcout << curVec << "\n";
//   //   while(findIt2 < nSteps){
//   //     pdfV = bothMissPDF_unord(curVec,
//   //                              Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_2, psi_2,
//   //                              h1,  h2, ksi_1, ksi_2, X, tau2_1, tau2_2, rho,
//   //                              0, 0);
//   //     // If pdf value is above limit, then update low and and high values accordingly
//   //     // low - minimum found, high - max found
//   //     if(pdfV > limit){
//   //       low1 = std::min(low1, curVec(0));
//   //       high1 = std::max(high1, curVec(0));
//   //       low2 = std::min(low2, curVec(1));
//   //       high2 = std::max(high2, curVec(1));
//   //     }
//   //     curVec(1) += stepVal2;
//   //     findIt2++;
//   //   }
//   //   curVec(0) += stepVal1;
//   //   findIt1++;
//   // }
//   
//   //// Set values to bounds and return;
//   bounds(0, 0) = low1;
//   bounds(1, 0) = low2;
//   bounds(0, 1) = high1;
//   bounds(1, 1) = high2;
//   return bounds;
// }


/////// Secondary Integration if also fitting a model for the product of mediators
// Only applicable if there are no covariates to account for
// Class for Multidimensional integration
class s1s2Int_unord2: public MFunc
{
private:
  const double sig2;          // sigma squared (variance term)
  const double rho;           // rho correlation term
  double a;             // (Y-beta*X)/sigma;
  double a1X;           // (alpha1*X)
  double a2X;           // (alpha2*X)
  double a3X;           // (alpha3*X)
  double tau_1;         // tau_1 term (SD not Var)
  double tau_2;         // tau_2 term (SD not Var)
  double tau_3;         // tau_3 term (SD not Var)
  double gS1;           // (gamma_1 + psi_1*G) / sigma  (sigma is sqrt(sig2) )
  double gS2;           // (gamma_2 + psi_2*G) / sigma
  double ht;            // (h1 + h2*G) / sigma
  int ps1;              // power of s1 in expectation
  int ps2;              // power of s2 in expectation
public:
  s1s2Int_unord2(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2_,
                const double & gamma_1, const double & psi_1,
                const double & gamma_2, const double & psi_2,
                const double & h1, const double & h2,
                arma::vec ksi_1, arma::vec ksi_2, arma::vec ksi_3, arma::rowvec X,  
                const double & tau2_1, const double & tau2_2, const double & tau2_3,
                const double & rho_, const int & ps1_, const int & ps2_) : sig2(sig2_), rho(rho_), ps1(ps1_), ps2(ps2_)
  {
    a     = (Y - as_scalar(Z*beta));
    a1X   = as_scalar(X*ksi_1);
    a2X   = as_scalar(X*ksi_2);
    a3X   = as_scalar(X*ksi_3); 
    tau_1 = std::sqrt(tau2_1);
    tau_2 = std::sqrt(tau2_2);
    tau_3 = std::sqrt(tau2_3);
    gS1   = (gamma_1 + psi_1*G);
    gS2   = (gamma_2 + psi_2*G);
    ht    = (h1 + h2*G);
  }
  
  // PDF of bivariate normal
  double operator()(Constvec& x)
  {
    double b = x[0]*gS1 + x[1]*gS2 + x[0]*x[1]*ht;
    double c = (x[0] - a1X)/tau_1;
    double d = (x[1] - a2X)/tau_2;
    double e = (x[0]*x[1] - a3X)/tau_3;
    double den1 = std::exp(-std::pow(a - b, 2.0)/(2*sig2)) / std::sqrt(2*M_PI*sig2);
    double den2 = std::exp(-(1 / 2.0*(1-std::pow(rho, 2.0)))*( std::pow(c, 2.0) + std::pow(d, 2.0) - 2.0*rho*c*d )) /
      (2*M_PI*tau_1*tau_2*std::sqrt(1 - rho*rho));
    double den3 = 1;
    // Only add density 3 if ht != 0;
    if(ht != 0){den3 = std::exp(-(1 / 2.0*std::pow(e, 2.0)));}
    
    return std::pow(x[0], ps1)*std::pow(x[1], ps2) * den1*den2*den3;
    // double baseInt = std::exp(-std::pow(a - b, 2.0)/(2*sig2) -
    //        (1 / 2.0*(1-std::pow(rho, 2.0)))*( std::pow(c, 2.0) + std::pow(d, 2.0) - 2.0*rho*c*d ));
    //
    // return std::pow(x[0], ps1)*std::pow(x[1], ps2) * baseInt;
  }
};


// [[Rcpp::export]]
double bothMissInt_unord2(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
                         const double & gamma_1, const double & psi_1,
                         const double & gamma_2, const double & psi_2,
                         const double & h1, const double & h2,
                         arma::vec ksi_1, arma::vec ksi_2, arma::vec ksi_3, arma::rowvec X,
                         const double & tau2_1, const double & tau2_2, const double & tau2_3, const double & rho,
                         const int & ps1, const int & ps2,
                         const double & lowS1, const double & lowS2,
                         const double & highS1, const double & highS2,
                         const int missKey1 = 0, const int missKey2 = 0){
  
  s1s2Int_unord2 f(Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_2, psi_2, h1, h2, ksi_1, ksi_2, ksi_3, X, tau2_1, tau2_2, tau2_3, rho,
                  ps1, ps2);
  double err_est;
  int err_code;
  Eigen::VectorXd lower(2);
  Eigen::VectorXd upper(2);
  lower << lowS1, lowS2;
  upper << highS1, highS2;
  
  long double res = integrate(f, lower, upper, err_est, err_code, 10000);
  long double newres = res + 1e-8;
  double myStep = 10.0;
  while(myStep > 1e-2){
    while(newres - res > 1e-12){
      res = newres;
      if(missKey1 >= 0){
        lower(0) -= myStep/2.0;
      }
      if(missKey2 >= 0){
        lower(1) -= myStep/2.0;
      }
      if(missKey1 <= 0){
        upper(0) += myStep/2.0;
      }
      if(missKey2 <= 0){
        upper(1) += myStep/2.0;
      }
      newres = integrate(f, lower, upper, err_est, err_code, 10000);
    }
    // after hit bad part put upper and lower limits back one step and update step
    if(missKey1 >= 0){
      lower(0) += myStep/2.0;
    }
    if(missKey2 >= 0){
      lower(1) += myStep/2.0;
    }
    if(missKey1 <= 0){
      upper(0) -= myStep/2.0;
    }
    if(missKey2 <= 0){
      upper(1) -= myStep/2.0;
    }
    newres = integrate(f, lower, upper, err_est, err_code, 10000);
    myStep /= 10.0;
  }
  return newres;
}




// calc_expectation takes the current estimates of the parameters and the
// data as arguments. Returns 8 vectors with expectations
// Use h1 and h2 values to determine if there is mediator-mediator interaction
// Need another function later for variance calculation
//[[Rcpp::export]]
List calc_expectation_twosim(const double gamma_1, const double gamma_2,
                             const double psi_1, const double psi_2,
                             const double h1, const double h2,
                             const double rho,
                             const double sig2, const double tau2_1, const double tau2_2,
                             const arma::vec & Y, const arma::mat & Z, const arma::mat & X, const arma::vec & G,
                             const arma::vec & beta, const arma::vec & ksi_1, const arma::vec & ksi_2,
                             arma::vec S1, arma::vec R1,
                             arma::vec S2, arma::vec R2,
                             const arma::vec & LLD1, const arma::vec & ULD1,
                             const arma::vec & LLD2, const arma::vec & ULD2,
                             const int & nDivisions = 5, 
                             const int MEASURE_TYPE_KNOWN = 1,
                             const int MEASURE_TYPE_MISSING = 0,
                             const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                             const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
{

  const unsigned int N = Z.n_rows;

  // Assign all values to expectation and expectation squared at beginning.
  // Update those that are missing/low/high;
  arma::vec ES1 = S1;
  arma::vec ES1_2 = arma::square(S1);
  arma::vec ES2 = S2;
  arma::vec ES2_2 = arma::square(S2);
  arma::vec ES1ES2 = S1%S2;
  arma::vec ES12ES2 = S1%S1%S2;   // ES1 squared, ES2
  arma::vec ES1ES22 = S1%S2%S2;   // ES2 squared, ES1
  arma::vec ES12ES22 = S1%S1%S2%S2;  // Both Squares


  for ( unsigned int index = 0; index < N; ++index )
  {
    // First here are cases with below or above detection limit
    // Theory not developed, but future work should address this
    // For now, half point imputation used for all missing lower
    // And upper limit imputation for higher
    // Values then treated as observed with respect to expected values
    // Update DEC2020 - for both missing (above or below), integrate to find expected values.
    bool detectFlag = false;
    if( ( R1(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT |
        R1(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT )  &
        ( R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT |
        R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT ) ){

      double lowLS1, lowLS2, highLS1, highLS2;
      if(R1(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT){lowLS1 = 0; highLS1 = LLD1(index);}
      if(R2(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT){lowLS2 = 0; highLS2 = LLD2(index);}
      if(R1(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT){lowLS1 = ULD1(index); highLS1 = 10.0*ULD1(index);}
      if(R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT){lowLS2 = ULD2(index); highLS2 = 10.0*ULD2(index);}

      /// Find limits for integration;
      arma::mat limits = bothMissInt_unord_limits(Y(index), beta, Z.row(index), G(index), sig2,
                                                  gamma_1, psi_1, gamma_2, psi_2,
                                                  h1, h2, ksi_1, ksi_2, X.row(index),
                                                  tau2_1, tau2_2, rho,
                                                  lowLS1, lowLS2,
                                                  highLS1, highLS2, R1(index), R2(index));
      // Find denominator integral for division
      double denomIntegral = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                                               gamma_1, psi_1, gamma_2, psi_2,
                                               h1, h2, ksi_1, ksi_2, X.row(index),
                                               tau2_1, tau2_2, rho,
                                               /* s1 power, s2 power */      0, 0,
                                               limits(0, 0), limits(1, 0),
                                               limits(0, 1), limits(1, 1), nDivisions);
      // find numerator and divide by denominator
      ES1(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
          gamma_1, psi_1, gamma_2, psi_2,
          h1, h2, ksi_1, ksi_2, X.row(index),
          tau2_1, tau2_2, rho,
          /* s1 power, s2 power */      1, 0,
          limits(0, 0), limits(1, 0),
          limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;

      ES1_2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
            gamma_1, psi_1, gamma_2, psi_2,
            h1, h2, ksi_1, ksi_2, X.row(index),
            tau2_1, tau2_2, rho,
            /* s1 power, s2 power */      2, 0,
            limits(0, 0), limits(1, 0),
            limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      ES2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
          gamma_1, psi_1, gamma_2, psi_2,
          h1, h2, ksi_1, ksi_2, X.row(index),
          tau2_1, tau2_2, rho,
          /* s1 power, s2 power */      0, 1,
          limits(0, 0), limits(1, 0),
          limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      ES2_2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
            gamma_1, psi_1, gamma_2, psi_2,
            h1, h2, ksi_1, ksi_2, X.row(index),
            tau2_1, tau2_2, rho,
            /* s1 power, s2 power */      0, 2,
            limits(0, 0), limits(1, 0),
            limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      ES1ES2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
             gamma_1, psi_1, gamma_2, psi_2,
             h1, h2, ksi_1, ksi_2, X.row(index),
             tau2_1, tau2_2, rho,
             /* s1 power, s2 power */      1, 1,
             limits(0, 0), limits(1, 0),
             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      ES12ES2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
              gamma_1, psi_1, gamma_2, psi_2,
              h1, h2, ksi_1, ksi_2, X.row(index),
              tau2_1, tau2_2, rho,
              /* s1 power, s2 power */      2, 1,
              limits(0, 0), limits(1, 0),
              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      ES1ES22(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
              gamma_1, psi_1, gamma_2, psi_2,
              h1, h2, ksi_1, ksi_2, X.row(index),
              tau2_1, tau2_2, rho,
              /* s1 power, s2 power */      1, 2,
              limits(0, 0), limits(1, 0),
              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      ES12ES22(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
               gamma_1, psi_1, gamma_2, psi_2,
               h1, h2, ksi_1, ksi_2, X.row(index),
               tau2_1, tau2_2, rho,
               /* s1 power, s2 power */      2, 2,
               limits(0, 0), limits(1, 0),
               limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
    }else{
      if (R1(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
      {
        S1(index) = LLD1(index) / 2.0;

        ES1(index) = S1(index);
        ES1_2(index) = std::pow(S1(index), 2.0);
        R1(index) = MEASURE_TYPE_KNOWN;
        detectFlag = true;
      }
      if (R1(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT)
      {
        S1(index) = ULD1(index);

        ES1(index) = S1(index);
        ES1_2(index) = std::pow(S1(index), 2.0);
        R1(index) = MEASURE_TYPE_KNOWN;
        detectFlag = true;
      }

      if (R2(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
      {
        S2(index) = LLD2(index) / 2.0;

        ES2(index) = S2(index);
        ES2_2(index) = std::pow(S2(index), 2.0);
        R2(index) = MEASURE_TYPE_KNOWN;
        detectFlag = true;
      }
      if (R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT)
      {
        S2(index) = ULD2(index);

        ES2(index) = S2(index);
        ES2_2(index) = std::pow(S2(index), 2.0);
        R2(index) = MEASURE_TYPE_KNOWN;
        detectFlag = true;
      }
      if(detectFlag){
        ES1ES2(index) = S1(index)*S2(index);
        ES12ES2(index) = std::pow(S1(index), 2.0)*S2(index);
        ES1ES22(index) = S1(index)*std::pow(S2(index), 2.0);
        ES12ES22(index) = std::pow(S1(index), 2.0)*std::pow(S2(index), 2.0);
      }
    }
    /********************************************************
     ********************************************************
     * Items below here for missing values.
     ********************************************************
     ********************************************************/
    // S1 missing, S2 observed
    if(R1(index) == MEASURE_TYPE_MISSING & R2(index) == MEASURE_TYPE_KNOWN)
    {
      const double a1_val = calc_a_val_twosim(gamma_1, psi_1, G(index), S2(index), h1, h2, sig2, tau2_1, rho);
      const double b1_val = a1_val*calc_b_val_twosim(gamma_1, gamma_2, psi_1, psi_2, h1, h2,
                                                     rho, tau2_1,  tau2_2, sig2,
                                                     Y(index), G(index), Z.row(index), beta,
                                                     S2(index), X.row(index), ksi_1, ksi_2);
      ES1(index) = b1_val ;
      ES1_2(index) = std::pow(b1_val, 2.0) + a1_val ;
      ES1ES2(index) = b1_val * S2(index);
      ES12ES2(index) = ES1_2(index) * S2(index);
      ES1ES22(index) = b1_val * std::pow(S2(index), 2);
      ES12ES22(index) = ES1_2(index) * std::pow(S2(index), 2.0);
      // Rcout << "b1: " << b1_val << "\n";
    }
    else
      // S2 missing, S1 observed
      if(R1(index) == MEASURE_TYPE_KNOWN & R2(index) == MEASURE_TYPE_MISSING)
      {
        const double a2_val = calc_a_val_twosim(gamma_2, psi_2, G(index), S1(index), h1, h2, sig2, tau2_2, rho);
        const double b2_val = a2_val*calc_b_val_twosim(gamma_2, gamma_1, psi_2, psi_1, h1, h2,
                                                       rho, tau2_2,  tau2_1, sig2,
                                                       Y(index), G(index), Z.row(index), beta,
                                                       S1(index), X.row(index), ksi_2, ksi_1);
        ES2(index) = b2_val ;
        ES2_2(index) = std::pow(b2_val, 2.0) + a2_val ;
        ES1ES2(index) = b2_val * S1(index);
        ES12ES2(index) = b2_val * std::pow(S1(index), 2);
        ES1ES22(index) = ES2_2(index) * S1(index);
        ES12ES22(index) = ES2_2(index) * std::pow(S1(index), 2.0);
        // Rcout << "b2: " << b2_val << "\n";
      }
      else
        // Both missing
        if(R1(index) == MEASURE_TYPE_MISSING & R2(index) == MEASURE_TYPE_MISSING)
        {
          // can use c and d values if h1 and h2 are both zero;
          if(h1 == 0 & h2 == 0){
            const double c1_val = calc_c_val_twosim(gamma_1, gamma_2, psi_1, psi_2, G(index), tau2_1, tau2_2, sig2, rho);
            const double c2_val = calc_c_val_twosim(gamma_2, gamma_1, psi_2, psi_1, G(index), tau2_2, tau2_1, sig2, rho);

            const double b = calc_corr_b_twosim(gamma_1, gamma_2, psi_1, psi_2, G(index),
                                                tau2_1, tau2_2, c1_val, c2_val, sig2, rho);
            const double r = calc_corr_r_twosim(b);

            const double d1_val = calc_d_val_twosim(gamma_1, gamma_2, psi_1, psi_2, rho, r,
                                                    tau2_1,  tau2_2, sig2,
                                                    Y(index), G(index), Z.row(index), X.row(index),
                                                    beta, ksi_1, ksi_2, c1_val, c2_val);

            const double d2_val = calc_d_val_twosim(gamma_2, gamma_1, psi_2, psi_1, rho, r,
                                                    tau2_2,  tau2_1, sig2,
                                                    Y(index), G(index), Z.row(index), X.row(index),
                                                    beta, ksi_2, ksi_1, c2_val, c1_val);

            //Rcout << "d1: " << d1_val << ", d2: " << d2_val << ", c1: " << c1_val << ", c2: " << c2_val << "\n";
            ES1(index) = d1_val;
            ES2(index) = d2_val;
            ES1_2(index) = std::pow(d1_val, 2.0) + c1_val;
            ES2_2(index) = std::pow(d2_val, 2.0) + c2_val;
            ES1ES2(index) = std::sqrt(c1_val*c2_val)*r + d1_val*d2_val;
            ES12ES2(index) = 2.0*d1_val*std::sqrt(c1_val*c2_val)*r + d2_val*(d1_val*d1_val + c1_val);
            ES1ES22(index) = 2.0*d2_val*std::sqrt(c1_val*c2_val)*r + d1_val*(d2_val*d2_val + c2_val);
            ES12ES22(index) = c1_val*c2_val + c1_val*d2_val*d2_val + 2.0*r*r*c1_val*c2_val + 4.0*d1_val*d2_val*std::sqrt(c1_val*c2_val)*r +
              c2_val*d1_val*d1_val + d1_val*d1_val*d2_val*d2_val;

            //Rcout << ES1(index) << " " << ES1_2(index) << " " << ES2(index) << " " << ES2_2(index) << " " << ES1ES2(index) << " " <<
            //                                         ES12ES2(index) << " " << ES1ES22(index) << " " << ES12ES22(index) << "\n\n";

          }else{
            double lowLS1 = LLD1(index);
            double lowLS2 = LLD2(index);
            double highLS1 = ULD1(index);
            double highLS2 = ULD2(index);

            // Find limits of integration;
            arma::mat limits = bothMissInt_unord_limits(Y(index), beta, Z.row(index), G(index), sig2,
                                                        gamma_1, psi_1, gamma_2, psi_2,
                                                        h1, h2, ksi_1, ksi_2, X.row(index),
                                                        tau2_1, tau2_2, rho,
                                                        lowLS1, lowLS2,
                                                        highLS1, highLS2, R1(index), R2(index));
            
            // Find denominator integral for division
            double denomIntegral = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                                                     gamma_1, psi_1, gamma_2, psi_2,
                                                     h1, h2, ksi_1, ksi_2, X.row(index),
                                                     tau2_1, tau2_2, rho,
                       /* s1 power, s2 power */      0, 0,
                                                     limits(0, 0), limits(1, 0),
                                                     limits(0, 1), limits(1, 1), nDivisions);
            //if(index == 10){
            // Rcout << "Parameters: Y: " << Y(index) << "\nbeta:\n " << beta << "\nZ\n" << Z.row(index) << "\n G" << G(index) <<
            //           ", sig2: " << sig2 << ", gamma_1" << gamma_1 << ", psi1: " << psi_1 << ", gamma_2: " << gamma_2 << ", psi2: " << psi_2 <<
            //             ", h1: " << h1 << ", h2: " << h2 << "\nksi1\n" << ksi_1 << "\nksi2\n" << ksi_2 << "\nX\n" << X.row(index) <<
            //               ", tau1: " << tau2_1 << ", tau2: " << tau2_2 << ", rho: " << rho << ", powers: " <<
            //                0 << ", " << 0 << ", Lows1: " << lowLS1 << ", lows2: " << lowLS2 << ", highs1: " <<
            //                highLS1 << ", highs2: " << highLS2<< "\n";
            // Rcout << "denomIntegral = " << denomIntegral << "\n";
            // Rcout << "ES1 integral: " << bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
            //                                  gamma_1, psi_1, gamma_2, psi_2,
            //                                  h1, h2, ksi_1, ksi_2, X.row(index),
            //                                  tau2_1, tau2_2, rho,
            //                                  /* s1 power, s2 power */      1, 0,
            //                                  lowLS1, lowLS2,
            //                                  highLS1, highLS2) << "\n";
            //}

            ES1(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                gamma_1, psi_1, gamma_2, psi_2,
                h1, h2, ksi_1, ksi_2, X.row(index),
                tau2_1, tau2_2, rho,
                /* s1 power, s2 power */      1, 0,
                limits(0, 0), limits(1, 0),
                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // if(index == 10){
            // Rcout << "Patient " << index + 1 << "Expected S1: " <<  ES1(index) << "\n";
            // }
            ES1_2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                  gamma_1, psi_1, gamma_2, psi_2,
                  h1, h2, ksi_1, ksi_2, X.row(index),
                  tau2_1, tau2_2, rho,
                  /* s1 power, s2 power */      2, 0,
                  limits(0, 0), limits(1, 0),
                  limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            ES2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                gamma_1, psi_1, gamma_2, psi_2,
                h1, h2, ksi_1, ksi_2, X.row(index),
                tau2_1, tau2_2, rho,
                /* s1 power, s2 power */      0, 1,
                limits(0, 0), limits(1, 0),
                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            ES2_2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                  gamma_1, psi_1, gamma_2, psi_2,
                  h1, h2, ksi_1, ksi_2, X.row(index),
                  tau2_1, tau2_2, rho,
                  /* s1 power, s2 power */      0, 2,
                  limits(0, 0), limits(1, 0),
                  limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            ES1ES2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                   gamma_1, psi_1, gamma_2, psi_2,
                   h1, h2, ksi_1, ksi_2, X.row(index),
                   tau2_1, tau2_2, rho,
                   /* s1 power, s2 power */      1, 1,
                   limits(0, 0), limits(1, 0),
                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            ES12ES2(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                    gamma_1, psi_1, gamma_2, psi_2,
                    h1, h2, ksi_1, ksi_2, X.row(index),
                    tau2_1, tau2_2, rho,
                    /* s1 power, s2 power */      2, 1,
                    limits(0, 0), limits(1, 0),
                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            ES1ES22(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                    gamma_1, psi_1, gamma_2, psi_2,
                    h1, h2, ksi_1, ksi_2, X.row(index),
                    tau2_1, tau2_2, rho,
                    /* s1 power, s2 power */      1, 2,
                    limits(0, 0), limits(1, 0),
                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            ES12ES22(index) = bothMissInt_unord(Y(index), beta, Z.row(index), G(index), sig2,
                     gamma_1, psi_1, gamma_2, psi_2,
                     h1, h2, ksi_1, ksi_2, X.row(index),
                     tau2_1, tau2_2, rho,
                     /* s1 power, s2 power */      2, 2,
                     limits(0, 0), limits(1, 0),
                     limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
          }
        }
        //Rcout << ES1(index) << " " << ES1_2(index) << " " << ES2(index) << " " << ES2_2(index) << " " << ES1ES2(index) << " " <<
        //                              ES12ES2(index) << " " << ES1ES22(index) << " " << ES12ES22(index) << "\n\n";
  } // outer for loop

  return List::create(Named("expectS1") = ES1, Named("expectS1_sq") = ES1_2,
                      Named("expectS2") = ES2, Named("expectS2_sq") = ES2_2,
                      Named("expectS1S2") = ES1ES2,
                      Named("expectS12S2") = ES12ES2,
                      Named("expectS1S22") = ES1ES22,
                      Named("expectS12S22") = ES12ES22);
}

// Use the expected values to calculated the new mle estimates
// of the parameters of interest
//[[Rcpp::export]]
arma::vec calc_beta_gamma_inter_twosim(const arma::vec & Y, const arma::mat & Z, const arma::vec & G,
                                       const arma::vec & ES1, const arma::vec & ES1_2,
                                       const arma::vec & ES2, const arma::vec & ES2_2,
                                       const arma::vec & ES1ES2, const arma::vec & ES12ES2,
                                       const arma::vec & ES1ES22, const arma::vec & ES12ES22,
                                       bool int_gs1 = true, bool int_gs2 = true,
                                       bool int_s1s2 = false, bool int_gs1s2 = false)
{
  const unsigned int N = Z.n_rows;
  const unsigned int N_VAR = Z.n_cols;
  arma::vec beta_gamma_inter_vec = arma::zeros( N_VAR + 6 );   // gamma 1 and 2, psi 1 and 2, h1 and h2
  arma::vec Y_expect_vec = arma::zeros( N_VAR + 6 );
  arma::mat Z_expect_mat = arma::zeros( N_VAR + 6, N_VAR + 6);
  for ( unsigned int index = 0; index < N; ++index )
  {
    arma::vec current_vec = arma::zeros( N_VAR + 6 );
    current_vec.subvec(0, N_VAR - 1) = trans(Y(index)*Z.row(index));
    current_vec(N_VAR)     = Y(index)*ES1(index);
    current_vec(N_VAR + 1) = Y(index)*ES2(index);
    // Only calculate interactions as needed
    if(int_gs1){current_vec(N_VAR + 2) = Y(index)*ES1(index)*G(index);}
    if(int_gs2){current_vec(N_VAR + 3) = Y(index)*ES2(index)*G(index);}
    if(int_s1s2){current_vec(N_VAR + 4) = Y(index)*ES1ES2(index);}
    if(int_gs1s2){current_vec(N_VAR + 5) = Y(index)*ES1ES2(index)*G(index);}
    Y_expect_vec += current_vec;

    arma::mat current_mat = arma::zeros( N_VAR + 6, N_VAR + 6 );
    // beta row and column
    current_mat.submat(0, 0, N_VAR - 1, N_VAR - 1) = trans(Z.row(index)) * Z.row(index);

    // gamma_1 row and column
    current_mat.submat(0, N_VAR, N_VAR-1, N_VAR) = ES1(index)*trans(Z.row(index));
    current_mat.submat(N_VAR, 0, N_VAR, N_VAR - 1) = ES1(index)*Z.row(index);
    current_mat(N_VAR, N_VAR) = ES1_2(index);

    // gamma_2 row and column
    current_mat.submat(0, N_VAR + 1, N_VAR - 1, N_VAR + 1) = ES2(index)*trans(Z.row(index));
    current_mat.submat(N_VAR + 1, 0, N_VAR + 1, N_VAR - 1) = ES2(index)*Z.row(index);
    current_mat(N_VAR, N_VAR + 1) = ES1ES2(index);
    current_mat(N_VAR + 1, N_VAR) = ES1ES2(index);
    current_mat(N_VAR + 1, N_VAR + 1) = ES2_2(index);

    //psi_1 row and column
    if(int_gs1){
      current_mat.submat(0, N_VAR + 2, N_VAR - 1, N_VAR + 2) = ES1(index)*trans(Z.row(index))*G(index);
      current_mat.submat(N_VAR + 2, 0, N_VAR + 2, N_VAR - 1) = ES1(index)*Z.row(index)*G(index);
      current_mat(N_VAR, N_VAR + 2) = ES1_2(index)*G(index);
      current_mat(N_VAR + 2, N_VAR) = ES1_2(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 1) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 2) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 2) = ES1_2(index)*G(index)*G(index);
    }

    // psi_2 row and column
    if(int_gs2){
      current_mat.submat(0, N_VAR + 3, N_VAR - 1, N_VAR + 3) = ES2(index)*trans(Z.row(index))*G(index);
      current_mat.submat(N_VAR + 3, 0, N_VAR + 3, N_VAR - 1) = ES2(index)*Z.row(index)*G(index);
      current_mat(N_VAR, N_VAR + 3) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 3, N_VAR) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 1) = ES2_2(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 3) = ES2_2(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 2) = ES1ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 3) = ES1ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 3) = ES2_2(index)*G(index)*G(index);
    }

    // h1 row and column - es1es2
    if(int_s1s2){
      current_mat.submat(0, N_VAR + 4, N_VAR - 1, N_VAR + 4) = ES1ES2(index)*trans(Z.row(index));
      current_mat.submat(N_VAR + 4, 0, N_VAR + 4, N_VAR - 1) = ES1ES2(index)*Z.row(index);
      current_mat(N_VAR, N_VAR + 4) = ES12ES2(index);
      current_mat(N_VAR + 4, N_VAR) = ES12ES2(index);
      current_mat(N_VAR + 4, N_VAR + 1) = ES1ES22(index);
      current_mat(N_VAR + 1, N_VAR + 4) = ES1ES22(index);
      current_mat(N_VAR + 4, N_VAR + 2) = ES12ES2(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 4) = ES12ES2(index)*G(index);
      current_mat(N_VAR + 4, N_VAR + 3) = ES1ES22(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 4) = ES1ES22(index)*G(index);
      current_mat(N_VAR + 4, N_VAR + 4) = ES12ES22(index);
    }

    // h2 row and column - g*es1es2
    if(int_gs1s2){
      current_mat.submat(0, N_VAR + 5, N_VAR - 1, N_VAR + 5) = ES1ES2(index)*trans(Z.row(index))*G(index);
      current_mat.submat(N_VAR + 5, 0, N_VAR + 5, N_VAR - 1) = ES1ES2(index)*Z.row(index)*G(index);
      current_mat(N_VAR, N_VAR + 5) = ES12ES2(index)*G(index);
      current_mat(N_VAR + 5, N_VAR) = ES12ES2(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 1) = ES1ES22(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 5) = ES1ES22(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 2) = ES12ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 5) = ES12ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 3) = ES1ES22(index)*G(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 5) = ES1ES22(index)*G(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 4) = ES12ES22(index)*G(index);
      current_mat(N_VAR + 4, N_VAR + 5) = ES12ES22(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 5) = ES12ES22(index)*G(index)*G(index);
    }


    Z_expect_mat += current_mat;
  }

  if(!int_gs1){Z_expect_mat(N_VAR + 2, N_VAR + 2) = 1;}
  if(!int_gs2){Z_expect_mat(N_VAR + 3, N_VAR + 3) = 1;}
  if(!int_s1s2){Z_expect_mat(N_VAR + 4, N_VAR + 4) = 1;}
  if(!int_gs1s2){Z_expect_mat(N_VAR + 5, N_VAR + 5) = 1;}

  arma::mat Z_expect_inv;
  bool invRes = arma::inv(Z_expect_inv, Z_expect_mat);
  if(!invRes){
    //Rcout << X_expect_mat << "\n\n";
    //Rcout <<   Z_expect_mat << "\n\n";
    Rcout << "Z'Z matrix not invertable\n";
    arma::vec temp = arma::zeros(N_VAR + 2);
    temp(0) = R_NaN;
    temp(N_VAR) = R_NaN;
    temp(N_VAR + 1) = R_NaN;
    return temp;
  }
  return Z_expect_inv*Y_expect_vec;
} //calc_beta_gamma_vec

//[[Rcpp::export]]
double calc_sig2_twosim(const double & gamma_1, const double & gamma_2,
                        const double & psi_1, const double & psi_2,
                        const double & h1, const double & h2,
                        bool & int_gs1, bool & int_gs2,
                        bool & int_s1s2, bool & int_gs1s2,
                        const arma::vec & Y, const arma::vec & beta,
                        const arma::mat & Z, const arma::vec G,
                        const arma::vec & ES1, const arma::vec & ES1_2,
                        const arma::vec & ES2, const arma::vec & ES2_2,
                        const arma::vec & ES1ES2, const arma::vec & ES12ES2,
                        const arma::vec & ES1ES22, const arma::vec & ES12ES22)
{
  const unsigned int N = Z.n_rows;
  const unsigned int N_VAR = Z.n_cols;

  // + std::pow(gamma2, 2.0)*ES2_2 -
  //2.0*(Y - X*beta)%(gamma1*ES1 + gamma2*ES2) +
  //2.0*gamma1*gamma2*ES1ES2;

  return (1.0/(N - N_VAR - 2.0 - int_gs1*1 - int_gs2*1 - int_s1s2*1 - int_gs1s2*1))*accu(pow(Y - Z*beta, 2.0) +
          pow(gamma_1 + psi_1*G, 2.0)%ES1_2 + pow(gamma_2 + psi_2*G, 2.0)%ES2_2 +
          pow(h1 + h2*G, 2.0)%ES12ES22 -
          2.0*(Y - Z*beta)%( (gamma_1 + psi_1*G)%ES1 + (gamma_2 + psi_2*G)%ES2 +
          (h1 + h2*G)%ES1ES2) +
          2.0*(gamma_1 + psi_1*G)%( (gamma_2 + psi_2*G)%ES1ES2 + (h1 + h2*G)%ES12ES2) +
          2.0*(gamma_2 + psi_2*G)%(h1+h2*G)%ES1ES22);
}  //calc_sig2

//calc_ksi : for Maximization step
// Can be used for ksi1 and ksi2
//[[Rcpp::export]]
arma::vec calc_ksi_twosim( const arma::mat & X,
                           const arma::vec & ES)
{
  const unsigned int N = X.n_rows;
  const unsigned int N_VAR = X.n_cols;
  arma::vec expect_X_vec = arma::zeros( N_VAR );
  arma::mat expect_X_mat = arma::zeros( N_VAR, N_VAR );
  for ( unsigned int index = 0; index < N; ++index)
  {
    expect_X_mat += trans(X.row(index)) * X.row(index);
    expect_X_vec += trans(X.row(index)) * ES(index);
  } //for

  arma::mat expect_X_inv;
  bool invRes = arma::inv(expect_X_inv, expect_X_mat);
  if(!invRes){
    Rcout << "X'X matrix not invertable\n";
    arma::vec ksi = arma::zeros(N_VAR);
    ksi(0) = R_NaN;
    return ksi;
  }
  return(expect_X_inv*expect_X_vec);
} //calc_ksi


//calc_tau_sqr : for Maximization step
//[[Rcpp::export]]
double calc_tau2_twosim( const arma::vec & ksi, const arma::mat & X,
                         const arma::vec ES,
                         const arma::vec ES2)
{
  const unsigned int N = X.n_rows;
  const unsigned int N_VAR = X.n_cols;
  return (1.0/(N - N_VAR))*accu(ES2 + pow(X*ksi, 2.0) - 2.0*(X*ksi)%ES);
} //calc_tau_sqr

//calc_rho : for Maximization step
//[[Rcpp::export]]
double calc_rho_twosim( const arma::vec & ksi_1, const arma::vec & ksi_2,
                        const arma::mat & X, const double tau2_1, const double tau2_2,
                        const arma::vec & ES1, const arma::vec & ES2, const arma::vec & ES1ES2)
{
  const unsigned int N = X.n_rows;
  const unsigned int N_VAR = X.n_cols;
  return (1.0/( (N - N_VAR) * std::sqrt(tau2_1*tau2_2))) *
    accu(ES1ES2 - ES1%(X*ksi_2) - ES2%(X*ksi_1) + (X*ksi_1)%(X*ksi_2));
} //calc_tau_sqr

// Function for the initial value of rho,
// Fits model and finds the correlation of the
// residuals
// [[Rcpp::export]]
double rhoInit(const arma::vec & y1, const arma::vec & y2, const arma::mat & X) {

  arma::colvec coef1 = arma::solve(X, y1);
  arma::colvec coef2 = arma::solve(X, y2);
  arma::colvec resid1 = y1 - X*coef1;
  arma::colvec resid2 = y2 - X*coef2;

  if(!arma::any(coef1) | !arma::any(coef2)){return 0;}

  return as_scalar(arma::cor(resid1, resid2));
}


// EM algorithm to iterate between Expectation and Maximization
//[[Rcpp::export]]
List twoSimMed_EM(const arma::vec & Y, const arma::vec & G, const arma::vec & S1, const arma::vec & R1,
                  const arma::vec & S2, const arma::vec & R2,
                  const arma::mat & Z, const arma::mat & X,
                  const arma::vec & lowerS1, const arma::vec & upperS1,
                  const arma::vec & lowerS2, const arma::vec & upperS2,
                  bool int_gs1 = true, bool int_gs2 = true,
                  bool int_s1s2 = false, bool int_gs1s2 = false,
                  const double convLimit = 1e-4, const double iterationLimit = 1e4,
                  const int & nDivisions = 5, 
                  const int MEASURE_TYPE_KNOWN = 1,
                  const int MEASURE_TYPE_MISSING = 0,
                  const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                  const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  // Initialize needed values;
  const unsigned int N = Y.n_elem;

  arma::mat Zuse = arma::zeros(N, Z.n_cols + 1);
  arma::mat Xuse = arma::zeros(N, X.n_cols + 1);

  // Update Z and X matrices to add in
  if(Z.n_elem < N){
    Zuse = arma::ones(N, 2);
    Zuse.submat(0, 1, N-1, 1) = G;
  }else{
    Zuse = arma::zeros(N, Z.n_cols + 1);
    //Rcout << Zuse << "\n1\n\n";
    Zuse.submat(0, 0, N-1, 0) = Z.submat(0, 0, N-1, 0);//Rcout << Zuse << "\n2\n\n";
    Zuse.submat(0, 1, N-1, 1) = G;//Rcout << Zuse << "\n3\n\n";
    Zuse.submat(0, 2, N-1, Zuse.n_cols - 1) = Z.submat(0, 1, N-1, Z.n_cols - 1);//Rcout << Zuse << "\n4\n\n";
  }
  if(X.n_elem < N){
    Xuse = arma::ones(N, 2);
    Xuse.submat(0, 1, N-1, 1) = G;
  }else{
    Xuse = arma::zeros(N, X.n_cols + 1);
    Xuse.submat(0, 0, N-1, 0) = X.submat(0, 0, N-1, 0);
    Xuse.submat(0, 1, N-1, 1) = G;
    Xuse.submat(0, 2, N-1, Xuse.n_cols - 1) = X.submat(0, 1, N-1, X.n_cols - 1);
  }



  arma::vec beta = arma::zeros(Zuse.n_cols);
  double gamma_1 = 0;
  double gamma_2 = 0;
  double psi_1 = 0;
  double psi_2 = 0;
  double h1 = 0;
  double h2 = 0;
  arma::vec ksi_1 = arma::zeros(Xuse.n_cols);
  arma::vec ksi_2 = arma::zeros(Xuse.n_cols);
  double sig2 = arma::var(Y);
  double tau2_1 = arma::var(S1.elem(find(R1 == 1)));
  double tau2_2 = arma::var(S2.elem(find(R2 == 1)));

  //Rcout << "tau2_2 check: " << S2.elem(find(R2 == 1)) << "\n\n";

  double rho = rhoInit(S2.elem(find((R2 == 1) % (R1 == 1))),
                       S1.elem(find((R2 == 1) % (R1 == 1))),
                       Xuse.rows(find((R2 == 1) % (R1 == 1))));

  bool converged = false;
  int iteration = 0;
  int rlow = 1, rhigh = 1;   // Counter for times rho has been too high or too low;
  double sigC = 2.0, tau1C = 2.0, tau2C = 2.0; // counter for times variance term has diverged;

  while(!converged & (iteration < iterationLimit)){
    /* Code to check if rho or variance terms are diverging
     * If diverging, reset all values and set divergent to specific value
     */
    if(rho > .99){
      beta = arma::zeros(Zuse.n_cols); gamma_1 = 0; gamma_2 = 0; ksi_1 = arma::zeros(Xuse.n_cols); ksi_2 = arma::zeros(Xuse.n_cols);
      sig2 = arma::var(Y); tau2_1 = arma::var(S1.elem(find(R1 == 1))); tau2_2 = arma::var(S2.elem(find(R2 == 1)));

      rho = -1.0 + (1.0/rlow);
      ++rlow;
      Rcout << "rho too large\n";
    }
    // rho approaching -1, restart with rho = .5;
    else if(rho < -.99){
      beta = arma::zeros(Zuse.n_cols); gamma_1 = 0; gamma_2 = 0; ksi_1 = arma::zeros(Xuse.n_cols); ksi_2 = arma::zeros(Xuse.n_cols);
      sig2 = arma::var(Y); tau2_1 = arma::var(S1.elem(find(R1 == 1))); tau2_2 = arma::var(S2.elem(find(R2 == 1)));

      rho = 1.0 - (1.0/rhigh);
      ++rhigh;
      Rcout << "rho too negative\n";
    }
    // Variance Components getting very very large (1e20);
    // sigma squared
    else if(sig2 > 1e10){
      beta = arma::zeros(Zuse.n_cols); gamma_1 = 0; gamma_2 = 0; ksi_1 = arma::zeros(Xuse.n_cols); ksi_2 = arma::zeros(Xuse.n_cols);
      tau2_1 = arma::var(S1.elem(find(R1 == 1))); tau2_2 = arma::var(S2.elem(find(R2 == 1)));
      rho = rhoInit(S2.elem(find((R2 == 1) % (R1 == 1))),
                    S1.elem(find((R2 == 1) % (R1 == 1))),
                    Xuse.rows(find((R2 == 1) % (R1 == 1))));

      sig2 = arma::var(Y) / sigC;

      ++sigC;
      Rcout << "sig too large\n";
    }
    // tau_1 squared
    else if(tau2_1> 1e10){
      beta = arma::zeros(Zuse.n_cols); gamma_1 = 0; gamma_2 = 0; ksi_1 = arma::zeros(Xuse.n_cols); ksi_2 = arma::zeros(Xuse.n_cols);
      sig2 = arma::var(Y); 
      rho = rhoInit(S2.elem(find((R2 == 1) % (R1 == 1))),
                    S1.elem(find((R2 == 1) % (R1 == 1))),
                    Xuse.rows(find((R2 == 1) % (R1 == 1))));
      tau2_2 = arma::var(S2.elem(find(R2 == 1)));
      
      tau2_1 = arma::var(S1.elem(find(R1 == 1))) / tau1C;
      ++tau1C;
      Rcout << "tau1 too large\n";
    }
    // tau_2 squared
    else if(tau2_2 > 1e10){
      beta = arma::zeros(Zuse.n_cols); gamma_1 = 0; gamma_2 = 0; ksi_1 = arma::zeros(Xuse.n_cols); ksi_2 = arma::zeros(Xuse.n_cols);
      sig2 = arma::var(Y); 
      rho = rhoInit(S2.elem(find((R2 == 1) % (R1 == 1))),
                    S1.elem(find((R2 == 1) % (R1 == 1))),
                    Xuse.rows(find((R2 == 1) % (R1 == 1))));

      tau2_1 = arma::var(S1.elem(find(R1 == 1))); 
      tau2_2 = arma::var(S2.elem(find(R2 == 1))) / tau2C;
      ++tau2C;
      Rcout << "tau2 too large\n";
    }
    // Create old holders;
    arma::vec oldBeta = beta;
    double oldGamma1 = gamma_1;
    double oldGamma2 = gamma_2;
    double oldPsi1 = psi_1;
    double oldPsi2 = psi_2;
    double oldH1 = h1;
    double oldH2 = h2;
    double oldSig2 = sig2;
    arma::vec oldKsi1 = ksi_1;
    arma::vec oldKsi2 = ksi_2;
    double oldTau2_1 = tau2_1;
    double oldTau2_2 = tau2_2;
    double oldRho = rho;

    //Update Expecation;
    List expRes = calc_expectation_twosim(gamma_1, gamma_2, psi_1, psi_2, h1, h2, rho,
                                          sig2, tau2_1, tau2_2,
                                          Y, Zuse, Xuse, G,
                                          beta, ksi_1, ksi_2,
                                          S1, R1,
                                          S2, R2,
                                          lowerS1, upperS1,
                                          lowerS2, upperS2,
                                          nDivisions,
                                          MEASURE_TYPE_KNOWN,
                                          MEASURE_TYPE_MISSING,
                                          MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                          MEASURE_TYPE_ABOVE_DETECTION_LIMIT);
    arma::vec expectS1 = expRes["expectS1"];
    arma::vec expectS1_sq = expRes["expectS1_sq"];
    arma::vec expectS2 = expRes["expectS2"];
    arma::vec expectS2_sq = expRes["expectS2_sq"];
    arma::vec expectS1S2 = expRes["expectS1S2"];
    arma::vec expectS12S2 = expRes["expectS12S2"];
    arma::vec expectS1S22 = expRes["expectS1S22"];
    arma::vec expectS12S22 = expRes["expectS12S22"];

    //Update Estimates;
    arma::vec beta_gamma_int = calc_beta_gamma_inter_twosim(Y, Zuse, G,
                                                            expectS1, expectS1_sq, expectS2, expectS2_sq, expectS1S2,
                                                            expectS12S2, expectS1S22, expectS12S22, int_gs1, int_gs2, int_s1s2, int_gs1s2);
    //Rcout << "beta vec: " << beta_gamma_int << "\n\n";
    beta = beta_gamma_int.subvec(0, Zuse.n_cols - 1);
    gamma_1 = beta_gamma_int(Zuse.n_cols);
    gamma_2 = beta_gamma_int(Zuse.n_cols + 1);
    psi_1   = beta_gamma_int(Zuse.n_cols + 2);
    psi_2   = beta_gamma_int(Zuse.n_cols + 3);
    h1      = beta_gamma_int(Zuse.n_cols + 4);
    h2      = beta_gamma_int(Zuse.n_cols + 5);

    //Rcout << "psi1: " << psi_1 << ", psi2: " << psi_2 << ", h1: " << h1 << ", h2: " << h2 << "\n";

    sig2 = calc_sig2_twosim(gamma_1, gamma_2, psi_1, psi_2, h1, h2, 
                            int_gs1, int_gs2, int_s1s2, int_gs1s2,
                            Y, beta, Zuse, G,
                            expectS1, expectS1_sq, expectS2, expectS2_sq,
                            expectS1S2, expectS12S2, expectS1S22, expectS12S22);

    ksi_1 = calc_ksi_twosim(Xuse, expectS1);
    ksi_2 = calc_ksi_twosim(Xuse, expectS2);
    
    tau2_1 = calc_tau2_twosim(ksi_1, Xuse, expectS1, expectS1_sq);
    tau2_2 = calc_tau2_twosim(ksi_2, Xuse, expectS2, expectS2_sq);

    rho = calc_rho_twosim(ksi_1, ksi_2, Xuse, tau2_1, tau2_2, expectS1, expectS2, expectS1S2);

    //if(iteration == 1){Rcout << expectS2 << "\n\n";}
    // Check for errors
    if(std::isnan(beta(0)) | std::isnan(ksi_1(0)) | std::isnan(ksi_2(0))){
      Rcout << "Values not calculable\n";
      gamma_1 = R_NaN;
      gamma_2 = R_NaN;
      psi_1 = R_NaN;
      psi_2 = R_NaN;
      h1 = R_NaN;
      h2 = R_NaN;
      sig2 = R_NaN;
      tau2_1 = R_NaN;
      tau2_2 = R_NaN;
      rho = R_NaN;
      return List::create(Named("beta") = beta, Named("gamma_1") = gamma_1, Named("gamma_2") = gamma_2,
                                Named("psi_1") = psi_1, Named("psi_2") = psi_2,
                                Named("h1") = h1, Named("h2") = h2,
                                Named("rho") = rho, Named("sig2") = sig2,
                                Named("ksi_1") = ksi_1, Named("ksi_2") = ksi_2,
                                Named("tau2_1") = tau2_1, Named("tau2_2") = tau2_2);
    }

    // Check for convergence;
    double maxDiff = -1;
    for(int betaIndex = 0; betaIndex < Zuse.n_cols; ++betaIndex){
      maxDiff = std::max(maxDiff,
                         std::max(fabs(beta(betaIndex) - oldBeta(betaIndex)),
                                  fabs(beta(betaIndex) - oldBeta(betaIndex)) / fabs(oldBeta(betaIndex))));
    }
    for(int ksiIndex = 0; ksiIndex < Z.n_cols; ++ksiIndex){
      maxDiff = std::max(maxDiff,
                         std::max(fabs(ksi_1(ksiIndex) - oldKsi1(ksiIndex)),
                                  fabs(ksi_1(ksiIndex) - oldKsi1(ksiIndex)) / fabs(oldKsi1(ksiIndex))));
      maxDiff = std::max(maxDiff,
                         std::max(fabs(ksi_2(ksiIndex) - oldKsi2(ksiIndex)),
                                  fabs(ksi_2(ksiIndex) - oldKsi2(ksiIndex)) / fabs(oldKsi2(ksiIndex))));
    }
    maxDiff = std::max(maxDiff, std::max(fabs(sig2 - oldSig2), fabs(sig2 - oldSig2) / fabs(oldSig2)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma_1 - oldGamma1), fabs(gamma_1 - oldGamma1) / fabs(oldGamma1)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma_2 - oldGamma2), fabs(gamma_2 - oldGamma2) / fabs(oldGamma2)));
    if(int_gs1){maxDiff = std::max(maxDiff, std::max(fabs(psi_1 - oldPsi1), fabs(psi_1 - oldPsi1) / fabs(oldPsi1)));}
    if(int_gs2){maxDiff = std::max(maxDiff, std::max(fabs(psi_2 - oldPsi2), fabs(psi_2 - oldPsi2) / fabs(oldPsi2)));}
    if(int_s1s2){maxDiff = std::max(maxDiff, std::max(fabs(h1 - oldH1), fabs(h1 - oldH1) / fabs(oldH1)));}
    if(int_gs1s2){maxDiff = std::max(maxDiff, std::max(fabs(h2 - oldH2), fabs(h2 - oldH2) / fabs(oldH2)));}
    maxDiff = std::max(maxDiff, std::max(fabs(tau2_1 - oldTau2_1), fabs(tau2_1 - oldTau2_1) / fabs(oldTau2_1)));
    maxDiff = std::max(maxDiff, std::max(fabs(tau2_2 - oldTau2_2), fabs(tau2_2 - oldTau2_2) / fabs(oldTau2_2)));
    maxDiff = std::max(maxDiff, std::max(fabs(rho - oldRho), fabs(rho - oldRho) / fabs(oldRho)));

    converged = maxDiff < convLimit;
    iteration++;
  }
  if(iteration == iterationLimit){Rcout << "Algorithm failed to converge\n";}
  return List::create(Named("beta") = beta, Named("gamma_1") = gamma_1, Named("gamma_2") = gamma_2,
                            Named("psi_1") = psi_1, Named("psi_2") = psi_2,
                            Named("h1") = h1, Named("h2") = h2,
                            Named("rho") = rho, Named("sig2") = sig2, Named("ksi_1") = ksi_1,
                                  Named("ksi_2") = ksi_2, Named("tau2_1") = tau2_1, Named("tau2_2") = tau2_2);
}

//[[Rcpp::export]]
arma::mat calc_Q_matrix_twosim(const double & sig2, const double & tau2_1, const double & tau2_2, const double & rho,
                               const arma::mat & Z, const arma::mat & X, const arma::vec & G,
                               const arma::vec & ES1, const arma::vec & ES1_2,
                               const arma::vec & ES2, const arma::vec & ES2_2,
                               const arma::vec & ES1S2, const arma::vec & ES12S2,
                               const arma::vec & ES1S22, const arma::vec & ES12S22,
                               bool int_gs1 = true, bool int_gs2 = true,
                               bool int_s1s2 = false, bool int_gs1s2 = false)
{
  const unsigned int X_VARS = X.n_cols;
  const unsigned int Z_VARS = Z.n_cols;
  const unsigned int N = Z.n_rows;

  // Z variables, gamma1 and gamma2, psi1, psi2, h1, h2, sigma2, X variables (twice) tau2_2, and tau2_1, rho;

  arma::mat Q = arma::zeros( Z_VARS + 7 + 2*X_VARS + 3, Z_VARS + 7 + 2*X_VARS + 3 );

  // Beta, gamma1, gamma2, psi1, psi2,  h1,   h2,   sig2, ksi2, ksi1, tau2, tau1, rho
  //   Z,    S1,     S2,    GS1   GS2  S1S2  GS1S2   1     X     X     1     1     p
  // Loop for summation elements
  for ( unsigned int index = 0; index < N; ++index )
  {
    // Beta - Z
    Q.submat(0,           0,           Z_VARS - 1, Z_VARS - 1) += (trans(Z.row(index)) * Z.row(index));

    // gamma_1 (beta, gamma1) - Z_VARS - S1 variable
    Q.submat(0,           Z_VARS,      Z_VARS - 1, Z_VARS    ) += (ES1(index)*trans(Z.row(index))) ;
    Q.submat(Z_VARS,      0,           Z_VARS,     Z_VARS - 1) += (ES1(index)*Z.row(index)) ;
    Q(Z_VARS,     Z_VARS) += ES1_2(index);

    // gamma_2 (beta, gamma1, gamma2) - Z_VARS + 1 - S2 variable
    Q.submat(0,           Z_VARS + 1,  Z_VARS - 1, Z_VARS + 1) += (ES2(index)*trans(Z.row(index))) ;
    Q.submat(Z_VARS + 1,  0,           Z_VARS + 1, Z_VARS - 1) += (ES2(index)*Z.row(index));
    Q(Z_VARS,     Z_VARS + 1) += ES1S2(index) ;
    Q(Z_VARS + 1, Z_VARS    ) += ES1S2(index) ;
    Q(Z_VARS + 1, Z_VARS + 1) += ES2_2(index) ;

    if(int_gs1){
      // psi_1 (beta, gamma1, gamma2, psi1) - Z_VARS + 2 - S1 * G variable
      Q.submat(0,           Z_VARS + 2,  Z_VARS - 1, Z_VARS + 2) += (ES1(index)*trans(Z.row(index))*G(index)) ;
      Q.submat(Z_VARS + 2,  0,           Z_VARS + 2, Z_VARS - 1) += (ES1(index)*Z.row(index)*G(index));
      Q(Z_VARS,     Z_VARS + 2) += ES1_2(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS    ) += ES1_2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS + 2) += ES1S2(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 1) += ES1S2(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 2) += ES1_2(index)*G(index)*G(index) ;
    }

    if(int_gs2){
      // psi_2 (beta, gamma1, gamma2, psi1, psi2) - Z_VARS + 3 - S2 * G variable
      Q.submat(0,           Z_VARS + 3,  Z_VARS - 1, Z_VARS + 3) += (ES2(index)*trans(Z.row(index))*G(index)) ;
      Q.submat(Z_VARS + 3,  0,           Z_VARS + 3, Z_VARS - 1) += (ES2(index)*Z.row(index)*G(index));
      Q(Z_VARS,     Z_VARS + 3) += ES1S2(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS    ) += ES1S2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS + 3) += ES2_2(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 1) += ES2_2(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 3) += ES1S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 2) += ES1S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 3) += ES2_2(index)*G(index)*G(index) ;
    }

    if(int_s1s2){
      // h1 (beta, gamma1, gamma2, psi1, psi2, h1) - Z_VARS + 4 - S1*S2  variable
      Q.submat(0,           Z_VARS + 4,  Z_VARS - 1, Z_VARS + 4) += (ES1S2(index)*trans(Z.row(index))) ;
      Q.submat(Z_VARS + 4,  0,           Z_VARS + 4, Z_VARS - 1) += (ES1S2(index)*Z.row(index));
      Q(Z_VARS,     Z_VARS + 4) += ES12S2(index) ;
      Q(Z_VARS + 4, Z_VARS    ) += ES12S2(index) ;
      Q(Z_VARS + 1, Z_VARS + 4) += ES1S22(index) ;
      Q(Z_VARS + 4, Z_VARS + 1) += ES1S22(index) ;
      Q(Z_VARS + 2, Z_VARS + 4) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 2) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 4) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 3) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 4) += ES12S22(index) ;
    }

    if(int_gs1s2){
      // h2 (beta, gamma1, gamma2, psi1, psi2, h1, h2) - Z_VARS + 5 - S1*S2  variable
      Q.submat(0,           Z_VARS + 5,  Z_VARS - 1, Z_VARS + 5) += (ES1S2(index)*trans(Z.row(index))*G(index)) ;
      Q.submat(Z_VARS + 5,  0,           Z_VARS + 5, Z_VARS - 1) += (ES1S2(index)*Z.row(index)*G(index));
      Q(Z_VARS,     Z_VARS + 5) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS    ) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS + 5) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 1) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 5) += ES12S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 2) += ES12S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 5) += ES1S22(index)*G(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 3) += ES1S22(index)*G(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 5) += ES12S22(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 4) += ES12S22(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 5) += ES12S22(index)*G(index)*G(index) ;
    }

    // ksi2 variables - Z_VARS + 7 for start of ksi2, Z_VARS + X_VARS + 7 for start of ksi2
    // ksi2  - Z_VARS + 7 and X_VARS long
    Q.submat(Z_VARS + 7, Z_VARS + 7, Z_VARS + 6 + X_VARS, Z_VARS + 6 + X_VARS) += (trans(X.row(index)) * X.row(index));

    // ksi1 -
    Q.submat(Z_VARS + 7,          Z_VARS + 7 + X_VARS,  Z_VARS + 6 + X_VARS, Z_VARS + 6 + 2*X_VARS) += (-rho*trans(X.row(index)) * X.row(index));
    Q.submat(Z_VARS + 7 + X_VARS, Z_VARS + 7,           Z_VARS + 6 + 2*X_VARS,  Z_VARS + 6 + X_VARS) += (-rho*trans(X.row(index)) * X.row(index));
    Q.submat(Z_VARS + 7 + X_VARS, Z_VARS + 7 + X_VARS,  Z_VARS + 6 + 2*X_VARS, Z_VARS + 6 + 2*X_VARS) += (trans(X.row(index)) * X.row(index));
  }

  //beta(ish)
  Q.submat(0, 0, Z_VARS + 5, Z_VARS + 5) /= sig2;

  // ksi2
  Q.submat(Z_VARS + 7, Z_VARS + 7, Z_VARS + 6 + X_VARS, Z_VARS + 6 + X_VARS) /= ((1 - rho*rho)*tau2_2);

  //ksi1xksi2
  Q.submat(Z_VARS + 7,          Z_VARS + 7 + X_VARS,  Z_VARS + 6 + X_VARS, Z_VARS + 6 + 2*X_VARS) /= ((1 - rho*rho) * std::sqrt(tau2_1*tau2_2));
  Q.submat(Z_VARS + 7 + X_VARS, Z_VARS + 7,           Z_VARS + 6 + 2*X_VARS,  Z_VARS + 6 + X_VARS) /= ((1 - rho*rho) * std::sqrt(tau2_1*tau2_2));

  //ksi1
  Q.submat(Z_VARS + 7 + X_VARS, Z_VARS + 7 + X_VARS,  Z_VARS + 6 + 2*X_VARS, Z_VARS + 6 + 2*X_VARS) /= ((1 - rho*rho)*tau2_1);


  //sig2
  Q(Z_VARS + 6, Z_VARS + 6) = N / (2.0*std::pow(sig2, 2.0));

  // tau2 and rho -
  // Diagonals;
  Q(Z_VARS + 6 + 2*X_VARS + 1, Z_VARS + 6 + 2*X_VARS + 1) = (N / (4.0 * tau2_2*tau2_2)) * ((2 - rho*rho) / (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 2, Z_VARS + 6 + 2*X_VARS + 2) = (N / (4.0 * tau2_1*tau2_1)) * ((2 - rho*rho) / (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 3, Z_VARS + 6 + 2*X_VARS + 3) = (N * (1 + rho*rho)) / std::pow(1 - rho*rho, 2.0);

  //Off Diag;
  Q(Z_VARS + 6 + 2*X_VARS + 1, Z_VARS + 6 + 2*X_VARS + 2) = -(N*rho*rho) / (4.0* tau2_1 * tau2_2 * (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 2, Z_VARS + 6 + 2*X_VARS + 1) = -(N*rho*rho) / (4.0* tau2_1 * tau2_2 * (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 1, Z_VARS + 6 + 2*X_VARS + 3) = -(N*rho) / (2.0* tau2_2 * (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 3, Z_VARS + 6 + 2*X_VARS + 1) = -(N*rho) / (2.0* tau2_2 * (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 2, Z_VARS + 6 + 2*X_VARS + 3) = -(N*rho) / (2.0* tau2_1 * (1 - rho*rho));
  Q(Z_VARS + 6 + 2*X_VARS + 3, Z_VARS + 6 + 2*X_VARS + 2) = -(N*rho) / (2.0* tau2_1 * (1 - rho*rho));

  return Q;
}

// Calc V mat
// calc U mat
// calc_V_mat returns V matrix for a single observation
// need 3 functions for each missing S scenario
// Name denotes value missing (S1, S2, S1S2 for both)
//[[Rcpp::export]]
arma::mat calc_V_mat_twosim_S1(const double & gamma_1, const double & gamma_2,
                               const double & psi_1, const double & psi_2,
                               const double & h1, const double & h2, const double & sig2,
                               const double & tau2_1, const double & tau2_2, const double & rho,
                               const double & Y, const double & S_obs, const double & G,
                               const arma::vec & beta,
                               const arma::vec & ksi_1, const arma::vec & ksi_2,
                               const arma::rowvec & Z,
                               const arma::rowvec & X)
{
  const unsigned int X_VARS = X.n_cols;
  const unsigned int Z_VARS = Z.n_cols;
  const double Y_residual = Y - as_scalar(Z*beta) - (gamma_2 + psi_2*G)*S_obs;
  const double s1InterTerms = gamma_1 + G*psi_1 + S_obs*(h1 + G*h2);
  const double S_residual = S_obs - as_scalar(X*ksi_2);
  const double ksi1_X = as_scalar(X*ksi_1);

  arma::mat V_mat = arma::zeros(Z_VARS + 2*X_VARS + 10, 3); // 10 is for 2 gammas, 2 psis, h1, h2, sig2, tau2, tau1, rho

  // Change and do by row (variable) as some rows are multiples of previous rows;
  // Beta rows
  V_mat.submat(0, 0, Z_VARS-1, 0) = (Y_residual ) * trans(Z) / sig2;
  V_mat.submat(0, 1, Z_VARS-1, 1) =  -s1InterTerms*trans(Z) / sig2;
  V_mat.submat(0, 2, Z_VARS-1, 2) =  arma::zeros(Z_VARS);

  // gamma1 rows
  V_mat(Z_VARS, 0) = 0;
  V_mat(Z_VARS, 1) = Y_residual / sig2;
  V_mat(Z_VARS, 2) = -s1InterTerms / sig2;

  //gamma2 rows
  V_mat(Z_VARS+1, 0) = Y_residual * S_obs / sig2;
  V_mat(Z_VARS+1, 1) = -s1InterTerms*S_obs / sig2;
  V_mat(Z_VARS+1, 2) = 0;

  //psi1 - S1 * G
  V_mat.submat(Z_VARS+2, 0, Z_VARS+2, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * G;
  //psi2 - S2*G
  V_mat.submat(Z_VARS+3, 0, Z_VARS+3, 2) = V_mat.submat(Z_VARS+1, 0, Z_VARS+1, 2) * G;
  //h1 - S1*S2
  V_mat.submat(Z_VARS+4, 0, Z_VARS+4, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * S_obs;
  //h2 - S1*S2*G
  V_mat.submat(Z_VARS+5, 0, Z_VARS+5, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * S_obs * G;

  //sig2 rows
  V_mat(Z_VARS+6, 0) = -1 / (2.0*sig2) +  std::pow(Y_residual, 2.0)/(2.0*sig2*sig2);
  V_mat(Z_VARS+6, 1) = -s1InterTerms * Y_residual / (sig2*sig2);
  V_mat(Z_VARS+6, 2) = std::pow(s1InterTerms, 2.0) / (2.0* sig2*sig2);

  // ksi2 rows
  V_mat.submat(Z_VARS + 7, 0, Z_VARS + 6 + X_VARS, 0) = S_residual * trans(X) / (tau2_2*(1- rho*rho)) +
                                                        rho*ksi1_X * trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7, 1, Z_VARS + 6 + X_VARS, 1) = -rho*trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7, 2, Z_VARS + 6 + X_VARS, 2) = arma::zeros(X_VARS);

  // ksi1 rows
  V_mat.submat(Z_VARS + 7 + X_VARS, 0, Z_VARS + 6 + 2*X_VARS, 0) = -rho*S_residual * trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
                                                                      ksi1_X * trans(X) / (tau2_1*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7 + X_VARS, 1, Z_VARS + 6 + 2*X_VARS, 1) = trans(X) / (tau2_1*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7 + X_VARS, 2, Z_VARS + 6 + 2*X_VARS, 2) = arma::zeros(X_VARS);

  // tau2 rows
  V_mat(Z_VARS + 7 + 2*X_VARS, 0) =                           -1 / (2.0*tau2_2) +
                                          std::pow(S_residual, 2.0) / (2.0*tau2_2*tau2_2*(1 - rho*rho)) +
                                          rho*S_residual*ksi1_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS, 1) = -rho*S_residual / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS, 2) = 0 ;

  // tau1 rows
  V_mat(Z_VARS + 8 + 2*X_VARS, 0) =                           -1 / (2.0*tau2_1) +
                                        std::pow(ksi1_X, 2.0) / (2.0*tau2_1*tau2_1*(1 - rho*rho)) +
                                        rho*S_residual*ksi1_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho));
  V_mat(Z_VARS + 8 + 2*X_VARS, 1) =                 -ksi1_X / (tau2_1*tau2_1*(1 - rho*rho)) -
                                            rho*S_residual / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho));
  V_mat(Z_VARS + 8 + 2*X_VARS, 2) = 1 / (2.0*tau2_1*tau2_1*(1 - rho*rho));

  // rho rows
  V_mat(Z_VARS + 9 + 2*X_VARS, 0) =     rho / (1 - rho*rho) -
                        S_residual*ksi1_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
                        rho/(std::pow(1 - rho*rho, 2.0)) * (
                              S_residual*S_residual / (tau2_2) -
                              ksi1_X*ksi1_X / (tau2_1) -
                              2.0*rho*S_residual*ksi1_X / (std::sqrt(tau2_1*tau2_2))
                              );                     
    // rho / (1 - rho*rho) -
    // S_residual*ksi1_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
    // rho*S_residual*S_residual / (tau2_2*std::pow(1 - rho*rho, 2.0)) -
    // rho*ksi1_X*ksi1_X / (tau2_1*std::pow(1 - rho*rho, 2.0)) -
    // 2.0*rho*rho*S_residual*ksi1_X / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0));
    
  V_mat(Z_VARS + 9 + 2*X_VARS, 1) =     S_residual/( (1 - rho*rho) * std::sqrt(tau2_1*tau2_2) ) + 
                                  rho/std::pow(1 - rho*rho, 2.0)*(
                                            2.0*ksi1_X / tau2_1 + 
                                    2.0*rho*S_residual / std::sqrt(tau2_1*tau2_2)
                                  );
    
    // S_residual / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) +
    // 2.0*rho*ksi1_X / (tau2_1*std::pow(1 - rho*rho, 2.0)) +
    // 2.0*rho*rho*S_residual / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0));
  V_mat(Z_VARS + 9 + 2*X_VARS, 2) = -rho / (tau2_1*std::pow(1 - rho*rho, 2.0));

  return V_mat;
} //calc_V_mat_twosim_V1


//[[Rcpp::export]]
arma::mat calc_V_mat_twosim_S2(const double & gamma_1, const double & gamma_2,
                               const double & psi_1, const double & psi_2,
                               const double & h1, const double & h2, const double & sig2,
                               const double & tau2_1, const double & tau2_2, const double & rho,
                               const double & Y, const double & S_obs, const double & G,
                               const arma::vec & beta,
                               const arma::vec & ksi_1, const arma::vec & ksi_2,
                               const arma::rowvec & Z,
                               const arma::rowvec & X)
{
  const unsigned int X_VARS = X.n_cols;
  const unsigned int Z_VARS = Z.n_cols;
  const double Y_residual = Y - as_scalar(X*beta) - (gamma_1 + psi_1*G)*S_obs;
  const double s2InterTerms = gamma_2 + G*psi_2 + S_obs*(h1 + G*h2);
  const double S_residual = S_obs - as_scalar(X*ksi_1);
  const double ksi2_X = as_scalar(X*ksi_2);

  arma::mat V_mat = arma::zeros(Z_VARS + 2*X_VARS + 10, 3); // 10 is for 2 gammas, 2 psis, h1, h2, sig2, tau2, tau1, rho

  // Change and do by row (variable) as some rows are multiples of previous rows;
  // Beta rows
  V_mat.submat(0, 0, Z_VARS-1, 0) = (Y_residual ) * trans(Z) / sig2; //*
  V_mat.submat(0, 1, Z_VARS-1, 1) =  -s2InterTerms*trans(Z) / sig2; //*
  V_mat.submat(0, 2, X_VARS-1, 2) =  arma::zeros(Z_VARS); //*

  // gamma1 rows
  V_mat(Z_VARS, 0) = Y_residual * S_obs / sig2; // *
  V_mat(Z_VARS, 1) = -s2InterTerms*S_obs / sig2; //*
  V_mat(Z_VARS, 2) = 0; //*

  //gamma2 rows
  V_mat(Z_VARS+1, 0) = 0; //*
  V_mat(Z_VARS+1, 1) = Y_residual / sig2; //*
  V_mat(Z_VARS+1, 2) = -s2InterTerms / sig2; //*

  //psi1 - S1*G
  V_mat.submat(Z_VARS+2, 0, Z_VARS+2, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * G; // *
  //psi2 - S2*G
  V_mat.submat(Z_VARS+3, 0, Z_VARS+3, 2) = V_mat.submat(Z_VARS+1, 0, Z_VARS+1, 2) * G; // *
  //h1 - S2*S1
  V_mat.submat(Z_VARS+4, 0, Z_VARS+4, 2) = V_mat.submat(Z_VARS+1, 0, Z_VARS+1, 2) * S_obs; // *
  //h2 - S2*S1*G
  V_mat.submat(Z_VARS+5, 0, Z_VARS+5, 2) = V_mat.submat(Z_VARS+1, 0, Z_VARS+1, 2) * S_obs * G;// *

  //sig2 rows
  V_mat(Z_VARS+6, 0) = -1 / (2.0*sig2) +  std::pow(Y_residual, 2.0)/(2.0*sig2*sig2); // *
  V_mat(Z_VARS+6, 1) = -s2InterTerms * Y_residual / (sig2*sig2); //*
  V_mat(Z_VARS+6, 2) = std::pow(s2InterTerms, 2.0) / (2.0* sig2*sig2); //*

  // ksi2 rows
  V_mat.submat(Z_VARS + 7, 0, Z_VARS + 6 + X_VARS, 0) = -rho*S_residual * trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
                                                              ksi2_X * trans(X) / (tau2_2*(1 - rho*rho)); // *
  V_mat.submat(Z_VARS + 7, 1, Z_VARS + 6 + X_VARS, 1) = trans(X) / (tau2_2*(1 - rho*rho));//*
  V_mat.submat(Z_VARS + 7, 2, Z_VARS + 6 + X_VARS, 2) = arma::zeros(X_VARS); //*

  // ksi1 rows
  V_mat.submat(Z_VARS + 7 + X_VARS, 0, Z_VARS + 6 + 2*X_VARS, 0) = S_residual * trans(X) / (tau2_1*(1- rho*rho)) +
                                                                    rho*ksi2_X * trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)); //*
  V_mat.submat(Z_VARS + 7 + X_VARS, 1, Z_VARS + 6 + 2*X_VARS, 1) = -rho*trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)); //*
  V_mat.submat(Z_VARS + 7 + X_VARS, 2, Z_VARS + 6 + 2*X_VARS, 2) = arma::zeros(X_VARS); //*

  // tau2 rows
  V_mat(Z_VARS + 7 + 2*X_VARS, 0) =                          -1 / (2.0*tau2_2) +
                                          std::pow(ksi2_X, 2.0) / (2.0*tau2_2*tau2_2*(1 - rho*rho)) +
                                          rho*S_residual*ksi2_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho)); //*
  V_mat(Z_VARS + 7 + 2*X_VARS, 1) =     -ksi2_X / (tau2_2*tau2_2*(1 - rho*rho)) -
    rho*S_residual / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho)); //*

  V_mat(Z_VARS + 7 + 2*X_VARS, 2) = 1 / (2.0*tau2_2*tau2_2*(1 - rho*rho)); //*

  // tau1 rows
  V_mat(Z_VARS + 8 + 2*X_VARS, 0) =                           -1 / (2.0*tau2_1) +
                                        std::pow(S_residual, 2.0) / (2.0*tau2_1*tau2_1*(1 - rho*rho)) +
                                        rho*S_residual*ksi2_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho)); //*
  V_mat(Z_VARS + 8 + 2*X_VARS, 1) =     -rho*S_residual / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho)); //*
  V_mat(Z_VARS + 8 + 2*X_VARS, 2) = 0; //*

  // rho rows
  V_mat(Z_VARS + 9 + 2*X_VARS, 0) =                          rho / (1 - rho*rho) -
                                    S_residual*ksi2_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
    rho/std::pow(1 - rho*rho, 2.0) * (
                          S_residual*S_residual / tau2_1 + 
                            ksi2_X*ksi2_X / tau2_2 + 
                            2.0*rho*ksi2_X*S_residual / std::sqrt(tau2_1*tau2_2)
                          );
    // rho / (1 - rho*rho) -
    // S_residual*ksi2_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
    // rho*ksi2_X*ksi2_X / (tau2_2*std::pow(1 - rho*rho, 2.0)) -
    // rho*S_residual*S_residual / (tau2_1*std::pow(1 - rho*rho, 2.0)) -
    // 2.0*rho*rho*S_residual*ksi2_X / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0)); //*
  V_mat(Z_VARS + 9 + 2*X_VARS, 1) =       S_residual / ( (1 - rho*rho) * std::sqrt(tau2_1*tau2_2)) + 
                                    rho / std::pow(1 - rho*rho, 2.0)*(
                                      2.0*ksi2_X / tau2_2 + 
                                      2.0*rho*S_residual / std::sqrt(tau2_1*tau2_2)
                                    );
    
    // S_residual / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) +
    // 2.0*rho*ksi2_X / (tau2_2*std::pow(1 - rho*rho, 2.0)) +
    // 2.0*rho*rho*S_residual / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0)); //*
  V_mat(Z_VARS + 9 + 2*X_VARS, 2) = -rho / (tau2_2*std::pow(1 - rho*rho, 2.0)); //*

  return V_mat;
} //calc_V_mat_twosim_V2


//[[Rcpp::export]]
arma::mat calc_V_mat_twosim_S1S2(const double & gamma_1, const double & gamma_2,
                                 const double & psi_1, const double & psi_2,
                                 const double & h1, const double & h2, const double & sig2,
                                 const double & tau2_1, const double & tau2_2, const double & rho,
                                 const double & Y, const double & G,
                                 const arma::vec & beta,
                                 const arma::vec & ksi_1, const arma::vec & ksi_2,
                                 const arma::rowvec & Z,
                                 const arma::rowvec & X)
{
  const unsigned int X_VARS = X.n_cols;
  const unsigned int Z_VARS = Z.n_cols;
  const double Y_residual = (Y - as_scalar(Z*beta))/sig2;
  const double s1T = (gamma_1 + G*psi_1) / sig2;
  const double s2T = (gamma_2 + G*psi_2) / sig2;
  const double hT  = (h1 + G*h2) / sig2;
  const double ksi1_X = as_scalar(X*ksi_1);
  const double ksi2_X = as_scalar(X*ksi_2);

  arma::mat V_mat = arma::zeros(Z_VARS + 2*X_VARS + 10, 9);

  // redo into parameters/rows (DEC2020)
  // Beta lines;
  V_mat.submat(0, 0, Z_VARS-1, 0) = Y_residual * trans(Z);
  V_mat.submat(0, 1, Z_VARS-1, 1) = -s1T * trans(Z);
  V_mat.submat(0, 2, Z_VARS-1, 2) = arma::zeros(Z_VARS);
  V_mat.submat(0, 3, Z_VARS-1, 3) = -s2T * trans(Z);
  V_mat.submat(0, 4, Z_VARS-1, 4) = arma::zeros(Z_VARS);
  V_mat.submat(0, 5, Z_VARS-1, 5) = -hT * trans(Z);
  V_mat.submat(0, 6, Z_VARS-1, 6) = arma::zeros(Z_VARS);
  V_mat.submat(0, 7, Z_VARS-1, 7) = arma::zeros(Z_VARS);
  V_mat.submat(0, 8, Z_VARS-1, 8) = arma::zeros(Z_VARS);

  // gamma1 lines - Z_VARS
  V_mat(Z_VARS, 0) = 0;
  V_mat(Z_VARS, 1) = Y_residual;
  V_mat(Z_VARS, 2) = -s1T;
  V_mat(Z_VARS, 3) = 0;
  V_mat(Z_VARS, 4) = 0;
  V_mat(Z_VARS, 5) = -s2T;
  V_mat(Z_VARS, 6) = -hT;
  V_mat(Z_VARS, 7) = 0;
  V_mat(Z_VARS, 8) = 0;

  // gamma2 lines - Z_VARS + 1
  V_mat(Z_VARS + 1, 0) = 0;
  V_mat(Z_VARS + 1, 1) = 0;
  V_mat(Z_VARS + 1, 2) = 0;
  V_mat(Z_VARS + 1, 3) = Y_residual;
  V_mat(Z_VARS + 1, 4) = -s2T;
  V_mat(Z_VARS + 1, 5) = -s1T;
  V_mat(Z_VARS + 1, 6) = 0;
  V_mat(Z_VARS + 1, 7) = -hT;
  V_mat(Z_VARS + 1, 8) = 0;

  // psi1 line = S1 line times G
  V_mat.submat(Z_VARS + 2, 0, Z_VARS + 2, 8) = V_mat.submat(Z_VARS, 0, Z_VARS, 8)*G;
  // psi2 line = S2 line times G
  V_mat.submat(Z_VARS + 3, 0, Z_VARS + 3, 8) = V_mat.submat(Z_VARS+1, 0, Z_VARS+1, 8)*G;

  // h1 line - Z_VARS + 4
  V_mat(Z_VARS + 4, 0) = 0;
  V_mat(Z_VARS + 4, 1) = 0;
  V_mat(Z_VARS + 4, 2) = 0;
  V_mat(Z_VARS + 4, 3) = 0;
  V_mat(Z_VARS + 4, 4) = 0;
  V_mat(Z_VARS + 4, 5) = Y_residual;
  V_mat(Z_VARS + 4, 6) = -s1T;
  V_mat(Z_VARS + 4, 7) = -s2T;
  V_mat(Z_VARS + 4, 8) = -hT;

  // h2 line = h1 line times G
  V_mat.submat(Z_VARS + 5, 0, Z_VARS + 5, 8) = V_mat.submat(Z_VARS+4, 0, Z_VARS+4, 8)*G;

  // sig2 line - Z_VARS + 6
  V_mat(Z_VARS + 6, 0) = -1/(2.0*sig2) + std::pow(Y_residual, 2.0) / 2.0;
  V_mat(Z_VARS + 6, 1) = -Y_residual*s1T;
  V_mat(Z_VARS + 6, 2) = std::pow(s1T, 2.0)/2.0;
  V_mat(Z_VARS + 6, 3) = -Y_residual*s2T;
  V_mat(Z_VARS + 6, 4) = std::pow(s2T, 2.0)/2.0;
  V_mat(Z_VARS + 6, 5) = -Y_residual*hT + s1T*s2T;
  V_mat(Z_VARS + 6, 6) = s1T*hT;
  V_mat(Z_VARS + 6, 7) = s2T*hT;
  V_mat(Z_VARS + 6, 8) = hT*hT/2.0;

  /// Lines below the same as original derivation - don't fill in last 3 columns
  //ksi 2 lines
  V_mat.submat(Z_VARS + 7, 0, Z_VARS + 7 + X_VARS -1, 0) = -ksi2_X * trans(X) / (tau2_2*(1 - rho*rho)) +
                                                              rho*ksi1_X * trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7, 1, Z_VARS + 7 + X_VARS -1, 1) = -rho*trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7, 2, Z_VARS + 7 + X_VARS -1, 2) = arma::zeros(Z_VARS);
  V_mat.submat(Z_VARS + 7, 3, Z_VARS + 7 + X_VARS -1, 3) = trans(X) / (tau2_2*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7, 4, Z_VARS + 7 + X_VARS -1, 4) = arma::zeros(Z_VARS);
  V_mat.submat(Z_VARS + 7, 5, Z_VARS + 7 + X_VARS -1, 5) = arma::zeros(Z_VARS);

  //ksi 1 lines
  V_mat.submat(Z_VARS + 7 + X_VARS, 0, Z_VARS + 7 + 2*X_VARS - 1, 0) = -ksi1_X * trans(X) / (tau2_1*(1 - rho*rho)) +
                                                            rho*ksi2_X * trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7 + X_VARS, 1, Z_VARS + 7 + 2*X_VARS - 1, 1) = trans(X) / (tau2_1*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7 + X_VARS, 2, Z_VARS + 7 + 2*X_VARS - 1, 2) = arma::zeros(Z_VARS);
  V_mat.submat(Z_VARS + 7 + X_VARS, 3, Z_VARS + 7 + 2*X_VARS - 1, 3) = -rho*trans(X) / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho));
  V_mat.submat(Z_VARS + 7 + X_VARS, 4, Z_VARS + 7 + 2*X_VARS - 1, 4) = arma::zeros(Z_VARS);
  V_mat.submat(Z_VARS + 7 + X_VARS, 5, Z_VARS + 7 + 2*X_VARS - 1, 5) = arma::zeros(Z_VARS);

  //tau2 lines
  V_mat(Z_VARS + 7 + 2*X_VARS, 0) =            -1 / (2.0*tau2_2) +
                                        ksi2_X*ksi2_X / (2.0*tau2_2*tau2_2*(1 - rho*rho)) -
                                        rho*ksi2_X*ksi1_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS, 1) = rho*ksi2_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS, 2) = 0;
  V_mat(Z_VARS + 7 + 2*X_VARS, 3) = -ksi2_X / (tau2_2*tau2_2*(1 - rho*rho)) +
                                  rho*ksi1_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS, 4) = 1 / (2.0*tau2_2*tau2_2*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS, 5) = -rho / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_2*(1 - rho*rho));

  //tau1 lines
  V_mat(Z_VARS + 7 + 2*X_VARS + 1, 0) =            -1 / (2.0*tau2_1) +
                                      ksi1_X*ksi1_X / (2.0*tau2_1*tau2_1*(1 - rho*rho)) -
                                      rho*ksi2_X*ksi1_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS + 1, 1) = -ksi1_X / (tau2_1*tau2_1*(1 - rho*rho)) +
                                      rho*ksi2_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS + 1, 2) = 1 / (2.0*tau2_1*tau2_1*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS + 1, 3) = rho*ksi1_X / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho));
  V_mat(Z_VARS + 7 + 2*X_VARS + 1, 4) = 0;
  V_mat(Z_VARS + 7 + 2*X_VARS + 1, 5) = -rho / (2.0*std::sqrt(tau2_1*tau2_2)*tau2_1*(1 - rho*rho));

  //rho lines
  V_mat(Z_VARS + 7 + 2*X_VARS + 2, 0) =           rho / (1 - rho*rho) +
                                        ksi2_X*ksi1_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
                                        rho/std::pow(1 - rho*rho, 2.0)*(
                                            ksi1_X*ksi1_X / tau2_1 + 
                                            ksi2_X*ksi2_X / tau2_2 - 
                                            2.0*rho*ksi1_X*ksi2_X / std::sqrt(tau2_1*tau2_2)
                                        );
    // rho / (1 - rho*rho) +
    // ksi2_X*ksi1_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
    // rho*ksi2_X*ksi2_X / (tau2_2*std::pow(1 - rho*rho, 2.0)) -
    // rho*ksi1_X*ksi1_X / (tau2_1*std::pow(1 - rho*rho, 2.0)) +
    // 2.0*rho*rho*ksi2_X*ksi1_X / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0));
  V_mat(Z_VARS + 7 + 2*X_VARS + 2, 1) = -ksi2_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
                                          rho/std::pow(1 - rho*rho, 2.0)*(
                                            2.0*rho*ksi2_X / std::sqrt(tau2_1*tau2_2) - 
                                            2.0*ksi1_X / tau2_1
                                          );
    // -ksi2_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) +
    // 2.0*rho*ksi1_X / (tau2_1*std::pow(1 - rho*rho, 2.0)) -
    // 2.0*rho*rho*ksi2_X / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0));
  V_mat(Z_VARS + 7 + 2*X_VARS + 2, 2) = -rho / (tau2_1*std::pow(1 - rho*rho, 2.0));
  V_mat(Z_VARS + 7 + 2*X_VARS + 2, 3) = -ksi1_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) -
                                      rho/std::pow(1 - rho*rho, 2.0)*(
                                          2.0*rho*ksi1_X / std::sqrt(tau2_1*tau2_2) - 
                                          2.0*ksi2_X / tau2_2
                                      );
    // -ksi1_X / (std::sqrt(tau2_1*tau2_2)*(1 - rho*rho)) +
    // 2.0*rho*ksi2_X / (tau2_2*std::pow(1 - rho*rho, 2.0)) -
    // 2.0*rho*rho*ksi1_X / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0));
  V_mat(Z_VARS + 7 + 2*X_VARS + 2, 4) = -rho / (tau2_2*std::pow(1 - rho*rho, 2.0));
  V_mat(Z_VARS + 7 + 2*X_VARS + 2, 5) = (1.0 + rho*rho) / (std::sqrt(tau2_1*tau2_2)*std::pow(1 - rho*rho, 2.0));
  //Rcout << "New Rho row: " << V_mat.submat(Z_VARS + 7 + 2*X_VARS + 2, 0, Z_VARS + 7 + 2*X_VARS + 2, 8) << "\n";
  return V_mat;
}

// Calculate the U Matrix for a single observation
// Need two U matrix functions: one for single missing, one for both missing
// Need to know detection limits and where observation falls
//[[Rcpp::export]]
arma::mat calc_U_mat_twosim_onemiss(const double & gamma_same, const double & gamma_diff,
                                    const double & psi_same, const double & psi_diff,
                                    const double & h1, const double & h2, const double & rho,
                                    const double & tau2_same, const double & tau2_diff, const double & sig2,
                                    const double & Y, const double & G, const arma::rowvec & Z, const arma::rowvec & X,
                                    const double & S_obs, const double & R,
                                    const arma::vec & beta,
                                    const arma::vec & ksi_same, const arma::vec & ksi_diff,
                                    const double & LLD, const double & ULD,
                                    const int MEASURE_TYPE_KNOWN = 1,
                                    const int MEASURE_TYPE_MISSING = 0,
                                    const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                                    const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
{
  arma::mat U_mat = arma::zeros( 3, 3 );

  if ( R == MEASURE_TYPE_KNOWN)
  {
    return U_mat;
  }

  const double a_val = calc_a_val_twosim(gamma_same, psi_same, G, S_obs, h1, h2, sig2, tau2_same, rho);
  const double b_val = a_val*calc_b_val_twosim(gamma_same, gamma_diff, psi_same, psi_diff, h1, h2,
                                               rho, tau2_same, tau2_diff, sig2,
                                               Y, G, Z, beta, S_obs, X, ksi_same, ksi_diff);

  if ( R == MEASURE_TYPE_MISSING)
  {
    const double mu_1 = b_val;
    const double mu_2 = b_val * b_val + a_val;
    const double mu_3 = 3.0*a_val*b_val + pow(b_val, 3);
    const double mu_4 = 3.0*a_val*a_val + 6.0*a_val*b_val*b_val + pow(b_val, 4);
    U_mat( 0, 0 ) = 1.0;
    U_mat( 0, 1 ) = mu_1;
    U_mat( 0, 2 ) = mu_2;
    U_mat( 1, 0 ) = mu_1;
    U_mat( 1, 1 ) = mu_2;
    U_mat( 1, 2 ) = mu_3;
    U_mat( 2, 0 ) = mu_2;
    U_mat( 2, 1 ) = mu_3;
    U_mat( 2, 2 ) = mu_4;
  }
  else
    if (  R == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
    {
      Rcout << "Method for Below Detection Limit not derived. Method for single mediator implimented and may be incorrect\n";
      const double sqrt_a_val = sqrt( a_val );
      const double arg_val = (LLD - b_val) / sqrt_a_val;
      const double pdf_val = R::dnorm(arg_val, 0.0, 1.0, 0);
      const double cdf_val = R::pnorm(arg_val, 0.0, 1.0, 1, 0);
      const double func_ratio = pdf_val / cdf_val;

      double f_arr[ 5 ];
      f_arr[ 0 ] = 1.0;
      f_arr[ 1 ] = -func_ratio;
      for ( int j = 2; j < 5; ++j )
      {
        const double second_term = (j-1)*f_arr[j-2];
        const double first_term =
          - ( std::pow(arg_val, j-1)  * func_ratio );
          f_arr[ j ] = first_term + second_term;
      }
      double mu_arr[5];
      //mu_arr[0] is unused
      for ( int k = 1; k < 5; ++k )
      {
        mu_arr[ k ] = 0.0;
        for ( int j = 0; j <= k; ++j )
        {
          const double k_choose_j =
            boost::math::factorial<double>( k ) /
              ( boost::math::factorial<double>( j ) *
                boost::math::factorial<double>( k-j ) );
          const double a_b_term =
            pow( a_val, j/2.0) * pow( b_val, k-j);
          mu_arr[k] += (k_choose_j * a_b_term * f_arr[ j ]);
        } //for j loop
      } //for k loop
      const double mu_1 = mu_arr[1];
      const double mu_2 = mu_arr[2];
      const double mu_3 = mu_arr[3];
      const double mu_4 = mu_arr[4];
      U_mat( 0, 0 ) = 1.0;
      U_mat( 0, 1 ) = mu_1;
      U_mat( 0, 2 ) = mu_2;
      U_mat( 1, 0 ) = mu_1;
      U_mat( 1, 1 ) = mu_2;
      U_mat( 1, 2 ) = mu_3;
      U_mat( 2, 0 ) = mu_2;
      U_mat( 2, 1 ) = mu_3;
      U_mat( 2, 2 ) = mu_4;
    } // MEASURE_TYPE_BELOW_DETECT_LIMIT
    else
      if (  R ==  MEASURE_TYPE_ABOVE_DETECTION_LIMIT )
      {
        Rcout << "Method for Above Detection Limit not derived. Method for single mediator implimented and may be incorrect\n";
        const double sqrt_a_val = sqrt( a_val );
        const double arg_val = (b_val - ULD) / sqrt_a_val;
        const double pdf_val = R::dnorm(arg_val, 0.0, 1.0, 0);
        const double cdf_val = R::pnorm(arg_val, 0.0, 1.0, 1, 0);
        const double func_ratio = pdf_val / cdf_val;

        double f_arr[ 5 ];
        f_arr[ 0 ] = 1.0;
        f_arr[ 1 ] = -func_ratio;
        for ( int j = 2; j < 5; ++j )
        {
          const double second_term = (j-1)*f_arr[j-2];
          const double first_term =
            - ( pow(arg_val, j-1)  * func_ratio );
            f_arr[ j ] = first_term + second_term;
        }
        double mu_arr[5];
        //mu_arr[0] is unused
        for ( int k = 1; k < 5; ++k )
        {
          mu_arr[ k ] = 0.0;
          for ( int j = 0; j <= k; ++j )
          {
            const double k_choose_j =
              boost::math::factorial<double>( k ) /
                ( boost::math::factorial<double>( j ) *
                  boost::math::factorial<double>( k-j ) );
            const double a_b_term =
              pow( a_val, j/2.0) * pow( -b_val, k-j);
            mu_arr[k] += (k_choose_j * a_b_term * f_arr[ j ]);
          } //for j loop
          if ( ( k % 2 ) == 1 )
          {
            mu_arr[k] = -mu_arr[k];
          }
        } //for k loop
        const double mu_1 = mu_arr[1];
        const double mu_2 = mu_arr[2];
        const double mu_3 = mu_arr[3];
        const double mu_4 = mu_arr[4];
        U_mat( 0, 0 ) = 1.0;
        U_mat( 0, 1 ) = mu_1;
        U_mat( 0, 2 ) = mu_2;
        U_mat( 1, 0 ) = mu_1;
        U_mat( 1, 1 ) = mu_2;
        U_mat( 1, 2 ) = mu_3;
        U_mat( 2, 0 ) = mu_2;
        U_mat( 2, 1 ) = mu_3;
        U_mat( 2, 2 ) = mu_4;
      } // MEASURE_TYPE_ABOVE_DETECT_LIMIT
      return U_mat;
}

// Update with interactions requires expected values up to S14S24
//[[Rcpp::export]]
arma::mat calc_U_mat_twosim_twomiss(const double & gamma_1, const double & gamma_2,
                                    const double & psi_1, const double & psi_2,
                                    const double & h1, const double & h2,
                                    const double & rho,
                                    const double & tau2_1, const double & tau2_2, const double & sig2,
                                    const double & Y, const double & G, const arma::rowvec & Z, const arma::rowvec & X,
                                    const double & R1, const double & R2,
                                    const arma::vec & beta,
                                    const arma::vec & ksi_1, const arma::vec & ksi_2,
                                    const double & LLD1, const double & ULD1,
                                    const double & LLD2, const double & ULD2,
                                    const int & nDivisions = 5, 
                                    const int MEASURE_TYPE_KNOWN = 1,
                                    const int MEASURE_TYPE_MISSING = 0,
                                    const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                                    const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
{
  arma::mat U_mat = arma::zeros( 9, 9 );

  if ( R1 == MEASURE_TYPE_KNOWN & R2 == MEASURE_TYPE_KNOWN )
  {
    return U_mat;
  }

  if ( R1 == MEASURE_TYPE_MISSING & R2 == MEASURE_TYPE_MISSING){
    double mu1, mu1_2, mu1_3, mu1_4, mu2, mu2_2, mu2_3, mu2_4,
    mu1_mu2, mu12_mu2, mu13_mu2, mu14_mu2, mu1_mu22, mu1_mu23, mu1_mu24,
    mu12_mu22, mu12_mu23, mu12_mu24, mu13_mu22, mu14_mu22,
    mu13_mu23, mu13_mu24, mu14_mu23, mu14_mu24;
    if(h1 == 0 & h2 == 0){
      const double c1_val = calc_c_val_twosim(gamma_1, gamma_2, psi_1, psi_2, G, tau2_1, tau2_2, sig2, rho);
      const double c2_val = calc_c_val_twosim(gamma_2, gamma_1, psi_2, psi_1, G, tau2_2, tau2_1, sig2, rho);
      const double b = calc_corr_b_twosim(gamma_1, gamma_2, psi_1, psi_2, G,
                                          tau2_1, tau2_2, c1_val, c2_val, sig2, rho);
      const double r = calc_corr_r_twosim(b);

      const double d1_val = calc_d_val_twosim(gamma_1, gamma_2, psi_1, psi_2, rho, r, tau2_1, tau2_2,
                                              sig2, Y, G, Z, X, beta, ksi_1, ksi_2, c1_val, c2_val);
      const double d2_val = calc_d_val_twosim(gamma_2, gamma_1, psi_2, psi_1, rho, r, tau2_2, tau2_1,
                                              sig2, Y, G, Z, X, beta, ksi_2, ksi_1, c2_val, c1_val);

      mu1 = d1_val;
      mu1_2 = d1_val*d1_val + c1_val;
      mu1_3 = 3.0*c1_val*d1_val + d1_val*d1_val*d1_val;
      mu1_4 = 3.0*c1_val*c1_val + 6.0*c1_val*d1_val*d1_val + d1_val*d1_val*d1_val*d1_val;
      
      mu2 = d2_val;
      mu2_2 = d2_val*d2_val + c2_val;
      mu2_3 = 3.0*c2_val*d2_val + d2_val*d2_val*d2_val;
      mu2_4 = 3.0*c2_val*c2_val + 6.0*c2_val*d2_val*d2_val + d2_val*d2_val*d2_val*d2_val;
      
      mu1_mu2 = std::sqrt(c1_val*c2_val)*r + d1_val*d2_val;
      mu12_mu2 = 2.0*d1_val*std::sqrt(c1_val*c2_val)*r + d2_val*(d1_val*d1_val + c1_val);
      mu13_mu2 = 3.0*std::sqrt(c1_val)*c1_val*r*std::sqrt(c2_val) + 
        3.0*c1_val*d1_val*d2_val +
        3.0*d1_val*d1_val*r*std::sqrt(c1_val*c2_val) + 
        d1_val*d1_val*d1_val*d2_val;
      mu14_mu2 = 12.0*c1_val*std::sqrt(c1_val*c2_val)*d1_val +
        4.0*std::pow(d1_val, 3.0)*r*std::sqrt(c1_val*c2_val) +
        6.0*c1_val*d2_val*std::pow(d1_val, 2.0) +
        3.0*std::pow(c1_val, 2.0)*d2_val + 
        std::pow(d1_val, 4.0)*d2_val;
      
      mu1_mu22 = 2.0*d2_val*std::sqrt(c1_val*c2_val)*r + 
        d1_val*(d2_val*d2_val + c2_val);
      mu1_mu23 = 3.0*std::sqrt(c2_val)*c2_val*r*std::sqrt(c1_val) + 
        3.0*c2_val*d1_val*d2_val +
        3.0*d2_val*d2_val*r*std::sqrt(c1_val*c2_val) + 
        d1_val*d2_val*d2_val*d2_val;
      mu1_mu24 = 12.0*c2_val*std::sqrt(c1_val*c2_val)*d2_val +
        4.0*std::pow(d2_val, 3.0)*r*std::sqrt(c1_val*c2_val) +
        6.0*c2_val*d1_val*std::pow(d2_val, 2.0) +
        3.0*std::pow(c2_val, 2.0)*d1_val + 
        std::pow(d2_val, 4.0)*d1_val;
      
      mu12_mu22 = c1_val*c2_val + 
        c1_val*d2_val*d2_val + 
        2.0*r*r*c1_val*c2_val + 
        4.0*d1_val*d2_val*std::sqrt(c1_val*c2_val)*r +
        c2_val*d1_val*d1_val + 
        d1_val*d1_val*d2_val*d2_val;
      
      mu12_mu23 = 6.0*r*r*c1_val*c2_val*d2_val + //
        6.0*r*std::sqrt(c1_val*c2_val)*c2_val*d1_val + //
        6.0*r*std::sqrt(c1_val*c2_val)*d1_val*std::pow(d2_val, 2.0) + //
        3.0*c1_val*c2_val*d2_val + //
        std::pow(d2_val, 3.0)*c1_val + //
        3.0*c2_val*std::pow(d1_val, 2.0)*d2_val + //
        std::pow(d1_val*d2_val, 2.0)*d2_val; //
      
      
      mu12_mu24 = 12.0*r*r*c1_val*std::pow(c2_val, 2.0) + //
        12.0*r*r*c1_val*c2_val*std::pow(d2_val, 2.0) + //
        24.0*r*std::sqrt(c1_val*c2_val)*c2_val*d1_val*d2_val + //
        8.0*r*std::sqrt(c1_val*c2_val)*d1_val*std::pow(d2_val, 3.0) + //
        3.0*c1_val*std::pow(c2_val, 2.0) + //
        6.0*c1_val*c2_val*std::pow(d2_val, 2.0) + //
        c1_val*std::pow(d2_val, 4.0) + //
        3.0*std::pow(c2_val, 2.0)*std::pow(d1_val, 2.0) + //
        6.0*c2_val*std::pow(d1_val*d2_val, 2.0) + //
        std::pow(d1_val*d2_val*d2_val, 2.0); //
      
      mu13_mu23 = 6.0*std::pow(r, 3.0)*std::pow(c1_val*c2_val, 1.5) + //
        18.0*r*r*c1_val*c2_val*d1_val*d2_val + // 
        9.0*r*std::pow(c1_val*c2_val, 1.5) + // 
        9.0*r*std::sqrt(c1_val*c2_val)*c1_val*std::pow(d2_val, 2.0)+ //
        9.0*r*std::sqrt(c1_val*c2_val)*c2_val*std::pow(d1_val, 2.0)+ //
        9.0*r*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 2.0) +//
        9.0*c1_val*c2_val*d1_val*d2_val + //
        3.0*c1_val*d1_val*std::pow(d2_val, 3.0) + // 
        3.0*c2_val*d2_val*std::pow(d1_val, 3.0) + //
        std::pow(d1_val*d2_val, 3.0); //
      
      mu13_mu24 = 24.0*std::pow(r, 3.0)*std::pow(c1_val*c2_val, 1.5)*d2_val + //
        36.0*r*r*c1_val*std::pow(c2_val, 2.0)*d1_val + //
        36.0*r*r*c1_val*c2_val*d1_val*std::pow(d2_val, 2.0) + //
        36.0*r*std::pow(c1_val*c2_val, 1.5)*d2_val + //
        12.0*r*std::sqrt(c1_val*c2_val)*c1_val*std::pow(d2_val, 3.0) + //
        36.0*r*std::sqrt(c1_val*c2_val)*c2_val*std::pow(d1_val, 2.0)*d2_val + //
        12.0*r*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 2.0)*d2_val + //
        9.0*c1_val*std::pow(c2_val, 2.0)*d1_val + //
        18.0*c1_val*c2_val*d1_val*std::pow(d2_val, 2.0) + //
        3.0*c1_val*d1_val*std::pow(d2_val, 4.0) + //
        3.0*std::pow(c2_val, 2.0)*std::pow(d1_val, 3.0) + //
        6.0*c2_val*std::pow(d1_val*d2_val, 2.0)*d1_val + //
        std::pow(d1_val*d2_val, 3.0)*d2_val; //
      
      mu14_mu24 = 24.0*std::pow(r, 4.0)*std::pow(c1_val*c2_val, 2.0) + //
        96.0*std::pow(r, 3.0)*std::pow(c1_val*c2_val, 1.5)*d1_val*d2_val + //
        72.0*r*r*std::pow(c1_val*c2_val, 2.0) + //
        72.0*r*r*c1_val*std::pow(c2_val, 2.0)*std::pow(d1_val, 2.0) + //
        72.0*r*r*c2_val*std::pow(c1_val, 2.0)*std::pow(d2_val, 2.0) + //
        72.0*r*r*c1_val*c2_val*std::pow(d1_val*d2_val, 2.0) + //
        144.0*r*std::pow(c1_val*c2_val, 1.5)*d1_val*d2_val + //
        48.0*r*std::sqrt(c1_val*c2_val)*c1_val*d1_val*std::pow(d2_val, 3.0) + //
        48.0*r*std::sqrt(c1_val*c2_val)*c2_val*d2_val*std::pow(d1_val, 3.0) + //
        16.0*r*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 3.0) + //
        9.0*std::pow(c1_val*c2_val, 2.0) + //
        18.0*c1_val*std::pow(c2_val, 2.0)*std::pow(d1_val, 2.0) + //
        18.0*c2_val*std::pow(c1_val, 2.0)*std::pow(d2_val, 2.0) + //
        36.0*c1_val*c2_val*std::pow(d1_val*d2_val, 2.0) + //
        3.0*std::pow(c1_val, 2.0)*std::pow(d2_val, 4.0) + //
        3.0*std::pow(c2_val, 2.0)*std::pow(d1_val, 4.0) + //
        6.0*c1_val*std::pow(d1_val, 2.0)*std::pow(d2_val, 4.0) + //
        6.0*c2_val*std::pow(d2_val, 2.0)*std::pow(d1_val, 4.0) + //
        std::pow(d1_val*d2_val, 4.0); //
      
      mu13_mu22 = 6.0*r*r*c1_val*c2_val*d1_val + //
        6.0*r*std::sqrt(c1_val*c2_val)*c1_val*d2_val + //
        6.0*r*std::sqrt(c1_val*c2_val)*d2_val*std::pow(d1_val, 2.0) + // 
        3.0*c1_val*c2_val*d1_val + // 
        std::pow(d1_val, 3.0)*c2_val + //
        3.0*c1_val*std::pow(d2_val, 2.0)*d1_val + //
        std::pow(d1_val*d2_val, 2.0)*d1_val; //
      
      mu14_mu22 = 12.0*r*r*c2_val*std::pow(c1_val, 2.0) + // checked
        12.0*r*r*c1_val*c2_val*std::pow(d1_val, 2.0) +
        24.0*r*std::sqrt(c1_val*c2_val)*c1_val*d1_val*d2_val +
        8.0*r*std::sqrt(c1_val*c2_val)*d2_val*std::pow(d1_val, 3.0) +
        3.0*c2_val*std::pow(c1_val, 2.0) +
        6.0*c1_val*c2_val*std::pow(d1_val, 2.0) +
        c2_val*std::pow(d1_val, 4.0) +
        3.0*std::pow(c1_val, 2.0)*std::pow(d2_val, 2.0) +
        6.0*c1_val*std::pow(d1_val*d2_val, 2.0) +
        std::pow(d1_val*d1_val*d2_val, 2.0);
      
      
      mu14_mu23 = 24.0*std::pow(r, 3.0)*std::pow(c1_val*c2_val, 1.5)*d1_val + // checked
        36.0*r*r*c2_val*std::pow(c1_val, 2.0)*d2_val +
        36.0*r*r*c1_val*c2_val*d2_val*std::pow(d1_val, 2.0) +
        36.0*r*std::pow(c1_val*c2_val, 1.5)*d1_val +
        12.0*r*std::sqrt(c1_val*c2_val)*c2_val*std::pow(d1_val, 3.0) +
        36.0*r*std::sqrt(c1_val*c2_val)*c1_val*std::pow(d2_val, 2.0)*d1_val +
        12.0*r*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 2.0)*d1_val +
        9.0*c2_val*std::pow(c1_val, 2.0)*d2_val +
        18.0*c1_val*c2_val*d2_val*std::pow(d1_val, 2.0) +
        3.0*c2_val*d2_val*std::pow(d1_val, 4.0) +
        3.0*std::pow(c1_val, 2.0)*std::pow(d2_val, 3.0) +
        6.0*c1_val*std::pow(d1_val*d2_val, 2.0)*d2_val +
        std::pow(d1_val*d2_val, 3.0)*d1_val;
    }else{
      double lowLS1 = LLD1;
      double lowLS2 = LLD2;
      double highLS1 = ULD1;
      double highLS2 = ULD2;
      
      arma::mat limits = bothMissInt_unord_limits(Y, beta, Z, G, sig2,
                                                     gamma_1, psi_1, gamma_2, psi_2,
                                                     h1, h2, ksi_1, ksi_2, X,
                                                     tau2_1, tau2_2, rho,
                                                     lowLS1, lowLS2,
                                                     highLS1, highLS2);

      double denomIntegral = bothMissInt_unord(Y, beta, Z, G, sig2,
                                               gamma_1, psi_1, gamma_2, psi_2,
                                               h1, h2, ksi_1, ksi_2, X,
                                               tau2_1, tau2_2, rho,
                 /* s1 power, s2 power */      0, 0,
                                               limits(0, 0), limits(1, 0),
                                               limits(0, 1), limits(1, 1), nDivisions);
      mu1  = bothMissInt_unord(Y, beta, Z, G, sig2,
                               gamma_1, psi_1, gamma_2, psi_2,
                               h1, h2, ksi_1, ksi_2, X,
                               tau2_1, tau2_2, rho,
                               /* s1 power, s2 power */      1, 0,
                               limits(0, 0), limits(1, 0),
                               limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                gamma_1, psi_1, gamma_2, psi_2,
                                h1, h2, ksi_1, ksi_2, X,
                                tau2_1, tau2_2, rho,
                                /* s1 power, s2 power */      2, 0,
                                limits(0, 0), limits(1, 0),
                                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_3 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                gamma_1, psi_1, gamma_2, psi_2,
                                h1, h2, ksi_1, ksi_2, X,
                                tau2_1, tau2_2, rho,
                                /* s1 power, s2 power */      3, 0,
                                limits(0, 0), limits(1, 0),
                                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_4 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                gamma_1, psi_1, gamma_2, psi_2,
                                h1, h2, ksi_1, ksi_2, X,
                                tau2_1, tau2_2, rho,
                                /* s1 power, s2 power */      4, 0,
                                limits(0, 0), limits(1, 0),
                                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                              gamma_1, psi_1, gamma_2, psi_2,
                              h1, h2, ksi_1, ksi_2, X,
                              tau2_1, tau2_2, rho,
                              /* s1 power, s2 power */      0, 1,
                              limits(0, 0), limits(1, 0),
                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2_2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                gamma_1, psi_1, gamma_2, psi_2,
                                h1, h2, ksi_1, ksi_2, X,
                                tau2_1, tau2_2, rho,
                                /* s1 power, s2 power */      0, 2,
                                limits(0, 0), limits(1, 0),
                                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2_3 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                gamma_1, psi_1, gamma_2, psi_2,
                                h1, h2, ksi_1, ksi_2, X,
                                tau2_1, tau2_2, rho,
                                /* s1 power, s2 power */      0, 3,
                                limits(0, 0), limits(1, 0),
                                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2_4 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                gamma_1, psi_1, gamma_2, psi_2,
                                h1, h2, ksi_1, ksi_2, X,
                                tau2_1, tau2_2, rho,
                                /* s1 power, s2 power */      0, 4,
                                limits(0, 0), limits(1, 0),
                                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;

      mu1_mu2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                  gamma_1, psi_1, gamma_2, psi_2,
                                  h1, h2, ksi_1, ksi_2, X,
                                  tau2_1, tau2_2, rho,
                                  /* s1 power, s2 power */      1, 1,
                                  limits(0, 0), limits(1, 0),
                                  limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                   gamma_1, psi_1, gamma_2, psi_2,
                                   h1, h2, ksi_1, ksi_2, X,
                                   tau2_1, tau2_2, rho,
                                   /* s1 power, s2 power */      2, 1,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                   gamma_1, psi_1, gamma_2, psi_2,
                                   h1, h2, ksi_1, ksi_2, X,
                                   tau2_1, tau2_2, rho,
                                   /* s1 power, s2 power */      3, 1,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu2 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                   gamma_1, psi_1, gamma_2, psi_2,
                                   h1, h2, ksi_1, ksi_2, X,
                                   tau2_1, tau2_2, rho,
                                   /* s1 power, s2 power */      4, 1,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu22 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                   gamma_1, psi_1, gamma_2, psi_2,
                                   h1, h2, ksi_1, ksi_2, X,
                                   tau2_1, tau2_2, rho,
                                   /* s1 power, s2 power */      1, 2,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu23 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                   gamma_1, psi_1, gamma_2, psi_2,
                                   h1, h2, ksi_1, ksi_2, X,
                                   tau2_1, tau2_2, rho,
                                   /* s1 power, s2 power */      1, 3,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu24 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                   gamma_1, psi_1, gamma_2, psi_2,
                                   h1, h2, ksi_1, ksi_2, X,
                                   tau2_1, tau2_2, rho,
                                   /* s1 power, s2 power */      1, 4,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;

      mu12_mu22 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       2, 2,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu23 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       2, 3,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu24 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       2, 4,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu22 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       3, 2,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu22 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       4, 2,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;

      mu13_mu23 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       3, 3,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu24 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       3, 4,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu23 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       4, 3,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu24 = bothMissInt_unord(Y, beta, Z, G, sig2,
                                    gamma_1, psi_1, gamma_2, psi_2,
                                    h1, h2, ksi_1, ksi_2, X,
                                    tau2_1, tau2_2, rho,
                                    /* s1 power, s2 power */       4, 4,
                                    limits(0, 0), limits(1, 0),
                                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
    }
    // 0
    U_mat( 0, 0 ) = 1.0;
    U_mat( 0, 1 ) = mu1;
    U_mat( 0, 2 ) = mu1_2;
    U_mat( 0, 3 ) = mu2;
    U_mat( 0, 4 ) = mu2_2;
    U_mat( 0, 5 ) = mu1_mu2;
    U_mat( 0, 6 ) = mu12_mu2;
    U_mat( 0, 7 ) = mu1_mu22;
    U_mat( 0, 8 ) = mu12_mu22;
    // mu1
    U_mat( 1, 0 ) = mu1;
    U_mat( 1, 1 ) = mu1_2;
    U_mat( 1, 2 ) = mu1_3;
    U_mat( 1, 3 ) = mu1_mu2;
    U_mat( 1, 4 ) = mu1_mu22;
    U_mat( 1, 5 ) = mu12_mu2;
    U_mat( 1, 6 ) = mu13_mu2;
    U_mat( 1, 7 ) = mu12_mu22;
    U_mat( 1, 8 ) = mu13_mu22;
    //mu1^2
    U_mat( 2, 0 ) = mu1_2;
    U_mat( 2, 1 ) = mu1_3;
    U_mat( 2, 2 ) = mu1_4;
    U_mat( 2, 3 ) = mu12_mu2;
    U_mat( 2, 4 ) = mu12_mu22;
    U_mat( 2, 5 ) = mu13_mu2;
    U_mat( 2, 6 ) = mu14_mu2;
    U_mat( 2, 7 ) = mu13_mu22;
    U_mat( 2, 8 ) = mu14_mu22;
    //mu2
    U_mat( 3, 0 ) = mu2;
    U_mat( 3, 1 ) = mu1_mu2;
    U_mat( 3, 2 ) = mu12_mu2;
    U_mat( 3, 3 ) = mu2_2;
    U_mat( 3, 4 ) = mu2_3;
    U_mat( 3, 5 ) = mu1_mu22;
    U_mat( 3, 6 ) = mu12_mu22;
    U_mat( 3, 7 ) = mu1_mu23;
    U_mat( 3, 8 ) = mu12_mu23;
    //mu2^2
    U_mat( 4, 0 ) = mu2_2;
    U_mat( 4, 1 ) = mu1_mu22;
    U_mat( 4, 2 ) = mu12_mu22;
    U_mat( 4, 3 ) = mu2_3;
    U_mat( 4, 4 ) = mu2_4;
    U_mat( 4, 5 ) = mu1_mu23;
    U_mat( 4, 6 ) = mu12_mu23;
    U_mat( 4, 7 ) = mu1_mu24;
    U_mat( 4, 8 ) = mu12_mu24;
    //mu1mu2
    U_mat( 5, 0 ) = mu1_mu2;
    U_mat( 5, 1 ) = mu12_mu2;
    U_mat( 5, 2 ) = mu13_mu2;
    U_mat( 5, 3 ) = mu1_mu22;
    U_mat( 5, 4 ) = mu1_mu23;
    U_mat( 5, 5 ) = mu12_mu22;
    U_mat( 5, 6 ) = mu13_mu22;
    U_mat( 5, 7 ) = mu12_mu23;
    U_mat( 5, 8 ) = mu13_mu23;
    //mu12mu2
    U_mat( 6, 0 ) = mu12_mu2;
    U_mat( 6, 1 ) = mu13_mu2;;
    U_mat( 6, 2 ) = mu14_mu2;;
    U_mat( 6, 3 ) = mu12_mu22;
    U_mat( 6, 4 ) = mu12_mu23;
    U_mat( 6, 5 ) = mu13_mu22;
    U_mat( 6, 6 ) = mu14_mu22;
    U_mat( 6, 7 ) = mu13_mu23;
    U_mat( 6, 8 ) = mu14_mu23;
    //mu1mu22
    U_mat( 7, 0 ) = mu1_mu22;
    U_mat( 7, 1 ) = mu12_mu22;
    U_mat( 7, 2 ) = mu13_mu22;
    U_mat( 7, 3 ) = mu1_mu23;
    U_mat( 7, 4 ) = mu1_mu24;
    U_mat( 7, 5 ) = mu12_mu23;
    U_mat( 7, 6 ) = mu13_mu23;
    U_mat( 7, 7 ) = mu12_mu24;
    U_mat( 7, 8 ) = mu13_mu24;
    //mu12mu22
    U_mat( 8, 0 ) = mu12_mu22;
    U_mat( 8, 1 ) = mu13_mu22;
    U_mat( 8, 2 ) = mu14_mu22;
    U_mat( 8, 3 ) = mu12_mu23;
    U_mat( 8, 4 ) = mu12_mu24;
    U_mat( 8, 5 ) = mu13_mu23;
    U_mat( 8, 6 ) = mu14_mu23;
    U_mat( 8, 7 ) = mu13_mu24;
    U_mat( 8, 8 ) = mu14_mu24;
  }
  else{
    if (  R1 == MEASURE_TYPE_BELOW_DETECTION_LIMIT | R2 == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
    {
      Rcout << "Method for Below Detection Limit not derived.\n";
      return U_mat;
    } // MEASURE_TYPE_BELOW_DETECT_LIMIT
    else
      if (  R1 ==  MEASURE_TYPE_ABOVE_DETECTION_LIMIT | R2 ==  MEASURE_TYPE_ABOVE_DETECTION_LIMIT )
      {
        Rcout << "Method for Above Detection Limit not derived.\n";
        return U_mat;
      } // MEASURE_TYPE_ABOVE_DETECT_LIMIT
  }

  return U_mat;
}

// Functions exist for U matrix and V matrix, each for a single observation;
//[[Rcpp::export]]
arma::mat calc_OMEGA_twosim(const arma::vec & Y,
                            const arma::vec & G,
                            arma::vec & S1, arma::vec & R1,
                            arma::vec & S2, arma::vec & R2,
                            const arma::vec & lowerS1, const arma::vec & upperS1,
                            const arma::vec & lowerS2, const arma::vec & upperS2,
                            const arma::vec & beta, const double & gamma_1, const double & gamma_2,
                            const double & psi_1, const double & psi_2,
                            const double & h1, const double & h2, const double & sig2,
                            const arma::vec & ksi_2, const double & tau2_2,
                            const arma::vec & ksi_1, const double & tau2_1, const double & rho,
                            const arma::mat & Z, const arma::mat & X,
                            bool int_gs1 = true, bool int_gs2 = true,
                            bool int_s1s2 = false, bool int_gs1s2 = false,
                            const int & nDivisions = 5, 
                            const int MEASURE_TYPE_KNOWN = 1,
                            const int MEASURE_TYPE_MISSING = 0,
                            const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                            const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  const unsigned int N = Y.n_elem;
  arma::mat Zuse;
  arma::mat Xuse;

  // Update Z and X matrices to add in
  if(Z.n_elem < N){
    Zuse = arma::ones(N, 2);
    Zuse.submat(0, 1, N-1, 1) = G;
  }else{
    Zuse = arma::zeros(N, Z.n_cols + 1);
    //Rcout << Zuse << "\n1\n\n";
    Zuse.submat(0, 0, N-1, 0) = Z.submat(0, 0, N-1, 0);//Rcout << Zuse << "\n2\n\n";
    Zuse.submat(0, 1, N-1, 1) = G;//Rcout << Zuse << "\n3\n\n";
    Zuse.submat(0, 2, N-1, Zuse.n_cols - 1) = Z.submat(0, 1, N-1, Z.n_cols - 1);//Rcout << Zuse << "\n4\n\n";
  }
  if(X.n_elem < N){
    Xuse = arma::ones(N, 2);
    Xuse.submat(0, 1, N-1, 1) = G;
  }else{
    Xuse = arma::zeros(N, X.n_cols + 1);
    Xuse.submat(0, 0, N-1, 0) = X.submat(0, 0, N-1, 0);
    Xuse.submat(0, 1, N-1, 1) = G;
    Xuse.submat(0, 2, N-1, Xuse.n_cols - 1) = X.submat(0, 1, N-1, X.n_cols - 1);
  }

  const unsigned int X_VARS = Xuse.n_cols;
  const unsigned int Z_VARS = Zuse.n_cols;

  // Update expectation using coverged estimates;
  List expRes = calc_expectation_twosim(gamma_1, gamma_2, psi_1, psi_2, h1, h2, rho,
                                        sig2, tau2_1, tau2_2,
                                        Y, Zuse, Xuse, G,
                                        beta, ksi_1, ksi_2,
                                        S1, R1,
                                        S2, R2,
                                        lowerS1, upperS1,
                                        lowerS2, upperS2,
                                        nDivisions,
                                        MEASURE_TYPE_KNOWN,
                                        MEASURE_TYPE_MISSING,
                                        MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                        MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

  arma::vec expectS1 = expRes["expectS1"];
  arma::vec expectS1_sq = expRes["expectS1_sq"];
  arma::vec expectS2 = expRes["expectS2"];
  arma::vec expectS2_sq = expRes["expectS2_sq"];
  arma::vec expectS1S2 = expRes["expectS1S2"];
  arma::vec expectS12S2 = expRes["expectS12S2"];
  arma::vec expectS1S22 = expRes["expectS1S22"];
  arma::vec expectS12S22 = expRes["expectS12S22"];

  // Calculate Q matrix
  arma::mat Qalt = arma::zeros( Z_VARS + 7 + 2*X_VARS + 3, Z_VARS + 7 + 2*X_VARS + 3);
  arma::mat Q = calc_Q_matrix_twosim( sig2, tau2_1, tau2_2, rho, Zuse, Xuse, G,
                                      expectS1, expectS1_sq,
                                      expectS2, expectS2_sq,
                                      expectS1S2, expectS12S2, expectS1S22, expectS12S22,
                                      int_gs1, int_gs2, int_s1s2, int_gs1s2);

  //Rcout << "Qdim: " << Q.n_rows << "x" << Q.n_cols << "\n";
  // Go through each subject calculating values and summing matrix;
  for(unsigned int index = 0; index < N; ++index ){
    /*
     * Methodology not developed for below or above limit detection
     */
    if (R1(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
    {

    }
    if (R1(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT)
    {

    }

    if (R2(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
    {

    }
    if (R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT)
    {

    }
    /*
     * These to be updated when method is developed to handle them
     */

    if( R1(index) == MEASURE_TYPE_KNOWN & R2(index) == MEASURE_TYPE_KNOWN ){ continue; }

    if( R1(index) == MEASURE_TYPE_MISSING & R2(index) == MEASURE_TYPE_KNOWN ){
      arma::vec uVec = arma::zeros(3);

      uVec(0) = 1;
      uVec(1) = expectS1(index);
      uVec(2) = expectS1_sq(index);

      arma::mat V = calc_V_mat_twosim_S1(gamma_1, gamma_2, psi_1, psi_2, h1, h2, sig2, tau2_1, tau2_2, rho,
                                         Y(index), S2(index), G(index),
                                         beta, ksi_1, ksi_2,
                                         Zuse.row(index), Xuse.row(index));

      arma::mat uMat = calc_U_mat_twosim_onemiss(gamma_1, gamma_2, psi_1, psi_2, h1, h2,
                                                 rho, tau2_1, tau2_2, sig2,
                                                 Y(index), G(index), Zuse.row(index), Xuse.row(index), S2(index), R1(index),
                                                 beta, ksi_1, ksi_2, lowerS1(index), upperS1(index),
                                                 MEASURE_TYPE_KNOWN,
                                                 MEASURE_TYPE_MISSING,
                                                 MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                                 MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

      arma::mat tempAll =  V * uMat * trans(V);
      
      Qalt += (trimatu(tempAll) + (trimatu(tempAll, 1)).t()) - (V*uVec * trans(V*uVec));
    }
    if( R1(index) == MEASURE_TYPE_KNOWN & R2(index) ==  MEASURE_TYPE_MISSING ){
      arma::vec uVec = arma::zeros(3);

      uVec(0) = 1;
      uVec(1) = expectS2(index);
      uVec(2) = expectS2_sq(index);

      arma::mat V = calc_V_mat_twosim_S2(gamma_1, gamma_2, psi_1, psi_2, h1, h2, sig2, tau2_1, tau2_2, rho,
                                         Y(index), S1(index), G(index),
                                         beta, ksi_1, ksi_2,
                                         Zuse.row(index), Xuse.row(index));

      arma::mat uMat = calc_U_mat_twosim_onemiss(gamma_2, gamma_1, psi_2, psi_1, h1, h2,
                                                 rho, tau2_2, tau2_1, sig2,
                                                 Y(index), G(index), Zuse.row(index), Xuse.row(index), S1(index), R2(index),
                                                 beta, ksi_2, ksi_1, lowerS2(index), upperS2(index),
                                                 MEASURE_TYPE_KNOWN,
                                                 MEASURE_TYPE_MISSING,
                                                 MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                                 MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

      arma::mat tempAll =  V * uMat * trans(V);
      
      Qalt += (trimatu(tempAll) + (trimatu(tempAll, 1)).t()) - (V*uVec * trans(V*uVec));
    }
    if( R1(index) == MEASURE_TYPE_MISSING  & R2(index) ==  MEASURE_TYPE_MISSING ){
      arma::vec uVec = arma::zeros(9);
      uVec(0) = 1;
      uVec(1) = expectS1(index);
      uVec(2) = expectS1_sq(index);
      uVec(3) = expectS2(index);
      uVec(4) = expectS2_sq(index);
      uVec(5) = expectS1S2(index);
      uVec(6) = expectS12S2(index);
      uVec(7) = expectS1S22(index);
      uVec(8) = expectS12S22(index);
      //Rcout << "1\n";
      
      arma::mat V = calc_V_mat_twosim_S1S2(gamma_1, gamma_2, psi_1, psi_2, h1, h2,
                                           sig2, tau2_1, tau2_2, rho,
                                           Y(index), G(index), beta, ksi_1, ksi_2, Zuse.row(index), Xuse.row(index));

      // if(h1 == 0 & h2 == 0){
      //   V.submat(0, 6, V.n_rows - 1, 8) = arma::zeros(V.n_rows, 3);
      // }
      //if(index == 32)Rcout << "V:\n" << V << "\n";
      //Rcout << "2\n";
      arma::mat uMat = calc_U_mat_twosim_twomiss(gamma_1, gamma_2, psi_1, psi_2, h1, h2,
                                                 rho, tau2_1, tau2_2, sig2,
                                                 Y(index), G(index), Zuse.row(index), Xuse.row(index), R1(index), R2(index),
                                                 beta, ksi_1, ksi_2,
                                                 lowerS1(index), upperS1(index), lowerS2(index), upperS2(index),
                                                 nDivisions,
                                                 MEASURE_TYPE_KNOWN,
                                                 MEASURE_TYPE_MISSING,
                                                 MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                                 MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

      arma::mat tempAll =  V * uMat * trans(V);
      
      Qalt += (trimatu(tempAll) + (trimatu(tempAll, 1)).t()) - (V*uVec * trans(V*uVec));
      // if(index == 32){
      // arma::mat temp = V * uMat * trans(V) - (V*uVec * trans(V*uVec));
      // Rcout << "New\n" << temp.submat(6, 11, 15, 11) << "\n";
      // }
      //Rcout << "4\n";
      //Rcout << expectS12S2(index) << ", " << expectS1S22(index) << ", " << expectS12S22(index) << "\n\n";
      //if(index == 17)Rcout << "New:\n" << arma::round((V * uMat * trans(V) - (V*uVec * trans(V*uVec)))*1000.0)/1000.0 << "\n\n";
    }
  }
//Rcout << "Qalt:\n" << Qalt.submat(0, 0, 4, 4) << "\n";
  arma::mat omega = Q - Qalt;
//Rcout << "Omega\n" << omega.submat(0, 0, 4, 4) << "\n";
  arma::mat omegaInv;

  //Rcout << omega << "\n\n";

  if(!int_gs1){
    omega.submat(Z_VARS + 2, 0, Z_VARS+2, omega.n_cols - 1) = arma::zeros(1, omega.n_cols); // beta, gamma1, gamma2
    omega.submat(0, Z_VARS+2, omega.n_cols - 1, Z_VARS+2) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS+2,Z_VARS+2) = 1;
  }
  if(!int_gs2){
    omega.submat(Z_VARS + 3, 0, Z_VARS+3, omega.n_cols - 1) = arma::zeros(1, omega.n_cols); // beta, gamma1, gamma2, psi1
    omega.submat(0, Z_VARS+3, omega.n_cols - 1, Z_VARS+3) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS+3,Z_VARS+3) = 1;
  }
  if(!int_s1s2){
    omega.submat(Z_VARS + 4, 0, Z_VARS+4, omega.n_cols - 1) = arma::zeros(1, omega.n_cols); // beta, gamma1, gamma2, psi1, psi2
    omega.submat(0, Z_VARS+4, omega.n_cols - 1, Z_VARS+4) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS+4,Z_VARS+4) = 1;
  }
  if(!int_gs1s2){
    omega.submat(Z_VARS + 5, 0, Z_VARS+5, omega.n_cols - 1) = arma::zeros(1, omega.n_cols); // beta, gamma1, gamma2, psi1, psi2, h1
    omega.submat(0, Z_VARS+5, omega.n_cols - 1, Z_VARS+5) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS+5,Z_VARS+5) = 1;
  }
  //Rcout << "new\n" << omega.submat(0, 0, omega.n_rows - 1, 15) << "\n\n";
  //Rcout << omega.submat(Z_VARS+2, 0, Z_VARS+5, omega.n_cols - 1) << "\n\n";
  //Rcout << "new:\n" <<omega.submat(0, 11, 5, 16) << "\n";
  bool badSig = inv(omegaInv, omega);
  if(badSig){
    /* if no interaction, force variance to 0 */
//Rcout << "Inverse\n" << omegaInv.submat(0, 0, 4, 4) << "\n";
    if(!int_gs1){
      omegaInv(Z_VARS+2,Z_VARS+2) = 0;
    }
    if(!int_gs2){
      omegaInv(Z_VARS+3,Z_VARS+3) = 0;
    }
    if(!int_s1s2){
      omegaInv(Z_VARS+4,Z_VARS+4) = 0;
    }
    if(!int_gs1s2){
      omegaInv(Z_VARS+5,Z_VARS+5) = 0;
    }

    return omegaInv;
  } else{
    Rcout << "Estimated Variance Matrix is Singular\n";
    return arma::zeros(Z_VARS + 7 + 2*X_VARS + 3, Z_VARS + 7 + 2*X_VARS + 3);
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////    Confidence Intervals   //////////////////////////////////////////////
///////////////////////  Same as single mediator  //////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     Delta Method    //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List deltaCI_twosim(const long double & mu1, const long double & sig1,
                 const long double & mu2, const long double & sig2, const long double & sig12,
                 const double & indL = 1,
                 const long double & mu3 = 0, const long double & sig3 = 0,
                 const long double & sig13 = 0, const long double & sig23 = 0,
                 const double alpha = 0.05){
  double zval = R::qnorm(1.0 - alpha/2.0, 0, 1, 1, 0);
  double deltaSE = 0;
  double deltaEst = mu1*(mu2 + indL*mu3);
  
  deltaSE = std::sqrt( std::pow(mu2 + indL*mu3, 2.0)*sig1 +
    std::pow(mu1, 2.0)*(sig2 + std::pow(indL, 2.0)*sig3) +
    2.0*( deltaEst*sig12 +
    std::pow(mu1, 2.0)*indL*sig23 +
    deltaEst*indL*sig13));
  
  arma::vec CI = arma::zeros(2);
  CI(0) = mu1*(mu2 + indL*mu3) - zval*deltaSE;
  CI(1) = mu1*(mu2 + indL*mu3) + zval*deltaSE;
  
  long double pval = R::pnorm(fabs(mu1*(mu2 + indL*mu3)) / deltaSE, 0, 1, 0, 0);
  
  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     bootstrap app    //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List bootstrapCI_twosim(const arma::vec & y, const arma::vec & g, const arma::vec & s1, const arma::vec & r1,
                        const arma::vec & s2, const arma::vec & r2,
                        const arma::mat & Z, const arma::mat & X,
                        const arma::vec & lowers1, const arma::vec & uppers1,
                        const arma::vec & lowers2, const arma::vec & uppers2,
                        const double & delta1, const double & delta2,
                        const double & alpha,
                        bool int_gs1 = true, bool int_gs2 = true,
                        bool int_s1s2 = false, bool int_gs1s2 = false,
                        const double & indL = 1,
                        const unsigned int bootStrapN = 1000,
                        const double convLimit = 1e-4, const double iterationLimit = 1e4,
                        const int & nDivisions = 5, 
                        const int MEASURE_TYPE_KNOWN = 1,
                        const int MEASURE_TYPE_MISSING = 0,
                        const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                        const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){
  unsigned int N = y.n_elem;
  //arma::uvec bootSample(N);
  arma::uvec pVec1;
  arma::uvec pVec2;


  double lowCut = alpha/2.0;
  unsigned int lowCutVal = round(bootStrapN * lowCut) - 1;
  double highCut = 1 - lowCut;
  unsigned int highCutVal = round(bootStrapN * highCut) + 1;

  if(highCutVal >= bootStrapN | lowCutVal <= 0){
    highCutVal = bootStrapN - 1;
    lowCutVal = 0;
    Rcout << "Too few iterations used for accurate Bootstrap CI\n";
  }

  //IntegerVector toSample =  seq(1, N) - 1;

  arma::vec delta1Hats = arma::zeros(bootStrapN);
  arma::vec delta2Hats = arma::zeros(bootStrapN);
  arma::uvec bootSample(N);
  int badSample  = 0;
  for(int i = 0; i < bootStrapN; i++){
    for(int j = 0; j < N; j++){
      bootSample(j) = R::runif(0,N-1);
    }

    /// Check to see if new X and Z are inevitable. If not, decrease i and start sample over
    arma::mat zBoot = Z.rows(bootSample);
    arma::mat xBoot = X.rows(bootSample);
    arma::vec gBoot = g.elem(bootSample);

    arma::mat zBootCheck = arma::zeros(N, zBoot.n_cols + 1);
    zBootCheck.submat(0, 0, N-1, zBoot.n_cols - 1) = zBoot;
    zBootCheck.submat(0, zBoot.n_cols, N-1, zBoot.n_cols) = gBoot;

    arma::mat xBootCheck = arma::zeros(N, xBoot.n_cols + 1);
    xBootCheck.submat(0, 0, N-1, xBoot.n_cols - 1) = xBoot;
    xBootCheck.submat(0, xBoot.n_cols, N-1, xBoot.n_cols) = gBoot;

    arma::mat t_X_inv;
    arma::mat t_Z_inv;

    bool xCheck = arma::inv(t_X_inv, trans(xBootCheck)*xBootCheck);
    bool zCheck = arma::inv(t_Z_inv, trans(zBootCheck)*zBootCheck);

    if(!xCheck | !zCheck){
      i--;
    }else{

      arma::vec yBoot = y.elem(bootSample);
      arma::vec s1Boot = s1.elem(bootSample);
      arma::vec r1Boot = r1.elem(bootSample);
      arma::vec s2Boot = s2.elem(bootSample);
      arma::vec r2Boot = r2.elem(bootSample);
      arma::vec low1Boot = lowers1.elem(bootSample);
      arma::vec up1Boot = uppers1.elem(bootSample);
      arma::vec low2Boot = lowers2.elem(bootSample);
      arma::vec up2Boot = uppers2.elem(bootSample);

      List itEM_res = twoSimMed_EM(yBoot, gBoot, s1Boot, r1Boot, s2Boot, r2Boot, zBoot, xBoot,
                                   low1Boot, up1Boot, low2Boot, up2Boot,
                                   int_gs1, int_gs2, int_s1s2, int_gs1s2,
                                   convLimit, iterationLimit,
                                   nDivisions,
                                   MEASURE_TYPE_KNOWN,
                                   MEASURE_TYPE_MISSING,
                                   MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                   MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

      double tGamma1 = itEM_res["gamma_1"];
      double tGamma2 = itEM_res["gamma_2"];
      double tPsi1 = itEM_res["psi_1"];
      double tPsi2 = itEM_res["psi_2"];
      arma::vec tKsi1 = itEM_res["ksi_1"];
      arma::vec tKsi2 = itEM_res["ksi_2"];

      delta1Hats(i) = tKsi1(1)*(tGamma1 + indL*tPsi1);
      delta2Hats(i) = tKsi2(1)*(tGamma2 + indL*tPsi2);

      if(!arma::is_finite(delta1Hats(i)) | isnan(delta1Hats(i)) |
         !arma::is_finite(delta2Hats(i)) | isnan(delta2Hats(i))){
         --i;
         ++badSample;
         // If bad sample/no estimate, redo
      }
      if(badSample >= (bootStrapN*1e3)){
        Rcout << "Too many bootstrap samples giving impossible results. returning with estimates available. Results may be inaccurate\n";
        if(i < 2){
          arma::vec CI1 = arma::zeros(2);
          arma::vec CI2 = arma::zeros(2);
          CI1(0) = -1e15;
          CI1(1) = 1e15;
          CI2(0) = -1e15;
          CI2(1) = 1e15;
          double deltaSE1 = 1e15;
          double deltaSE2 = 1e15;
          double pval1 = 1;
          double pval2 = 1;
          return List::create(Named("CI1") = CI1,
                              Named("deltaSE1") = deltaSE1,
                              Named("pval1") = pval1,
                              Named("CI2") = CI2,
                              Named("deltaSE2") = deltaSE2,
                              Named("pval2") = pval2);
        }
        arma::vec temp2_1 = sort(delta1Hats.subvec(0, i - 1));
        arma::vec temp2_2 = sort(delta2Hats.subvec(0, i - 1));
        arma::vec CI1 = arma::zeros(2);
        arma::vec CI2 = arma::zeros(2);
        CI1(0) = temp2_1(std::max(floor(lowCutVal * i / bootStrapN), 0.0));
        CI1(1) = temp2_1(std::min(ceil(highCutVal * i / bootStrapN), i - 1.0));
        CI2(0) = temp2_2(std::max(floor(lowCutVal * i / bootStrapN), 0.0));
        CI2(1) = temp2_2(std::min(ceil(highCutVal * i / bootStrapN), i - 1.0));
        
        double deltaSE1 = stddev(temp2_1);
        double deltaSE2 = stddev(temp2_2);

        if(delta1 < 0){
          pVec1 = arma::find(temp2_1 >= 0);
        }else if(delta1 > 0){
          pVec1 = arma::find(temp2_1 <= 0);
        }else if(delta1 == 0){
          pVec1 = arma::find(temp2_1 > -999999999999999999);
        }
        if(delta2 < 0){
          pVec2 = arma::find(temp2_2 >= 0);
        }else if(delta2 > 0){
          pVec2 = arma::find(temp2_2 <= 0);
        }else if(delta2 == 0){
          pVec2 = arma::find(temp2_2 > -999999999999999999);
        }

        double pval1 = std::min(1.0, 2.0*pVec1.n_elem/i);
        double pval2 = std::min(1.0, 2.0*pVec2.n_elem/i);

        return List::create(Named("CI1") = CI1,
                            Named("delta1SE") = deltaSE1,
                            Named("pval1") = pval1,
                            Named("CI2") = CI2,
                            Named("delta2SE") = deltaSE2,
                            Named("pval2") = pval2);
      }
    }
  }

  arma::vec temp1 = sort(delta1Hats);
  arma::vec temp2 = sort(delta2Hats);
  arma::vec CI1 = arma::zeros(2);
  arma::vec CI2 = arma::zeros(2);
  CI1(0) = temp1(floor(lowCutVal));
  CI1(1) = temp1(ceil(highCutVal));
  CI2(0) = temp2(floor(lowCutVal));
  CI2(1) = temp2(ceil(highCutVal));

  double deltaSE1 = stddev(temp1);
  double deltaSE2 = stddev(temp2);

  if(delta1 < 0){
    pVec1 = arma::find(temp1 >= 0);
  }else if(delta1 > 0){
    pVec1 = arma::find(temp1 <= 0);
  }else if(delta1 == 0){
    pVec1 = arma::find(temp1 > -999999999999999999);
  }
  if(delta2 < 0){
    pVec2 = arma::find(temp2 >= 0);
  }else if(delta2 > 0){
    pVec2 = arma::find(temp2 <= 0);
  }else if(delta2 == 0){
    pVec2 = arma::find(temp2 > -999999999999999999);
  }

  double pval1 = std::min(1.0, 2.0*pVec1.n_elem/bootStrapN);
  double pval2 = std::min(1.0, 2.0*pVec2.n_elem/bootStrapN);

  return List::create(Named("CI1") = CI1,
                      Named("deltaSE1") = deltaSE1,
                      Named("pval1") = pval1,
                      Named("CI2") = CI2,
                      Named("deltaSE2") = deltaSE2,
                      Named("pval2") = pval2);
}



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     MC Approach    ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// order for all components: alphaG1, gamma1, psi1, alphaG2, gamma2, psi2
// delta - delta1, delta2

// [[Rcpp::export]]
List mcCI_twosim(const arma::vec & mu, const arma::mat & sig, const arma::vec & delta,
                 const double & indL = 1.0,
                 const double nIt = 10000, const double alpha = 0.05){
  double outVal = std::ceil((alpha/2) * nIt);
  double itGen = outVal + 1;
  double itGen2 = itGen;
  double totGen = 0;
  double pMean_1 = 0, pMean_2 = 0, cMean_1 = 0, cMean_2 = 0;
  double pVar_1 = 0, pVar_2 = 0, cVar_1 = 0, cVar_2 = 0;
  double aMean_1 = 0, aMean_2 = 0;
  double pval_1 = 0, pval_2 = 0;
  
  double indL1Mult = indL;
  double indL2Mult = indL;
  arma::mat sigUse = sig;
  
  if(sig(2, 2) == 0){
    indL1Mult = 0;
    sigUse(2, 2) = 1.0;
  }
  if(sig(5, 5) == 0){
    indL2Mult = 0;
    sigUse(5, 5) = 1.0;
  }

  arma::mat tGen = rmvnorm(itGen2 * 2, mu, sigUse);
  arma::vec genValues1_1 = tGen.col(0) % (tGen.col(1) + indL1Mult*tGen.col(2));
  arma::vec genValues1_2 = tGen.col(3) % (tGen.col(4) + indL2Mult*tGen.col(5));
  arma::vec low_1 = sort(genValues1_1);
  arma::vec high_1 = sort(genValues1_1, "descend");
  arma::vec low_2 = sort(genValues1_2);
  arma::vec high_2 = sort(genValues1_2, "descend");

  pMean_1 = arma::mean(genValues1_1);
  pVar_1 = arma::var(genValues1_1);
  pMean_2 = arma::mean(genValues1_2);
  pVar_2 = arma::var(genValues1_2);

  if(delta(0) < 0){
    arma::uvec s1_1 = arma::find(genValues1_1 >= 0);
    pval_1 += s1_1.n_elem;
  }else if(delta(0) > 0){
    arma::uvec s1_1 = arma::find(genValues1_1 <= 0);
    pval_1 += s1_1.n_elem;
  }else if(delta(0) == 0){
    pval_1 += genValues1_1.n_elem;
  }

  if(delta(1) < 0){
    arma::uvec s1_2 = arma::find(genValues1_2 >= 0);
    pval_2 += s1_2.n_elem;
  }else if(delta(1) > 0){
    arma::uvec s1_2 = arma::find(genValues1_2 <= 0);
    pval_2 += s1_2.n_elem;
  }else if(delta(1) == 0){
    pval_2 += genValues1_2.n_elem;
  }


  totGen = itGen2 * 2;
  while(totGen < nIt){
    itGen2 = std::min(nIt - totGen, itGen);

    arma::mat tGen = rmvnorm(itGen2, mu, sigUse);
    arma::vec genValues_1 = tGen.col(0) % (tGen.col(1) + indL1Mult*tGen.col(2));
    arma::vec genValues_2 = tGen.col(3) % (tGen.col(4) + indL2Mult*tGen.col(5));

    low_1.subvec(low_1.n_elem - genValues_1.n_elem, low_1.n_elem - 1) = genValues_1;
    high_1.subvec(high_1.n_elem - genValues_1.n_elem, high_1.n_elem - 1) = genValues_1;
    low_2.subvec(low_2.n_elem - genValues_2.n_elem, low_2.n_elem - 1) = genValues_2;
    high_2.subvec(high_2.n_elem - genValues_2.n_elem, high_2.n_elem - 1) = genValues_2;

    low_1 = sort(low_1);
    high_1 = sort(high_1, "descend");
    low_2 = sort(low_2);
    high_2 = sort(high_2, "descend");

    cMean_1 = arma::mean(genValues_1);
    cVar_1 = arma::var(genValues_1);
    aMean_1 = (cMean_1*itGen2 + pMean_1*totGen) / (totGen + itGen2);
    cMean_2 = arma::mean(genValues_2);
    cVar_2 = arma::var(genValues_2);
    aMean_2 = (cMean_2*itGen2 + pMean_2*totGen) / (totGen + itGen2);

    pVar_1 = (((totGen * (pVar_1 + std::pow(pMean_1, 2))) + (itGen2 * (cVar_1 + std::pow(cMean_1, 2)))) /
      (totGen + itGen2)) - std::pow(aMean_1, 2);
    pVar_2 = (((totGen * (pVar_2 + std::pow(pMean_2, 2))) + (itGen2 * (cVar_2 + std::pow(cMean_2, 2)))) /
      (totGen + itGen2)) - std::pow(aMean_2, 2);

    totGen += itGen2;
    pMean_1 = aMean_1;
    pMean_2 = aMean_2;

    if(delta(0) < 0){
      arma::uvec s1_1 = arma::find(genValues_1 >= 0);
      pval_1 += s1_1.n_elem;
    }else if(delta(0) > 0){
      arma::uvec s1_1 = arma::find(genValues_1 <= 0);
      pval_1 += s1_1.n_elem;
    }else if(delta(0) == 0){
      pval_1 += genValues1_1.n_elem;
    }

    if(delta(1) < 0){
      arma::uvec s1_2 = arma::find(genValues_2 >= 0);
      pval_2 += s1_2.n_elem;
    }else if(delta(1) > 0){
      arma::uvec s1_2 = arma::find(genValues_2 <= 0);
      pval_2 += s1_2.n_elem;
    }else if(delta(1) == 0){
      pval_2 += genValues1_2.n_elem;
    }
  }

  arma::vec CI_1 = arma::zeros(2);
  CI_1(0) = low_1(outVal);
  CI_1(1) = high_1(outVal);
  arma::vec CI_2 = arma::zeros(2);
  CI_2(0) = low_2(outVal);
  CI_2(1) = high_2(outVal);
  double deltaSE_1 = std::sqrt(pVar_1 * nIt / (nIt-1));
  double deltaSE_2 = std::sqrt(pVar_2 * nIt / (nIt-1));

  pval_1 = (2.0*pval_1) / (1.0 * nIt);
  pval_2 = (2.0*pval_2) / (1.0 * nIt);

  return List::create(Named("CI_1") = CI_1,
                      Named("deltaSE_1") = deltaSE_1,
                      Named("pval_1") = pval_1,
                      Named("CI_2") = CI_2,
                      Named("deltaSE_2") = deltaSE_2,
                      Named("pval_2") = pval_2);
}

