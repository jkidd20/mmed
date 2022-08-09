#include <RcppDist.h>
#include <RcppNumerical.h>
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>
#include <math.h>
#include <Rcpp.h>
#include <ctime>


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(BH)]]

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
//        X_n - matrix (or row vector) for independent to mediator n regression
//        h1  - mediator-mediator interaction term
//        h2  - independent-mediator-mediator interaction term
//        sig2- variance of mediator to response regression
//      tau2_n- variance of independent to mediator n regression
//        rho - parameter for correlation between mediators
//        r   - correlation between mediators in proportional distributions
//        gamma_tilde, psi_tilde - effect of mediator 1 on response, interaction of mediator/indpendent on response

//[[Rcpp::export]]
double calc_a1_val_twoseq(const double & gamma_tilde, const double & psi_tilde, 
                          const double & h1, const double & h2, 
                          const double & sig2, 
                          const double & gamma_1, const double psi_1, 
                          const double & tau2_2, const double tau2_1, 
                          const double & S2, const double & G){
  
  double p1n = std::pow(gamma_tilde + psi_tilde*G + S2*(h1 + h2*G), 2.0); 
  double p2n = std::pow(gamma_1 + psi_1*G, 2.0);
  return 1 / ( (p1n / sig2) + (p2n / tau2_2) + (1 / tau2_1) );
}

/// b1-value - don't include a1 term here - multiply it in other functions;
//[[Rcpp::export]]
double calc_b1_val_twoseq(const double & gamma_1, const double & psi_1, 
                          const double & gamma_tilde, const double & psi_tilde,
                          const double & gamma_2, const double & psi_2,
                          const double & h1, const double & h2, 

                          const double & sig2, const double & tau2_1, const double & tau2_2, 
                          const arma::vec & beta, const arma::vec & ksi_1, const arma::vec & ksi_2,
                          const double & y, const arma::rowvec & Z, const double & G,
                          const arma::rowvec & X1, const arma::rowvec & X2,
                          const double & S2){
  double p1 = (gamma_tilde + psi_tilde*G + S2*(h1 + h2*G))*(y - as_scalar(Z*beta) - S2*(gamma_2 + psi_2*G)) / sig2;
    
    
  double p2 = (gamma_1 + psi_1*G)*( S2 - as_scalar(X2*ksi_2) ) / tau2_2;
  double p3 = as_scalar(X1*ksi_1) / tau2_1;
  // Rcout << p1 << ", " << p2 << ", " << p3 << "\n";
  return (p1 + p2 + p3);
}

//[[Rcpp::export]]
double calc_a2_val_twoseq(const double & gamma_2, const double & psi_2, 
                          const double & h1, const double & h2, 
                          const double & S1, const double & G,
                          const double & sig2, const double & tau2_2){
  double p1 = std::pow(gamma_2 + psi_2*G + S1*(h1 + h2*G), 2.0);
  return 1 / (  p1/sig2 + (1 / tau2_2) );
}

/// b2-value - don't include a2 term here - multiply it in other functions;
//[[Rcpp::export]]
double calc_b2_val_twoseq(const double & gamma_2, const double & psi_2, 
                          const double & h1, const double & h2, 
                          const double & gamma_tilde, const double & psi_tilde, 
                          const arma::vec & beta,
                          const double & gamma_1, const double & psi_1,
                          const arma::vec & ksi_2,
                          const double & sig2, const double & tau2_2, 
                          const double & y, const double & G, 
                          const arma::rowvec & Z, const arma::rowvec & X2,
                          const double & S1){
  
  double p1 = (gamma_2 + psi_2*G + S1*(h1 + h2*G))*(y - as_scalar(Z*beta) - S1*(gamma_tilde + psi_tilde*G)) / sig2;
  double p2 = ( as_scalar(X2*ksi_2) + S1*(gamma_1 + psi_1*G)) / tau2_2;
  return (p1 + p2);
}


//[[Rcpp::export]]
double calc_c1_val_twoseq(const double & gamma_1, const double & psi_1, 
                          const double & gamma_2, const double & psi_2,
                          const double & gamma_tilde, const double & psi_tilde,
                         const double & tau2_1,  const double & tau2_2,
                         const double & sig2, const double & G){
  return ( ( tau2_1 * (tau2_2* std::pow(gamma_2 + psi_2*G, 2.0) + sig2) ) /
           (tau2_1*std::pow(gamma_1 + psi_1*G, 2.0)*std::pow(gamma_2 + psi_2*G, 2.0) + 
              2.0*(gamma_tilde + psi_tilde*G)*tau2_1*(gamma_1 + psi_1*G)*(gamma_2 + psi_2*G) +
              std::pow(gamma_tilde + psi_tilde*G, 2.0)*tau2_1 + tau2_2*std::pow(gamma_2 + psi_2*G, 2.0) + sig2)
         );
}

//[[Rcpp::export]]
double calc_d1_val_twoseq(const double & gamma_1, const double & psi_1, 
                          const double & gamma_2, const double & psi_2,
                          const double & gamma_tilde, const double & psi_tilde,
                          const double & rho,
                          const double & tau2_1,  const double & tau2_2, const double & sig2,
                          const double & y, const double & G, const arma::rowvec & Z,
                          const arma::rowvec & X1, const arma::rowvec & X2,
                          const arma::vec & beta, const arma::vec & ksi_1, const arma::vec & ksi_2,
                          const double & c1, const double c2){
  double p1 = (y - as_scalar(Z*beta))*(std::sqrt(c2)*(gamma_2 + psi_2*G)*rho + std::sqrt(c1)*(gamma_tilde + psi_tilde*G)) / sig2;
  double p2 = as_scalar(X2*ksi_2)*(std::sqrt(c2)*rho - std::sqrt(c1)*(gamma_1 + psi_1*G)) / tau2_2;
  double p3 = as_scalar(X1*ksi_1)*std::sqrt(c1) / tau2_1;
  return std::sqrt(c1)*(p1 + p2 + p3);
}


//[[Rcpp::export]]
double calc_c2_val_twoseq(const double & gamma_1, const double & psi_1, 
                          const double & gamma_2, const double & psi_2,
                          const double & gamma_tilde, const double & psi_tilde,
                          const double & tau2_1,  const double & tau2_2,
                          const double & sig2, const double & G){
  return ( ( tau2_2*(std::pow(gamma_tilde + psi_tilde*G, 2.0)*tau2_1 + sig2) + std::pow(gamma_1 + psi_1*G, 2.0)*tau2_1*sig2 ) /
           (tau2_1*std::pow(gamma_1 + psi_1*G, 2.0)*std::pow(gamma_2 + psi_2*G, 2.0) + 
             2.0*(gamma_tilde + psi_tilde*G)*tau2_1*(gamma_1 + psi_1*G)*(gamma_2 + psi_2*G) +
             std::pow(gamma_tilde + psi_tilde*G, 2.0)*tau2_1 + tau2_2*std::pow(gamma_2 + psi_2*G, 2.0) + sig2)
  );
}



//[[Rcpp::export]]
double calc_d2_val_twoseq(const double & gamma_1, const double & psi_1, 
                          const double & gamma_2, const double & psi_2,
                          const double & gamma_tilde, const double & psi_tilde,
                          const double & rho,
                          const double & tau2_1,  const double & tau2_2, const double & sig2,
                          const double & y, const double & G, const arma::rowvec & Z,
                          const arma::rowvec & X1, const arma::rowvec & X2,
                          const arma::vec & beta, const arma::vec & ksi_1, const arma::vec & ksi_2,
                          const double & c1, const double c2){
  double p1 = (y - as_scalar(Z*beta))*(std::sqrt(c2)*(gamma_2 + psi_2*G) + std::sqrt(c1)*(gamma_tilde + psi_tilde*G)*rho) / sig2;
  double p2 = as_scalar(X2*ksi_2)*(std::sqrt(c2) - std::sqrt(c1)*(gamma_1 + psi_1*G)*rho) / tau2_2;
  double p3 = as_scalar(X1*ksi_1)*std::sqrt(c1)*rho / tau2_1;
  return std::sqrt(c2)*(p1 + p2 + p3);
}


//[[Rcpp::export]]
double calc_corr_b_twoseq(const double & gamma_1, const double & psi_1, 
                          const double & gamma_2, const double & psi_2,
                          const double & gamma_tilde, const double & psi_tilde,
                          const double & tau2_2, const double & sig2, const double & G,
                          const double & c1, const double & c2){
  return ( (tau2_2 * sig2) /
            (std::sqrt(c1*c2) * ((gamma_1 + psi_1*G)*sig2 - (gamma_2 + psi_2*G)*(gamma_tilde + psi_tilde*G)*tau2_2)) );
}

//[[Rcpp::export]]
double calc_corr_rho_twoseq(const double & b){
  double r1 = (-b + std::sqrt(b*b + 4)) / 2.0;
  double r2 = (-b - std::sqrt(b*b + 4)) / 2.0;

  if(r1 >= -1 & r1 <= 1){return r1;}
  if(r2 >= -1 & r2 <= 1){return r2;}

  return 0;
}


// Class for Multidimensional integration
class s1s2Int_ord: public MFunc
{
private:
  const double sig2;          // sigma squared (variance term)
  double a;             // (Y-beta*X)/sigma;
  double a1X;           // (alpha1*X1)
  double a2X;           // (alpha2*X2)
  double tau_1;         // tau_1 term (SD not Var)
  double tau_2;         // tau_2 term (SD not Var)
  double gS1;           // (gamma_1 + psi_1*G) / sigma  (sigma is sqrt(sig2) )
  double gS1y;          // (gamma_tilde + psi_tilde*G) / sigma
  double gS2;           // (gamma_2 + psi_2*G) / sigma
  double ht;            // (h1 + h2*G) / sigma
  int ps1;              // power of s1 in expectation
  int ps2;              // power of s2 in expectation
public:
  s1s2Int_ord(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2_,
                const double & gamma_1, const double & psi_1,
                const double & gamma_tilde, const double & psi_tilde,
                const double & gamma_2, const double & psi_2,
                const double & h1, const double & h2,
                arma::vec ksi_1, arma::vec ksi_2, 
                arma::rowvec X1, arma::rowvec X2,
                const double & tau2_1, const double & tau2_2,
                const int & ps1_, const int & ps2_) : sig2(sig2_), ps1(ps1_), ps2(ps2_)
  {
    a     = (Y - as_scalar(Z*beta));
    a1X   = as_scalar(X1*ksi_1);
    a2X   = as_scalar(X2*ksi_2);
    tau_1 = std::sqrt(tau2_1);
    tau_2 = std::sqrt(tau2_2);
    gS1   = (gamma_1 + psi_1*G);
    gS1y  = (gamma_tilde + psi_tilde*G);
    gS2   = (gamma_2 + psi_2*G);
    ht    = (h1 + h2*G);
  }
  
  // PDF of bivariate normal
  double operator()(Constvec& x)
  {
    double b = x[0]*gS1y + x[1]*gS2 + x[0]*x[1]*ht;
    double c = (x[1] - a2X - x[0]*gS1);
    double d = (x[0] - a1X);
    double den1 = std::exp(-std::pow(a - b, 2.0)/(2.0*sig2)) / std::sqrt(2.0*M_PI*sig2);
    double den2 = std::exp(-std::pow(c, 2.0)/(2.0*std::pow(tau_2, 2.0))) / (tau_2*std::sqrt(2.0*M_PI));
    double den3 = std::exp(-std::pow(d, 2.0)/(2.0*std::pow(tau_1, 2.0))) / (tau_1*std::sqrt(2.0*M_PI));
    
    return std::pow(x[0], ps1)*std::pow(x[1], ps2) * den1*den2*den3;
  }
};


// [[Rcpp::export]]
double bothMissInt_ord(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
                         const double & gamma_1, const double & psi_1,
                         const double & gamma_tilde, const double & psi_tilde,
                         const double & gamma_2, const double & psi_2,
                         const double & h1, const double & h2,
                         arma::vec ksi_1, arma::vec ksi_2, 
                         arma::rowvec X1, arma::rowvec X2,
                         const double & tau2_1, const double & tau2_2,
                         const int & ps1, const int & ps2,
                         const double & lowS1, const double & lowS2,
                         const double & highS1, const double & highS2,
                         const int & nDivisions = 5){
  
  s1s2Int_ord f(Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_tilde, psi_tilde, 
                gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2, tau2_1, tau2_2,
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
  return res;
  
  // long double res = integrate(f, lower, upper, err_est, err_code, 10000);
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


// [[Rcpp::export]]
arma::mat bothMissInt_ord_limits(const double & Y, arma::vec beta, arma::rowvec Z, const double & G, const double & sig2,
                                 const double & gamma_1, const double & psi_1,
                                 const double & gamma_tilde, const double & psi_tilde,
                                 const double & gamma_2, const double & psi_2,
                                 const double & h1, const double & h2,
                                 arma::vec ksi_1, arma::vec ksi_2, 
                                 arma::rowvec X1, arma::rowvec X2,
                                 const double & tau2_1, const double & tau2_2,
                                 const double & lowS1, const double & lowS2,
                                 const double & highS1, const double & highS2,
                                 const int missKey1 = 0, const int missKey2 = 0,
                                 const double limit = 1e-14, 
                                 const double stepCorrect = 1000.0){
  s1s2Int_ord f(Y, beta, Z, G, sig2, gamma_1, psi_1, gamma_tilde, psi_tilde, 
                gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2, tau2_1, tau2_2,
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
  //Rcout << "res: " << newres << "\n";
  return bounds;
}



// calc_expectation takes the current estimates of the parameters and the
// data as arguments.
//[[Rcpp::export]]
List calc_expectation_twoseq(const double & gamma_1, const double & psi_1,
                             const double & gamma_2, const double & psi_2, 
                             const double & gamma_tilde, const double & psi_tilde, 
                             const double & h1, const double & h2,
                             const double sig2, const double tau2_1, const double tau2_2,
                      const arma::vec & Y, const arma::mat & Z, const arma::mat & X1, const arma::mat & X2, const arma::vec & G,
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
  arma::vec ES12ES2 = S1%S1%S2;
  arma::vec ES1ES22 = S1%S2%S2;
  arma::vec ES12ES22 = S1%S1%S2%S2;

  for ( unsigned int index = 0; index < N; ++index )
  {
    // First here are cases with below or above detection limit
    // Theory not developed, but future work should address this
    // For now, half point imputation used for all missing lower
    // And upper limit imputation for higher
    // Values then treated as observed with respect to expected values
    bool detectFlag = false;
  
  // Integration will work for both values missing beyond limits. 
    if( ( R1(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT |
        R1(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT )  &
        ( R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT |
        R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT ) ){
      
      double lowLS1, lowLS2, highLS1, highLS2;
      if(R1(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT){lowLS1 = 0; highLS1 = LLD1(index);}
      if(R2(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT){lowLS2 = 0; highLS2 = LLD2(index);}
      if(R1(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT){lowLS1 = ULD1(index); highLS1 = 10.0*ULD1(index);}
      if(R2(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT){lowLS2 = ULD2(index); highLS2 = 10.0*ULD2(index);}
      
      arma::mat limits = bothMissInt_ord_limits(Y(index), beta, Z.row(index), G(index), sig2,
                                                gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                                h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                                tau2_1, tau2_2,
                                                lowLS1, lowLS2,
                                                highLS1, highLS2, R1(index), R2(index)); 
      
      double denomIntegral = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      0, 0,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions);
      // find numerator and divide by denominator
      ES1(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      1, 0,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES1_2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      2, 0,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      0, 1,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES2_2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      0, 2,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES1ES2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      1, 1,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES12ES2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      2, 1,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES1ES22(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
                                              /* s1 power, s2 power */      1, 2,
                                              limits(0, 0), limits(1, 0),
                                              limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      // find numerator and divide by denominator
      ES12ES22(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                              gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                              h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                              tau2_1, tau2_2, 
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
        const double a1_val = calc_a1_val_twoseq(gamma_tilde, psi_tilde, h1, h2, sig2, 
                                                 gamma_1, psi_1, tau2_2, tau2_1, S2(index), G(index));
        const double b1_val = a1_val*calc_b1_val_twoseq(gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2, h1, h2, 
                                                        sig2, tau2_1,  tau2_2,
                                                        beta, ksi_1, ksi_2,
                                                Y(index), Z.row(index), G(index), 
                                                X1.row(index), X2.row(index), S2(index));
        ES1(index) = b1_val ;
        ES1_2(index) = std::pow(b1_val, 2.0) + a1_val ;
        ES1ES2(index) = b1_val * S2(index);
        ES12ES2(index) = ES1_2(index) * S2(index);
        ES1ES22(index) = b1_val * std::pow(S2(index), 2);
        ES12ES22(index) = ES1_2(index) * std::pow(S2(index), 2.0);
      }
    else
    // S2 missing, S1 observed
      if(R1(index) == MEASURE_TYPE_KNOWN & R2(index) == MEASURE_TYPE_MISSING)
      {
        const double a2_val = calc_a2_val_twoseq(gamma_2, psi_2, h1, h2, S1(index), G(index), sig2, tau2_2);
        const double b2_val = a2_val*calc_b2_val_twoseq(gamma_2, psi_2, h1, h2, gamma_tilde, psi_tilde, beta, 
                                                        gamma_1, psi_1, ksi_2, sig2, tau2_2,
                                                        Y(index), G(index), Z.row(index), X2.row(index), S1(index));
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
          if(h1 == 0 & h2 == 0){
            const double c1_val = calc_c1_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, 
                                                     gamma_tilde, psi_tilde, tau2_1,  tau2_2, sig2, G(index));
              
            const double c2_val = calc_c2_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, 
                                                     gamma_tilde, psi_tilde, tau2_1, tau2_2, sig2, G(index));
            
            const double b = calc_corr_b_twoseq(gamma_1, psi_1, gamma_2, psi_2, gamma_tilde, psi_tilde, 
                                                tau2_2, sig2, G(index), c1_val, c2_val);
            
            const double rho = calc_corr_rho_twoseq(b);
              
            const double d1_val = calc_d1_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, gamma_tilde, psi_tilde, 
                                                     rho, tau2_1,  tau2_2, sig2,
                                                    Y(index),  G(index), Z.row(index), X1.row(index), X2.row(index),
                                                    beta, ksi_1, ksi_2, c1_val, c2_val);
            const double d2_val = calc_d2_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, gamma_tilde, psi_tilde, 
                                                     rho, tau2_1,  tau2_2, sig2,
                                                     Y(index),  G(index), Z.row(index), X1.row(index), X2.row(index),
                                                     beta, ksi_1, ksi_2, c1_val, c2_val);
            //Rcout << "new: c1: " << c1_val << ", c2: " << c2_val << ", d1: " << d1_val << ", d2: " << d2_val << "\n";
            ES1(index) = d1_val;
            ES2(index) = d2_val;
            ES1_2(index) = std::pow(d1_val, 2.0) + c1_val;
            ES2_2(index) = std::pow(d2_val, 2.0) + c2_val;
            ES1ES2(index) = std::sqrt(c1_val*c2_val)*rho + d1_val*d2_val;
            ES12ES2(index) = 2.0*d1_val*std::sqrt(c1_val*c2_val)*rho + d2_val*(d1_val*d1_val + c1_val);
            ES1ES22(index) = 2.0*d2_val*std::sqrt(c1_val*c2_val)*rho + d1_val*(d2_val*d2_val + c2_val);
            ES12ES22(index) = c1_val*c2_val + c1_val*d2_val*d2_val + 2.0*rho*rho*c1_val*c2_val + 4.0*d1_val*d2_val*std::sqrt(c1_val*c2_val)*rho +
              c2_val*d1_val*d1_val + d1_val*d1_val*d2_val*d2_val;
            // Rcout << "d1: " << d1_val << "\n";
            // Rcout << "d2: " << d2_val << "\n";
          }else{
            double lowLS1 = LLD1(index);
            double lowLS2 = LLD2(index);
            double highLS1 = ULD1(index);
            double highLS2 = ULD2(index);
            
            arma::mat limits = bothMissInt_ord_limits(Y(index), beta, Z.row(index), G(index), sig2,
                                                      gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                                      h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                                      tau2_1, tau2_2,
                                                      lowLS1, lowLS2,
                                                      highLS1, highLS2, R1(index), R2(index)); 
            
            double denomIntegral = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                                                   gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                                                   h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                                                   tau2_1, tau2_2, 
                                                   /* s1 power, s2 power */      0, 0,
                                                   limits(0, 0), limits(1, 0),
                                                   limits(0, 1), limits(1, 1), nDivisions);
            // find numerator and divide by denominator
            ES1(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                tau2_1, tau2_2, 
                /* s1 power, s2 power */      1, 0,
                limits(0, 0), limits(1, 0),
                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES1_2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                  gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                  h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                  tau2_1, tau2_2, 
                  /* s1 power, s2 power */      2, 0,
                  limits(0, 0), limits(1, 0),
                  limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                tau2_1, tau2_2, 
                /* s1 power, s2 power */      0, 1,
                limits(0, 0), limits(1, 0),
                limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES2_2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                  gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                  h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                  tau2_1, tau2_2, 
                  /* s1 power, s2 power */      0, 2,
                  limits(0, 0), limits(1, 0),
                  limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES1ES2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                   gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                   h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                   tau2_1, tau2_2, 
                   /* s1 power, s2 power */      1, 1,
                   limits(0, 0), limits(1, 0),
                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES12ES2(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                    gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                    h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                    tau2_1, tau2_2, 
                    /* s1 power, s2 power */      2, 1,
                    limits(0, 0), limits(1, 0),
                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES1ES22(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                    gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                    h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                    tau2_1, tau2_2, 
                    /* s1 power, s2 power */      1, 2,
                    limits(0, 0), limits(1, 0),
                    limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
            // find numerator and divide by denominator
            ES12ES22(index) = bothMissInt_ord(Y(index), beta, Z.row(index), G(index), sig2,
                     gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2,
                     h1, h2, ksi_1, ksi_2, X1.row(index), X2.row(index),
                     tau2_1, tau2_2, 
                     /* s1 power, s2 power */      2, 2,
                     limits(0, 0), limits(1, 0),
                     limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
          }
        }
  } // outer for loop
  //Rcout << ES1.elem(find((R2 == 0) % (R1 == 0))) << "\n\n"
        //<< ES2.elem(find((R2 == 0) % (R1 == 0))) << "\n\n"
        //<< ES1_2.elem(find((R2 == 0) % (R1 == 0))) << "\n\n"
        //<< ES2_2.elem(find((R2 == 0) % (R1 == 0))) << "\n\n"
        //<< ES1ES2.elem(find((R2 == 0) % (R1 == 0))) << "\n\n"

        return List::create(Named("expectS1") = ES1, Named("expectS1_sq") = ES1_2,
                            Named("expectS2") = ES2, Named("expectS2_sq") = ES2_2,
                            Named("expectS1S2") = ES1ES2,
                            Named("expectS12S2") = ES12ES2,
                            Named("expectS1S22") = ES1ES22,
                            Named("expectS12S22") = ES12ES22);
}

// Use the expected values to calculate the new mle estimates
// of the parameters of interest
//[[Rcpp::export]]
arma::vec calc_beta_tilde_gamma2_twoseq(const arma::vec & Y, const arma::mat & Z, const arma::vec & G, 
                         const arma::vec & ES1, const arma::vec & ES1_2,
                         const arma::vec & ES2, const arma::vec & ES2_2,
                         const arma::vec & ES1ES2, const arma::vec & ES12ES2,
                         const arma::vec & ES1ES22, const arma::vec & ES12ES22,
                         const bool & int_gs1_Y = true, const bool & int_gs2 = true,  // psi_tilde, psi_2
                         const bool & int_s1s2 = false, const bool & int_gs1s2 = false) // h1 and h2
{
  const unsigned int N = Z.n_rows;
  const unsigned int N_VAR = Z.n_cols;
  arma::vec beta_tilde_gamma_vec = arma::zeros( N_VAR + 6 );  // gamma_tilde, psi_tilde, gamma_2, psi_2, h1, h2
  arma::vec Y_expect_vec = arma::zeros( N_VAR + 6 );
  arma::mat Z_expect_mat = arma::zeros( N_VAR + 6, N_VAR + 6);


  for ( unsigned int index = 0; index < N; ++index )
  {
    arma::vec current_vec = arma::zeros( N_VAR + 6 );
    current_vec.subvec(0, N_VAR - 1) = trans(Y(index)*Z.row(index));
    current_vec(N_VAR)     = Y(index)*ES1(index); // gamma_tilde
    current_vec(N_VAR+2)   = Y(index)*ES2(index); // gamma_2
    if(int_gs1_Y)current_vec(N_VAR + 1) = Y(index)*ES1(index)*G(index); // psi_tilde
    if(int_gs2)current_vec(N_VAR + 3)   = Y(index)*ES2(index)*G(index); // psi_2
    if(int_s1s2){current_vec(N_VAR + 4) = Y(index)*ES1ES2(index);}      // h1
    if(int_gs1s2){current_vec(N_VAR + 5)= Y(index)*ES1ES2(index)*G(index);} // h2
    
    Y_expect_vec += current_vec;

    
    arma::mat current_mat = arma::zeros( N_VAR + 6, N_VAR + 6 );
    
    // beta row and column
    current_mat.submat(0, 0, N_VAR - 1, N_VAR - 1) = trans(Z.row(index)) * Z.row(index);
    
    // gamma_tilde row and column
    current_mat.submat(0, N_VAR, N_VAR-1, N_VAR) = ES1(index)*trans(Z.row(index));
    current_mat.submat(N_VAR, 0, N_VAR, N_VAR - 1) = ES1(index)*Z.row(index);
    current_mat(N_VAR, N_VAR) = ES1_2(index);
    
    // gamma_2 row and column - do crosses with psi_intercept in next section
    current_mat.submat(0, N_VAR + 2, N_VAR - 1, N_VAR + 2) = ES2(index)*trans(Z.row(index));
    current_mat.submat(N_VAR + 2, 0, N_VAR + 2, N_VAR - 1) = ES2(index)*Z.row(index);
    current_mat(N_VAR, N_VAR + 2) = ES1ES2(index);
    current_mat(N_VAR + 2, N_VAR) = ES1ES2(index);
    current_mat(N_VAR + 2, N_VAR + 2) = ES2_2(index);
    
    //psi_tilde row and column
    if(int_gs1_Y){
      current_mat.submat(0, N_VAR + 1, N_VAR - 1, N_VAR + 1) = ES1(index)*trans(Z.row(index))*G(index);
      current_mat.submat(N_VAR + 1, 0, N_VAR + 1, N_VAR - 1) = ES1(index)*Z.row(index)*G(index);
      current_mat(N_VAR, N_VAR + 1) = ES1_2(index)*G(index);
      current_mat(N_VAR + 1, N_VAR) = ES1_2(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 1) = ES1_2(index)*G(index)*G(index);
      
      current_mat(N_VAR + 1, N_VAR + 2) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 1) = ES1ES2(index)*G(index);
    }
    
    // psi_2 row and column
    if(int_gs2){
      current_mat.submat(0, N_VAR + 3, N_VAR - 1, N_VAR + 3) = ES2(index)*trans(Z.row(index))*G(index);
      current_mat.submat(N_VAR + 3, 0, N_VAR + 3, N_VAR - 1) = ES2(index)*Z.row(index)*G(index);
      current_mat(N_VAR, N_VAR + 3) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 3, N_VAR) = ES1ES2(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 1) = ES1ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 3) = ES1ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 2) = ES2_2(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 3) = ES2_2(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 3) = ES2_2(index)*G(index)*G(index);
    }
    
    // h1 row and column - es1es2
    if(int_s1s2){
      current_mat.submat(0, N_VAR + 4, N_VAR - 1, N_VAR + 4) = ES1ES2(index)*trans(Z.row(index));
      current_mat.submat(N_VAR + 4, 0, N_VAR + 4, N_VAR - 1) = ES1ES2(index)*Z.row(index);
      current_mat(N_VAR, N_VAR + 4) = ES12ES2(index);
      current_mat(N_VAR + 4, N_VAR) = ES12ES2(index);
      current_mat(N_VAR + 4, N_VAR + 1) = ES12ES2(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 4) = ES12ES2(index)*G(index);
      current_mat(N_VAR + 4, N_VAR + 2) = ES1ES22(index);
      current_mat(N_VAR + 2, N_VAR + 4) = ES1ES22(index);
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
      current_mat(N_VAR + 5, N_VAR + 1) = ES12ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 1, N_VAR + 5) = ES12ES2(index)*G(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 2) = ES1ES22(index)*G(index);
      current_mat(N_VAR + 2, N_VAR + 5) = ES1ES22(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 3) = ES1ES22(index)*G(index)*G(index);
      current_mat(N_VAR + 3, N_VAR + 5) = ES1ES22(index)*G(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 4) = ES12ES22(index)*G(index);
      current_mat(N_VAR + 4, N_VAR + 5) = ES12ES22(index)*G(index);
      current_mat(N_VAR + 5, N_VAR + 5) = ES12ES22(index)*G(index)*G(index);
    }
    
    
    Z_expect_mat += current_mat;
  }

  // Set diagonals to 1 for inversion if no interaction term;
  if(!int_gs1_Y){Z_expect_mat(N_VAR + 1, N_VAR + 1) = 1;}
  if(!int_gs2){Z_expect_mat(N_VAR + 3, N_VAR + 3) = 1;}
  if(!int_s1s2){Z_expect_mat(N_VAR + 4, N_VAR + 4) = 1;}
  if(!int_gs1s2){Z_expect_mat(N_VAR + 5, N_VAR + 5) = 1;}
  
  //Rcout << Z_expect_mat << "\n\n";
  arma::mat Z_expect_inv;
  bool invRes = arma::inv(Z_expect_inv, Z_expect_mat);
  if(!invRes){
    //Rcout << X_expect_mat << "\n\n";
    Rcout << "Z'Z matrix not invertable\n";
    beta_tilde_gamma_vec = arma::zeros(N_VAR + 6);
    beta_tilde_gamma_vec(N_VAR) = R_NaN;
    beta_tilde_gamma_vec(N_VAR + 2) = R_NaN;
    return beta_tilde_gamma_vec;
  }
  beta_tilde_gamma_vec = Z_expect_inv*Y_expect_vec;
  return(beta_tilde_gamma_vec);

} //calc_beta_tilde_gamma_vec


//[[Rcpp::export]]
double calc_sig2_twoseq(const double & gamma_tilde, const double & psi_tilde, 
                        const double & gamma_2, const double psi_2,
                        const double & h1, const double & h2,
                        const bool & int_gs1_Y, const bool & int_gs2, 
                        const bool & int_s1s2, const bool & int_gs1s2,
                       const arma::vec & Y, const arma::vec & beta, const arma::vec & G,
                       const arma::mat & Z,
                       const arma::vec & ES1, const arma::vec & ES1_2,
                       const arma::vec & ES2, const arma::vec & ES2_2,
                       const arma::vec & ES1ES2, const arma::vec & ES12ES2,
                       const arma::vec & ES1ES22, const arma::vec & ES12ES22)
{
  const unsigned int N = Z.n_rows;
  const unsigned int N_VAR = Z.n_cols;
                        // gamma tilde, gamma_2, additional
  return (1.0/(N - N_VAR - 2.0 - int_gs1_Y*1 - int_gs2*1 - int_s1s2*1 - int_gs1s2*1))*accu(pow(Y - Z*beta, 2.0) +
          pow(gamma_tilde + psi_tilde*G, 2.0)%ES1_2 + pow(gamma_2 + psi_2*G, 2.0)%ES2_2 +
          pow(h1 + h2*G, 2.0)%ES12ES22 -
          2.0*(Y - Z*beta)%( (gamma_tilde + psi_tilde*G)%ES1 + (gamma_2 + psi_2*G)%ES2 +
          (h1 + h2*G)%ES1ES2) +
          2.0*(gamma_tilde + psi_tilde*G)%( (gamma_2 + psi_2*G)%ES1ES2 + (h1 + h2*G)%ES12ES2) +
          2.0*(gamma_2 + psi_2*G)%(h1+h2*G)%ES1ES22);
  
}  //calc_sig2

//calc_ksi : for Maximization step
//[[Rcpp::export]]
arma::vec calc_ksi2_gamma1_twoseq( const arma::mat & X2, const arma::vec & G,
              const arma::vec & ES1, const arma::vec & ES1_2,
              const arma::vec & ES2, const arma::vec & ES1ES2, 
              const arma::vec & ES12ES2, const bool & gs1 = true )
{
  const unsigned int N = X2.n_rows;
  const unsigned int N_VAR = X2.n_cols;
  arma::vec ksi2_gamma1_vec = arma::zeros( N_VAR + 2 );  // ksi2, gamma1, psi1
  arma::vec expect_X2_vec = arma::zeros( N_VAR + 2);
  arma::mat expect_X2_mat = arma::zeros( N_VAR + 2, N_VAR + 2);


  for ( unsigned int index = 0; index < N; ++index)
  {
    arma::vec current_vec = arma::zeros( N_VAR + 2 );
    
    current_vec.subvec(0, N_VAR - 1) = trans(ES2(index)*X2.row(index)); // ksi2
    current_vec(N_VAR) = ES1ES2(index);                                 // gamma_1
    
    if(gs1){current_vec(N_VAR+1) = ES1ES2(index)*G(index);}                // psi_1
    expect_X2_vec += current_vec;

    arma::mat current_mat = arma::zeros( N_VAR + 2, N_VAR + 2 );
    
    current_mat.submat(0, 0, N_VAR - 1, N_VAR - 1) = trans(X2.row(index)) * X2.row(index);
    current_mat.submat(0, N_VAR, N_VAR-1, N_VAR) = ES1(index)*trans(X2.row(index));
    current_mat.submat(N_VAR, 0, N_VAR, N_VAR - 1) = ES1(index)*X2.row(index);
    current_mat(N_VAR, N_VAR) = ES1_2(index);
    
    if(gs1){
      current_mat.submat(0, N_VAR+1, N_VAR-1, N_VAR+1) = ES1(index)*trans(X2.row(index))*G(index);     // beta/psi
      current_mat.submat(N_VAR+1, 0, N_VAR+1, N_VAR-1) = ES1(index)*X2.row(index)*G(index);            // beta/psi
      current_mat(N_VAR, N_VAR+1) = ES1_2(index)*G(index);                                             // gamma/psi
      current_mat(N_VAR+1, N_VAR) = ES1_2(index)*G(index);                                             // gamma/psi
      current_mat(N_VAR+1, N_VAR+1) = ES1_2(index)*G(index)*G(index);                                  // psi/psi
    }

    expect_X2_mat += current_mat;
  } //for

  // if no interaction, set diagonal to 1 for inverse
  if(!gs1){
  expect_X2_mat(N_VAR+1, N_VAR+1) = 1;
  }
  
  arma::mat expect_X2_inv;
  bool invRes = arma::inv(expect_X2_inv, expect_X2_mat);
  if(!invRes){
    Rcout << "X2'X2 matrix not invertable\n";
    arma::vec temp = arma::zeros(N_VAR + 2);
    temp(0) = R_NaN;
    temp(N_VAR) = R_NaN;
    temp(N_VAR + 1) = R_NaN;
    return temp;
  }

  ksi2_gamma1_vec =  expect_X2_inv*expect_X2_vec;
  if(!gs1)ksi2_gamma1_vec(N_VAR + 1) = 0;
  return ksi2_gamma1_vec;
} //calc_ksi2_gamma1

//[[Rcpp::export]]
arma::vec calc_ksi1_twoseq( const arma::mat & X1,
                      const arma::vec & ES1)
{
  const unsigned int N = X1.n_rows;
  const unsigned int N_VAR = X1.n_cols;
  arma::vec expect_X1_vec = arma::zeros( N_VAR );
  arma::mat expect_X1_mat = arma::zeros( N_VAR, N_VAR );
  for ( unsigned int index = 0; index < N; ++index)
  {
    expect_X1_mat += trans(X1.row(index)) * X1.row(index);
    expect_X1_vec += trans(X1.row(index)) * ES1(index);
  } //for

  arma::mat expect_X1_inv;
  bool invRes = arma::inv(expect_X1_inv, expect_X1_mat);
  if(!invRes){
    Rcout << "X1'X1 matrix not invertable\n";
    arma::vec temp = arma::zeros(N_VAR);
    temp(0) = R_NaN;
    return temp;
  }
  return expect_X1_inv*expect_X1_vec;
} //calc_ksi1


//calc_tau_sqr : for Maximization step
//[[Rcpp::export]]
double calc_tau2_twoseq( const arma::vec & ksi_2, const arma::mat & X2, const arma::vec & G,
                         const double & gamma_1, const double psi_1, const bool & int_gs1,
                     const arma::vec & ES1, const arma::vec & ES1_2,
                     const arma::vec & ES2, const arma::vec & ES2_2,
                     const arma::vec & ES1ES2)
{
  const unsigned int N = X2.n_rows;
  const unsigned int N_VAR = X2.n_cols;
  return (1.0/(N - N_VAR - 1 - int_gs1*1))*accu(ES2_2 - 2.0*ES2%(X2*ksi_2) + pow(X2*ksi_2, 2.0) +
                                      pow(gamma_1 + psi_1*G, 2.0)%ES1_2 - 
                                          2.0*( (gamma_1 + psi_1*G)%ES1ES2 - (X2*ksi_2) % ES1 % (gamma_1 + psi_1*G) ));
} //calc_tau2_sqr

//[[Rcpp::export]]
double calc_tau1_twoseq( const arma::vec & ksi_1, const arma::mat & X1,
                           const arma::vec & ES1,
                           const arma::vec & ES1_2)
{
  const unsigned int N = X1.n_rows;
  const unsigned int N_VAR = X1.n_cols;
  return (1.0/(N - N_VAR))*accu(ES1_2 + pow(X1*ksi_1, 2.0) - 2.0*(X1*ksi_1)%ES1);
} //calc_tau_sqr


// Current location
// EM algorithm to iterate between Expectation and Maximization
//[[Rcpp::export]]
List twoSeqMed_EM(const arma::vec & Y, const arma::vec & G, const arma::vec & S1, const arma::vec & R1,
                  const arma::vec & S2, const arma::vec & R2, 
               const arma::mat & Z, const arma::mat & X1, const arma::mat & X2,
               const arma::vec & lowerS1, const arma::vec & upperS1,
               const arma::vec & lowerS2, const arma::vec & upperS2,
               bool int_gs1 = true, bool int_gs1_Y = true, bool int_gs2 = true,
               bool int_s1s2 = false, bool int_gs1s2 = false,
               const double convLimit = 1e-4, const double iterationLimit = 1e4,
               const int & nDivisions = 5, 
               const int MEASURE_TYPE_KNOWN = 1,
               const int MEASURE_TYPE_MISSING = 0,
               const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
               const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  // Initialize needed values;
  //const unsigned int N = Y.n_elem;

  const unsigned int N = Y.n_elem;
  
  arma::mat Zuse = arma::zeros(N, Z.n_cols + 1);
  arma::mat X1use = arma::zeros(N, X1.n_cols + 1);
  arma::mat X2use = arma::zeros(N, X2.n_cols + 1);
  
  
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
  if(X1.n_elem < N){
    X1use = arma::ones(N, 2);
    X1use.submat(0, 1, N-1, 1) = G;
  }else{
    X1use = arma::zeros(N, X1.n_cols + 1);
    X1use.submat(0, 0, N-1, 0) = X1.submat(0, 0, N-1, 0);
    X1use.submat(0, 1, N-1, 1) = G;
    X1use.submat(0, 2, N-1, X1use.n_cols - 1) = X1.submat(0, 1, N-1, X1.n_cols - 1);
  }
  if(X2.n_elem < N){
    X2use = arma::ones(N, 2);
    X2use.submat(0, 1, N-1, 1) = G;
  }else{
    X2use = arma::zeros(N, X2.n_cols + 1);
    X2use.submat(0, 0, N-1, 0) = X2.submat(0, 0, N-1, 0);
    X2use.submat(0, 1, N-1, 1) = G;
    X2use.submat(0, 2, N-1, X2use.n_cols - 1) = X2.submat(0, 1, N-1, X2.n_cols - 1);
  }
  
  arma::vec beta = arma::zeros(Zuse.n_cols);
  double gamma_1 = 0;
  double gamma_tilde = 0;
  double gamma_2 = 0;
  double psi_1 = 0;
  double psi_tilde = 0;
  double psi_2 = 0;
  double h1 = 0;
  double h2 = 0;
  
  arma::vec ksi_1 = arma::zeros(X1use.n_cols);
  arma::vec ksi_2 = arma::zeros(X2use.n_cols);
  double sig2 = arma::var(Y);
  double tau2_1 = arma::var(S1.elem(find(R1 == 1)));
  double tau2_2 = arma::var(S2.elem(find(R2 == 1)));

  bool converged = false;
  int iteration = 0;

  while(!converged & (iteration < iterationLimit)){

    // Create old holders;
    arma::vec oldBeta = beta;
    double oldGamma1 = gamma_1;
    double oldGammat = gamma_tilde;
    double oldGamma2 = gamma_2;
    double oldPsi1   = psi_1;
    double oldPsit   = psi_tilde;
    double oldPsi2   = psi_2;
    double oldSig2 = sig2;
    arma::vec oldKsi1 = ksi_1;
    arma::vec oldKsi2 = ksi_2;
    double oldTau2_1 = tau2_1;
    double oldTau2_2 = tau2_2;
    double oldH1 = h1;
    double oldH2 = h2;

    //Update Expectation;
    List expRes = calc_expectation_twoseq(gamma_1, psi_1, gamma_2, psi_2, 
                                          gamma_tilde, psi_tilde, h1, h2, 
                                          sig2, tau2_1, tau2_2,
                                          Y, Zuse, X1use, X2use, G,
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
    arma::vec beta_kappa_gamma2 = calc_beta_tilde_gamma2_twoseq(Y, Zuse, G,
                           expectS1, expectS1_sq, expectS2, expectS2_sq, expectS1S2, 
                           expectS12S2, expectS1S22, expectS12S22, int_gs1_Y, int_gs2, int_s1s2, int_gs1s2);

    beta = beta_kappa_gamma2.subvec(0, Zuse.n_cols - 1);
    gamma_tilde = beta_kappa_gamma2(Zuse.n_cols);
    psi_tilde = beta_kappa_gamma2(Zuse.n_cols + 1);
    gamma_2 = beta_kappa_gamma2(Zuse.n_cols + 2);
    psi_2 = beta_kappa_gamma2(Zuse.n_cols + 3);
    h1  = beta_kappa_gamma2(Zuse.n_cols + 4);
    h2  = beta_kappa_gamma2(Zuse.n_cols + 5);

      
    sig2 = calc_sig2_twoseq(gamma_tilde, psi_tilde, gamma_2, psi_2, h1, h2, 
                            int_gs1_Y, int_gs2, int_s1s2, int_gs1s2,
                            Y, beta, G, Zuse,
                           expectS1, expectS1_sq, expectS2, expectS2_sq, expectS1S2, 
                           expectS12S2, expectS1S22, expectS12S22);

      
    arma::vec ksi2_gamma1 = calc_ksi2_gamma1_twoseq(X2use, G, expectS1, expectS1_sq, expectS2, expectS1S2, expectS12S2, int_gs1);

    ksi_2 = ksi2_gamma1.subvec(0, X2use.n_cols - 1);
    gamma_1 = ksi2_gamma1(X2use.n_cols);
    psi_1 = ksi2_gamma1(X2use.n_cols + 1);

    ksi_1 = calc_ksi1_twoseq(X1use, expectS1);

    tau2_1 = calc_tau1_twoseq(ksi_1, X1use, expectS1, expectS1_sq);

    tau2_2 = calc_tau2_twoseq(ksi_2, X2use, G, gamma_1, psi_1, int_gs1, expectS1, expectS1_sq, expectS2, expectS2_sq, expectS1S2);
    
    // Check for errors
    if(std::isnan(gamma_1) | std::isnan(ksi_1(0)) | std::isnan(beta(0)) | std::isnan(ksi_2(0))){
      //Rcout << "Errors\n";
      gamma_1 = R_NaN;
      psi_1 = R_NaN;
      gamma_2 = R_NaN;
      psi_2 = R_NaN;
      sig2 = R_NaN;
      tau2_1 = R_NaN;
      tau2_2 = R_NaN;
      gamma_tilde = R_NaN;
      psi_tilde = R_NaN;
      h1 = R_NaN;
      h2 = R_NaN;
      return List::create(Named("beta") = beta, Named("gamma_1") = gamma_1, Named("psi_1") = psi_1, 
                                Named("gamma_2") = gamma_2, Named("psi_2") = psi_2,
                                Named("h1") = h1, Named("h2") = h2,
                                Named("gamma_tilde") = gamma_tilde, Named("psi_tilde") = psi_tilde,
                                Named("sig2") = sig2, Named("ksi_1") = ksi_1,
                                      Named("ksi_2") = ksi_2, Named("tau2_1") = tau2_1, Named("tau2_2") = tau2_2);
    }

    // Check for convergence;
    double maxDiff = -1;
    for(int betaIndex = 0; betaIndex < beta.n_elem; ++betaIndex){
      maxDiff = std::max(maxDiff,
                          std::max(fabs(beta(betaIndex) - oldBeta(betaIndex)),
                                   fabs(beta(betaIndex) - oldBeta(betaIndex)) / fabs(oldBeta(betaIndex))));
    }
    for(int ksiIndex = 0; ksiIndex < ksi_1.n_elem; ++ksiIndex){
      maxDiff = std::max(maxDiff,
                         std::max(fabs(ksi_1(ksiIndex) - oldKsi1(ksiIndex)),
                                  fabs(ksi_1(ksiIndex) - oldKsi1(ksiIndex)) / fabs(oldKsi1(ksiIndex))));
    }
    for(int ksiIndex = 0; ksiIndex < ksi_2.n_elem; ++ksiIndex){
      maxDiff = std::max(maxDiff,
                       std::max(fabs(ksi_2(ksiIndex) - oldKsi2(ksiIndex)),
                                fabs(ksi_2(ksiIndex) - oldKsi2(ksiIndex)) / fabs(oldKsi2(ksiIndex))));
    }
    maxDiff = std::max(maxDiff, std::max(fabs(sig2 - oldSig2), fabs(sig2 - oldSig2) / fabs(oldSig2)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma_1 - oldGamma1), fabs(gamma_1 - oldGamma1) / fabs(oldGamma1)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma_2 - oldGamma2), fabs(gamma_2 - oldGamma2) / fabs(oldGamma2)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma_tilde - oldGammat), fabs(gamma_tilde - oldGammat) / fabs(oldGammat)));
    maxDiff = std::max(maxDiff, std::max(fabs(tau2_1 - oldTau2_1), fabs(tau2_1 - oldTau2_1) / fabs(oldTau2_1)));
    maxDiff = std::max(maxDiff, std::max(fabs(tau2_2 - oldTau2_2), fabs(tau2_2 - oldTau2_2) / fabs(oldTau2_2)));
    
    if(int_gs1_Y)maxDiff = std::max(maxDiff, std::max(fabs(psi_tilde - oldPsit), fabs(psi_tilde - oldPsit) / fabs(oldPsit)));
    if(int_gs1)maxDiff = std::max(maxDiff, std::max(fabs(psi_1 - oldPsi1), fabs(psi_1 - oldPsi1) / fabs(oldPsi1)));
    if(int_gs2)maxDiff = std::max(maxDiff, std::max(fabs(psi_2 - oldPsi2), fabs(psi_2 - oldPsi2) / fabs(oldPsi2)));
    if(int_s1s2)maxDiff = std::max(maxDiff, std::max(fabs(h1 - oldH1), fabs(h1-oldH1) / fabs(oldH1)));
    if(int_gs1s2)maxDiff = std::max(maxDiff, std::max(fabs(h2 - oldH2), fabs(h2-oldH2) / fabs(oldH2)));

   converged = maxDiff < convLimit;
   iteration++;
  }
  if(iteration == iterationLimit){Rcout << "Algorithm failed to converge\n";}

  return List::create(Named("beta") = beta, Named("gamma_1") = gamma_1, Named("psi_1") = psi_1, 
                            Named("gamma_2") = gamma_2, Named("psi_2") = psi_2,
                            Named("h1") = h1, Named("h2") = h2,
                            Named("gamma_tilde") = gamma_tilde, Named("psi_tilde") = psi_tilde,
                            Named("sig2") = sig2, Named("ksi_1") = ksi_1,
                            Named("ksi_2") = ksi_2, Named("tau2_1") = tau2_1, Named("tau2_2") = tau2_2);
}


//[[Rcpp::export]]
arma::mat calc_Q_matrix_twoseq(const double & sig2, const double & tau2_1, const double & tau2_2,
                    const arma::mat & Z, const arma::mat & X1, const arma::mat & X2, const arma::vec & G,
                    const arma::vec & ES1, const arma::vec & ES1_2,
                    const arma::vec & ES2, const arma::vec & ES2_2,
                    const arma::vec & ES1S2, const arma::vec & ES12S2,
                    const arma::vec & ES1S22, const arma::vec & ES12S22,
                    bool int_gs1 = true, bool int_gs1_Y = true, bool int_gs2 = true,
                    bool int_s1s2 = false, bool int_gs1s2 = false)
{
  const unsigned int Z_VARS = Z.n_cols;
  const unsigned int X1_VARS = X1.n_cols;
  const unsigned int X2_VARS = X2.n_cols;
  const unsigned int N = Z.n_rows;

  // Z variables, two gamma and two psi with two h and sigma, X2 variables, gamma1 and psi1 with tau2, X1 variables, tau1

  arma::mat Q = arma::zeros( Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1);


  // Loop for summation elements
  for ( unsigned int index = 0; index < N; ++index )
  {
    // Beta - Z
    Q.submat(0,           0,           Z_VARS - 1, Z_VARS - 1) += (trans(Z.row(index)) * Z.row(index));
 
    // gamma_tilde (beta, gamma_tilde) - Z_VARS - S1 variable
    Q.submat(0,           Z_VARS,      Z_VARS - 1, Z_VARS    ) += (ES1(index)*trans(Z.row(index))) ;
    Q.submat(Z_VARS,      0,           Z_VARS,     Z_VARS - 1) += (ES1(index)*Z.row(index)) ;
    Q(Z_VARS,     Z_VARS) += ES1_2(index);
   
    if(int_gs1_Y){
      // psi_tilde (beta, gamma_tilde, psi_tilde) - Z_VARS + 1 - S1 * G variable
      Q.submat(0,           Z_VARS + 1,  Z_VARS - 1, Z_VARS + 1) += (ES1(index)*trans(Z.row(index))*G(index)) ;
      Q.submat(Z_VARS + 1,  0,           Z_VARS + 1, Z_VARS - 1) += (ES1(index)*Z.row(index)*G(index));
      Q(Z_VARS,     Z_VARS + 1) += ES1_2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS    ) += ES1_2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS + 1) += ES1_2(index)*G(index)*G(index) ;
    }
  
    // gamma_2 (beta, gamma_tilde, psi_tilde, gamma2) - Z_VARS + 2 - S2 variable
    Q.submat(0,           Z_VARS + 2,  Z_VARS - 1, Z_VARS + 2) += (ES2(index)*trans(Z.row(index))) ;
    Q.submat(Z_VARS + 2,  0,           Z_VARS + 2, Z_VARS - 1) += (ES2(index)*Z.row(index));
    Q(Z_VARS,     Z_VARS + 2) += ES1S2(index) ;
    Q(Z_VARS + 2, Z_VARS    ) += ES1S2(index) ;
    Q(Z_VARS + 1, Z_VARS + 2) += ES1S2(index)*G(index) ;
    Q(Z_VARS + 2, Z_VARS + 1) += ES1S2(index)*G(index) ;
    Q(Z_VARS + 2, Z_VARS + 2) += ES2_2(index) ;

    if(int_gs2){
      // psi_2 (beta, gamma_tilde, psi_tilde, gamma2, psi2) - Z_VARS + 3 - S2 * G variable
      Q.submat(0,           Z_VARS + 3,  Z_VARS - 1, Z_VARS + 3) += (ES2(index)*trans(Z.row(index))*G(index)) ;
      Q.submat(Z_VARS + 3,  0,           Z_VARS + 3, Z_VARS - 1) += (ES2(index)*Z.row(index)*G(index));
      Q(Z_VARS,     Z_VARS + 3) += ES1S2(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS    ) += ES1S2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS + 3) += ES1S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 1) += ES1S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 3) += ES2_2(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 2) += ES2_2(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 3) += ES2_2(index)*G(index)*G(index) ;
    }
  
    if(int_s1s2){
      // h1 (beta, gamma_tilde, psi_tilde, gamma2, psi2, h1) - Z_VARS + 4 - S1*S2  variable
      Q.submat(0,           Z_VARS + 4,  Z_VARS - 1, Z_VARS + 4) += (ES1S2(index)*trans(Z.row(index))) ;
      Q.submat(Z_VARS + 4,  0,           Z_VARS + 4, Z_VARS - 1) += (ES1S2(index)*Z.row(index));
      Q(Z_VARS,     Z_VARS + 4) += ES12S2(index) ;
      Q(Z_VARS + 4, Z_VARS    ) += ES12S2(index) ;
      Q(Z_VARS + 1, Z_VARS + 4) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 1) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 4) += ES1S22(index) ;
      Q(Z_VARS + 4, Z_VARS + 2) += ES1S22(index) ;
      Q(Z_VARS + 3, Z_VARS + 4) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 3) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 4) += ES12S22(index) ;
    }
    
    if(int_gs1s2){
      // h2 (beta, gamma_tilde, psi_tilde, gamma2, psi2, h1, h2) - Z_VARS + 5 - S1*S2*G  variable
      Q.submat(0,           Z_VARS + 5,  Z_VARS - 1, Z_VARS + 5) += (ES1S2(index)*trans(Z.row(index))*G(index)) ;
      Q.submat(Z_VARS + 5,  0,           Z_VARS + 5, Z_VARS - 1) += (ES1S2(index)*Z.row(index)*G(index));
      Q(Z_VARS,     Z_VARS + 5) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS    ) += ES12S2(index)*G(index) ;
      Q(Z_VARS + 1, Z_VARS + 5) += ES12S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 1) += ES12S2(index)*G(index)*G(index) ;
      Q(Z_VARS + 2, Z_VARS + 5) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 2) += ES1S22(index)*G(index) ;
      Q(Z_VARS + 3, Z_VARS + 5) += ES1S22(index)*G(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 3) += ES1S22(index)*G(index)*G(index) ;
      Q(Z_VARS + 4, Z_VARS + 5) += ES12S22(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 4) += ES12S22(index)*G(index) ;
      Q(Z_VARS + 5, Z_VARS + 5) += ES12S22(index)*G(index)*G(index) ;
    }
  
    // ksi2 variables - Z_VARS + 7 for start of ksi2, Z_VARS + X_VARS + 7 for start of ksi2
    // ksi2  - Z_VARS + 7 and X2_VARS long
    Q.submat(Z_VARS + 7, Z_VARS + 7, Z_VARS + 7 + X2_VARS - 1, Z_VARS + 7 + X2_VARS - 1) += (trans(X2.row(index)) * X2.row(index));
   
    // gamma_1 - Z_VARS + 7 + X2_VARS
    Q.submat(Z_VARS + 7, Z_VARS + 7 + X2_VARS, Z_VARS + 7 + X2_VARS - 1, Z_VARS + 7 + X2_VARS) += (ES1(index)*trans(X2.row(index)));
    Q.submat(Z_VARS + 7 + X2_VARS, Z_VARS + 7, Z_VARS + 7 + X2_VARS, Z_VARS + 7 + X2_VARS - 1) += (ES1(index)*X2.row(index));
    Q(Z_VARS + 7 + X2_VARS, Z_VARS + 7 + X2_VARS) += ES1_2(index);

    if(int_gs1){
      // psi_1 - Z_VARS + 7 + X2_VARS + 1
      Q.submat(Z_VARS + 7, Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7 + X2_VARS - 1, Z_VARS + 7 + X2_VARS + 1) += (ES1(index)*trans(X2.row(index))*G(index));
      Q.submat(Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7, Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7 + X2_VARS - 1) += (ES1(index)*X2.row(index)*G(index));
      Q(Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7 + X2_VARS) += ES1_2(index)*G(index);
      Q(Z_VARS + 7 + X2_VARS, Z_VARS + 7 + X2_VARS + 1) += ES1_2(index)*G(index);
      Q(Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7 + X2_VARS + 1) += ES1_2(index)*G(index)*G(index);
    }
   
    // ksi1 - Z_VARS + 7 + X2_VARS + 3
    Q.submat(Z_VARS + 7 + X2_VARS + 3, Z_VARS + 7 + X2_VARS + 3, 
             Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1) += (trans(X1.row(index)) * X1.row(index));
 }

  Q.submat(0, 0, Z_VARS + 5, Z_VARS + 5) /= sig2;
  Q(Z_VARS + 6, Z_VARS + 6) = N / (2.0*std::pow(sig2, 2.0));
  Q.submat(Z_VARS + 7, Z_VARS + 7, Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7 + X2_VARS + 1) /= tau2_2;
  Q(Z_VARS + 7 + X2_VARS + 2, Z_VARS + 7 + X2_VARS + 2) = N / (2.0*std::pow(tau2_2, 2.0));
  Q.submat(Z_VARS + 7 + X2_VARS + 3, Z_VARS + 7 + X2_VARS + 3, 
           Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1) /= tau2_1;
  Q(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, Z_VARS + 7 + X2_VARS + 3 + X1_VARS) = N / (2.0*std::pow(tau2_1, 2.0));

  return Q;
}

// Calc V mat
// calc U mat
// calc_V_mat returns V matrix for a single observation
// need 3 functions for each missing S scenario
// Name denotes value missing (S1, S2, S1S2 for both)
//[[Rcpp::export]]
arma::mat calc_V_mat_twoseq_S1(const double & gamma_tilde, const double & psi_tilde, 
                               const double & gamma_2, const double & psi_2, 
                               const double & gamma_1, const double & psi_1,
                               const double & h1, const double & h2,
                               const double & sig2, const double & tau2_1, const double & tau2_2,
                     const double & Y, const double & G, const double & S_obs, 
                     const arma::vec & beta,
                     const arma::vec & ksi_1, const arma::vec & ksi_2,
                     const arma::rowvec & Z,
                     const arma::rowvec & X1, const arma::rowvec & X2)
{
    const unsigned int Z_VARS = beta.n_elem;
    const unsigned int X1_VARS = ksi_1.n_elem;
    const unsigned int X2_VARS = ksi_2.n_elem;
    const double Y_residual = Y - as_scalar(Z*beta) - (gamma_2 + psi_2*G)*S_obs;
    const double S_residual = S_obs - as_scalar(X2*ksi_2);
    const double s1InterTermsY = gamma_tilde + G*psi_tilde + S_obs*(h1 + G*h2);
    const double s1InterTermsS2 = gamma_1 + G*psi_1;
    const double ksi1_X = as_scalar(X1*ksi_1);

    arma::mat V_mat = arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, 3);
   
    // Change and do by row (variable) as some rows are multiples of previous rows;
    // Beta rows
    V_mat.submat(0, 0, Z_VARS-1, 0) = (Y_residual) * trans(Z) / sig2;
    V_mat.submat(0, 1, Z_VARS-1, 1) =  -(s1InterTermsY)*trans(Z) / sig2;
    V_mat.submat(0, 2, Z_VARS-1, 2) =  arma::zeros(Z_VARS);

    // gamma_tilde row
    V_mat(Z_VARS, 0) = 0;
    V_mat(Z_VARS, 1) = Y_residual / sig2;
    V_mat(Z_VARS, 2) = -(s1InterTermsY) / sig2;

    // psi_tilde row - gamma_tilde row times G
    V_mat.submat(Z_VARS + 1, 0, Z_VARS + 1, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * G;
   
    // gamma_2 row
    V_mat(Z_VARS + 2, 0) = (Y_residual) * S_obs / sig2;
    V_mat(Z_VARS + 2, 1) = -(s1InterTermsY)*S_obs / sig2;
    V_mat(Z_VARS + 2, 2) = 0;
  
    // psi_2 row - gamma_2 row times G
    V_mat.submat(Z_VARS + 3, 0, Z_VARS + 3, 2) = V_mat.submat(Z_VARS + 2, 0, Z_VARS + 2, 2) * G;
    // h1 row - gamma_tilde row times S_obs
    V_mat.submat(Z_VARS + 4, 0, Z_VARS + 4, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * S_obs;
    // h2 row - gamma_tilde row times S_obs*G
    V_mat.submat(Z_VARS + 5, 0, Z_VARS + 5, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * S_obs * G;
   
    //sig2 rows
    V_mat(Z_VARS + 6, 0) = -1 / (2.0*sig2) + std::pow(Y_residual, 2.0)/(2.0*sig2*sig2);
    V_mat(Z_VARS + 6, 1) = -(s1InterTermsY) * Y_residual / (sig2*sig2);
    V_mat(Z_VARS + 6, 2) = std::pow(s1InterTermsY, 2.0) / (2.0* sig2*sig2);
  
    // ksi2 rows
    V_mat.submat(Z_VARS + 7, 0, Z_VARS + 7 + X2_VARS - 1, 0) = S_residual * trans(X2) / (tau2_2);
    V_mat.submat(Z_VARS + 7, 1, Z_VARS + 7 + X2_VARS - 1, 1) = -s1InterTermsS2 * trans(X2) / (tau2_2);
    V_mat.submat(Z_VARS + 7, 2, Z_VARS + 7 + X2_VARS - 1, 2) = arma::zeros(X2_VARS);
 
    // gamma_1 rows
    V_mat(Z_VARS + 7 + X2_VARS, 0) = 0;
    V_mat(Z_VARS + 7 + X2_VARS, 1) = S_residual / (tau2_2);
    V_mat(Z_VARS + 7 + X2_VARS, 2) = -s1InterTermsS2 / (tau2_2);
  
    // psi_1 row - gamma_1 row G
    V_mat.submat(Z_VARS + 7 + X2_VARS + 1, 0, Z_VARS + 7 + X2_VARS + 1, 2) =
                V_mat.submat(Z_VARS + 7 + X2_VARS, 0, Z_VARS + 7 + X2_VARS, 2) *  G;
    
    // tau2_2 rows
    V_mat(Z_VARS + 7 + X2_VARS + 2, 0) = -1 / (2.0*tau2_2) + std::pow(S_residual, 2.0) / (2.0*tau2_2*tau2_2);
    V_mat(Z_VARS + 7 + X2_VARS + 2, 1) = -S_residual*s1InterTermsS2  / (tau2_2*tau2_2);
    V_mat(Z_VARS + 7 + X2_VARS + 2, 2) = std::pow(s1InterTermsS2, 2.0) / (2.0*tau2_2*tau2_2);
  
    // ksi1 rows
    V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 0, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 0) = -ksi1_X * trans(X1) / (tau2_1);
    V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 1) = trans(X1) / (tau2_1);
    V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 2, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 2) = arma::zeros(X1_VARS);
  
    // tau2_1 rows
    V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 0) = -1 / (2.0*tau2_1) + std::pow(ksi1_X, 2.0) / (2.0*tau2_1*tau2_1);
    V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 1) = -ksi1_X  / (tau2_1*tau2_1);
    V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 2) = 1/ (2.0*tau2_1*tau2_1);

    return V_mat;
} //calc_V_mat_twoseq_V1

//[[Rcpp::export]]
arma::mat calc_V_mat_twoseq_S2(const double & gamma_tilde, const double & psi_tilde, 
                               const double & gamma_2, const double & psi_2, 
                               const double & gamma_1, const double & psi_1,
                               const double & h1, const double & h2,
                               const double & sig2, const double & tau2_1, const double & tau2_2,
                               const double & Y, const double & G, const double & S_obs, 
                               const arma::vec & beta,
                               const arma::vec & ksi_1, const arma::vec & ksi_2,
                               const arma::rowvec & Z,
                               const arma::rowvec & X1, const arma::rowvec & X2)
{
  const unsigned int Z_VARS = beta.n_elem;
  const unsigned int X1_VARS = ksi_1.n_elem;
  const unsigned int X2_VARS = ksi_2.n_elem;
  const double Y_residual = Y - as_scalar(Z*beta) - (gamma_tilde + psi_tilde*G)*S_obs;
  const double s2InterTermsY = gamma_2 + G*psi_2 + S_obs*(h1 + G*h2);
  const double ksi2_X_S1 = as_scalar(X2*ksi_2) + S_obs*(gamma_1+psi_1*G);
  const double ksi1_X = as_scalar(X1*ksi_1);
  
  arma::mat V_mat = arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, 3);
  
  // Change and do by row (variable) as some rows are multiples of previous rows;
  // Beta rows
  V_mat.submat(0, 0, Z_VARS-1, 0) = (Y_residual) * trans(Z) / sig2;
  V_mat.submat(0, 1, Z_VARS-1, 1) =  -(s2InterTermsY)*trans(Z) / sig2;
  V_mat.submat(0, 2, Z_VARS-1, 2) =  arma::zeros(Z_VARS);
  
  // gamma_tilde row
  V_mat(Z_VARS, 0) = Y_residual*S_obs / sig2;
  V_mat(Z_VARS, 1) = -(s2InterTermsY)*S_obs / sig2;
  V_mat(Z_VARS, 2) = 0;
  
  // psi_tilde row - gamma_tilde row times G
  V_mat.submat(Z_VARS + 1, 0, Z_VARS + 1, 2) = V_mat.submat(Z_VARS, 0, Z_VARS, 2) * G;
  
  // gamma_2 row
  V_mat(Z_VARS + 2, 0) = 0;
  V_mat(Z_VARS + 2, 1) = (Y_residual) / sig2;
  V_mat(Z_VARS + 2, 2) = -(s2InterTermsY) / sig2;
  
  // psi_2 row - gamma_2 row times G
  V_mat.submat(Z_VARS + 3, 0, Z_VARS + 3, 2) = V_mat.submat(Z_VARS + 2, 0, Z_VARS + 2, 2) * G;
  // h1 row - gamma_2 row times S_obs
  V_mat.submat(Z_VARS + 4, 0, Z_VARS + 4, 2) = V_mat.submat(Z_VARS + 2, 0, Z_VARS + 2, 2) * S_obs;
  // h2 row - gamma_2 row times S_obs*G
  V_mat.submat(Z_VARS + 5, 0, Z_VARS + 5, 2) = V_mat.submat(Z_VARS + 2, 0, Z_VARS + 2, 2) * S_obs * G;
  
  //sig2 rows
  V_mat(Z_VARS + 6, 0) = -1 / (2.0*sig2) + std::pow(Y_residual, 2.0)/(2.0*sig2*sig2);
  V_mat(Z_VARS + 6, 1) = -(s2InterTermsY) * Y_residual / (sig2*sig2);
  V_mat(Z_VARS + 6, 2) = std::pow(s2InterTermsY, 2.0) / (2.0* sig2*sig2);
  
  // ksi2 rows
  V_mat.submat(Z_VARS + 7, 0, Z_VARS + 7 + X2_VARS - 1, 0) = -ksi2_X_S1 * trans(X2) / (tau2_2);
  V_mat.submat(Z_VARS + 7, 1, Z_VARS + 7 + X2_VARS - 1, 1) = trans(X2) / (tau2_2);
  V_mat.submat(Z_VARS + 7, 2, Z_VARS + 7 + X2_VARS - 1, 2) = arma::zeros(X2_VARS);
  
  // gamma_1 rows
  V_mat(Z_VARS + 7 + X2_VARS, 0) = -ksi2_X_S1*S_obs / (tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS, 1) = S_obs / (tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS, 2) = 0;
  
  // psi_1 row - gamma_1 row G
  V_mat.submat(Z_VARS + 7 + X2_VARS + 1, 0, Z_VARS + 7 + X2_VARS + 1, 2) =
    V_mat.submat(Z_VARS + 7 + X2_VARS, 0, Z_VARS + 7 + X2_VARS, 2) *  G;
  
  // tau2_2 rows
  V_mat(Z_VARS + 7 + X2_VARS + 2, 0) = -1 / (2.0*tau2_2) + std::pow(ksi2_X_S1, 2.0) / (2.0*tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 1) = -ksi2_X_S1  / (tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 2) = 1 / (2.0*tau2_2*tau2_2);
  
  // ksi1 rows
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 0, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 0) = (S_obs-ksi1_X) * trans(X1) / (tau2_1);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 1) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 2, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 2) = arma::zeros(X1_VARS);
  
  // tau2_1 rows
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 0) = -1 / (2.0*tau2_1) + std::pow(S_obs-ksi1_X, 2.0) / (2.0*tau2_1*tau2_1);
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 1) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 2) = 0;
  
  return V_mat;
} //calc_V_mat_twoseq_V2


//[[Rcpp::export]]
arma::mat calc_V_mat_twoseq_S1S2(const double & gamma_tilde, const double & psi_tilde, 
                                 const double & gamma_2, const double & psi_2, 
                                 const double & gamma_1, const double & psi_1,
                                 const double & h1, const double & h2,
                                 const double & sig2, const double & tau2_1, const double & tau2_2,
                                 const double & Y, const double & G,
                                 const arma::vec & beta,
                                 const arma::vec & ksi_1, const arma::vec & ksi_2,
                                 const arma::rowvec & Z,
                                 const arma::rowvec & X1, const arma::rowvec & X2)
{
  const unsigned int Z_VARS = beta.n_elem;
  const unsigned int X1_VARS = ksi_1.n_elem;
  const unsigned int X2_VARS = ksi_2.n_elem;
  const double Y_residual = Y - as_scalar(Z*beta); 
  const double s1InterTermsY = gamma_tilde + psi_tilde*G;
  const double s2InterTermsY = gamma_2 + G*psi_2;
  const double hTerms = h1 + G*h2;
  const double s1InterTermsS2 = (gamma_1+psi_1*G);
  const double ksi2_X = as_scalar(X2*ksi_2);
  const double ksi1_X = as_scalar(X1*ksi_1);
  
  arma::mat V_mat = arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, 9);
  
  // Change and do by row (variable) as some rows are multiples of previous rows;
  // Beta rows
  V_mat.submat(0, 0, Z_VARS-1, 0) = (Y_residual) * trans(Z) / sig2;
  V_mat.submat(0, 1, Z_VARS-1, 1) =  -(s1InterTermsY)*trans(Z) / sig2;
  V_mat.submat(0, 2, Z_VARS-1, 2) =  arma::zeros(Z_VARS);
  V_mat.submat(0, 3, Z_VARS-1, 3) =  -(s2InterTermsY)*trans(Z) / sig2;
  V_mat.submat(0, 4, Z_VARS-1, 4) =  arma::zeros(Z_VARS);
  V_mat.submat(0, 5, Z_VARS-1, 5) =  -(hTerms)*trans(Z) / sig2;
  V_mat.submat(0, 6, Z_VARS-1, 6) =  arma::zeros(Z_VARS);
  V_mat.submat(0, 7, Z_VARS-1, 7) =  arma::zeros(Z_VARS);
  V_mat.submat(0, 8, Z_VARS-1, 8) =  arma::zeros(Z_VARS);
  
  
  // gamma_tilde row
  V_mat(Z_VARS, 0) = 0;
  V_mat(Z_VARS, 1) = Y_residual / sig2;
  V_mat(Z_VARS, 2) = -(s1InterTermsY) / sig2;
  V_mat(Z_VARS, 3) = 0;
  V_mat(Z_VARS, 4) = 0;
  V_mat(Z_VARS, 5) = -s2InterTermsY / sig2;
  V_mat(Z_VARS, 6) = - hTerms / sig2;
  V_mat(Z_VARS, 7) = 0;
  V_mat(Z_VARS, 8) = 0;
  
  // psi_tilde row - gamma_tilde row times G
  V_mat.submat(Z_VARS + 1, 0, Z_VARS + 1, 8) = V_mat.submat(Z_VARS, 0, Z_VARS, 8) * G;
  
  // gamma_2 row
  V_mat(Z_VARS + 2, 0) = 0;
  V_mat(Z_VARS + 2, 1) = 0;
  V_mat(Z_VARS + 2, 2) = 0;
  V_mat(Z_VARS + 2, 3) = (Y_residual) / sig2;
  V_mat(Z_VARS + 2, 4) = -(s2InterTermsY)/ sig2;
  V_mat(Z_VARS + 2, 5) = -s1InterTermsY / sig2;
  V_mat(Z_VARS + 2, 6) = 0;
  V_mat(Z_VARS + 2, 7) = -hTerms / sig2;
  V_mat(Z_VARS + 2, 8) = 0;
  
  // psi_2 row - gamma_2 row times G
  V_mat.submat(Z_VARS + 3, 0, Z_VARS + 3, 8) = V_mat.submat(Z_VARS + 2, 0, Z_VARS + 2, 8) * G;
  // h1 row;
  V_mat(Z_VARS + 4, 0) = 0;
  V_mat(Z_VARS + 4, 1) = 0;
  V_mat(Z_VARS + 4, 2) = 0;
  V_mat(Z_VARS + 4, 3) = 0;
  V_mat(Z_VARS + 4, 4) = 0; 
  V_mat(Z_VARS + 4, 5) = (Y_residual) / sig2; 
  V_mat(Z_VARS + 4, 6) = -s1InterTermsY / sig2;
  V_mat(Z_VARS + 4, 7) = -(s2InterTermsY)/ sig2;
  V_mat(Z_VARS + 4, 8) = -hTerms / sig2;
  // h2 row - h1 row times G
  V_mat.submat(Z_VARS + 5, 0, Z_VARS + 5, 8) = V_mat.submat(Z_VARS + 4, 0, Z_VARS + 4, 8) * G;
  
  //sig2 rows
  V_mat(Z_VARS + 6, 0) = -1 / (2.0*sig2) + std::pow(Y_residual, 2.0)/(2.0*sig2*sig2);
  V_mat(Z_VARS + 6, 1) = -(s1InterTermsY) * Y_residual / (sig2*sig2);
  V_mat(Z_VARS + 6, 2) = std::pow(s1InterTermsY, 2.0) / (2.0* sig2*sig2);
  V_mat(Z_VARS + 6, 3) = -(s2InterTermsY) * Y_residual / (sig2*sig2);
  V_mat(Z_VARS + 6, 4) = std::pow(s2InterTermsY, 2.0) / (2.0* sig2*sig2);
  V_mat(Z_VARS + 6, 5) = ( (s1InterTermsY*s2InterTermsY) - (hTerms*Y_residual) ) / (sig2*sig2);
  V_mat(Z_VARS + 6, 6) = (s1InterTermsY) * hTerms / (sig2*sig2);
  V_mat(Z_VARS + 6, 7) = (s2InterTermsY) * hTerms / (sig2*sig2);
  V_mat(Z_VARS + 6, 8) = std::pow(hTerms, 2.0) / (2.0* sig2*sig2);
  
  
  // ksi2 rows
  V_mat.submat(Z_VARS + 7, 0, Z_VARS + 7 + X2_VARS - 1, 0) = -ksi2_X * trans(X2) / (tau2_2);
  V_mat.submat(Z_VARS + 7, 1, Z_VARS + 7 + X2_VARS - 1, 1) = -s1InterTermsS2*trans(X2) / (tau2_2);
  V_mat.submat(Z_VARS + 7, 2, Z_VARS + 7 + X2_VARS - 1, 2) = arma::zeros(X2_VARS);
  V_mat.submat(Z_VARS + 7, 3, Z_VARS + 7 + X2_VARS - 1, 3) = trans(X2) / tau2_2;
  V_mat.submat(Z_VARS + 7, 4, Z_VARS + 7 + X2_VARS - 1, 4) = arma::zeros(X2_VARS);
  V_mat.submat(Z_VARS + 7, 5, Z_VARS + 7 + X2_VARS - 1, 5) = arma::zeros(X2_VARS);
  V_mat.submat(Z_VARS + 7, 6, Z_VARS + 7 + X2_VARS - 1, 6) = arma::zeros(X2_VARS);
  V_mat.submat(Z_VARS + 7, 7, Z_VARS + 7 + X2_VARS - 1, 7) = arma::zeros(X2_VARS);
  V_mat.submat(Z_VARS + 7, 8, Z_VARS + 7 + X2_VARS - 1, 8) = arma::zeros(X2_VARS);
  // gamma_1 rows
  V_mat(Z_VARS + 7 + X2_VARS, 0) = 0;
  V_mat(Z_VARS + 7 + X2_VARS, 1) = -ksi2_X / (tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS, 2) = -s1InterTermsS2 / tau2_2;
  V_mat(Z_VARS + 7 + X2_VARS, 3) = 0;
  V_mat(Z_VARS + 7 + X2_VARS, 4) = 0;
  V_mat(Z_VARS + 7 + X2_VARS, 5) =   1.0 / (tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS, 6) = 0;
  V_mat(Z_VARS + 7 + X2_VARS, 7) = 0;
  V_mat(Z_VARS + 7 + X2_VARS, 8) = 0;
  
  // psi_1 row - gamma_1 row G
  V_mat.submat(Z_VARS + 7 + X2_VARS + 1, 0, Z_VARS + 7 + X2_VARS + 1, 8) =
    V_mat.submat(Z_VARS + 7 + X2_VARS, 0, Z_VARS + 7 + X2_VARS, 8) *  G;
  
  // tau2_2 rows
  V_mat(Z_VARS + 7 + X2_VARS + 2, 0) = -1 / (2.0*tau2_2) + std::pow(ksi2_X, 2.0) / (2.0*tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 1) = ksi2_X*s1InterTermsS2  / (tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 2) = std::pow(s1InterTermsS2, 2.0)  / (2.0*tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 3) = -ksi2_X / (tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 4) = 1 / (2.0*tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 5) = -s1InterTermsS2 / (tau2_2*tau2_2);
  V_mat(Z_VARS + 7 + X2_VARS + 2, 6) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 2, 7) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 2, 8) = 0;
  
  // ksi1 rows
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 0, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 0) = (-ksi1_X) * trans(X1) / (tau2_1);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 1) = trans(X1) / (tau2_1);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 2, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 2) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 3, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 3) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 4, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 4) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 5, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 5) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 6, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 6) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 7, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 7) = arma::zeros(X1_VARS);
  V_mat.submat(Z_VARS + 7 + X2_VARS + 3, 8, Z_VARS + 7 + X2_VARS + 3 + X1_VARS - 1, 8) = arma::zeros(X1_VARS);
  
  // tau2_1 rows
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 0) = -1 / (2.0*tau2_1) + std::pow(ksi1_X, 2.0) / (2.0*tau2_1*tau2_1);
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 1) = -ksi1_X / (tau2_1*tau2_1);
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 2) = 1 / (2.0*tau2_1*tau2_1);
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 3) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 4) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 5) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 6) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 7) = 0;
  V_mat(Z_VARS + 7 + X2_VARS + 3 + X1_VARS, 8) = 0;
  
  return V_mat;
}

// Calculate the U Matrix for a single observation
// Need two U matrix functions: one for single missing, one for both missing
// Need to know detection limits and where observation falls
//[[Rcpp::export]]
arma::mat calc_U_mat_twoseq_S1miss(const double & gamma_tilde, const double & psi_tilde, 
                                   const double & gamma_2, const double & psi_2, 
                                   const double & gamma_1, const double & psi_1,
                                   const double & h1, const double & h2,
                                   const double & sig2, const double & tau2_1, const double & tau2_2,
                                   const double & Y, const double & G, const double & S_obs, const double & R,
                                   const arma::vec & beta,
                                   const arma::vec & ksi_1, const arma::vec & ksi_2,
                                   const arma::rowvec & Z,
                                   const arma::rowvec & X1, const arma::rowvec & X2,
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
  
  const double a_val = calc_a1_val_twoseq(gamma_tilde, psi_tilde, h1, h2, sig2, 
                                           gamma_1, psi_1, tau2_2, tau2_1, S_obs, G);
  const double b_val = a_val*calc_b1_val_twoseq(gamma_1, psi_1, gamma_tilde, psi_tilde, gamma_2, psi_2, h1, h2, 
                                                  sig2, tau2_1,  tau2_2,
                                                  beta, ksi_1, ksi_2,
                                                  Y, Z, G, X1, X2, S_obs);
  if ( R == MEASURE_TYPE_MISSING)
  {
    const double mu_1 = b_val;
    const double mu_2 = b_val * b_val + a_val;
    const double mu_3 = 3*a_val*b_val + pow(b_val, 3);
    const double mu_4 = 3*a_val*a_val + 6*a_val*b_val*b_val + pow(b_val, 4);
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

//[[Rcpp::export]]
arma::mat calc_U_mat_twoseq_S2miss(const double & gamma_tilde, const double & psi_tilde, 
                                   const double & gamma_2, const double & psi_2, 
                                   const double & gamma_1, const double & psi_1,
                                   const double & h1, const double & h2,
                                   const double & sig2, const double & tau2_1, const double & tau2_2,
                                   const double & Y, const double & G, const double & S_obs, const double & R,
                                   const arma::vec & beta,
                                   const arma::vec & ksi_1, const arma::vec & ksi_2,
                                   const arma::rowvec & Z,
                                  const arma::rowvec & X2,
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
  const double a_val = calc_a2_val_twoseq(gamma_2, psi_2, h1, h2, S_obs, G, sig2, tau2_2);
  const double b_val = a_val*calc_b2_val_twoseq(gamma_2, psi_2, h1, h2, gamma_tilde, psi_tilde, beta, 
                                                  gamma_1, psi_1, ksi_2, sig2, tau2_2,
                                                  Y, G, Z, X2, S_obs);
    if ( R == MEASURE_TYPE_MISSING)
    {
      const double mu_1 = b_val;
      const double mu_2 = b_val * b_val + a_val;
      const double mu_3 = 3*a_val*b_val + pow(b_val, 3);
      const double mu_4 = 3*a_val*a_val + 6*a_val*b_val*b_val + pow(b_val, 4);
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

//[[Rcpp::export]]
arma::mat calc_U_mat_twoseq_twomiss(const double & gamma_tilde, const double & psi_tilde, 
                                    const double & gamma_2, const double & psi_2, 
                                    const double & gamma_1, const double & psi_1,
                                    const double & h1, const double & h2,
                                    const double & sig2, const double & tau2_1, const double & tau2_2,
                                    const double & Y, const double & G, const double & R1, const double & R2,
                                    const arma::vec & beta,
                                    const arma::vec & ksi_1, const arma::vec & ksi_2,
                                    const arma::rowvec & Z,
                                    const arma::rowvec & X1, const arma::rowvec & X2,
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

  if ( R1 == MEASURE_TYPE_MISSING & R2 == MEASURE_TYPE_MISSING)
  {
    double mu1, mu1_2, mu1_3, mu1_4, mu2, mu2_2, mu2_3, mu2_4,
    mu1_mu2, mu12_mu2, mu13_mu2, mu14_mu2, mu1_mu22, mu1_mu23, mu1_mu24,
    mu12_mu22, mu12_mu23, mu12_mu24, mu13_mu22, mu14_mu22,
    mu13_mu23, mu13_mu24, mu14_mu23, mu14_mu24;
    if(h1 == 0 & h2 == 0){
      const double c1_val = calc_c1_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, 
                                               gamma_tilde, psi_tilde, tau2_1,  tau2_2, sig2, G);
      
      const double c2_val = calc_c2_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, 
                                               gamma_tilde, psi_tilde, tau2_1, tau2_2, sig2, G);
      
      const double b = calc_corr_b_twoseq(gamma_1, psi_1, gamma_2, psi_2, gamma_tilde, psi_tilde, 
                                          tau2_2, sig2, G, c1_val, c2_val);
      
      const double rho = calc_corr_rho_twoseq(b);
      
      const double d1_val = calc_d1_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, gamma_tilde, psi_tilde, 
                                               rho, tau2_1,  tau2_2, sig2,
                                               Y,  G, Z, X1, X2,
                                               beta, ksi_1, ksi_2, c1_val, c2_val);
      const double d2_val = calc_d2_val_twoseq(gamma_1, psi_1, gamma_2, psi_2, gamma_tilde, psi_tilde, 
                                               rho, tau2_1,  tau2_2, sig2,
                                               Y,  G, Z, X1, X2,
                                               beta, ksi_1, ksi_2, c1_val, c2_val);
      mu1 = d1_val;
      mu1_2 = d1_val*d1_val + c1_val;
      mu1_3 = 3.0*c1_val*d1_val + d1_val*d1_val*d1_val;
      mu1_4 = 3.0*c1_val*c1_val + 6.0*c1_val*d1_val*d1_val + d1_val*d1_val*d1_val*d1_val;
      
      mu2 = d2_val;
      mu2_2 = d2_val*d2_val + c2_val;
      mu2_3 = 3.0*c2_val*d2_val + d2_val*d2_val*d2_val;
      mu2_4 = 3.0*c2_val*c2_val + 6.0*c2_val*d2_val*d2_val + d2_val*d2_val*d2_val*d2_val;
      
      mu1_mu2 = std::sqrt(c1_val*c2_val)*rho + d1_val*d2_val;
      mu12_mu2 = 2.0*d1_val*std::sqrt(c1_val*c2_val)*rho + d2_val*(d1_val*d1_val + c1_val);
      mu13_mu2 = 3.0*std::sqrt(c1_val)*c1_val*rho*std::sqrt(c2_val) + 
                    3.0*c1_val*d1_val*d2_val +
                    3.0*d1_val*d1_val*rho*std::sqrt(c1_val*c2_val) + 
                    d1_val*d1_val*d1_val*d2_val;
      mu14_mu2 = 12.0*c1_val*std::sqrt(c1_val*c2_val)*d1_val +
                    4.0*std::pow(d1_val, 3.0)*rho*std::sqrt(c1_val*c2_val) +
                    6.0*c1_val*d2_val*std::pow(d1_val, 2.0) +
                    3.0*std::pow(c1_val, 2.0)*d2_val + 
                    std::pow(d1_val, 4.0)*d2_val;
      
      mu1_mu22 = 2.0*d2_val*std::sqrt(c1_val*c2_val)*rho + 
                    d1_val*(d2_val*d2_val + c2_val);
      mu1_mu23 = 3.0*std::sqrt(c2_val)*c2_val*rho*std::sqrt(c1_val) + 
                    3.0*c2_val*d1_val*d2_val +
                    3.0*d2_val*d2_val*rho*std::sqrt(c1_val*c2_val) + 
                    d1_val*d2_val*d2_val*d2_val;
      mu1_mu24 = 12.0*c2_val*std::sqrt(c1_val*c2_val)*d2_val +
                    4.0*std::pow(d2_val, 3.0)*rho*std::sqrt(c1_val*c2_val) +
                    6.0*c2_val*d1_val*std::pow(d2_val, 2.0) +
                    3.0*std::pow(c2_val, 2.0)*d1_val + 
                    std::pow(d2_val, 4.0)*d1_val;
      
      mu12_mu22 = c1_val*c2_val + 
                  c1_val*d2_val*d2_val + 
                  2.0*rho*rho*c1_val*c2_val + 
                  4.0*d1_val*d2_val*std::sqrt(c1_val*c2_val)*rho +
                  c2_val*d1_val*d1_val + 
                  d1_val*d1_val*d2_val*d2_val;
      
      mu12_mu23 = 6.0*rho*rho*c1_val*c2_val*d2_val + //
                    6.0*rho*std::sqrt(c1_val*c2_val)*c2_val*d1_val + //
                    6.0*rho*std::sqrt(c1_val*c2_val)*d1_val*std::pow(d2_val, 2.0) + //
                    3.0*c1_val*c2_val*d2_val + //
                    std::pow(d2_val, 3.0)*c1_val + //
                    3.0*c2_val*std::pow(d1_val, 2.0)*d2_val + //
                    std::pow(d1_val*d2_val, 2.0)*d2_val; //
      
      
      mu12_mu24 = 12.0*rho*rho*c1_val*std::pow(c2_val, 2.0) + //
        12.0*rho*rho*c1_val*c2_val*std::pow(d2_val, 2.0) + //
        24.0*rho*std::sqrt(c1_val*c2_val)*c2_val*d1_val*d2_val + //
        8.0*rho*std::sqrt(c1_val*c2_val)*d1_val*std::pow(d2_val, 3.0) + //
        3.0*c1_val*std::pow(c2_val, 2.0) + //
        6.0*c1_val*c2_val*std::pow(d2_val, 2.0) + //
        c1_val*std::pow(d2_val, 4.0) + //
        3.0*std::pow(c2_val, 2.0)*std::pow(d1_val, 2.0) + //
        6.0*c2_val*std::pow(d1_val*d2_val, 2.0) + //
        std::pow(d1_val*d2_val*d2_val, 2.0); //
      
      mu13_mu23 = 6.0*std::pow(rho, 3.0)*std::pow(c1_val*c2_val, 1.5) + //
                    18.0*rho*rho*c1_val*c2_val*d1_val*d2_val + // 
                    9.0*rho*std::pow(c1_val*c2_val, 1.5) + // 
                    9.0*rho*std::sqrt(c1_val*c2_val)*c1_val*std::pow(d2_val, 2.0)+ //
                    9.0*rho*std::sqrt(c1_val*c2_val)*c2_val*std::pow(d1_val, 2.0)+ //
                    9.0*rho*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 2.0) +//
                    9.0*c1_val*c2_val*d1_val*d2_val + //
                    3.0*c1_val*d1_val*std::pow(d2_val, 3.0) + // 
                    3.0*c2_val*d2_val*std::pow(d1_val, 3.0) + //
                    std::pow(d1_val*d2_val, 3.0); //
                  
      mu13_mu24 = 24.0*std::pow(rho, 3.0)*std::pow(c1_val*c2_val, 1.5)*d2_val + //
                    36.0*rho*rho*c1_val*std::pow(c2_val, 2.0)*d1_val + //
                    36.0*rho*rho*c1_val*c2_val*d1_val*std::pow(d2_val, 2.0) + //
                    36.0*rho*std::pow(c1_val*c2_val, 1.5)*d2_val + //
                    12.0*rho*std::sqrt(c1_val*c2_val)*c1_val*std::pow(d2_val, 3.0) + //
                    36.0*rho*std::sqrt(c1_val*c2_val)*c2_val*std::pow(d1_val, 2.0)*d2_val + //
                    12.0*rho*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 2.0)*d2_val + //
                    9.0*c1_val*std::pow(c2_val, 2.0)*d1_val + //
                    18.0*c1_val*c2_val*d1_val*std::pow(d2_val, 2.0) + //
                    3.0*c1_val*d1_val*std::pow(d2_val, 4.0) + //
                    3.0*std::pow(c2_val, 2.0)*std::pow(d1_val, 3.0) + //
                    6.0*c2_val*std::pow(d1_val*d2_val, 2.0)*d1_val + //
                    std::pow(d1_val*d2_val, 3.0)*d2_val; //
      
      mu14_mu24 = 24.0*std::pow(rho, 4.0)*std::pow(c1_val*c2_val, 2.0) + //
                      96.0*std::pow(rho, 3.0)*std::pow(c1_val*c2_val, 1.5)*d1_val*d2_val + //
                      72.0*rho*rho*std::pow(c1_val*c2_val, 2.0) + //
                      72.0*rho*rho*c1_val*std::pow(c2_val, 2.0)*std::pow(d1_val, 2.0) + //
                      72.0*rho*rho*c2_val*std::pow(c1_val, 2.0)*std::pow(d2_val, 2.0) + //
                      72.0*rho*rho*c1_val*c2_val*std::pow(d1_val*d2_val, 2.0) + //
                      144.0*rho*std::pow(c1_val*c2_val, 1.5)*d1_val*d2_val + //
                      48.0*rho*std::sqrt(c1_val*c2_val)*c1_val*d1_val*std::pow(d2_val, 3.0) + //
                      48.0*rho*std::sqrt(c1_val*c2_val)*c2_val*d2_val*std::pow(d1_val, 3.0) + //
                      16.0*rho*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 3.0) + //
                      9.0*std::pow(c1_val*c2_val, 2.0) + //
                      18.0*c1_val*std::pow(c2_val, 2.0)*std::pow(d1_val, 2.0) + //
                      18.0*c2_val*std::pow(c1_val, 2.0)*std::pow(d2_val, 2.0) + //
                      36.0*c1_val*c2_val*std::pow(d1_val*d2_val, 2.0) + //
                      3.0*std::pow(c1_val, 2.0)*std::pow(d2_val, 4.0) + //
                      3.0*std::pow(c2_val, 2.0)*std::pow(d1_val, 4.0) + //
                      6.0*c1_val*std::pow(d1_val, 2.0)*std::pow(d2_val, 4.0) + //
                      6.0*c2_val*std::pow(d2_val, 2.0)*std::pow(d1_val, 4.0) + //
                      std::pow(d1_val*d2_val, 4.0); //
      
      mu13_mu22 = 6.0*rho*rho*c1_val*c2_val*d1_val + //
                    6.0*rho*std::sqrt(c1_val*c2_val)*c1_val*d2_val + //
                    6.0*rho*std::sqrt(c1_val*c2_val)*d2_val*std::pow(d1_val, 2.0) + // 
                    3.0*c1_val*c2_val*d1_val + // 
                    std::pow(d1_val, 3.0)*c2_val + //
                    3.0*c1_val*std::pow(d2_val, 2.0)*d1_val + //
                    std::pow(d1_val*d2_val, 2.0)*d1_val; //
      
      mu14_mu22 = 12.0*rho*rho*c2_val*std::pow(c1_val, 2.0) + // checked
        12.0*rho*rho*c1_val*c2_val*std::pow(d1_val, 2.0) +
        24.0*rho*std::sqrt(c1_val*c2_val)*c1_val*d1_val*d2_val +
        8.0*rho*std::sqrt(c1_val*c2_val)*d2_val*std::pow(d1_val, 3.0) +
        3.0*c2_val*std::pow(c1_val, 2.0) +
        6.0*c1_val*c2_val*std::pow(d1_val, 2.0) +
        c2_val*std::pow(d1_val, 4.0) +
        3.0*std::pow(c1_val, 2.0)*std::pow(d2_val, 2.0) +
        6.0*c1_val*std::pow(d1_val*d2_val, 2.0) +
        std::pow(d1_val*d1_val*d2_val, 2.0);

      
      mu14_mu23 = 24.0*std::pow(rho, 3.0)*std::pow(c1_val*c2_val, 1.5)*d1_val + // checked
        36.0*rho*rho*c2_val*std::pow(c1_val, 2.0)*d2_val +
        36.0*rho*rho*c1_val*c2_val*d2_val*std::pow(d1_val, 2.0) +
        36.0*rho*std::pow(c1_val*c2_val, 1.5)*d1_val +
        12.0*rho*std::sqrt(c1_val*c2_val)*c2_val*std::pow(d1_val, 3.0) +
        36.0*rho*std::sqrt(c1_val*c2_val)*c1_val*std::pow(d2_val, 2.0)*d1_val +
        12.0*rho*std::sqrt(c1_val*c2_val)*std::pow(d1_val*d2_val, 2.0)*d1_val +
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
      
      arma::mat limits = bothMissInt_ord_limits(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                                                gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                                                tau2_1, tau2_2, 
                                                lowLS1, lowLS2,
                                                highLS1, highLS2, R1, R2); 
      
      double denomIntegral = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                                             tau2_1, tau2_2, 
                                             /* s1 power, s2 power */      0, 0,
                                             limits(0, 0), limits(1, 0),
                                             limits(0, 1), limits(1, 1), nDivisions);
      mu1  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                                             tau2_1, tau2_2, 
                                             /* s1 power, s2 power */      1, 0,
                                             limits(0, 0), limits(1, 0),
                                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      2, 0,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_3  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      3, 0,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_4  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      4, 0,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      0, 1,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2_2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      0, 2,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2_3  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      0, 3,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu2_4  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      0, 4,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      1, 1,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      2, 1,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      3, 1,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu2  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      4, 1,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu22  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      1, 2,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu23  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      1, 3,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu1_mu24  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      1, 4,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu22  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      2, 2,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu23  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      2, 3,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu12_mu24  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      2, 4,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu22  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      3, 2,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu23  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      3, 3,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu13_mu24  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                             gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                             tau2_1, tau2_2, 
                             /* s1 power, s2 power */      3, 4,
                             limits(0, 0), limits(1, 0),
                             limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu22  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                                   gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                                   tau2_1, tau2_2, 
                                   /* s1 power, s2 power */      4, 2,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu23  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                                   gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                                   tau2_1, tau2_2, 
                                   /* s1 power, s2 power */      4, 3,
                                   limits(0, 0), limits(1, 0),
                                   limits(0, 1), limits(1, 1), nDivisions) / denomIntegral;
      mu14_mu24  = bothMissInt_ord(Y, beta, Z, G, sig2,gamma_1, psi_1, gamma_tilde, psi_tilde, 
                                   gamma_2, psi_2, h1, h2, ksi_1, ksi_2, X1, X2,
                                   tau2_1, tau2_2, 
                                   /* s1 power, s2 power */      4, 4,
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
    return U_mat;
  }
  else
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

  return U_mat;
}

// Functions exist for U matrix and V matrix, each for a single observation;
//[[Rcpp::export]]
arma::mat calc_OMEGA_twoseq(const arma::vec & Y, const arma::vec & G, const arma::vec & S1, const arma::vec & R1,
                        const arma::vec & S2, const arma::vec & R2,
                const arma::mat & Z, const arma::mat & X1, const arma::mat & X2,
                const arma::vec & lowerS1, const arma::vec & upperS1,
                const arma::vec & lowerS2, const arma::vec & upperS2,
                const arma::vec & beta, const double & gamma_tilde, const double & psi_tilde,
                const double & gamma_1, const double & psi_1, const double & gamma_2, const double & psi_2, 
                const double & h1, const double & h2, 
                const double & sig2, const double & tau2_1, const double & tau2_2,
                const arma::vec & ksi_1, const arma::vec & ksi_2,
                bool int_gs1 = true, bool int_gs1_Y = true, bool int_gs2 = true,
                bool int_s1s2 = false, bool int_gs1s2 = false,
                const int & nDivisions = 5, 
                const int MEASURE_TYPE_KNOWN = 1,
                const int MEASURE_TYPE_MISSING = 0,
                const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  const unsigned int N = Y.n_elem;
  arma::mat Zuse;
  arma::mat X1use;
  arma::mat X2use;

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
  if(X1.n_elem < N){
    X1use = arma::ones(N, 2);
    X1use.submat(0, 1, N-1, 1) = G;
  }else{
    X1use = arma::zeros(N, X1.n_cols + 1);
    X1use.submat(0, 0, N-1, 0) = X1.submat(0, 0, N-1, 0);
    X1use.submat(0, 1, N-1, 1) = G;
    X1use.submat(0, 2, N-1, X1use.n_cols - 1) = X1.submat(0, 1, N-1, X1.n_cols - 1);
  }
  if(X2.n_elem < N){
    X2use = arma::ones(N, 2);
    X2use.submat(0, 1, N-1, 1) = G;
  }else{
    X2use = arma::zeros(N, X2.n_cols + 1);
    X2use.submat(0, 0, N-1, 0) = X2.submat(0, 0, N-1, 0);
    X2use.submat(0, 1, N-1, 1) = G;
    X2use.submat(0, 2, N-1, X2use.n_cols - 1) = X2.submat(0, 1, N-1, X2.n_cols - 1);
  }
  const unsigned int Z_VARS = Zuse.n_cols;
  const unsigned int X1_VARS = X1use.n_cols;
  const unsigned int X2_VARS = X2use.n_cols;

  // Update expectation using coverged estimates;
  List expRes = calc_expectation_twoseq(gamma_1, psi_1, gamma_2, psi_2, 
                          gamma_tilde, psi_tilde, h1, h2, 
                          sig2, tau2_1, tau2_2,
                          Y, Zuse, X1use, X2use, G,
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
  arma::mat Q = arma::zeros( Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1); // beta, 2 gamma, 2 psi, 2 h, sig, ksi2, gamma, psi, tau, ksi1, tau
  arma::mat Qalt = arma::zeros( Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1);
  Q = calc_Q_matrix_twoseq( sig2, tau2_1, tau2_2, Zuse, X1use, X2use, G,
                       expectS1, expectS1_sq,
                       expectS2, expectS2_sq,
                       expectS1S2, expectS12S2,
                       expectS1S22, expectS12S22, 
                       int_gs1, int_gs1_Y, int_gs2, int_s1s2, int_gs1s2);
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
      arma::mat V = arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, 3);
      arma::mat uMat = arma::zeros(3, 3);
      arma::vec uVec = arma::zeros(3);

      uVec(0) = 1;
      uVec(1) = expectS1(index);
      uVec(2) = expectS1_sq(index);

      V = calc_V_mat_twoseq_S1(gamma_tilde, psi_tilde, gamma_2, psi_2, gamma_1, psi_1, 
                               h1, h2, sig2, tau2_1, tau2_2,
                           Y(index), G(index), S2(index), beta, ksi_1, ksi_2,
                           Zuse.row(index), X1use.row(index), X2use.row(index));

      uMat = calc_U_mat_twoseq_S1miss(gamma_tilde, psi_tilde, gamma_2, psi_2, gamma_1, psi_1, 
                                      h1, h2, sig2, tau2_1, tau2_2,
                                      Y(index), G(index), S2(index), R1(index), beta, ksi_1, ksi_2,
                                      Zuse.row(index), X1use.row(index), X2use.row(index),
                                      lowerS1(index), upperS1(index),
                                MEASURE_TYPE_KNOWN,
                                MEASURE_TYPE_MISSING,
                                MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                MEASURE_TYPE_ABOVE_DETECTION_LIMIT);
  
      Qalt += V * uMat * trans(V) - (V*uVec * trans(V*uVec));
    }
    if( R1(index) == MEASURE_TYPE_KNOWN & R2(index) ==  MEASURE_TYPE_MISSING ){
      arma::mat V = arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, 3);
      arma::mat uMat = arma::zeros(3, 3);
      arma::vec uVec = arma::zeros(3);

      uVec(0) = 1;
      uVec(1) = expectS2(index);
      uVec(2) = expectS2_sq(index);

        
      V = calc_V_mat_twoseq_S2(gamma_tilde, psi_tilde, gamma_2, psi_2, gamma_1, psi_1, 
                               h1, h2, sig2, tau2_1, tau2_2,
                               Y(index), G(index), S1(index), beta, ksi_1, ksi_2,
                               Zuse.row(index), X1use.row(index), X2use.row(index));
  
      uMat = calc_U_mat_twoseq_S2miss(gamma_tilde, psi_tilde, gamma_2, psi_2, gamma_1, psi_1, 
                                      h1, h2, sig2, tau2_1, tau2_2,
                                      Y(index), G(index), S1(index), R2(index), beta, ksi_1, ksi_2,
                                      Zuse.row(index), X2use.row(index), lowerS2(index), upperS2(index),
                                MEASURE_TYPE_KNOWN,
                                MEASURE_TYPE_MISSING,
                                MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

      Qalt += V * uMat * trans(V) - (V*uVec * trans(V*uVec));
    }
    if( R1(index) == MEASURE_TYPE_MISSING  & R2(index) ==  MEASURE_TYPE_MISSING ){
      arma::mat V = arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, 9);
      arma::mat uMat = arma::zeros(9, 9);
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
        
      V = calc_V_mat_twoseq_S1S2(gamma_tilde, psi_tilde, gamma_2, psi_2, gamma_1, psi_1, h1, h2, 
                                 sig2, tau2_1, tau2_2,
                             Y(index), G(index), beta, ksi_1, ksi_2, Zuse.row(index), X1use.row(index), X2use.row(index));
//Rcout << "new V" << index<< "\n" << V.submat(0, 0, V.n_rows - 1, V.n_cols - 1) << "\n";      
      uMat = calc_U_mat_twoseq_twomiss(gamma_tilde, psi_tilde, gamma_2, psi_2, gamma_1, psi_1, h1, h2, 
                                       sig2, tau2_1, tau2_2,
                                Y(index), G(index), R1(index), R2(index),
                                beta, ksi_1, ksi_2,
                                Zuse.row(index), X1use.row(index), X2use.row(index),
                                lowerS1(index), upperS1(index), lowerS2(index), upperS2(index),
                                nDivisions, 
                                MEASURE_TYPE_KNOWN,
                                MEASURE_TYPE_MISSING,
                                MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                MEASURE_TYPE_ABOVE_DETECTION_LIMIT);
      
      Qalt += V * uMat * trans(V) - (V*uVec * trans(V*uVec));
    }
  }
  arma::mat omega = Q - Qalt;
  arma::mat omegaInv;
  
  if(!int_gs1){
    omega.submat(Z_VARS + 7 + X2_VARS + 1, 0, Z_VARS + 7 + X2_VARS + 1, omega.n_cols - 1) = arma::zeros(1, omega.n_cols); // beta, gamma1, gamma2
    omega.submat(0, Z_VARS + 7 + X2_VARS + 1, omega.n_cols - 1, Z_VARS + 7 + X2_VARS + 1) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS + 7 + X2_VARS + 1, Z_VARS + 7 + X2_VARS + 1) = 1;
  }
  if(!int_gs1_Y){
    omega.submat(Z_VARS + 1, 0, Z_VARS+1, omega.n_cols - 1) = arma::zeros(1, omega.n_cols); // beta, gamma1, gamma2
    omega.submat(0, Z_VARS+1, omega.n_cols - 1, Z_VARS+1) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS+1,Z_VARS+1) = 1;
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
  
  bool badSig = inv(omegaInv, omega);
  if(badSig){
    /* if no interaction, force variance to 0 */
    
    if(!int_gs1){
      omegaInv(Z_VARS + 7 + X2_VARS + 1,Z_VARS + 7 + X2_VARS + 1) = 0;
    }
    if(!int_gs1_Y){
      omegaInv(Z_VARS+1,Z_VARS+1) = 0;
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
    return arma::zeros(Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1, Z_VARS + 7 + X2_VARS + 3 + X1_VARS + 1);
  }
}



///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
////////////////////////// Confidence Intervals ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     Delta Method    //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List deltaCI_twoseq(const long double & mu1, const long double & sig1,
                 const long double & mu2, const long double & sig2,
                 const long double & mu2b, const long double & sig2b,
                 const long double & mu3, const long double & sig3,
                 const long double & mu3b, const long double & sig3b,
                 const long double & sig12, const long double & sig12b, 
                 const long double & sig13, const long double & sig13b, 
                 const long double & sig22b, const long double & sig23,
                 const long double & sig23b,
                 const long double & sig2b3, const long double & sig2b3b,
                 const long double & sig33b,
                 const int & indL = 1, const double alpha = 0.05){
  double zval = R::qnorm(1.0 - alpha/2.0, 0, 1, 1, 0);
  
  double g1 = (mu2 + mu2b*indL)*(mu3 + mu3b*indL);
  double g2 = mu1*(mu3 + mu3b*indL);
  double g2b = indL*g2;
  double g3 = mu1*(mu2 + mu2b*indL);
  double g3b = indL*g3;
  
  double p1  = g1*sig1   + g2*sig12  + g2b*sig12b  + g3*sig13  + g3b*sig13b;
  double p2  = g1*sig12  + g2*sig2   + g2b*sig22b  + g3*sig23  + g3b*sig23b;
  double p2b = g1*sig12b + g2*sig22b + g2b*sig2b   + g3*sig2b3 + g3b*sig2b3b;
  double p3  = g1*sig13  + g2*sig23  + g2b*sig2b3  + g3*sig3   + g3b*sig33b;
  double p3b = g1*sig13b + g2*sig23b + g2b*sig2b3b + g3*sig33b + g3b*sig3b;
  
  double deltaSE = std::sqrt( g1*p1 + g2*p2 + g2b*p2b + g3*p3 + g3b*p3b );

  arma::vec CI = arma::zeros(2);
  CI(0) = mu1*(mu2+indL*mu2b)*(mu3+indL*mu3b) - zval*deltaSE;
  CI(1) = mu1*(mu2+indL*mu2b)*(mu3+indL*mu3b) + zval*deltaSE;

  long double pval = R::pnorm(mu1*(mu2+indL*mu2b)*(mu3+indL*mu3b) / deltaSE, 0, 1, 0, 0);

  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     Bootstrap Method    //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List bootstrapCI_twoseq(const arma::vec & y, const arma::vec & g, const arma::vec & s1, const arma::vec & r1,
                     const arma::vec & s2, const arma::vec & r2,
                     const arma::mat & Z, const arma::mat & X1, const arma::mat & X2,
                     const arma::vec & lowerS1, const arma::vec & upperS1,
                     const arma::vec & lowerS2, const arma::vec & upperS2, const double & delta,
                     const double & alpha,                        
                     bool int_gs1 = true, bool int_gs1_Y = true, bool int_gs2 = true,
                     bool int_s1s2 = false, bool int_gs1s2 = false,
                     const double & indL = 1, const unsigned int bootStrapN = 1000,
                     const double convLimit = 1e-4, const double iterationLimit = 1e4,
                     const int & nDivisions = 5, 
                     const int MEASURE_TYPE_KNOWN = 1,
                     const int MEASURE_TYPE_MISSING = 0,
                     const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                     const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  unsigned int N = y.n_elem;
  arma::uvec bootSample(N);
  arma::uvec pvalS1;

  double lowCut = alpha/2.0;
  unsigned int lowCutVal = round(bootStrapN * lowCut) - 1;
  double highCut = 1 - lowCut;
  unsigned int highCutVal = round(bootStrapN * highCut) + 1;
  
  if(highCutVal >= bootStrapN | lowCutVal <= 0){
    highCutVal = bootStrapN - 1;
    lowCutVal = 0;
    Rcout << "Too few iterations used for accurate Bootstrap CI\n";
  }

  IntegerVector toSample =  seq(1, N) - 1;

  arma::vec deltaHats = arma::zeros(bootStrapN);
  arma::vec CI = arma::zeros(2);
  double deltaSE = 0;

  int badSample = 0;
  // Add something in this loop for the bootstrap sample to make sure it is valid;
  for(int i = 0; i < bootStrapN; i++){
    for(int j = 0; j < N; j++){
      bootSample(j) = R::runif(0,N-1);
    }
    /// Check to see if new X and Z are inevitable. If not, decrease i and start sample over
    arma::mat zBoot = Z.rows(bootSample);
    arma::mat x1Boot = X1.rows(bootSample);
    arma::mat x2Boot = X2.rows(bootSample);
    arma::vec gBoot = g.elem(bootSample);
    
    arma::mat zBootCheck = arma::zeros(N, zBoot.n_cols + 1);
    zBootCheck.submat(0, 0, N-1, zBoot.n_cols - 1) = zBoot;
    zBootCheck.submat(0, zBoot.n_cols, N-1, zBoot.n_cols) = gBoot;
    
    arma::mat x1BootCheck = arma::zeros(N, x1Boot.n_cols + 1);
    x1BootCheck.submat(0, 0, N-1, x1Boot.n_cols - 1) = x1Boot;
    x1BootCheck.submat(0, x1Boot.n_cols, N-1, x1Boot.n_cols) = gBoot;
    
    arma::mat x2BootCheck = arma::zeros(N, x2Boot.n_cols + 1);
    x2BootCheck.submat(0, 0, N-1, x2Boot.n_cols - 1) = x2Boot;
    x2BootCheck.submat(0, x2Boot.n_cols, N-1, x2Boot.n_cols) = gBoot;
    
    arma::mat t_X1_inv;
    arma::mat t_X2_inv;
    arma::mat t_Z_inv;
    
    bool x1Check = arma::inv(t_X1_inv, trans(x1BootCheck)*x1BootCheck);
    bool x2Check = arma::inv(t_X2_inv, trans(x2BootCheck)*x2BootCheck);
    bool zCheck = arma::inv(t_Z_inv, trans(zBootCheck)*zBootCheck);
    if(!x1Check | !x2Check | !zCheck){
      i--;
    }else{
      arma::vec yBoot = y.elem(bootSample);
      arma::vec s1Boot = s1.elem(bootSample);
      arma::vec s2Boot = s2.elem(bootSample);
      arma::vec r1Boot = r1.elem(bootSample);
      arma::vec r2Boot = r2.elem(bootSample);
      arma::vec ls1Boot = lowerS1.elem(bootSample);
      arma::vec us1Boot = upperS1.elem(bootSample);
      arma::vec ls2Boot = lowerS2.elem(bootSample);
      arma::vec us2Boot = upperS2.elem(bootSample);
      
      List itEM_res = twoSeqMed_EM(yBoot, gBoot, s1Boot, r1Boot, s2Boot, r2Boot, zBoot, x1Boot, x2Boot,
                                 ls1Boot, us1Boot, ls2Boot, us2Boot, int_gs1, int_gs1_Y, int_gs2, int_s1s2, int_gs1s2, 
                                 convLimit, iterationLimit, nDivisions, 
                                 MEASURE_TYPE_KNOWN, MEASURE_TYPE_MISSING,
                                 MEASURE_TYPE_BELOW_DETECTION_LIMIT, MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

    arma::vec tKsi1 = itEM_res["ksi_1"];
    double tGamma1 = itEM_res["gamma_1"];
    double tPsi1 = itEM_res["psi_1"];
    double tGamma2 = itEM_res["gamma_2"];
    double tPsi2 = itEM_res["psi_2"];


    deltaHats(i) = tKsi1(1)*(tGamma1 + indL*tPsi1)*(tGamma2 + indL*tPsi2);

    if(!arma::is_finite(deltaHats(i)) | isnan(deltaHats(i))){
      --i;
      ++badSample;
    }
    if(badSample >= (bootStrapN*1e3)){
      Rcout << "Too many bootstrap samples giving impossible results. returning with estimates available. Results may be inaccurate\n";
      if(i < 2){
        arma::vec CI = arma::zeros(2);
        CI(0) = -1e15;
        CI(1) = 1e15;
        double deltaSE = 1e15;
        double pval = 1;
        return List::create(Named("CI") = CI,
                            Named("deltaSE") = deltaSE,
                            Named("pval") = pval);
      }
      arma::vec temp2 = sort(deltaHats.subvec(0, i - 1));
      arma::vec CI = arma::zeros(2);
      CI(0) = temp2(std::max(floor(lowCutVal * i / bootStrapN), 0.0));
      CI(1) = temp2(std::min(ceil(highCutVal * i / bootStrapN), i - 1.0));
      double deltaSE = stddev(temp2);

      if(delta < 0){
        pvalS1 = arma::find(temp2 >= 0);
      }else if(delta > 0){
        pvalS1 = arma::find(temp2 <= 0);
      }else if(delta == 0){
        pvalS1 = arma::find(temp2 > -999999999999999999);
      }

      double pval = std::min(1.0, 2.0*s1.n_elem/i);

      return List::create(Named("CI") = CI,
                          Named("deltaSE") = deltaSE,
                          Named("pval") = pval);
      }
    }
      // If bad sample/no estimate, redo
  }

  arma::vec temp = sort(deltaHats);

  if(delta < 0){
    pvalS1 = arma::find(temp >= 0);
  }else if(delta > 0){
    pvalS1 = arma::find(temp <= 0);
  }else if(delta == 0){
    pvalS1 = arma::find(temp > -999999999999);
  }

  CI(0) = temp(lowCutVal);
  CI(1) = temp(highCutVal);
  deltaSE = stddev(deltaHats);

  double pval = std::min(1.0, 2.0*pvalS1.n_elem/bootStrapN);

  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////      MC Method     ?//////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// order for all components: alphaG, gamma1, psi1, gamma2, psi2
// [[Rcpp::export]]
List mcCI_twoseq(const arma::vec & mu, const arma::mat & sig, const double & delta,
                 const double & indL = 1.0, 
              const double nIt = 10000, const double alpha = 0.05){
  
  // mu and sig are of dimension 5 - alphaG, gamma1, psi1, gamma2, psi2;
  double outVal = std::ceil((alpha/2) * nIt);
  double itGen = outVal + 1;
  double itGen2 = itGen;
  double totGen = 0;
  double pMean, cMean = 0;
  double pVar, cVar = 0;
  double aMean = 0;
  double pval = 0;
  
  double indL1Mult = indL;
  double indL2Mult = indL;
  arma::mat sigUse = sig;
  
  if(sig(2, 2) == 0){
    indL1Mult = 0;
    sigUse(2, 2) = 1.0;
  }
  if(sig(4, 4) == 0){
    indL2Mult = 0;
    sigUse(4, 4) = 1.0;
  }
  
  //Rcout << sigUse << "\n";

  arma::mat tGen = rmvnorm(itGen2 * 2, mu, sigUse);
  //Rcout << "2\n";
  arma::vec genValues1 = tGen.col(0) % (tGen.col(1) + tGen.col(2)*indL1Mult) % (tGen.col(3) + tGen.col(4)*indL2Mult) ;
  arma::vec low = sort(genValues1);
  arma::vec high = sort(genValues1, "descend");

  pMean = arma::mean(genValues1);
  pVar = arma::var(genValues1);

  if(delta < 0){
    arma::uvec s1 = arma::find(genValues1 >= 0);
    pval += s1.n_elem;
  }else if(delta > 0){
    arma::uvec s1 = arma::find(genValues1 <= 0);
    pval += s1.n_elem;
  }else if(delta == 0){
    pval += genValues1.n_elem;
  }

  totGen = itGen2 * 2;
  while(totGen < nIt){
    itGen2 = std::min(nIt - totGen, itGen);

    arma::mat tGen = rmvnorm(itGen2, mu, sigUse);
    arma::vec genValues = tGen.col(0) % (tGen.col(1) + tGen.col(2)*indL1Mult) % (tGen.col(3) + tGen.col(4)*indL2Mult) ;

    low.subvec(low.n_elem - genValues.n_elem, low.n_elem - 1) = genValues;
    high.subvec(high.n_elem - genValues.n_elem, high.n_elem - 1) = genValues;

    low = sort(low);
    high = sort(high, "descend");

    cMean = arma::mean(genValues);
    cVar = arma::var(genValues);
    aMean = (cMean*itGen2 + pMean*totGen) / (totGen + itGen2);

    pVar = (((totGen * (pVar + std::pow(pMean, 2))) + (itGen2 * (cVar + std::pow(cMean, 2)))) /
      (totGen + itGen2)) - std::pow(aMean, 2);

    totGen += itGen2;
    pMean = aMean;

    if(delta < 0){
      arma::uvec s1 = arma::find(genValues >= 0);
      pval += s1.n_elem;
    }else if(delta > 0){
      arma::uvec s1 = arma::find(genValues <= 0);
      pval += s1.n_elem;
    }else if(delta == 0){
      pval += genValues.n_elem;
    }
  }

  arma::vec CI = arma::zeros(2);
  CI(0) = low(outVal);
  CI(1) = high(outVal);
  double deltaSE = std::sqrt(pVar * nIt / (nIt-1));
  pval = (2.0*pval) / (1.0 * nIt);

  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}

